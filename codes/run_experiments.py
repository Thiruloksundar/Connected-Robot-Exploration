"""
Experiment runner for the final project.

Runs 5 conditions × 3 maps × 3 seeds and collects metrics.
Generates comparison plots and (optionally) frame-by-frame videos.

Usage (from multi_robot_project/):
    python -m final_project.run_experiments
    python -m final_project.run_experiments --quick   # fast debug run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Ensure project root is importable ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.env import GridWorldEnv
from simulation.metrics import compute_coverage
from simulation.sensors import reveal_within_radius
from simulation.types import (
    ACTION_DELTAS,
    Action,
    CellState,
    EnvConfig,
    Observation,
    RobotState,
)

from final_project.maps import (
    generate_corridor_map,
    generate_open_map,
    generate_room_map,
)
from final_project.connectivity import algebraic_connectivity, is_connected
from final_project.pipeline import ExplorationPipeline, PipelineConfig


# ─── Custom env that accepts a pre-built truth grid ───────────

class CustomGridWorldEnv:
    """Lightweight env wrapper that uses a pre-built truth grid and starts."""

    def __init__(self, truth: np.ndarray, starts: List[Tuple[int, int]],
                 sensor_radius: int = 4, max_steps: int = 500,
                 allow_overlap: bool = False):
        self.truth = truth.copy()
        self.height, self.width = truth.shape
        self.sensor_radius = sensor_radius
        self.max_steps = max_steps
        self.allow_overlap = allow_overlap
        self._starts = starts

        self.robots: List[RobotState] = []
        self.fused_map: Optional[np.ndarray] = None
        self.robot_maps: List[np.ndarray] = []
        self.step_index: int = 0

    def reset(self) -> Observation:
        self.robots = [RobotState(x=x, y=y) for (x, y) in self._starts]
        self.fused_map = np.full((self.height, self.width),
                                 int(CellState.UNKNOWN), dtype=np.int8)
        self.robot_maps = [
            np.full((self.height, self.width), int(CellState.UNKNOWN), dtype=np.int8)
            for _ in self.robots
        ]
        self.step_index = 0

        # Initial reveal
        for i, robot in enumerate(self.robots):
            reveal_within_radius(self.truth, self.robot_maps[i],
                                 [(robot.x, robot.y)], self.sensor_radius)
            reveal_within_radius(self.truth, self.fused_map,
                                 [(robot.x, robot.y)], self.sensor_radius)
        return self._observe()

    def _observe(self) -> Observation:
        return Observation(
            fused_map=self.fused_map,
            robot_positions=[(r.x, r.y) for r in self.robots],
            step_index=self.step_index,
            max_steps=self.max_steps,
        )

    def step(self, actions: List[Action]):
        intended: List[Tuple[int, int]] = []
        for robot, action in zip(self.robots, actions):
            dx, dy = ACTION_DELTAS[action]
            nx, ny = robot.x + dx, robot.y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                nx, ny = robot.x, robot.y
            intended.append((nx, ny))

        # Block moves into obstacles
        for idx, (nx, ny) in enumerate(intended):
            if int(self.truth[ny, nx]) == int(CellState.OCCUPIED):
                intended[idx] = (self.robots[idx].x, self.robots[idx].y)

        # Handle robot-robot collisions
        if not self.allow_overlap:
            seen = {}
            for i, pos in enumerate(intended):
                if pos in seen:
                    j = seen[pos]
                    intended[i] = (self.robots[i].x, self.robots[i].y)
                    intended[j] = (self.robots[j].x, self.robots[j].y)
                else:
                    seen[pos] = i

        # Apply moves
        for robot, (nx, ny) in zip(self.robots, intended):
            if (nx, ny) != (robot.x, robot.y):
                robot.distance_traveled += 1
            robot.x, robot.y = nx, ny

        # Reveal
        for i, robot in enumerate(self.robots):
            reveal_within_radius(self.truth, self.robot_maps[i],
                                 [(robot.x, robot.y)], self.sensor_radius)

        # Recompute fused map
        np.copyto(self.fused_map, np.maximum.reduce(self.robot_maps))

        self.step_index += 1
        coverage = compute_coverage(self.fused_map)
        done = self.step_index >= self.max_steps or coverage >= 0.999

        return self._observe(), coverage, done

    def get_positions(self) -> List[Tuple[int, int]]:
        return [(r.x, r.y) for r in self.robots]


# ─── Experiment result ────────────────────────────────────────

@dataclass
class ExperimentResult:
    condition: str
    map_name: str
    seed: int
    steps_to_90: int           # -1 if not reached
    final_coverage: float
    total_steps: int
    min_lambda2: float
    mean_lambda2: float
    formation_error_mean: float
    disconnections: int        # number of steps where graph was disconnected
    coverage_curve: List[float]
    lambda2_curve: List[float]
    formation_error_curve: List[float]


# ─── Map generators ──────────────────────────────────────────

MAP_GENERATORS = {
    "corridor": generate_corridor_map,
    "rooms": generate_room_map,
    "open": generate_open_map,
}


def _generate_map(map_name: str, num_robots: int, seed: int):
    if map_name == "corridor":
        return generate_corridor_map(width=50, height=50, num_robots=num_robots, seed=seed)
    elif map_name == "rooms":
        return generate_room_map(width=50, height=50, num_robots=num_robots, seed=seed)
    elif map_name == "open":
        return generate_open_map(width=50, height=50, num_robots=num_robots, seed=seed)
    else:
        raise ValueError(f"Unknown map: {map_name}")


# ─── Condition configs ────────────────────────────────────────

def _make_config(condition: str, r_comm: int = 10, alpha: float = 0.35,
                 num_robots: int = 4, conn_penalty_weight: float = 10.0) -> PipelineConfig:
    cfg = PipelineConfig(
        condition=condition,
        alpha=alpha,
        r_comm=r_comm,
        sensor_radius=4,
        replanning_period=8,
        cohesion_radius=float(r_comm) * 2.0,  # allow relay-chain spread before cohesion fires
        conn_penalty_weight=conn_penalty_weight,
    )
    if condition == "adaptive_adversarial":
        cfg.adversarial_robot = num_robots - 1  # last robot is adversarial
    return cfg


# ─── Single experiment run ────────────────────────────────────

def run_single(
    condition: str,
    map_name: str,
    seed: int,
    num_robots: int = 4,
    max_steps: int = 500,
    r_comm: int = 8,
    alpha: float = 0.5,
    save_frames: bool = False,
    output_dir: str = "results",
    conn_penalty_weight: float = 10.0,
) -> ExperimentResult:
    """Run one experiment (one condition, one map, one seed)."""
    # Generate map
    truth, starts = _generate_map(map_name, num_robots, seed)

    # Create env and pipeline
    env = CustomGridWorldEnv(truth, starts, sensor_radius=4, max_steps=max_steps)
    cfg = _make_config(condition, r_comm=r_comm, alpha=alpha,
                       num_robots=num_robots,
                       conn_penalty_weight=conn_penalty_weight)
    pipeline = ExplorationPipeline(cfg)

    obs = env.reset()
    pipeline.reset(obs)

    frames = [] if save_frames else None
    steps_to_90 = -1

    for step in range(max_steps):
        actions = pipeline.act(obs)
        obs, coverage, done = env.step(actions)

        if steps_to_90 < 0 and coverage >= 0.90:
            steps_to_90 = step + 1

        if save_frames and step % 5 == 0:
            frames.append({
                "step": step + 1,
                "positions": env.get_positions(),
                "coverage": coverage,
                "fused_map": env.fused_map.copy(),
            })

        if done:
            break

    # Compile results
    cov_curve = pipeline.coverage_history
    lam2_curve = pipeline.lambda2_history
    ferr_curve = pipeline.formation_error_history

    result = ExperimentResult(
        condition=condition,
        map_name=map_name,
        seed=seed,
        steps_to_90=steps_to_90,
        final_coverage=cov_curve[-1] if cov_curve else 0.0,
        total_steps=env.step_index,
        min_lambda2=float(np.min(lam2_curve)) if lam2_curve else 0.0,
        mean_lambda2=float(np.mean(lam2_curve)) if lam2_curve else 0.0,
        formation_error_mean=float(np.mean(ferr_curve)) if ferr_curve else 0.0,
        disconnections=sum(1 for c in pipeline.connectivity_maintained if not c),
        coverage_curve=cov_curve,
        lambda2_curve=lam2_curve,
        formation_error_curve=ferr_curve,
    )

    # Optionally save frames for video generation
    if save_frames and frames:
        frame_dir = os.path.join(output_dir, "frames",
                                 f"{condition}_{map_name}_s{seed}")
        os.makedirs(frame_dir, exist_ok=True)
        for f in frames:
            np.save(os.path.join(frame_dir, f"fused_{f['step']:04d}.npy"),
                    f["fused_map"])

    return result


# ─── Batch runner ─────────────────────────────────────────────

CONDITIONS = [
    "pure_frontier",
    "frontier_conn",
    "rigid_formation",
    "adaptive_formation",
    "adaptive_adversarial",
]

MAP_NAMES = ["corridor", "rooms", "open"]
SEEDS = [0, 1, 2]


def run_all_experiments(
    conditions: List[str] = CONDITIONS,
    map_names: List[str] = MAP_NAMES,
    seeds: List[int] = SEEDS,
    num_robots: int = 4,
    max_steps: int = 500,
    r_comm: int = 8,
    alpha: float = 0.5,
    save_frames: bool = False,
    output_dir: str = "results",
    conn_penalty_weight: float = 10.0,
) -> List[ExperimentResult]:
    """Run full experiment grid."""
    os.makedirs(output_dir, exist_ok=True)
    results: List[ExperimentResult] = []
    total = len(conditions) * len(map_names) * len(seeds)
    count = 0

    for condition in conditions:
        for map_name in map_names:
            for seed in seeds:
                count += 1
                print(f"[{count}/{total}] {condition} | {map_name} | seed={seed} ...",
                      end=" ", flush=True)
                t0 = time.time()
                result = run_single(
                    condition=condition,
                    map_name=map_name,
                    seed=seed,
                    num_robots=num_robots,
                    max_steps=max_steps,
                    r_comm=r_comm,
                    alpha=alpha,
                    save_frames=save_frames,
                    output_dir=output_dir,
                    conn_penalty_weight=conn_penalty_weight,
                )
                dt = time.time() - t0
                print(f"done in {dt:.1f}s | cov={result.final_coverage:.3f} "
                      f"| s90={result.steps_to_90} | min_λ2={result.min_lambda2:.3f}")
                results.append(result)

    # Save summary
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    summary = []
    for r in results:
        entry = {
            "condition": r.condition,
            "map": r.map_name,
            "seed": r.seed,
            "steps_to_90": r.steps_to_90,
            "final_coverage": round(r.final_coverage, 4),
            "total_steps": r.total_steps,
            "min_lambda2": round(r.min_lambda2, 4),
            "mean_lambda2": round(r.mean_lambda2, 4),
            "formation_error_mean": round(r.formation_error_mean, 4),
            "disconnections": r.disconnections,
        }
        summary.append(entry)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return results


# ─── Visualization ────────────────────────────────────────────

def plot_results(results: List[ExperimentResult], output_dir: str = "results"):
    """Generate comparison plots from experiment results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Group results by condition and map
    from collections import defaultdict
    by_condition_map = defaultdict(list)
    for r in results:
        by_condition_map[(r.condition, r.map_name)].append(r)

    conditions = list(dict.fromkeys(r.condition for r in results))
    maps = list(dict.fromkeys(r.map_name for r in results))

    cond_labels = {
        "pure_frontier": "Frontier Only",
        "frontier_conn": "Frontier+Conn",
        "rigid_formation": "Rigid Form+Conn",
        "adaptive_formation": "Adaptive Form+Conn",
        "adaptive_adversarial": "Adaptive+Adversary",
    }
    cond_colors = {
        "pure_frontier": "#e41a1c",
        "frontier_conn": "#377eb8",
        "rigid_formation": "#4daf4a",
        "adaptive_formation": "#984ea3",
        "adaptive_adversarial": "#ff7f00",
    }

    # ── Plot 1: Steps to 90% coverage (bar chart) ────────────
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 5), sharey=True)
    if len(maps) == 1:
        axes = [axes]

    for ax, map_name in zip(axes, maps):
        x_pos = np.arange(len(conditions))
        means = []
        stds = []
        for cond in conditions:
            vals = [r.steps_to_90 for r in by_condition_map[(cond, map_name)]
                    if r.steps_to_90 > 0]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(500)  # did not reach 90%
                stds.append(0)

        bars = ax.bar(x_pos, means, yerr=stds, capsize=4,
                      color=[cond_colors[c] for c in conditions], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([cond_labels[c] for c in conditions],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Map: {map_name}", fontsize=11)
        ax.set_ylabel("Steps to 90% coverage" if ax == axes[0] else "")
        ax.set_ylim(0, 550)

    fig.suptitle("Exploration Speed: Steps to 90% Coverage", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "steps_to_90.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved steps_to_90.png")

    # ── Plot 2: Min algebraic connectivity (bar chart) ────────
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 5), sharey=True)
    if len(maps) == 1:
        axes = [axes]

    for ax, map_name in zip(axes, maps):
        x_pos = np.arange(len(conditions))
        means = []
        stds = []
        for cond in conditions:
            vals = [r.min_lambda2 for r in by_condition_map[(cond, map_name)]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.bar(x_pos, means, yerr=stds, capsize=4,
               color=[cond_colors[c] for c in conditions], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([cond_labels[c] for c in conditions],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Map: {map_name}", fontsize=11)
        ax.set_ylabel("Min λ₂" if ax == axes[0] else "")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle("Connectivity Robustness: Minimum Algebraic Connectivity (λ₂)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "min_lambda2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved min_lambda2.png")

    # ── Plot 3: Coverage curves (time series) ─────────────────
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4.5), sharey=True)
    if len(maps) == 1:
        axes = [axes]

    for ax, map_name in zip(axes, maps):
        for cond in conditions:
            runs = by_condition_map[(cond, map_name)]
            # Average coverage curve across seeds (truncate to shortest)
            curves = [r.coverage_curve for r in runs if r.coverage_curve]
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            avg = np.mean([c[:min_len] for c in curves], axis=0)
            ax.plot(avg, label=cond_labels[cond], color=cond_colors[cond], linewidth=1.5)

        ax.set_title(f"Map: {map_name}", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Coverage" if ax == axes[0] else "")
        ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(0, 1.05)
        if ax == axes[-1]:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Coverage Over Time", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "coverage_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved coverage_curves.png")

    # ── Plot 4: λ₂ curves (time series) ──────────────────────
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4.5), sharey=True)
    if len(maps) == 1:
        axes = [axes]

    for ax, map_name in zip(axes, maps):
        for cond in conditions:
            runs = by_condition_map[(cond, map_name)]
            curves = [r.lambda2_curve for r in runs if r.lambda2_curve]
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            avg = np.mean([c[:min_len] for c in curves], axis=0)
            ax.plot(avg, label=cond_labels[cond], color=cond_colors[cond], linewidth=1.5)

        ax.set_title(f"Map: {map_name}", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("λ₂" if ax == axes[0] else "")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        if ax == axes[-1]:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Algebraic Connectivity (λ₂) Over Time", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lambda2_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved lambda2_curves.png")

    # ── Plot 5: Disconnections count ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = []
    disc_vals = []
    colors = []
    for map_name in maps:
        for cond in conditions:
            runs = by_condition_map[(cond, map_name)]
            avg_disc = np.mean([r.disconnections for r in runs])
            x_labels.append(f"{map_name}\n{cond_labels[cond]}")
            disc_vals.append(avg_disc)
            colors.append(cond_colors[cond])

    ax.bar(range(len(disc_vals)), disc_vals, color=colors, alpha=0.8)
    ax.set_xticks(range(len(disc_vals)))
    ax.set_xticklabels(x_labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Average Disconnections")
    ax.set_title("Number of Disconnection Events (avg over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "disconnections.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved disconnections.png")

    # ── Plot 6: Summary table as image ────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    # Build table data
    col_labels = ["Condition", "Map", "Steps→90%", "Coverage", "Min λ₂",
                  "Mean λ₂", "Form. Error", "Disconn."]
    table_data = []
    for map_name in maps:
        for cond in conditions:
            runs = by_condition_map[(cond, map_name)]
            s90 = np.mean([r.steps_to_90 for r in runs if r.steps_to_90 > 0]) \
                  if any(r.steps_to_90 > 0 for r in runs) else float('nan')
            cov = np.mean([r.final_coverage for r in runs])
            ml2 = np.mean([r.min_lambda2 for r in runs])
            al2 = np.mean([r.mean_lambda2 for r in runs])
            fe = np.mean([r.formation_error_mean for r in runs])
            dc = np.mean([r.disconnections for r in runs])
            table_data.append([
                cond_labels[cond], map_name,
                f"{s90:.0f}" if not np.isnan(s90) else "N/A",
                f"{cov:.3f}", f"{ml2:.3f}", f"{al2:.3f}",
                f"{fe:.2f}" if fe > 0 else "—",
                f"{dc:.1f}",
            ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Color header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Experiment Summary (averaged over 3 seeds)", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved summary_table.png")


# ─── Video generation ─────────────────────────────────────────

def generate_video(
    condition: str,
    map_name: str,
    seed: int,
    num_robots: int = 4,
    max_steps: int = 500,
    r_comm: int = 8,
    alpha: float = 0.5,
    output_dir: str = "results",
    fps: int = 10,
):
    """Generate a video for a single run showing exploration progress."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    truth, starts = _generate_map(map_name, num_robots, seed)
    env = CustomGridWorldEnv(truth, starts, sensor_radius=4, max_steps=max_steps)
    cfg = _make_config(condition, r_comm=r_comm, alpha=alpha, num_robots=num_robots)
    pipeline = ExplorationPipeline(cfg)

    obs = env.reset()
    pipeline.reset(obs)

    # Collect frames
    frame_maps = [env.fused_map.copy()]
    frame_positions = [env.get_positions()]
    frame_coverages = [compute_coverage(env.fused_map)]
    frame_lambda2s = [algebraic_connectivity(env.get_positions(), r_comm)]

    for step in range(max_steps):
        actions = pipeline.act(obs)
        obs, coverage, done = env.step(actions)
        frame_maps.append(env.fused_map.copy())
        frame_positions.append(env.get_positions())
        frame_coverages.append(coverage)
        frame_lambda2s.append(algebraic_connectivity(env.get_positions(), r_comm))
        if done:
            break

    # Create animation
    fig, (ax_map, ax_cov, ax_lam) = plt.subplots(1, 3, figsize=(16, 5.5),
                                                   gridspec_kw={"width_ratios": [2, 1, 1]})

    cond_label = {
        "pure_frontier": "Frontier Only",
        "frontier_conn": "Frontier+Conn",
        "rigid_formation": "Rigid Form+Conn",
        "adaptive_formation": "Adaptive Form+Conn",
        "adaptive_adversarial": "Adaptive+Adversary",
    }[condition]

    robot_colors = plt.cm.tab10(np.linspace(0, 1, num_robots))

    def _render_map(fmap):
        img = np.zeros((*fmap.shape, 3))
        img[fmap == int(CellState.FREE)] = [1.0, 1.0, 1.0]
        img[fmap == int(CellState.OCCUPIED)] = [0.15, 0.15, 0.15]
        img[fmap == int(CellState.UNKNOWN)] = [0.7, 0.7, 0.7]
        return img

    im = ax_map.imshow(_render_map(frame_maps[0]), origin="upper")
    scatter = ax_map.scatter([], [], s=80, edgecolors="black", linewidths=1.2, zorder=5)
    ax_map.set_title(f"{cond_label} | {map_name} | seed={seed}")
    ax_map.axis("off")

    line_cov, = ax_cov.plot([], [], "b-", linewidth=1.5)
    ax_cov.set_xlim(0, len(frame_maps))
    ax_cov.set_ylim(0, 1.05)
    ax_cov.set_xlabel("Step")
    ax_cov.set_ylabel("Coverage")
    ax_cov.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8)
    ax_cov.set_title("Coverage")

    line_lam, = ax_lam.plot([], [], "r-", linewidth=1.5)
    ax_lam.set_xlim(0, len(frame_maps))
    ax_lam.set_ylim(min(0, min(frame_lambda2s) - 0.5),
                    max(frame_lambda2s) + 0.5)
    ax_lam.set_xlabel("Step")
    ax_lam.set_ylabel("λ₂")
    ax_lam.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_lam.set_title("Algebraic Connectivity")

    fig.tight_layout()

    def update(frame_idx):
        im.set_data(_render_map(frame_maps[frame_idx]))
        positions = frame_positions[frame_idx]
        scatter.set_offsets(np.array(positions))
        scatter.set_facecolors(robot_colors[:len(positions)])

        xs = list(range(frame_idx + 1))
        line_cov.set_data(xs, frame_coverages[:frame_idx + 1])
        line_lam.set_data(xs, frame_lambda2s[:frame_idx + 1])
        return im, scatter, line_cov, line_lam

    # Sample every N frames to keep video short
    total_frames = len(frame_maps)
    sample_rate = max(1, total_frames // 200)
    frame_indices = list(range(0, total_frames, sample_rate))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)

    anim = FuncAnimation(fig, update, frames=frame_indices, blit=True)

    video_path = os.path.join(output_dir, f"video_{condition}_{map_name}_s{seed}.mp4")
    anim.save(video_path, writer="ffmpeg", fps=fps, dpi=100)
    plt.close(fig)
    print(f"  Saved {video_path}")


# ─── Parameter sweep (alpha) ─────────────────────────────────

def run_alpha_sweep(
    alphas: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    map_names: List[str] = MAP_NAMES,
    seeds: List[int] = SEEDS,
    num_robots: int = 4,
    max_steps: int = 500,
    r_comm: int = 8,
    output_dir: str = "results",
) -> Dict:
    """Sweep blending weight alpha for adaptive_formation condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    sweep_results = {}

    total = len(alphas) * len(map_names) * len(seeds)
    count = 0

    for alpha in alphas:
        for map_name in map_names:
            for seed in seeds:
                count += 1
                print(f"[Alpha sweep {count}/{total}] α={alpha:.1f} | {map_name} | seed={seed}",
                      end=" ", flush=True)
                result = run_single(
                    condition="adaptive_formation",
                    map_name=map_name,
                    seed=seed,
                    num_robots=num_robots,
                    max_steps=max_steps,
                    r_comm=r_comm,
                    alpha=alpha,
                )
                key = (alpha, map_name)
                sweep_results.setdefault(key, []).append(result)
                print(f"| s90={result.steps_to_90} cov={result.final_coverage:.3f}")

    # Plot Pareto frontier: steps_to_90 vs min_lambda2
    fig, axes = plt.subplots(1, len(map_names), figsize=(5 * len(map_names), 5))
    if len(map_names) == 1:
        axes = [axes]

    for ax, map_name in zip(axes, map_names):
        s90_means = []
        lam2_means = []
        alpha_vals = []

        for alpha in alphas:
            runs = sweep_results.get((alpha, map_name), [])
            if not runs:
                continue
            s90 = np.mean([r.steps_to_90 if r.steps_to_90 > 0 else 500 for r in runs])
            ml2 = np.mean([r.min_lambda2 for r in runs])
            s90_means.append(s90)
            lam2_means.append(ml2)
            alpha_vals.append(alpha)

        scatter = ax.scatter(s90_means, lam2_means, c=alpha_vals, cmap="viridis",
                             s=100, edgecolors="black", linewidths=0.8, zorder=5)
        for i, a in enumerate(alpha_vals):
            ax.annotate(f"α={a:.1f}", (s90_means[i], lam2_means[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

        ax.set_xlabel("Steps to 90% Coverage")
        ax.set_ylabel("Min λ₂")
        ax.set_title(f"Map: {map_name}")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle("Pareto Frontier: Exploration Speed vs Connectivity (α sweep)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    cbar = fig.colorbar(scatter, ax=axes, label="Blending weight α", shrink=0.8)
    fig.savefig(os.path.join(output_dir, "alpha_sweep_pareto.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved alpha_sweep_pareto.png")

    return sweep_results


# ─── R_comm sweep ─────────────────────────────────────────────

def run_rcomm_sweep(
    r_comms: List[int] = [4, 6, 8, 10, 12, 15],
    map_names: List[str] = MAP_NAMES,
    seeds: List[int] = SEEDS,
    num_robots: int = 4,
    max_steps: int = 500,
    alpha: float = 0.5,
    output_dir: str = "results",
):
    """Sweep communication range r_comm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    sweep_results = {}

    total = len(r_comms) * len(map_names) * len(seeds)
    count = 0

    for rc in r_comms:
        for map_name in map_names:
            for seed in seeds:
                count += 1
                print(f"[R_comm sweep {count}/{total}] r={rc} | {map_name} | seed={seed}",
                      end=" ", flush=True)
                result = run_single(
                    condition="adaptive_formation",
                    map_name=map_name,
                    seed=seed,
                    num_robots=num_robots,
                    max_steps=max_steps,
                    r_comm=rc,
                    alpha=alpha,
                )
                key = (rc, map_name)
                sweep_results.setdefault(key, []).append(result)
                print(f"| s90={result.steps_to_90} cov={result.final_coverage:.3f}")

    # Plot
    fig, axes = plt.subplots(1, len(map_names), figsize=(5 * len(map_names), 5))
    if len(map_names) == 1:
        axes = [axes]

    for ax, map_name in zip(axes, map_names):
        s90_means = []
        disc_means = []
        rc_vals = []

        for rc in r_comms:
            runs = sweep_results.get((rc, map_name), [])
            if not runs:
                continue
            s90 = np.mean([r.steps_to_90 if r.steps_to_90 > 0 else 500 for r in runs])
            disc = np.mean([r.disconnections for r in runs])
            s90_means.append(s90)
            disc_means.append(disc)
            rc_vals.append(rc)

        ax2 = ax.twinx()
        line1, = ax.plot(rc_vals, s90_means, "b-o", linewidth=1.5, label="Steps→90%")
        line2, = ax2.plot(rc_vals, disc_means, "r-s", linewidth=1.5, label="Disconnections")
        ax.set_xlabel("Communication Range (r_comm)")
        ax.set_ylabel("Steps to 90%", color="blue")
        ax2.set_ylabel("Disconnections", color="red")
        ax.set_title(f"Map: {map_name}")
        ax.legend([line1, line2], ["Steps→90%", "Disconnections"], fontsize=8)

    fig.suptitle("Effect of Communication Range", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rcomm_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved rcomm_sweep.png")

    return sweep_results


# ─── Snapshot visualization ───────────────────────────────────

def generate_snapshots(
    condition: str = "adaptive_formation",
    map_name: str = "rooms",
    seed: int = 0,
    num_robots: int = 4,
    max_steps: int = 500,
    r_comm: int = 8,
    alpha: float = 0.5,
    snapshot_steps: List[int] = [1, 50, 100, 200, 350, 500],
    output_dir: str = "results",
):
    """Generate snapshot images at specific timesteps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    truth, starts = _generate_map(map_name, num_robots, seed)
    env = CustomGridWorldEnv(truth, starts, sensor_radius=4, max_steps=max_steps)
    cfg = _make_config(condition, r_comm=r_comm, alpha=alpha, num_robots=num_robots)
    pipeline = ExplorationPipeline(cfg)

    obs = env.reset()
    pipeline.reset(obs)

    snapshots = {}
    snapshot_set = set(snapshot_steps)

    # Capture step 0/1
    if 0 in snapshot_set or 1 in snapshot_set:
        snapshots[0] = (env.fused_map.copy(), env.get_positions(),
                        compute_coverage(env.fused_map))

    for step in range(1, max_steps + 1):
        actions = pipeline.act(obs)
        obs, coverage, done = env.step(actions)

        if step in snapshot_set:
            snapshots[step] = (env.fused_map.copy(), env.get_positions(), coverage)

        if done:
            # Fill remaining snapshots with final state
            for s in snapshot_steps:
                if s not in snapshots:
                    snapshots[s] = (env.fused_map.copy(), env.get_positions(), coverage)
            break

    # Plot snapshots
    n_snaps = len(snapshot_steps)
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4))
    if n_snaps == 1:
        axes = [axes]

    robot_colors = plt.cm.tab10(np.linspace(0, 1, num_robots))

    cond_label = {
        "pure_frontier": "Frontier Only",
        "frontier_conn": "Frontier+Conn",
        "rigid_formation": "Rigid Form+Conn",
        "adaptive_formation": "Adaptive Form+Conn",
        "adaptive_adversarial": "Adaptive+Adversary",
    }.get(condition, condition)

    for ax, step_t in zip(axes, snapshot_steps):
        if step_t not in snapshots:
            ax.set_title(f"Step {step_t}\n(not reached)")
            ax.axis("off")
            continue

        fmap, positions, cov = snapshots[step_t]
        img = np.zeros((*fmap.shape, 3))
        img[fmap == int(CellState.FREE)] = [1.0, 1.0, 1.0]
        img[fmap == int(CellState.OCCUPIED)] = [0.15, 0.15, 0.15]
        img[fmap == int(CellState.UNKNOWN)] = [0.75, 0.75, 0.75]

        ax.imshow(img, origin="upper")
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.scatter(xs, ys, c=robot_colors[:len(positions)], s=60,
                   edgecolors="black", linewidths=1, zorder=5)
        ax.set_title(f"Step {step_t}\ncov={cov:.2f}", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"{cond_label} | {map_name} | seed={seed}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                f"snapshots_{condition}_{map_name}_s{seed}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved snapshots_{condition}_{map_name}_s{seed}.png")


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Final project experiment runner")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with 1 seed and reduced steps")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--num-robots", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--r-comm", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--conn-penalty-weight", type=float, default=10.0,
                        help="Soft connectivity penalty weight w_p")
    parser.add_argument("--videos", action="store_true",
                        help="Generate videos (requires ffmpeg)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweeps (alpha and r_comm)")
    parser.add_argument("--snapshots", action="store_true",
                        help="Generate exploration snapshots")
    args = parser.parse_args()

    seeds = [0] if args.quick else SEEDS
    max_steps = 200 if args.quick else args.max_steps

    print("=" * 60)
    print("  Final Project: Multi-Robot Exploration Experiments")
    print("=" * 60)

    # Run main experiments
    results = run_all_experiments(
        conditions=CONDITIONS,
        map_names=MAP_NAMES,
        seeds=seeds,
        num_robots=args.num_robots,
        max_steps=max_steps,
        r_comm=args.r_comm,
        alpha=args.alpha,
        output_dir=args.output_dir,
        conn_penalty_weight=args.conn_penalty_weight,
    )

    # Generate plots
    print("\nGenerating plots...")
    plot_results(results, output_dir=args.output_dir)

    # Generate snapshots
    if args.snapshots or not args.quick:
        print("\nGenerating snapshots...")
        for cond in CONDITIONS:
            for map_name in MAP_NAMES:
                generate_snapshots(
                    condition=cond, map_name=map_name, seed=0,
                    num_robots=args.num_robots, max_steps=max_steps,
                    r_comm=args.r_comm, alpha=args.alpha,
                    output_dir=args.output_dir,
                )

    # Videos
    if args.videos:
        print("\nGenerating videos...")
        for cond in CONDITIONS:
            for map_name in MAP_NAMES:
                try:
                    generate_video(
                        condition=cond, map_name=map_name, seed=0,
                        num_robots=args.num_robots, max_steps=max_steps,
                        r_comm=args.r_comm, alpha=args.alpha,
                        output_dir=args.output_dir,
                    )
                except Exception as e:
                    print(f"  Video failed for {cond}/{map_name}: {e}")

    # Parameter sweeps
    if args.sweep:
        print("\nRunning alpha sweep...")
        run_alpha_sweep(
            map_names=MAP_NAMES, seeds=seeds,
            num_robots=args.num_robots, max_steps=max_steps,
            r_comm=args.r_comm, output_dir=args.output_dir,
        )
        print("\nRunning r_comm sweep...")
        run_rcomm_sweep(
            map_names=MAP_NAMES, seeds=seeds,
            num_robots=args.num_robots, max_steps=max_steps,
            alpha=args.alpha, output_dir=args.output_dir,
        )

    print("\n" + "=" * 60)
    print("  All experiments complete!")
    print(f"  Results in: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
