"""
Combined exploration + formation + connectivity pipeline.

Five experimental conditions
─────────────────────────────
1. pure_frontier       — greedy frontier exploration, no connectivity constraint
2. frontier_conn       — frontier + hard connectivity filter
3. rigid_formation     — all robots share one group target; connectivity filter applied
4. adaptive_formation  — individual frontier targets + cohesion pull + connectivity filter
5. adaptive_adversarial — condition 4 with one adversarial robot

Blending weight alpha (adaptive conditions only)
────────────────────────────────────────────────
  alpha=0.0  → pure frontier (ignore cohesion)
  alpha=1.0  → pure cohesion (ignore frontier)
  default    → 0.35  (mostly frontier-driven; cohesion kicks in only when spread too wide)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.types import Action, CellState, Observation

from a_heuristic_exploration.frontiers import extract_frontiers
from a_heuristic_exploration.clustering import (
    cluster_representative_point,
    frontier_clusters_region_pca,
)
from a_heuristic_exploration.information_gain import information_gain_count_unknown
from a_heuristic_exploration.assignment import hungarian_assignment, greedy_assignment
from a_heuristic_exploration.planner_astar import astar_path

from final_project.connectivity import (
    connectivity_filter,
    is_connected,
    algebraic_connectivity,
    _count_components,
)
from final_project.formation import (
    AdaptiveFormationController,
    GroupFormationController,
    weighted_frontier_centroid,
    nearest_free_to,
)


# ─── Action helpers ────────────────────────────────────────────

ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    Action.UP:    (0, -1),
    Action.DOWN:  (0,  1),
    Action.LEFT:  (-1, 0),
    Action.RIGHT: (1,  0),
    Action.STAY:  (0,  0),
}


def _action_to_pos(pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
    dx, dy = ACTION_DELTAS[action]
    return (pos[0] + dx, pos[1] + dy)


def _next_action_from_path(path: List[Tuple[int, int]]) -> Action:
    if len(path) < 2:
        return Action.STAY
    (x0, y0), (x1, y1) = path[0], path[1]
    dx, dy = x1 - x0, y1 - y0
    for action, (adx, ady) in ACTION_DELTAS.items():
        if adx == dx and ady == dy:
            return action
    return Action.STAY


def _pos_to_action(current: Tuple[int, int], target: Tuple[int, int]) -> Action:
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    if abs(dx) > 1 or abs(dy) > 1 or (dx != 0 and dy != 0):
        return Action.STAY
    for action, (adx, ady) in ACTION_DELTAS.items():
        if adx == dx and ady == dy:
            return action
    return Action.STAY


# ─── Adversarial policy ───────────────────────────────────────

def _adversarial_action(
    fused_map: np.ndarray,
    robot_positions: List[Tuple[int, int]],
    adv_idx: int,
) -> Action:
    """Adversarial robot maximises distance from team centroid."""
    pos = robot_positions[adv_idx]
    others = [p for i, p in enumerate(robot_positions) if i != adv_idx]
    if not others:
        return Action.STAY

    cx = float(np.mean([p[0] for p in others]))
    cy = float(np.mean([p[1] for p in others]))
    h, w = fused_map.shape

    best_action = Action.STAY
    best_dist2 = -1.0
    for action, (dx, dy) in ACTION_DELTAS.items():
        if dx == 0 and dy == 0:
            continue
        nx, ny = pos[0] + dx, pos[1] + dy
        if not (0 <= nx < w and 0 <= ny < h):
            continue
        if int(fused_map[ny, nx]) != int(CellState.FREE):
            continue
        d2 = (nx - cx) ** 2 + (ny - cy) ** 2
        if d2 > best_dist2:
            best_dist2 = d2
            best_action = action
    return best_action


# ─── Configuration ────────────────────────────────────────────

@dataclass
class PipelineConfig:
    condition: str = "adaptive_formation"

    # Blending weight: 0=pure frontier, 1=pure cohesion
    alpha: float = 0.35

    # Communication range (Chebyshev, ignores walls)
    r_comm: int = 10

    # Cohesion radius: robots beyond this distance from centroid get pulled back
    cohesion_radius: float = 6.0

    # Sensor / IG radius
    sensor_radius: int = 4

    # How often to fully replan
    replanning_period: int = 8

    # Frontier clustering
    min_frontier_cluster_size: int = 6
    frontier_pca_eigen_threshold: float = 25.0
    max_targets: int = 30

    # Assignment method
    assignment_method: str = "hungarian"

    # Adversarial robot index (-1 = none)
    adversarial_robot: int = -1

    # Rigid formation: steps with zero movement before deadlock recovery kicks in
    stuck_threshold: int = 15

    # Soft connectivity penalty weight for frontier_conn.
    # Score = frontier_progress - conn_penalty_weight * lambda2_loss
    # Higher = stricter connectivity; 0 = pure frontier (no penalty).
    # With weight~5, disconnection costs ~5x a single step of progress.
    conn_penalty_weight: float = 5.0


# ─── Pipeline ─────────────────────────────────────────────────

class ExplorationPipeline:
    """Unified policy for all five experimental conditions."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._paths: Dict[int, List[Tuple[int, int]]] = {}
        self._targets: Dict[int, Tuple[int, int]] = {}
        self._group_ctrl: Optional[GroupFormationController] = None
        self._adapt_ctrl: Optional[AdaptiveFormationController] = None

        # Metrics
        self.lambda2_history: List[float] = []
        self.coverage_history: List[float] = []
        self.formation_error_history: List[float] = []
        self.connectivity_maintained: List[bool] = []

    def reset(self, obs: Observation) -> None:
        self._paths.clear()
        self._targets.clear()
        n = len(obs.robot_positions)

        if self.config.condition == "rigid_formation":
            self._group_ctrl = GroupFormationController(
                cohesion_radius=self.config.cohesion_radius
            )
            self._adapt_ctrl = None
            self._rigid_targets: Dict[int, Tuple[int, int]] = {}
            self._rigid_paths: Dict[int, List[Tuple[int, int]]] = {}
            self._rigid_stuck_steps: int = 0
            self._rigid_skipped_clusters: int = 0
        elif self.config.condition in ("adaptive_formation", "adaptive_adversarial"):
            self._adapt_ctrl = AdaptiveFormationController(
                num_robots=n,
                cohesion_radius=self.config.cohesion_radius,
            )
            self._group_ctrl = None
            self._rigid_paths: Dict[int, List[Tuple[int, int]]] = {}
        else:
            self._group_ctrl = None
            self._adapt_ctrl = None
            self._rigid_paths = {}

        self.lambda2_history.clear()
        self.coverage_history.clear()
        self.formation_error_history.clear()
        self.connectivity_maintained.clear()

    # ── Main act ──────────────────────────────────────────────

    def act(self, obs: Observation) -> List[Action]:
        fused_map = obs.fused_map
        positions = list(obs.robot_positions)
        n = len(positions)

        # Record metrics
        lam2 = algebraic_connectivity(positions, self.config.r_comm)
        self.lambda2_history.append(lam2)
        self.connectivity_maintained.append(is_connected(positions, self.config.r_comm))
        self.coverage_history.append(
            float(np.mean(fused_map != int(CellState.UNKNOWN)))
        )

        # Record formation error
        ctrl = self._adapt_ctrl or self._group_ctrl
        if ctrl is not None:
            self.formation_error_history.append(ctrl.get_formation_error(positions))

        # Dispatch to condition
        cond = self.config.condition
        if cond == "pure_frontier":
            return self._frontier_actions(fused_map, positions, obs.step_index)

        elif cond == "frontier_conn":
            f_acts = self._frontier_actions(fused_map, positions, obs.step_index)
            return self._soft_conn_actions(fused_map, positions, f_acts)

        elif cond == "rigid_formation":
            return self._rigid_formation_act(fused_map, positions, obs.step_index)

        elif cond in ("adaptive_formation", "adaptive_adversarial"):
            return self._adaptive_formation_act(fused_map, positions, obs.step_index,
                                                 cond)
        else:
            raise ValueError(f"Unknown condition: {cond}")

    # ── Rigid formation ───────────────────────────────────────

    def _rigid_formation_act(
        self,
        fused_map: np.ndarray,
        positions: List[Tuple[int, int]],
        step_index: int,
    ) -> List[Action]:
        """Rigid formation: the whole team navigates toward the same frontier
        cluster. Each robot gets a separate cell within the cluster as its
        A*-planned target so robots don't collide at one cell."""
        assert self._group_ctrl is not None
        n = len(positions)

        # Replan when needed
        missing = any(i not in self._rigid_paths or len(self._rigid_paths[i]) <= 1
                      for i in range(n))
        if step_index % self.config.replanning_period == 0 or missing:
            self._replan_rigid(fused_map, positions)

        actions = []
        for i, pos in enumerate(positions):
            path = self._rigid_paths.get(i)
            if not path:
                actions.append(Action.STAY)
                continue
            if path[0] != pos:
                self._rigid_paths.pop(i, None)
                actions.append(Action.STAY)
                continue
            actions.append(_next_action_from_path(path))
            if len(path) >= 2:
                self._rigid_paths[i] = path[1:]

        proposed = [_action_to_pos(p, a) for p, a in zip(positions, actions)]
        filtered = connectivity_filter(positions, proposed, self.config.r_comm)
        final = [_pos_to_action(c, t) for c, t in zip(positions, filtered)]

        # Invalidate blocked paths
        for i, (ia, aa) in enumerate(zip(actions, final)):
            if ia != aa:
                self._rigid_paths.pop(i, None)

        # Deadlock detection: if no robot moved, increment stuck counter.
        # After stuck_threshold steps, skip to the next cluster in the ranked list.
        if any(a != Action.STAY for a in final):
            self._rigid_stuck_steps = 0
        else:
            self._rigid_stuck_steps += 1
            if self._rigid_stuck_steps >= self.config.stuck_threshold:
                self._rigid_skipped_clusters += 1
                self._replan_rigid(fused_map, positions)
                self._rigid_stuck_steps = 0

        return final

    def _replan_rigid(
        self,
        fused_map: np.ndarray,
        positions: List[Tuple[int, int]],
    ) -> None:
        """Find the best reachable frontier cluster; assign each robot a cell in it.

        Tries clusters in descending IG order starting at _rigid_skipped_clusters
        (incremented each time deadlock is detected). Picks the first cluster
        where at least (n-1) robots have a valid A* path. Resets skip on success.
        Falls back to best cluster if nothing passes reachability.
        """
        n = len(positions)
        frontiers = extract_frontiers(fused_map)
        if not frontiers:
            self._rigid_paths.clear()
            return

        clusters = frontier_clusters_region_pca(
            frontiers,
            min_cluster_size=self.config.min_frontier_cluster_size,
            eigenvalue_threshold=self.config.frontier_pca_eigen_threshold,
        )
        if not clusters:
            self._rigid_paths.clear()
            return

        def _cluster_ig(cluster):
            return sum(information_gain_count_unknown(
                fused_map, t, self.config.sensor_radius) for t in cluster)

        def _sample_targets(cluster):
            step = max(1, len(cluster) // n)
            targets, used = [], set()
            for i in range(n):
                t = cluster[(i * step) % len(cluster)]
                if t in used:
                    t = next((c for c in cluster if c not in used), t)
                targets.append(t)
                used.add(t)
            return targets

        clusters_sorted = sorted(clusters, key=_cluster_ig, reverse=True)
        skip = self._rigid_skipped_clusters % len(clusters_sorted)
        ordered = clusters_sorted[skip:] + clusters_sorted[:skip]

        for cluster in ordered:
            targets = _sample_targets(cluster)
            new_paths: Dict[int, List] = {}
            for i, (start, target) in enumerate(zip(positions, targets)):
                path = astar_path(fused_map, start=start, goal=target)
                if path is not None:
                    new_paths[i] = path
            if len(new_paths) >= max(1, n - 1):
                self._rigid_paths = new_paths
                self._rigid_skipped_clusters = 0
                return

        # Nothing passed — fall back to highest-IG cluster regardless
        targets = _sample_targets(clusters_sorted[0])
        new_paths = {}
        for i, (start, target) in enumerate(zip(positions, targets)):
            path = astar_path(fused_map, start=start, goal=target)
            if path is not None:
                new_paths[i] = path
        self._rigid_paths = new_paths

    # ── Adaptive formation ────────────────────────────────────

    def _adaptive_formation_act(
        self,
        fused_map: np.ndarray,
        positions: List[Tuple[int, int]],
        step_index: int,
        condition: str,
    ) -> List[Action]:
        """Per-robot frontier targets with cohesion override.

        Priority rules (in order):
          1. Adversarial robot: move away from team centroid.
          2. Drifted robot (dist > effective_cohesion): cohesion action.
          3. Otherwise: follow frontier target.

        Connectivity filter applied as hard constraint on all actions.

        alpha controls effective_cohesion_radius:
          - alpha=0   → never override (pure frontier, same as frontier_conn)
          - alpha=1   → always override (pure cohesion)
          - alpha=0.35 → override when drift > 0.65 * cohesion_radius
        """
        assert self._adapt_ctrl is not None

        # Frontier actions (individual targets, A*-planned)
        f_acts = self._frontier_actions(fused_map, positions, step_index)

        # Cohesion actions (pull toward group centroid)
        coh_acts = self._adapt_ctrl.cohesion_actions(fused_map, positions)

        alpha = self.config.alpha
        # Effective cohesion radius scaled by cohesion_radius (set to 2×r_comm),
        # so alpha=0.35 → eff_radius=13: only very spread robots get pulled back.
        #   alpha=0 → cohesion_radius (rarely fires)
        #   alpha=1 → 0               (always fires)
        eff_radius = self.config.cohesion_radius * (1.0 - alpha)

        # Compute distances from centroid
        cx = float(np.mean([p[0] for p in positions]))
        cy = float(np.mean([p[1] for p in positions]))

        blended = []
        for i, (f_a, c_a) in enumerate(zip(f_acts, coh_acts)):
            # 1. Adversarial override
            if condition == "adaptive_adversarial" and i == self.config.adversarial_robot:
                blended.append(_adversarial_action(fused_map, positions, i))
                continue

            # 2. Cohesion override if drifted beyond effective radius
            dist = max(abs(positions[i][0] - cx), abs(positions[i][1] - cy))
            if dist > eff_radius and c_a != Action.STAY:
                blended.append(c_a)
                continue

            # 3. Follow frontier target
            blended.append(f_a)

        # Soft connectivity penalty (same as frontier_conn)
        return self._soft_conn_actions(fused_map, positions, blended)

    # ── Soft connectivity filter (frontier_conn) ──────────────

    def _soft_conn_actions(
        self,
        fused_map: np.ndarray,
        positions: List[Tuple[int, int]],
        f_acts: List[Action],
    ) -> List[Action]:
        """Soft connectivity penalty for frontier_conn.

        For each robot, score all 5 possible actions by:
            score = frontier_progress - conn_penalty_weight * lambda2_loss

        Where:
          frontier_progress = Chebyshev distance reduction toward assigned target
                              (positive = moving closer, negative = moving away)
          lambda2_loss      = max(0, lambda2_before - lambda2_after)

        Connected moves that make progress always win (lambda2_loss=0, progress>0).
        The penalty kicks in only when no connected move makes forward progress,
        allowing the robot to advance at a connectivity cost rather than freezing.
        """
        h, w = fused_map.shape
        n = len(positions)
        penalty = self.config.conn_penalty_weight
        # Use joint greedy evaluation: update working_positions after each robot
        # so that robot i+1 sees the already-committed moves of robots 0..i.
        working = list(positions)

        result: List[Action] = []
        for i, (pos, f_act) in enumerate(zip(positions, f_acts)):
            current_components = _count_components(working, self.config.r_comm)

            # Fast path: proposed action keeps or improves connectivity
            proposed = _action_to_pos(pos, f_act)
            trial = list(working)
            trial[i] = proposed
            if _count_components(trial, self.config.r_comm) <= current_components:
                working[i] = proposed
                result.append(f_act)
                continue

            # Proposed action worsens connectivity — score all alternatives.
            # score = frontier_progress - penalty * delta_components
            # Reconnecting moves (delta < 0) earn a bonus; splitting moves pay a cost.
            target = self._targets.get(i)
            best_action = f_act
            best_score = -np.inf

            for action in Action:
                np_pos = _action_to_pos(pos, action)
                if not (0 <= np_pos[0] < w and 0 <= np_pos[1] < h):
                    continue
                if int(fused_map[np_pos[1], np_pos[0]]) == int(CellState.OCCUPIED):
                    continue

                trial2 = list(working)
                trial2[i] = np_pos
                delta_comp = _count_components(trial2, self.config.r_comm) - current_components

                if target is not None:
                    tx, ty = target
                    d_before = max(abs(pos[0] - tx), abs(pos[1] - ty))
                    d_after  = max(abs(np_pos[0] - tx), abs(np_pos[1] - ty))
                    step_progress = float(d_before - d_after)   # +1, 0, or -1
                    ig_target = information_gain_count_unknown(
                        fused_map, target, self.config.sensor_radius)
                    # Scale progress by target IG so penalty is on the same order.
                    # A robot only disconnects if the target is valuable enough.
                    progress = step_progress * float(ig_target)
                else:
                    progress = 0.0

                score = progress - penalty * delta_comp
                if score > best_score:
                    best_score = score
                    best_action = action

            working[i] = _action_to_pos(pos, best_action)
            result.append(best_action)
            if best_action != f_act:
                self._paths.pop(i, None)
                self._targets.pop(i, None)

        return result

    # ── Frontier planning ─────────────────────────────────────

    def _frontier_actions(
        self,
        fused_map: np.ndarray,
        positions: List[Tuple[int, int]],
        step_index: int,
    ) -> List[Action]:
        should_replan = step_index % self.config.replanning_period == 0
        missing = any(i not in self._paths or len(self._paths[i]) <= 1
                      for i in range(len(positions)))
        if should_replan or missing:
            self._replan(fused_map, positions)

        actions = []
        for i, pos in enumerate(positions):
            path = self._paths.get(i)
            if not path:
                actions.append(Action.STAY)
                continue
            if path[0] != pos:
                self._paths.pop(i, None)
                actions.append(Action.STAY)
                continue
            actions.append(_next_action_from_path(path))
            if len(path) >= 2:
                self._paths[i] = path[1:]
        return actions

    def _replan(
        self,
        fused_map: np.ndarray,
        positions: List[Tuple[int, int]],
    ) -> None:
        frontiers = extract_frontiers(fused_map)
        if not frontiers:
            self._paths.clear()
            self._targets.clear()
            return

        clusters = frontier_clusters_region_pca(
            frontiers,
            min_cluster_size=self.config.min_frontier_cluster_size,
            eigenvalue_threshold=self.config.frontier_pca_eigen_threshold,
        )
        targets = [cluster_representative_point(c) for c in clusters]
        if not targets:
            return

        # Cap and score targets
        if len(targets) > self.config.max_targets:
            scored = sorted(
                [(information_gain_count_unknown(fused_map, t, self.config.sensor_radius), t)
                 for t in targets],
                reverse=True,
            )
            targets = [t for _, t in scored[:self.config.max_targets]]

        n, nt = len(positions), len(targets)
        utilities = np.full((n, nt), -np.inf)
        paths_cache: List[List] = []

        for i, start in enumerate(positions):
            row: List = []
            for j, t in enumerate(targets):
                path = astar_path(fused_map, start=start, goal=t)
                row.append(path)
                if path is not None:
                    cost = max(0, len(path) - 1)
                    ig = information_gain_count_unknown(
                        fused_map, target=t, radius=self.config.sensor_radius
                    )
                    utilities[i, j] = float(ig) - 100.0 * float(cost)
            paths_cache.append(row)

        if self.config.assignment_method == "hungarian":
            assignment = hungarian_assignment(utilities)
        else:
            assignment = greedy_assignment(utilities)

        new_paths: Dict[int, List] = {}
        new_targets: Dict[int, Tuple[int, int]] = {}
        for ri, ti in assignment.robot_to_target.items():
            path = paths_cache[ri][ti]
            if path is None:
                continue
            new_targets[ri] = targets[ti]
            new_paths[ri] = path

        self._targets = new_targets
        self._paths = new_paths

    # ── Helpers ───────────────────────────────────────────────

    def _invalidate_blocked(
        self,
        intended: List[Action],
        actual: List[Action],
    ) -> None:
        """Invalidate path cache for robots blocked by the connectivity filter."""
        for i, (ia, aa) in enumerate(zip(intended, actual)):
            if ia != aa:
                self._paths.pop(i, None)
                self._targets.pop(i, None)

