"""
Structured map generators for the final project.
Produces truth grids + start positions compatible with GridWorldEnv.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from simulation.types import CellState

FREE = int(CellState.FREE)
OCC  = int(CellState.OCCUPIED)
UNK  = int(CellState.UNKNOWN)


# ──────────────────────────────────────────────
#  Helper: carve a rectangular room into the grid
# ──────────────────────────────────────────────
def _carve_rect(grid: np.ndarray, y0: int, x0: int, h: int, w: int):
    """Set a rectangular region to FREE (interior only)."""
    grid[y0:y0 + h, x0:x0 + w] = FREE


def _carve_hline(grid: np.ndarray, y: int, x0: int, x1: int):
    grid[y, x0:x1 + 1] = FREE


def _carve_vline(grid: np.ndarray, y0: int, y1: int, x: int):
    grid[y0:y1 + 1, x] = FREE


def _place_robots_in_region(grid: np.ndarray, n: int,
                            y0: int, x0: int, h: int, w: int,
                            rng: np.random.Generator,
                            r_comm: int = 6) -> List[Tuple[int, int]]:
    """Place n robots in a rectangular FREE region, clustered within r_comm.

    Picks a seed cell, then places remaining robots within Chebyshev
    distance r_comm of the seed so the team starts connected.
    Returns (x, y) list.
    """
    free = []
    for y in range(y0, y0 + h):
        for x in range(x0, x0 + w):
            if grid[y, x] == FREE:
                free.append((x, y))
    if len(free) < n:
        return free[:n]

    rng.shuffle(free)
    seed = free[0]
    placed = [seed]
    used = {seed}

    # Place remaining robots near the seed
    candidates = [p for p in free if p != seed and
                  max(abs(p[0] - seed[0]), abs(p[1] - seed[1])) <= r_comm]
    rng.shuffle(candidates)

    for c in candidates:
        if len(placed) >= n:
            break
        if c not in used:
            placed.append(c)
            used.add(c)

    # Fallback: if not enough within r_comm, use nearest free cells
    if len(placed) < n:
        remaining = [p for p in free if p not in used]
        remaining.sort(key=lambda p: max(abs(p[0] - seed[0]), abs(p[1] - seed[1])))
        for p in remaining:
            if len(placed) >= n:
                break
            placed.append(p)

    return placed[:n]


# ──────────────────────────────────────────────
#  Map 1: Corridor / Maze
# ──────────────────────────────────────────────
def generate_corridor_map(
    width: int = 50,
    height: int = 50,
    num_robots: int = 4,
    seed: int = 0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    A map with long horizontal corridors connected by vertical passages.
    Stresses connectivity — robots must stay close when single-file in corridors.

    Layout (schematic):
        ██████████████████
        █                █
        █ ██████████████ █
        █                █
        █ ██████████████ █
        █                █
        ██████████████████
    """
    rng = np.random.default_rng(seed)
    grid = np.full((height, width), OCC, dtype=np.int8)

    corridor_h = 3          # corridor height (free space)
    wall_h = 2              # wall thickness between corridors
    margin = 2              # outer wall thickness
    passage_w = 3           # vertical passage width

    y = margin
    corridor_idx = 0
    while y + corridor_h < height - margin:
        # Carve horizontal corridor
        _carve_rect(grid, y, margin, corridor_h, width - 2 * margin)

        # Add internal walls with gaps (make it maze-like)
        if corridor_idx > 0:
            # Place some random obstacles inside corridors
            for _ in range(width // 8):
                ox = rng.integers(margin + 2, width - margin - 2)
                oy = rng.integers(y, y + corridor_h)
                grid[oy, ox] = OCC

        y += corridor_h
        if y + wall_h + corridor_h >= height - margin:
            break

        # Wall between corridors — leave vertical passages
        wall_top = y
        y += wall_h

        # Carve 2-3 vertical passages through the wall
        n_passages = max(2, width // 15)
        passage_xs = np.linspace(margin + 3, width - margin - 4, n_passages, dtype=int)
        # Alternate passage positions between corridor pairs
        if corridor_idx % 2 == 1:
            passage_xs = passage_xs + rng.integers(-2, 3, size=len(passage_xs))
            passage_xs = np.clip(passage_xs, margin + 1, width - margin - passage_w - 1)

        for px in passage_xs:
            _carve_rect(grid, wall_top, int(px), wall_h, passage_w)

        corridor_idx += 1

    # Place robots in the top-left corridor
    starts = _place_robots_in_region(grid, num_robots, margin, margin, corridor_h, width // 3, rng)
    return grid, starts


# ──────────────────────────────────────────────
#  Map 2: Multi-Room Floorplan
# ──────────────────────────────────────────────
def generate_room_map(
    width: int = 50,
    height: int = 50,
    num_robots: int = 4,
    num_rooms_x: int = 3,
    num_rooms_y: int = 3,
    seed: int = 0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    An office-like floorplan with rooms connected by doorways.
    Tests formation — robots must coordinate to cover rooms efficiently.

    Layout (schematic):
        ████████████████████
        █     █     █      █
        █  R1 █  R2 █  R3  █
        █     D     D      █
        ████D██████D████████
        █     █     █      █
        █  R4 D  R5 █  R6  █
        █     █     D      █
        ████████████████████
    """
    rng = np.random.default_rng(seed)
    grid = np.full((height, width), OCC, dtype=np.int8)

    margin = 1
    wall_t = 1  # wall thickness

    # Compute room dimensions
    usable_w = width - 2 * margin - (num_rooms_x - 1) * wall_t
    usable_h = height - 2 * margin - (num_rooms_y - 1) * wall_t
    room_w = usable_w // num_rooms_x
    room_h = usable_h // num_rooms_y

    rooms = []  # Store (y0, x0, h, w) for each room

    for ry in range(num_rooms_y):
        for rx in range(num_rooms_x):
            x0 = margin + rx * (room_w + wall_t)
            y0 = margin + ry * (room_h + wall_t)
            _carve_rect(grid, y0, x0, room_h, room_w)
            rooms.append((y0, x0, room_h, room_w))

            # Add random furniture (small obstacles) inside rooms
            n_furniture = rng.integers(1, 4)
            for _ in range(n_furniture):
                fx = rng.integers(x0 + 2, max(x0 + 3, x0 + room_w - 2))
                fy = rng.integers(y0 + 2, max(y0 + 3, y0 + room_h - 2))
                fw = rng.integers(1, 3)
                fh = rng.integers(1, 3)
                grid[fy:fy + fh, fx:fx + fw] = OCC

    # Add doorways between adjacent rooms
    door_w = 2

    for ry in range(num_rooms_y):
        for rx in range(num_rooms_x):
            y0, x0, rh, rw = rooms[ry * num_rooms_x + rx]

            # Door to the right neighbor
            if rx < num_rooms_x - 1:
                wall_x = x0 + rw
                door_y = y0 + rh // 2 - door_w // 2
                for dy in range(door_w):
                    for dx in range(wall_t):
                        grid[door_y + dy, wall_x + dx] = FREE

            # Door to the bottom neighbor
            if ry < num_rooms_y - 1:
                wall_y = y0 + rh
                door_x = x0 + rw // 2 - door_w // 2
                for dx in range(door_w):
                    for dy in range(wall_t):
                        grid[wall_y + dy, door_x + dx] = FREE

    # Place robots in the top-left room
    r0_y, r0_x, r0_h, r0_w = rooms[0]
    starts = _place_robots_in_region(grid, num_robots, r0_y + 1, r0_x + 1,
                                     r0_h - 2, r0_w - 2, rng)
    return grid, starts


# ──────────────────────────────────────────────
#  Map 3: Open field with scattered obstacles
# ──────────────────────────────────────────────
def generate_open_map(
    width: int = 50,
    height: int = 50,
    num_robots: int = 4,
    obstacle_prob: float = 0.12,
    seed: int = 0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Open environment with scattered obstacles (similar to existing random map).
    Baseline — connectivity is easy, exploration is straightforward.
    """
    rng = np.random.default_rng(seed)
    grid = np.full((height, width), FREE, dtype=np.int8)

    # Borders
    grid[0, :] = OCC
    grid[-1, :] = OCC
    grid[:, 0] = OCC
    grid[:, -1] = OCC

    # Random obstacles
    interior = rng.random((height - 2, width - 2)) < obstacle_prob
    grid[1:-1, 1:-1] = np.where(interior, OCC, FREE).astype(np.int8)

    starts = _place_robots_in_region(grid, num_robots, 1, 1, 8, 8, rng)
    return grid, starts


# ──────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────
def visualize_truth_map(grid, starts=None, title="", ax=None):
    """Quick visualization of a truth map."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    display = np.zeros((*grid.shape, 3))
    display[grid == FREE] = [1.0, 1.0, 1.0]      # white
    display[grid == OCC]  = [0.15, 0.15, 0.15]    # dark gray

    ax.imshow(display, origin='upper')

    if starts:
        sx = [s[0] for s in starts]
        sy = [s[1] for s in starts]
        colors = plt.cm.tab10(np.linspace(0, 1, len(starts)))
        ax.scatter(sx, sy, c=colors, s=100, edgecolors='black',
                   linewidths=1.5, zorder=5, marker='o')
        for i, (x, y) in enumerate(starts):
            ax.annotate(f'R{i}', (x, y), textcoords='offset points',
                        xytext=(5, 5), fontsize=9, fontweight='bold')

    ax.set_title(title, fontsize=12)
    ax.axis('off')
    return fig, ax
