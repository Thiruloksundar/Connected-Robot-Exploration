"""
Formation controller for discrete multi-robot exploration.

Instead of enforcing exact circle positions (which breaks in obstacle-rich
environments when ideal slots land inside walls), we use a cohesion-based
approach:

  - rigid_formation : all robots share a single group target (nearest frontier
    cluster to the team centroid). Robots move as a tight group.
    Formation error = mean distance from group centroid.

  - adaptive_formation : robots get individual frontier targets from the
    assignment, but a cohesion pull biases them back toward the group
    when they drift beyond a cohesion radius.  The alpha parameter
    controls the weight of cohesion vs. exploration.

This is the discrete analog of the cohesion term in flocking / virtual
structure formation control from the multi-robot systems literature.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.types import Action, CellState

# Cardinal moves: (dx, dy) -> Action
_DELTA_TO_ACTION: Dict[Tuple[int, int], Action] = {
    (1, 0): Action.RIGHT,
    (-1, 0): Action.LEFT,
    (0, -1): Action.UP,
    (0, 1): Action.DOWN,
    (0, 0): Action.STAY,
}


def cohesion_action(
    pos: Tuple[int, int],
    group_centroid: Tuple[float, float],
    fused_map: np.ndarray,
    occupied_cells: set,
) -> Action:
    """Return the cardinal action that moves pos closest to group_centroid.

    Only considers FREE cells. Falls back to STAY if every neighbor is blocked.
    """
    cx, cy = pos
    gx, gy = group_centroid
    height, width = fused_map.shape

    best_action = Action.STAY
    best_dist2 = (cx - gx) ** 2 + (cy - gy) ** 2

    for (dx, dy), action in _DELTA_TO_ACTION.items():
        if dx == 0 and dy == 0:
            continue
        nx, ny = cx + dx, cy + dy
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        if int(fused_map[ny, nx]) != int(CellState.FREE):
            continue
        if (nx, ny) in occupied_cells:
            continue
        dist2 = (nx - gx) ** 2 + (ny - gy) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_action = action

    return best_action


def toward_target_action(
    pos: Tuple[int, int],
    target: Tuple[int, int],
    fused_map: np.ndarray,
    occupied_cells: set,
) -> Action:
    """Greedy one-step action toward an integer target on free cells."""
    return cohesion_action(pos, (float(target[0]), float(target[1])),
                           fused_map, occupied_cells)


class GroupFormationController:
    """Rigid formation: all robots share one group target.

    The group target is the nearest FREE cell to the weighted frontier centroid.
    All robots navigate toward this single point as a cohesive unit.
    Formation error = mean Chebyshev distance from group centroid.

    Parameters
    ----------
    cohesion_radius : float
        When a robot is within this radius of the group centroid it may
        stay; otherwise it must move toward the centroid.
    """

    def __init__(self, cohesion_radius: float = 3.0):
        self.cohesion_radius = cohesion_radius
        self._group_target: Optional[Tuple[int, int]] = None

    def reset(self):
        self._group_target = None

    def set_group_target(self, target: Tuple[int, int]):
        self._group_target = target

    def compute_actions(
        self,
        fused_map: np.ndarray,
        robot_positions: List[Tuple[int, int]],
        group_target: Tuple[int, int],
    ) -> List[Action]:
        """Move all robots toward group_target as a cohesive unit."""
        occupied = set(robot_positions)
        actions = []
        for pos in robot_positions:
            action = toward_target_action(pos, group_target, fused_map,
                                          occupied - {pos})
            actions.append(action)
        return actions

    def get_formation_error(self, robot_positions: List[Tuple[int, int]]) -> float:
        """Mean Chebyshev distance from group centroid."""
        if not robot_positions:
            return 0.0
        cx = np.mean([p[0] for p in robot_positions])
        cy = np.mean([p[1] for p in robot_positions])
        return float(np.mean([max(abs(p[0] - cx), abs(p[1] - cy))
                               for p in robot_positions]))


class AdaptiveFormationController:
    """Adaptive formation: individual frontier targets + cohesion pull.

    Each robot follows its own assigned frontier target. When a robot
    drifts more than `cohesion_radius` (Chebyshev) from the group centroid,
    its action is replaced by the cohesion action (move toward centroid).

    The `alpha` blending weight is handled externally in the pipeline:
    - alpha=0  → pure frontier (no cohesion)
    - alpha=1  → pure cohesion (no frontier)

    This class just computes the cohesion actions; the pipeline blends them.
    """

    def __init__(self, num_robots: int, cohesion_radius: float = 5.0):
        self.num_robots = num_robots
        self.cohesion_radius = cohesion_radius

    def reset(self):
        pass

    def cohesion_actions(
        self,
        fused_map: np.ndarray,
        robot_positions: List[Tuple[int, int]],
    ) -> List[Action]:
        """Compute cohesion actions pulling each robot toward the group centroid."""
        cx = float(np.mean([p[0] for p in robot_positions]))
        cy = float(np.mean([p[1] for p in robot_positions]))
        occupied = set(robot_positions)
        actions = []
        for pos in robot_positions:
            actions.append(cohesion_action(pos, (cx, cy), fused_map,
                                           occupied - {pos}))
        return actions

    def needs_cohesion(
        self,
        robot_positions: List[Tuple[int, int]],
    ) -> List[bool]:
        """Return True for each robot that is outside the cohesion radius."""
        cx = float(np.mean([p[0] for p in robot_positions]))
        cy = float(np.mean([p[1] for p in robot_positions]))
        result = []
        for pos in robot_positions:
            dist = max(abs(pos[0] - cx), abs(pos[1] - cy))
            result.append(dist > self.cohesion_radius)
        return result

    def get_formation_error(self, robot_positions: List[Tuple[int, int]]) -> float:
        if not robot_positions:
            return 0.0
        cx = float(np.mean([p[0] for p in robot_positions]))
        cy = float(np.mean([p[1] for p in robot_positions]))
        return float(np.mean([max(abs(p[0] - cx), abs(p[1] - cy))
                               for p in robot_positions]))


# ─── Frontier centroid helpers ────────────────────────────────

def frontier_centroid(frontiers: List[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
    """Unweighted centroid of frontier cells. Returns None if empty."""
    if not frontiers:
        return None
    return (float(np.mean([f[0] for f in frontiers])),
            float(np.mean([f[1] for f in frontiers])))


def weighted_frontier_centroid(
    frontiers: List[Tuple[int, int]],
    fused_map: np.ndarray,
    sensor_radius: int,
) -> Optional[Tuple[float, float]]:
    """Centroid of frontiers weighted by local information gain (UNKNOWN count)."""
    if not frontiers:
        return None
    h, w = fused_map.shape
    weights = []
    for (fx, fy) in frontiers:
        y0 = max(0, fy - sensor_radius)
        y1 = min(h, fy + sensor_radius + 1)
        x0 = max(0, fx - sensor_radius)
        x1 = min(w, fx + sensor_radius + 1)
        ig = float(np.sum(fused_map[y0:y1, x0:x1] == int(CellState.UNKNOWN)))
        weights.append(max(ig, 1.0))

    total_w = sum(weights)
    cx = sum(f[0] * wt for f, wt in zip(frontiers, weights)) / total_w
    cy = sum(f[1] * wt for f, wt in zip(frontiers, weights)) / total_w
    return (cx, cy)


def nearest_free_to(
    centroid: Tuple[float, float],
    fused_map: np.ndarray,
    max_search: int = 20,
) -> Optional[Tuple[int, int]]:
    """Find the nearest FREE cell to a continuous centroid via BFS."""
    from collections import deque
    h, w = fused_map.shape
    gx, gy = int(round(centroid[0])), int(round(centroid[1]))

    # Clamp
    gx = max(0, min(w - 1, gx))
    gy = max(0, min(h - 1, gy))

    if int(fused_map[gy, gx]) == int(CellState.FREE):
        return (gx, gy)

    # BFS outward
    visited = {(gx, gy)}
    queue = deque([(gx, gy, 0)])
    while queue:
        cx, cy, dist = queue.popleft()
        if dist > max_search:
            break
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            if int(fused_map[ny, nx]) == int(CellState.FREE):
                return (nx, ny)
            queue.append((nx, ny, dist + 1))

    return None
