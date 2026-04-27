"""
Graph connectivity checks for discrete multi-robot exploration.
Discrete analog of the CBF λ₂ constraint from Problem 4.4.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

import numpy as np


def communication_graph(positions: List[Tuple[int, int]], r_comm: int) -> Dict[int, Set[int]]:
    """Build adjacency list for robots within Chebyshev distance r_comm."""
    n = len(positions)
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(positions[i][0] - positions[j][0])
            dy = abs(positions[i][1] - positions[j][1])
            if max(dx, dy) <= r_comm:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def is_connected(positions: List[Tuple[int, int]], r_comm: int) -> bool:
    """Check if the communication graph is connected (BFS from robot 0)."""
    if len(positions) <= 1:
        return True
    adj = communication_graph(positions, r_comm)
    visited = set()
    queue = deque([0])
    visited.add(0)
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == len(positions)


def algebraic_connectivity(positions: List[Tuple[int, int]], r_comm: int) -> float:
    """Compute λ₂ of the communication Laplacian (0-1 adjacency)."""
    n = len(positions)
    if n <= 1:
        return 0.0
    adj = communication_graph(positions, r_comm)
    L = np.zeros((n, n))
    for i in range(n):
        for j in adj[i]:
            L[i, j] = -1.0
            L[i, i] += 1.0
    eigvals = np.linalg.eigvalsh(L)
    return float(eigvals[1])


def would_disconnect(positions: List[Tuple[int, int]], robot_idx: int,
                     new_pos: Tuple[int, int], r_comm: int) -> bool:
    """Check if moving robot_idx to new_pos would disconnect the graph."""
    trial = list(positions)
    trial[robot_idx] = new_pos
    return not is_connected(trial, r_comm)


def connectivity_filter(positions: List[Tuple[int, int]],
                        proposed_positions: List[Tuple[int, int]],
                        r_comm: int) -> List[Tuple[int, int]]:
    """
    Filter proposed moves to maintain connectivity.

    Uses an iterative multi-pass greedy approach: runs up to n passes so that
    relay-chain coordination is possible. In a relay chain A-B-C-D all moving
    in the same direction, a single-pass greedy starting from A would block A
    (B hasn't moved yet to relay). With multiple passes, B advances in pass 1,
    then A can advance in pass 2. Converges when no more robots can be unblocked.

    If the current configuration is already disconnected, allow all moves
    that don't increase the number of connected components (soft recovery).
    """
    n = len(positions)
    currently_connected = is_connected(positions, r_comm)

    if currently_connected:
        result = list(positions)
        for _ in range(n):
            changed = False
            for i in range(n):
                if result[i] == proposed_positions[i]:
                    continue
                trial = list(result)
                trial[i] = proposed_positions[i]
                if is_connected(trial, r_comm):
                    result[i] = proposed_positions[i]
                    changed = True
            if not changed:
                break
        return result
    else:
        # Recovery mode: accept moves that don't increase # of components
        current_components = _count_components(positions, r_comm)
        result = list(positions)
        for _ in range(n):
            changed = False
            for i in range(n):
                if result[i] == proposed_positions[i]:
                    continue
                trial = list(result)
                trial[i] = proposed_positions[i]
                trial_components = _count_components(trial, r_comm)
                if trial_components <= current_components:
                    result[i] = proposed_positions[i]
                    current_components = trial_components
                    changed = True
            if not changed:
                break
        return result


def _count_components(positions: List[Tuple[int, int]], r_comm: int) -> int:
    """Count the number of connected components in the communication graph."""
    if len(positions) <= 1:
        return len(positions)
    adj = communication_graph(positions, r_comm)
    visited = set()
    components = 0
    for start in range(len(positions)):
        if start in visited:
            continue
        components += 1
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    return components
