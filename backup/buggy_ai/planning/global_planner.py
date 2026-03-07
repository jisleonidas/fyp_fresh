from __future__ import annotations

import heapq
from typing import Dict, Optional, Tuple

import numpy as np

from buggy_ai.types import Path2D, Waypoint


class AStarGlobalPlanner:
    def __init__(self, cfg: dict) -> None:
        self.obstacle_cost = int(cfg["planning"]["global"]["obstacle_cost"])

    def _neighbors(self, node: Tuple[int, int]) -> list[Tuple[int, int]]:
        x, y = node
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        return [(x + dx, y + dy) for dx, dy in offsets]

    def plan(self, grid: np.ndarray, start: Waypoint, goal: Waypoint, map_resolution_m: float) -> Path2D:
        h, w = grid.shape

        sx = int(start.x / map_resolution_m + w // 2)
        sy = int(start.y / map_resolution_m + h // 2)
        gx = int(goal.x / map_resolution_m + w // 2)
        gy = int(goal.y / map_resolution_m + h // 2)

        sx = max(0, min(w - 1, sx))
        sy = max(0, min(h - 1, sy))
        gx = max(0, min(w - 1, gx))
        gy = max(0, min(h - 1, gy))

        start_n = (sx, sy)
        goal_n = (gx, gy)

        frontier: list[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(frontier, (0.0, start_n))
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_n: None}
        cost_so_far: Dict[Tuple[int, int], float] = {start_n: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_n:
                break

            for nxt in self._neighbors(current):
                nx, ny = nxt
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue

                base_cost = 1.4 if nx != current[0] and ny != current[1] else 1.0
                obstacle_penalty = self.obstacle_cost if grid[ny, nx] > 0 else 0
                new_cost = cost_so_far[current] + base_cost + obstacle_penalty

                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    heuristic = ((gx - nx) ** 2 + (gy - ny) ** 2) ** 0.5
                    priority = new_cost + heuristic
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal_n not in came_from:
            return Path2D(points=[start, goal])

        path_cells = []
        cur = goal_n
        while cur is not None:
            path_cells.append(cur)
            cur = came_from[cur]
        path_cells.reverse()

        points = [
            Waypoint(
                x=(x - w // 2) * map_resolution_m,
                y=(y - h // 2) * map_resolution_m,
            )
            for x, y in path_cells
        ]
        return Path2D(points=points)
