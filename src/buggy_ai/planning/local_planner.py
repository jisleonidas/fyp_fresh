from __future__ import annotations

import math

import numpy as np

from buggy_ai.types import MotionCommand, OccupancyGrid2D, Path2D, Pose2D, VehicleState, Waypoint


class DWALocalPlanner:
    def __init__(self, cfg: dict) -> None:
        local = cfg["planning"]["local"]
        self.horizon_s = float(local["horizon_s"])
        self.dt_s = float(local["dt_s"])
        self.max_speed_mps = float(local["max_speed_mps"])
        self.min_speed_mps = float(local["min_speed_mps"])
        self.max_yaw_rate_rps = float(local["max_yaw_rate_rps"])
        self.samples_v = int(local["samples_v"])
        self.samples_w = int(local["samples_w"])

        self.w_goal = float(local["weights"]["goal"])
        self.w_clearance = float(local["weights"]["clearance"])
        self.w_heading = float(local["weights"]["heading"])
        self.w_speed = float(local["weights"]["speed"])

    def pick_local_goal(self, path: Path2D, pose: Pose2D) -> Waypoint:
        if not path.points:
            return Waypoint(pose.x + 1.0, pose.y)

        lookahead = 1.5
        best = path.points[-1]
        for pt in path.points:
            dist = math.hypot(pt.x - pose.x, pt.y - pose.y)
            if dist >= lookahead:
                best = pt
                break
        return best

    def _simulate_endpoint(self, state: VehicleState, v: float, w: float) -> tuple[float, float, float]:
        x, y, yaw = state.pose.x, state.pose.y, state.pose.yaw
        steps = int(self.horizon_s / self.dt_s)
        for _ in range(steps):
            yaw += w * self.dt_s
            x += v * math.cos(yaw) * self.dt_s
            y += v * math.sin(yaw) * self.dt_s
        return x, y, yaw

    def _clearance_score(self, endpoint_x: float, endpoint_y: float, occ: OccupancyGrid2D) -> float:
        gx = int((endpoint_x - occ.origin_x_m) / occ.resolution_m)
        gy = int((endpoint_y - occ.origin_y_m) / occ.resolution_m)
        if gx < 0 or gx >= occ.grid.shape[1] or gy < 0 or gy >= occ.grid.shape[0]:
            return -2.0

        if occ.grid[gy, gx] > 0:
            return -1.0

        radius = 6
        y0 = max(0, gy - radius)
        y1 = min(occ.grid.shape[0], gy + radius + 1)
        x0 = max(0, gx - radius)
        x1 = min(occ.grid.shape[1], gx + radius + 1)

        window = occ.grid[y0:y1, x0:x1]
        occupied = np.count_nonzero(window)
        total = max(1, window.size)
        return 1.0 - (occupied / total)

    def plan(self, state: VehicleState, local_goal: Waypoint, occ: OccupancyGrid2D) -> MotionCommand:
        best_score = -1e9
        best_cmd = MotionCommand(target_speed_mps=0.0, target_yaw_rate_rps=0.0)

        v_samples = np.linspace(self.min_speed_mps, self.max_speed_mps, self.samples_v)
        w_samples = np.linspace(-self.max_yaw_rate_rps, self.max_yaw_rate_rps, self.samples_w)

        for v in v_samples:
            for w in w_samples:
                ex, ey, eyaw = self._simulate_endpoint(state, float(v), float(w))

                dist_goal = math.hypot(local_goal.x - ex, local_goal.y - ey)
                score_goal = -dist_goal

                heading_to_goal = math.atan2(local_goal.y - ey, local_goal.x - ex)
                heading_error = abs((heading_to_goal - eyaw + math.pi) % (2 * math.pi) - math.pi)
                score_heading = -heading_error

                score_clearance = self._clearance_score(ex, ey, occ)
                score_speed = float(v) / max(1e-6, self.max_speed_mps)

                score = (
                    self.w_goal * score_goal
                    + self.w_heading * score_heading
                    + self.w_clearance * score_clearance
                    + self.w_speed * score_speed
                )

                if score > best_score:
                    best_score = score
                    best_cmd = MotionCommand(target_speed_mps=float(v), target_yaw_rate_rps=float(w))

        return best_cmd
