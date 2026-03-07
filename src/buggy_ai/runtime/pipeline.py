from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from buggy_ai.control.controller import DriveController
from buggy_ai.localization.ekf import LocalizationEKF
from buggy_ai.perception.occupancy import OccupancyMapper
from buggy_ai.perception.stereo import StereoProcessor
from buggy_ai.planning.global_planner import AStarGlobalPlanner
from buggy_ai.planning.local_planner import DWALocalPlanner
from buggy_ai.safety.supervisor import SafetySupervisor
from buggy_ai.types import Path2D, Pose2D, VehicleState, Velocity2D, Waypoint


class AutonomyPipeline:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.stereo = StereoProcessor(cfg)
        self.occupancy = OccupancyMapper(cfg)
        self.localization = LocalizationEKF()
        self.global_planner = AStarGlobalPlanner(cfg)
        self.local_planner = DWALocalPlanner(cfg)
        self.controller = DriveController(cfg)
        self.safety = SafetySupervisor(cfg)

        self.global_map = np.zeros((120, 120), dtype=np.uint8)
        self.goal = Waypoint(x=8.0, y=0.0)
        self.path: Optional[Path2D] = None

        self.last_sensor_update = time.time()
        self.last_planner_update = time.time()
        self.last_control_update = time.time()

    def _mock_stereo_frames(self) -> tuple[np.ndarray, np.ndarray]:
        height, width = 240, 320
        left = np.random.randint(0, 80, (height, width), dtype=np.uint8)
        right = left.copy()

        cx = int(width * 0.65)
        cy = int(height * 0.5)
        radius = 35
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
        left[mask] = 220
        right[np.roll(mask, shift=-7, axis=1)] = 220
        return left, right

    def _simulate_odometry(self, state: VehicleState, dt: float) -> tuple[float, float, float]:
        v = state.velocity.v
        yaw_rate = state.velocity.yaw_rate
        dx = v * math.cos(state.pose.yaw) * dt
        dy = v * math.sin(state.pose.yaw) * dt
        return dx, dy, yaw_rate * dt

    def run(self) -> None:
        loop_hz = float(self.cfg["system"]["loop_hz"])
        max_runtime_s = float(self.cfg["system"]["max_runtime_s"])
        dt = 1.0 / loop_hz

        state = VehicleState(
            pose=Pose2D(0.0, 0.0, 0.0),
            velocity=Velocity2D(0.0, 0.0),
            timestamp=time.time(),
        )

        self.path = self.global_planner.plan(
            self.global_map,
            start=Waypoint(state.pose.x, state.pose.y),
            goal=self.goal,
            map_resolution_m=self.cfg["planning"]["global"]["map_resolution_m"],
        )

        start_time = time.time()
        while time.time() - start_time < max_runtime_s:
            tick_start = time.time()

            left, right = self._mock_stereo_frames()
            depth = self.stereo.compute_depth(left, right)
            self.last_sensor_update = time.time()

            grid = self.occupancy.from_depth(depth)

            dx, dy, dyaw = self._simulate_odometry(state, dt)
            state = self.localization.predict_and_update(state, dx, dy, dyaw)

            local_goal = self.local_planner.pick_local_goal(self.path, state.pose)
            motion_cmd = self.local_planner.plan(state, local_goal, grid)
            self.last_planner_update = time.time()

            actuator_cmd = self.controller.compute(state, motion_cmd)
            self.last_control_update = time.time()

            safe_cmd, safety_state = self.safety.gate_command(
                actuator_cmd=actuator_cmd,
                current_speed_mps=state.velocity.v,
                occupancy=grid,
                now=time.time(),
                last_sensor_update=self.last_sensor_update,
                last_planner_update=self.last_planner_update,
                last_control_update=self.last_control_update,
            )

            state.velocity.v = max(0.0, min(2.0, motion_cmd.target_speed_mps))
            state.velocity.yaw_rate = motion_cmd.target_yaw_rate_rps
            state.pose.yaw += state.velocity.yaw_rate * dt
            state.pose.x += state.velocity.v * math.cos(state.pose.yaw) * dt
            state.pose.y += state.velocity.v * math.sin(state.pose.yaw) * dt
            state.timestamp = time.time()

            print(
                f"state={safety_state.value} x={state.pose.x:.2f} y={state.pose.y:.2f} "
                f"v={state.velocity.v:.2f} steer={safe_cmd.steering_angle_rad:.2f} "
                f"thr={safe_cmd.throttle:.2f} brk={safe_cmd.brake:.2f}"
            )

            elapsed = time.time() - tick_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
