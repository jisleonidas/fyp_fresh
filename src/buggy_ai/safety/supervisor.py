from __future__ import annotations

import math

import numpy as np

from buggy_ai.types import ActuatorCommand, OccupancyGrid2D, SafetyState


class SafetySupervisor:
    def __init__(self, cfg: dict) -> None:
        safety = cfg["safety"]
        self.sensor_timeout_s = float(safety["sensor_timeout_s"])
        self.planner_timeout_s = float(safety["planner_timeout_s"])
        self.ttc_threshold_s = float(safety["ttc_threshold_s"])
        self.degraded_speed_limit_mps = float(safety["degraded_speed_limit_mps"])
        self.state = SafetyState.INIT

    def _estimate_min_ttc(self, occupancy: OccupancyGrid2D, speed: float) -> float:
        if speed <= 0.05:
            return 999.0

        center_x = occupancy.grid.shape[1] // 2
        col = occupancy.grid[:, center_x]
        obstacle_indices = np.where(col > 0)[0]
        if obstacle_indices.size == 0:
            return 999.0

        nearest_cell = obstacle_indices.min()
        distance_m = nearest_cell * occupancy.resolution_m
        return distance_m / max(speed, 1e-3)

    def gate_command(
        self,
        actuator_cmd: ActuatorCommand,
        current_speed_mps: float,
        occupancy: OccupancyGrid2D,
        now: float,
        last_sensor_update: float,
        last_planner_update: float,
        last_control_update: float,
    ) -> tuple[ActuatorCommand, SafetyState]:
        sensor_fresh = (now - last_sensor_update) <= self.sensor_timeout_s
        planner_fresh = (now - last_planner_update) <= self.planner_timeout_s
        control_fresh = (now - last_control_update) <= self.planner_timeout_s

        min_ttc = self._estimate_min_ttc(occupancy, current_speed_mps)

        if min_ttc < self.ttc_threshold_s or (not sensor_fresh) or (not planner_fresh) or (not control_fresh):
            self.state = SafetyState.EMERGENCY_STOP
        elif self.state == SafetyState.INIT:
            self.state = SafetyState.READY
        elif self.state in (SafetyState.READY, SafetyState.DEGRADED):
            self.state = SafetyState.AUTONOMOUS

        if self.state == SafetyState.EMERGENCY_STOP:
            return ActuatorCommand(steering_angle_rad=0.0, throttle=0.0, brake=1.0), self.state

        if not sensor_fresh:
            self.state = SafetyState.DEGRADED
            capped = max(0.0, min(actuator_cmd.throttle, self.degraded_speed_limit_mps / 2.0))
            return ActuatorCommand(
                steering_angle_rad=actuator_cmd.steering_angle_rad,
                throttle=capped,
                brake=max(actuator_cmd.brake, 0.2),
            ), self.state

        return actuator_cmd, self.state
