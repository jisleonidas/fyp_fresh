from __future__ import annotations

import math

from buggy_ai.types import ActuatorCommand, MotionCommand, VehicleState


class PID:
    def __init__(self, kp: float, ki: float, kd: float, i_limit: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.integral = 0.0
        self.last_error = 0.0

    def step(self, target: float, measured: float, dt: float) -> float:
        error = target - measured
        self.integral += error * dt
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        derivative = (error - self.last_error) / max(dt, 1e-4)
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class DriveController:
    def __init__(self, cfg: dict) -> None:
        pp = cfg["control"]["pure_pursuit"]
        pid_cfg = cfg["control"]["speed_pid"]

        self.lookahead_min_m = float(pp["lookahead_min_m"])
        self.lookahead_gain = float(pp["lookahead_gain"])
        self.wheelbase_m = float(pp["wheelbase_m"])

        self.pid = PID(
            kp=float(pid_cfg["kp"]),
            ki=float(pid_cfg["ki"]),
            kd=float(pid_cfg["kd"]),
            i_limit=float(pid_cfg["i_limit"]),
        )

    def compute(self, state: VehicleState, motion: MotionCommand, dt: float = 0.05) -> ActuatorCommand:
        speed = max(0.0, state.velocity.v)
        lookahead = self.lookahead_min_m + self.lookahead_gain * speed

        curvature = 0.0
        if motion.target_speed_mps > 1e-3:
            curvature = motion.target_yaw_rate_rps / max(motion.target_speed_mps, 1e-3)

        steering = math.atan(self.wheelbase_m * curvature)

        u = self.pid.step(target=motion.target_speed_mps, measured=speed, dt=dt)

        throttle = max(0.0, min(1.0, u))
        brake = max(0.0, min(1.0, -u))

        steering = max(-0.6, min(0.6, steering))

        _ = lookahead
        return ActuatorCommand(steering_angle_rad=steering, throttle=throttle, brake=brake)
