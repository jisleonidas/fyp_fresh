from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class Velocity2D:
    v: float
    yaw_rate: float


@dataclass
class VehicleState:
    pose: Pose2D
    velocity: Velocity2D
    timestamp: float


@dataclass
class OccupancyGrid2D:
    grid: np.ndarray
    resolution_m: float
    origin_x_m: float
    origin_y_m: float


@dataclass
class Waypoint:
    x: float
    y: float


@dataclass
class Path2D:
    points: List[Waypoint] = field(default_factory=list)


@dataclass
class MotionCommand:
    target_speed_mps: float
    target_yaw_rate_rps: float


@dataclass
class ActuatorCommand:
    steering_angle_rad: float
    throttle: float
    brake: float


class SafetyState(str, Enum):
    INIT = "INIT"
    READY = "READY"
    AUTONOMOUS = "AUTONOMOUS"
    DEGRADED = "DEGRADED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class HealthStatus:
    sensor_fresh: bool
    planner_fresh: bool
    control_fresh: bool
    min_ttc_s: float
