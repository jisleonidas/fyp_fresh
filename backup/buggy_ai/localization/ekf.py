from __future__ import annotations

from buggy_ai.types import Pose2D, VehicleState, Velocity2D


class LocalizationEKF:
    def predict_and_update(self, state: VehicleState, dx: float, dy: float, dyaw: float) -> VehicleState:
        return VehicleState(
            pose=Pose2D(
                x=state.pose.x + dx,
                y=state.pose.y + dy,
                yaw=state.pose.yaw + dyaw,
            ),
            velocity=Velocity2D(v=state.velocity.v, yaw_rate=state.velocity.yaw_rate),
            timestamp=state.timestamp,
        )
