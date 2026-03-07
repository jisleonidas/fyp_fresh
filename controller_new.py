"""
controller_new.py

ROS2 arbitration node for command fusion across:
  1) Pure pursuit node output
  2) Lane-keeping node output
  3) Obstacle occupancy grid from stereo perception

This node does not replace controller.py; it is a new integration module.

Subscriptions:
    /pure_pursuit/cmd_vel         (geometry_msgs/Twist)
    /lane_keep/cmd_vel            (geometry_msgs/Twist)      optional
    /cv/lane_offset               (std_msgs/Float32)         optional
    /cv/confidence                (std_msgs/Float32)         optional
    /perception/occupancy_grid    (nav_msgs/OccupancyGrid)
    /obstacle/nearest_distance    (std_msgs/Float32)         optional

Publishes:
    /cmd_vel                      (geometry_msgs/Twist)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32


@dataclass
class ArbitrationConfig:
    control_rate_hz: float = 20.0

    max_linear_mps: float = 1.5
    min_linear_mps: float = 0.0
    max_angular_rps: float = 1.2

    lane_max_trim_rps: float = 0.40
    lane_default_confidence: float = 0.0

    obstacle_influence_start_m: float = 4.0
    obstacle_stop_m: float = 0.65
    obstacle_stop_hold_s: float = 1.0
    obstacle_max_trim_rps: float = 0.90
    obstacle_min_clearance_weight: float = 1.4
    obstacle_heading_weight: float = 0.8

    stale_input_timeout_s: float = 0.5


class CommandArbiterV2:
    """
    Blend strategy:
      - Pure pursuit: baseline longitudinal and heading intent
      - Lane keeping: bounded angular trim
      - Obstacle planner: DWA-style sampled steering scored by occupancy clearance + heading
    """

    def __init__(self, cfg: ArbitrationConfig):
        self.cfg = cfg

        self.pp_linear = 0.0
        self.pp_angular = 0.0
        self.pp_stamp = 0.0

        self.lane_angular = 0.0
        self.lane_confidence = cfg.lane_default_confidence
        self.lane_stamp = 0.0

        self.occupancy_grid: np.ndarray | None = None
        self.grid_res = 0.1
        self.grid_origin_x = -6.0
        self.grid_origin_y = 0.0
        self.grid_width = 0
        self.grid_height = 0
        self.occ_stamp = 0.0

        self.nearest_obstacle_m = float("inf")

        self.stop_start_time: float | None = None

    def update_pure_pursuit(self, linear: float, angular: float, stamp: float) -> None:
        self.pp_linear = linear
        self.pp_angular = angular
        self.pp_stamp = stamp

    def update_lane(self, angular: float, confidence: float, stamp: float) -> None:
        self.lane_angular = angular
        self.lane_confidence = max(0.0, min(1.0, confidence))
        self.lane_stamp = stamp

    def update_occupancy(
        self,
        occupancy_msg: OccupancyGrid,
        occ_array: np.ndarray,
        stamp: float,
    ) -> None:
        self.occupancy_grid = occ_array
        self.grid_res = float(occupancy_msg.info.resolution)
        self.grid_origin_x = float(occupancy_msg.info.origin.position.x)
        self.grid_origin_y = float(occupancy_msg.info.origin.position.y)
        self.grid_width = int(occupancy_msg.info.width)
        self.grid_height = int(occupancy_msg.info.height)
        self.occ_stamp = stamp

    def update_nearest_obstacle(self, distance_m: float) -> None:
        if distance_m <= 0.0:
            self.nearest_obstacle_m = float("inf")
        else:
            self.nearest_obstacle_m = distance_m

    def _input_fresh(self, now: float, stamp: float) -> bool:
        return (now - stamp) <= self.cfg.stale_input_timeout_s

    def _projected_clearance_score(self, v: float, w: float, horizon_s: float = 1.6, dt_s: float = 0.1) -> float:
        if self.occupancy_grid is None:
            return 1.0

        x = 0.0
        y = 0.0
        yaw = 0.0
        steps = max(1, int(horizon_s / dt_s))

        for _ in range(steps):
            yaw += w * dt_s
            x += v * math.cos(yaw) * dt_s
            y += v * math.sin(yaw) * dt_s

            gx = int((y - self.grid_origin_x) / self.grid_res)
            gy = int((x - self.grid_origin_y) / self.grid_res)

            if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
                return -2.0

            if self.occupancy_grid[gy, gx] > 50:
                return -1.5

        gx_end = int((y - self.grid_origin_x) / self.grid_res)
        gy_end = int((x - self.grid_origin_y) / self.grid_res)
        radius = max(1, int(0.8 / self.grid_res))

        y0 = max(0, gy_end - radius)
        y1 = min(self.grid_height, gy_end + radius + 1)
        x0 = max(0, gx_end - radius)
        x1 = min(self.grid_width, gx_end + radius + 1)

        window = self.occupancy_grid[y0:y1, x0:x1]
        occ_ratio = float(np.count_nonzero(window > 50)) / max(1, window.size)
        return 1.0 - occ_ratio

    def _obstacle_trim(self, base_linear: float, base_angular: float) -> tuple[float, float]:
        if self.occupancy_grid is None:
            return base_linear, base_angular

        candidate_ws = np.linspace(
            -self.cfg.obstacle_max_trim_rps,
            self.cfg.obstacle_max_trim_rps,
            13,
        )

        best_w = base_angular
        best_score = -1e9

        for w in candidate_ws:
            clearance = self._projected_clearance_score(base_linear, float(w))
            heading = -abs(float(w) - base_angular)
            score = (
                self.cfg.obstacle_min_clearance_weight * clearance
                + self.cfg.obstacle_heading_weight * heading
            )
            if score > best_score:
                best_score = score
                best_w = float(w)

        nearest = self.nearest_obstacle_m
        if math.isfinite(nearest):
            if nearest <= self.cfg.obstacle_stop_m:
                scaled_linear = 0.0
            elif nearest < self.cfg.obstacle_influence_start_m:
                ratio = (nearest - self.cfg.obstacle_stop_m) / max(
                    1e-3,
                    self.cfg.obstacle_influence_start_m - self.cfg.obstacle_stop_m,
                )
                scaled_linear = base_linear * max(0.2, min(1.0, ratio))
            else:
                scaled_linear = base_linear
        else:
            scaled_linear = base_linear

        return scaled_linear, best_w

    def blend(self) -> tuple[float, float]:
        now = time.time()

        if not self._input_fresh(now, self.pp_stamp):
            return 0.0, 0.0

        linear = self.pp_linear
        angular = self.pp_angular

        if self._input_fresh(now, self.lane_stamp):
            lane_trim = self.lane_angular * self.lane_confidence
            lane_trim = max(-self.cfg.lane_max_trim_rps, min(self.cfg.lane_max_trim_rps, lane_trim))
            angular += lane_trim

        if self._input_fresh(now, self.occ_stamp):
            linear, angular = self._obstacle_trim(linear, angular)

        if self.nearest_obstacle_m <= self.cfg.obstacle_stop_m:
            if self.stop_start_time is None:
                self.stop_start_time = now
            if (now - self.stop_start_time) >= self.cfg.obstacle_stop_hold_s:
                return 0.0, 0.0
        else:
            self.stop_start_time = None

        linear = max(self.cfg.min_linear_mps, min(self.cfg.max_linear_mps, linear))
        angular = max(-self.cfg.max_angular_rps, min(self.cfg.max_angular_rps, angular))

        return linear, angular


class ControllerNewNode(Node):
    def __init__(self) -> None:
        super().__init__("controller_new")

        self.cfg = ArbitrationConfig()
        self.arbiter = CommandArbiterV2(self.cfg)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_subscription(Twist, "/pure_pursuit/cmd_vel", self._pp_callback, 10)
        self.create_subscription(Twist, "/lane_keep/cmd_vel", self._lane_cmd_callback, 10)
        self.create_subscription(Float32, "/cv/lane_offset", self._lane_offset_callback, 10)
        self.create_subscription(Float32, "/cv/confidence", self._lane_conf_callback, 10)
        self.create_subscription(OccupancyGrid, "/perception/occupancy_grid", self._occupancy_callback, 10)
        self.create_subscription(Float32, "/obstacle/nearest_distance", self._nearest_callback, 10)

        self._lane_offset_m = 0.0
        self._lane_conf = self.cfg.lane_default_confidence

        self.create_timer(1.0 / self.cfg.control_rate_hz, self._control_tick)
        self.get_logger().info("[ControllerNew] Running arbitration over PP + lane + occupancy.")

    def _pp_callback(self, msg: Twist) -> None:
        self.arbiter.update_pure_pursuit(msg.linear.x, msg.angular.z, time.time())

    def _lane_cmd_callback(self, msg: Twist) -> None:
        self.arbiter.update_lane(msg.angular.z, self._lane_conf, time.time())

    def _lane_offset_callback(self, msg: Float32) -> None:
        self._lane_offset_m = msg.data
        lane_k = 0.7
        inferred_angular = -lane_k * self._lane_offset_m
        self.arbiter.update_lane(inferred_angular, self._lane_conf, time.time())

    def _lane_conf_callback(self, msg: Float32) -> None:
        self._lane_conf = max(0.0, min(1.0, msg.data))

    def _occupancy_callback(self, msg: OccupancyGrid) -> None:
        data = np.array(msg.data, dtype=np.int16)
        if data.size != msg.info.width * msg.info.height:
            self.get_logger().warning("[ControllerNew] OccupancyGrid size mismatch; frame ignored.")
            return

        occ = data.reshape((msg.info.height, msg.info.width))
        self.arbiter.update_occupancy(msg, occ, time.time())

    def _nearest_callback(self, msg: Float32) -> None:
        self.arbiter.update_nearest_obstacle(msg.data)

    def _control_tick(self) -> None:
        linear, angular = self.arbiter.blend()
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerNewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
