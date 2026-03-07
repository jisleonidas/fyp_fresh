"""
occupancy_grid_publisher.py

ROS2 node that converts stereo depth frames into a local occupancy grid.
The grid is re-generated and published on every depth frame.

Subscriptions:
    /stereo/depth                       (sensor_msgs/Image, 32FC1 or 16UC1)

Publications:
    /perception/occupancy_grid          (nav_msgs/OccupancyGrid)
    /obstacle/nearest_distance          (std_msgs/Float32)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class OccupancyGridPublisher(Node):
    def __init__(self) -> None:
        super().__init__("occupancy_grid_publisher")

        self.declare_parameter("grid_resolution_m", 0.10)
        self.declare_parameter("grid_width_cells", 120)
        self.declare_parameter("grid_height_cells", 120)
        self.declare_parameter("origin_x_m", -6.0)
        self.declare_parameter("origin_y_m", 0.0)
        self.declare_parameter("camera_fx_px", 700.0)
        self.declare_parameter("camera_fy_px", 700.0)
        self.declare_parameter("camera_cx_px", 320.0)
        self.declare_parameter("camera_cy_px", 240.0)
        self.declare_parameter("obstacle_height_min_m", -0.20)
        self.declare_parameter("obstacle_height_max_m", 1.50)
        self.declare_parameter("depth_min_m", 0.30)
        self.declare_parameter("depth_max_m", 15.0)
        self.declare_parameter("inflation_radius_m", 0.35)
        self.declare_parameter("downsample_stride", 2)

        self.grid_resolution_m = float(self.get_parameter("grid_resolution_m").value)
        self.grid_width_cells = int(self.get_parameter("grid_width_cells").value)
        self.grid_height_cells = int(self.get_parameter("grid_height_cells").value)
        self.origin_x_m = float(self.get_parameter("origin_x_m").value)
        self.origin_y_m = float(self.get_parameter("origin_y_m").value)

        self.fx = float(self.get_parameter("camera_fx_px").value)
        self.fy = float(self.get_parameter("camera_fy_px").value)
        self.cx = float(self.get_parameter("camera_cx_px").value)
        self.cy = float(self.get_parameter("camera_cy_px").value)

        self.obstacle_height_min_m = float(self.get_parameter("obstacle_height_min_m").value)
        self.obstacle_height_max_m = float(self.get_parameter("obstacle_height_max_m").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.inflation_radius_m = float(self.get_parameter("inflation_radius_m").value)
        self.downsample_stride = max(1, int(self.get_parameter("downsample_stride").value))

        self.occupancy_pub = self.create_publisher(OccupancyGrid, "/perception/occupancy_grid", 10)
        self.nearest_pub = self.create_publisher(Float32, "/obstacle/nearest_distance", 10)

        self.create_subscription(Image, "/stereo/depth", self._depth_callback, 10)
        self.get_logger().info("[OccupancyGridPublisher] Started. Waiting for /stereo/depth frames.")

    @staticmethod
    def _to_numpy_depth(msg: Image) -> np.ndarray:
        if msg.encoding == "32FC1":
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            return arr.copy()

        if msg.encoding == "16UC1":
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            return (arr.astype(np.float32) / 1000.0).copy()

        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")

    @staticmethod
    def _dilate(binary_grid: np.ndarray, cells: int) -> np.ndarray:
        if cells <= 0:
            return binary_grid

        padded = np.pad(binary_grid, ((cells, cells), (cells, cells)), mode="constant")
        out = np.zeros_like(binary_grid)
        window = 2 * cells + 1

        for gy in range(binary_grid.shape[0]):
            y0, y1 = gy, gy + window
            for gx in range(binary_grid.shape[1]):
                x0, x1 = gx, gx + window
                out[gy, gx] = 1 if np.any(padded[y0:y1, x0:x1]) else 0

        return out

    def _project_to_grid(self, depth_m: np.ndarray) -> Tuple[np.ndarray, float]:
        depth = depth_m[:: self.downsample_stride, :: self.downsample_stride]
        h, w = depth.shape

        yy, xx = np.indices((h, w), dtype=np.float32)
        xx = xx * self.downsample_stride
        yy = yy * self.downsample_stride

        z = depth
        valid_depth = np.isfinite(z) & (z > self.depth_min_m) & (z < self.depth_max_m)

        x = (xx - self.cx) * z / self.fx
        y = (yy - self.cy) * z / self.fy

        forward = z
        lateral = x
        up = -y

        obstacle_mask = (
            valid_depth
            & (up >= self.obstacle_height_min_m)
            & (up <= self.obstacle_height_max_m)
            & (forward > 0.0)
        )

        grid = np.zeros((self.grid_height_cells, self.grid_width_cells), dtype=np.uint8)

        gx = ((lateral[obstacle_mask] - self.origin_x_m) / self.grid_resolution_m).astype(np.int32)
        gy = ((forward[obstacle_mask] - self.origin_y_m) / self.grid_resolution_m).astype(np.int32)

        valid = (
            (gx >= 0)
            & (gx < self.grid_width_cells)
            & (gy >= 0)
            & (gy < self.grid_height_cells)
        )
        grid[gy[valid], gx[valid]] = 1

        inflation_cells = int(math.ceil(self.inflation_radius_m / self.grid_resolution_m))
        grid = self._dilate(grid, inflation_cells)

        if np.any(obstacle_mask):
            nearest = float(np.min(forward[obstacle_mask]))
        else:
            nearest = float("inf")

        return grid, nearest

    def _publish_occupancy(self, grid: np.ndarray, stamp) -> None:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"

        msg.info.map_load_time = stamp
        msg.info.resolution = float(self.grid_resolution_m)
        msg.info.width = self.grid_width_cells
        msg.info.height = self.grid_height_cells
        msg.info.origin.position.x = float(self.origin_x_m)
        msg.info.origin.position.y = float(self.origin_y_m)
        msg.info.origin.orientation.w = 1.0

        # ROS OccupancyGrid convention: -1 unknown, 0 free, 100 occupied
        data = np.where(grid > 0, 100, 0).astype(np.int8)
        msg.data = data.flatten().tolist()

        self.occupancy_pub.publish(msg)

    def _depth_callback(self, msg: Image) -> None:
        try:
            depth = self._to_numpy_depth(msg)
            grid, nearest = self._project_to_grid(depth)
        except Exception as exc:
            self.get_logger().error(f"[OccupancyGridPublisher] Failed to process frame: {exc}")
            return

        self._publish_occupancy(grid, msg.header.stamp)

        nearest_msg = Float32()
        nearest_msg.data = nearest if math.isfinite(nearest) else -1.0
        self.nearest_pub.publish(nearest_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OccupancyGridPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
