"""
stereo_depth_node.py

ROS2 node that computes depth from synchronized stereo images.
Publishes depth every stereo frame pair.

Subscriptions:
    /stereo/left/image_raw      (sensor_msgs/Image)
    /stereo/right/image_raw     (sensor_msgs/Image)

Publications:
    /stereo/depth               (sensor_msgs/Image, 32FC1, depth in meters)
    /stereo/disparity_debug     (sensor_msgs/Image, 32FC1, disparity in pixels)
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class StereoDepthNode(Node):
    def __init__(self) -> None:
        super().__init__("stereo_depth_node")

        self.declare_parameter("baseline_m", 0.12)
        self.declare_parameter("focal_px", 700.0)
        self.declare_parameter("min_disparity", 1)
        self.declare_parameter("num_disparities", 64)
        self.declare_parameter("sad_window", 5)
        self.declare_parameter("depth_min_m", 0.30)
        self.declare_parameter("depth_max_m", 20.0)

        self.baseline_m = float(self.get_parameter("baseline_m").value)
        self.focal_px = float(self.get_parameter("focal_px").value)
        self.min_disparity = int(self.get_parameter("min_disparity").value)
        self.num_disparities = int(self.get_parameter("num_disparities").value)
        self.sad_window = max(1, int(self.get_parameter("sad_window").value))
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)

        self._left_msg: Image | None = None
        self._right_msg: Image | None = None

        self.depth_pub = self.create_publisher(Image, "/stereo/depth", 10)
        self.disparity_pub = self.create_publisher(Image, "/stereo/disparity_debug", 10)

        self.create_subscription(Image, "/stereo/left/image_raw", self._left_callback, 10)
        self.create_subscription(Image, "/stereo/right/image_raw", self._right_callback, 10)

        self.get_logger().info("[StereoDepthNode] Waiting for stereo image pairs.")

    @staticmethod
    def _image_to_gray(msg: Image) -> np.ndarray:
        if msg.encoding in ("mono8", "8UC1"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return arr.copy()

        if msg.encoding in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == "rgb8":
                r = arr[:, :, 0].astype(np.float32)
                g = arr[:, :, 1].astype(np.float32)
                b = arr[:, :, 2].astype(np.float32)
            else:
                b = arr[:, :, 0].astype(np.float32)
                g = arr[:, :, 1].astype(np.float32)
                r = arr[:, :, 2].astype(np.float32)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return np.clip(gray, 0, 255).astype(np.uint8)

        raise ValueError(f"Unsupported encoding for stereo input: {msg.encoding}")

    @staticmethod
    def _mean_filter(img: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return img
        pad = k // 2
        padded = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
        out = np.zeros_like(img, dtype=np.float32)
        area = float(k * k)

        for y in range(img.shape[0]):
            y0, y1 = y, y + k
            for x in range(img.shape[1]):
                x0, x1 = x, x + k
                out[y, x] = float(np.sum(padded[y0:y1, x0:x1])) / area
        return out

    def _compute_disparity(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        left = left_gray.astype(np.float32)
        right = right_gray.astype(np.float32)

        if self.sad_window > 1:
            left = self._mean_filter(left, self.sad_window)
            right = self._mean_filter(right, self.sad_window)

        disparities = np.arange(self.min_disparity, self.min_disparity + self.num_disparities)
        cost = np.full((len(disparities), left.shape[0], left.shape[1]), np.inf, dtype=np.float32)

        for i, d in enumerate(disparities):
            shifted = np.roll(right, shift=d, axis=1)
            shifted[:, :d] = right[:, :1]
            cost[i] = np.abs(left - shifted)

        best = np.argmin(cost, axis=0)
        disparity = disparities[best].astype(np.float32)
        disparity[disparity <= 0.1] = np.nan
        return disparity

    def _disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        depth = (self.focal_px * self.baseline_m) / disparity
        depth = np.clip(depth, self.depth_min_m, self.depth_max_m)
        depth[np.isnan(depth)] = self.depth_max_m
        return depth.astype(np.float32)

    @staticmethod
    def _array_to_image(arr: np.ndarray, template: Image, encoding: str) -> Image:
        out = Image()
        out.header = template.header
        out.height = template.height
        out.width = template.width
        out.encoding = encoding
        out.is_bigendian = 0
        out.step = int(template.width * 4)
        out.data = arr.astype(np.float32).tobytes()
        return out

    def _publish_depth_from_pair(self) -> None:
        if self._left_msg is None or self._right_msg is None:
            return

        if self._left_msg.header.stamp != self._right_msg.header.stamp:
            # Keep simple nearest-pair policy for now; rely on upstream sync where possible.
            return

        try:
            left_gray = self._image_to_gray(self._left_msg)
            right_gray = self._image_to_gray(self._right_msg)
            disparity = self._compute_disparity(left_gray, right_gray)
            depth = self._disparity_to_depth(disparity)
        except Exception as exc:
            self.get_logger().error(f"[StereoDepthNode] Failed stereo processing: {exc}")
            return

        depth_msg = self._array_to_image(depth, self._left_msg, "32FC1")
        disp = np.nan_to_num(disparity, nan=0.0).astype(np.float32)
        disp_msg = self._array_to_image(disp, self._left_msg, "32FC1")

        self.depth_pub.publish(depth_msg)
        self.disparity_pub.publish(disp_msg)

    def _left_callback(self, msg: Image) -> None:
        self._left_msg = msg
        self._publish_depth_from_pair()

    def _right_callback(self, msg: Image) -> None:
        self._right_msg = msg
        self._publish_depth_from_pair()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
