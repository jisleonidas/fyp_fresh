from __future__ import annotations

import numpy as np


class StereoProcessor:
    def __init__(self, cfg: dict) -> None:
        cam_cfg = cfg["camera"]
        self.baseline_m = float(cam_cfg["baseline_m"])
        self.focal_px = float(cam_cfg["focal_px"])
        self.depth_min_m = float(cam_cfg["depth_min_m"])
        self.depth_max_m = float(cam_cfg["depth_max_m"])
        self.min_disparity = int(cam_cfg["min_disparity"])
        self.num_disparities = int(cam_cfg["num_disparities"])

    def compute_depth(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        left = left_gray.astype(np.float32)
        right = right_gray.astype(np.float32)

        disparities = np.arange(self.min_disparity, self.min_disparity + self.num_disparities)
        cost_volume = np.full((len(disparities), left.shape[0], left.shape[1]), np.inf, dtype=np.float32)

        for idx, d in enumerate(disparities):
            shifted = np.roll(right, shift=d, axis=1)
            if d > 0:
                shifted[:, :d] = right[:, :1]
            cost_volume[idx] = np.abs(left - shifted)

        best_idx = np.argmin(cost_volume, axis=0)
        disparity = disparities[best_idx].astype(np.float32)
        disparity[disparity <= 0.1] = np.nan

        depth = (self.focal_px * self.baseline_m) / disparity
        depth = np.clip(depth, self.depth_min_m, self.depth_max_m)
        depth[np.isnan(depth)] = self.depth_max_m
        return depth
