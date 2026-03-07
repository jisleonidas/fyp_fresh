from __future__ import annotations

import numpy as np

from buggy_ai.types import OccupancyGrid2D


class OccupancyMapper:
    def __init__(self, cfg: dict) -> None:
        occ = cfg["occupancy"]
        self.resolution_m = float(occ["resolution_m"])
        self.width = int(occ["width_cells"])
        self.height = int(occ["height_cells"])
        self.origin_x_m = float(occ["origin_x_m"])
        self.origin_y_m = float(occ["origin_y_m"])
        self.inflation_radius_m = float(occ["inflation_radius_m"])
        self.obstacle_height_threshold_m = float(occ["obstacle_height_threshold_m"])

    def from_depth(self, depth: np.ndarray) -> OccupancyGrid2D:
        h, w = depth.shape
        cx = w / 2.0
        cy = h / 2.0
        fx = 700.0
        fy = 700.0

        sample = depth[::2, ::2]
        yy, xx = np.indices(sample.shape)
        x_pix = xx * 2.0
        y_pix = yy * 2.0

        z = sample
        x = (x_pix - cx) * z / fx
        y = (y_pix - cy) * z / fy

        forward = z
        lateral = x
        up = -y

        obstacle_mask = (up > self.obstacle_height_threshold_m) & (forward > 0.2) & (forward < 8.0)

        occ = np.zeros((self.height, self.width), dtype=np.uint8)
        gx = ((lateral[obstacle_mask] - self.origin_x_m) / self.resolution_m).astype(int)
        gy = ((forward[obstacle_mask] - self.origin_y_m) / self.resolution_m).astype(int)

        valid = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        occ[gy[valid], gx[valid]] = 1

        inflate_cells = max(1, int(self.inflation_radius_m / self.resolution_m))
        pad = inflate_cells
        padded = np.pad(occ, ((pad, pad), (pad, pad)), mode="constant")
        occ_inflated = np.zeros_like(occ)
        for y in range(occ.shape[0]):
            y0 = y
            y1 = y + 2 * pad + 1
            for x in range(occ.shape[1]):
                x0 = x
                x1 = x + 2 * pad + 1
                occ_inflated[y, x] = 1 if np.any(padded[y0:y1, x0:x1]) else 0

        return OccupancyGrid2D(
            grid=occ_inflated,
            resolution_m=self.resolution_m,
            origin_x_m=self.origin_x_m,
            origin_y_m=self.origin_y_m,
        )
