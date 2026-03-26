from __future__ import annotations

import numpy as np

from ..core.oriented_query import project_points_xy


class CrosswalkDetector:
    def __init__(self, forward_half_m: float = 2.5, lateral_half_m: float = 2.0, bin_size_m: float = 0.10):
        self.forward_half_m = float(forward_half_m)
        self.lateral_half_m = float(lateral_half_m)
        self.bin_size_m = float(bin_size_m)

    def score(self, xyz: np.ndarray, intensity: np.ndarray, indices: np.ndarray, center_xyz: np.ndarray, tangent_xy: np.ndarray) -> tuple[float, dict]:
        if indices.size == 0:
            return 0.0, {"reason": "empty_indices"}
        pts = xyz[indices]
        along, lateral = project_points_xy(pts[:, :2], center_xyz[:2], tangent_xy)
        mask = ((along >= 0.0) & (along <= self.forward_half_m) & (np.abs(lateral) <= self.lateral_half_m))
        if not np.any(mask):
            return 0.0, {"reason": "empty_forward_window"}
        a = along[mask]
        w = intensity[indices][mask].astype(float)
        bins = np.arange(0.0, self.forward_half_m + self.bin_size_m, self.bin_size_m)
        hist, _ = np.histogram(a, bins=bins, weights=w)
        cnt, _ = np.histogram(a, bins=bins)
        hist = hist / np.maximum(cnt, 1)
        if hist.size < 3:
            return 0.0, {"reason": "too_few_bins"}
        peaks = 0
        thr = max(float(np.mean(hist)) + float(np.std(hist)), 0.0)
        for i in range(1, hist.size - 1):
            if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] >= thr:
                peaks += 1
        return min(1.0, peaks / 5.0), {"peaks": peaks, "hist_mean": float(np.mean(hist)), "hist_std": float(np.std(hist))}
