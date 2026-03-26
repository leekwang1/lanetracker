from __future__ import annotations

import laspy
import numpy as np


class LasData:
    def __init__(self, xyz: np.ndarray, intensity: np.ndarray):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.intensity = np.asarray(intensity, dtype=np.float32)


def load_las_xyz_intensity(path: str) -> LasData:
    las = laspy.read(path)
    xyz = np.empty((len(las.x), 3), dtype=np.float64)
    xyz[:, 0] = np.asarray(las.x, dtype=np.float64)
    xyz[:, 1] = np.asarray(las.y, dtype=np.float64)
    xyz[:, 2] = np.asarray(las.z, dtype=np.float64)
    if not hasattr(las, "intensity"):
        raise ValueError("LAS intensity attribute is required.")
    intensity = np.asarray(las.intensity, dtype=np.float32)
    return LasData(xyz=xyz, intensity=intensity)
