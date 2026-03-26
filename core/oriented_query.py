from __future__ import annotations

import numpy as np


def normalize_xy(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vv = np.asarray(v[:2], dtype=float)
    n = float(np.linalg.norm(vv))
    if n < eps:
        return np.array([1.0, 0.0], dtype=float)
    return vv / n


def make_frame(tangent_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = normalize_xy(tangent_xy)
    n = np.array([-t[1], t[0]], dtype=float)
    return t, n


def project_points_xy(pts_xy: np.ndarray, origin_xy: np.ndarray, tangent_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, n = make_frame(tangent_xy)
    d = pts_xy - origin_xy[None, :]
    along = d @ t
    lateral = d @ n
    return along, lateral


def filter_oriented_strip(
    pts_xy: np.ndarray,
    origin_xy: np.ndarray,
    tangent_xy: np.ndarray,
    along_half: float,
    lateral_half: float,
) -> np.ndarray:
    along, lateral = project_points_xy(pts_xy, origin_xy, tangent_xy)
    return (np.abs(along) <= along_half) & (np.abs(lateral) <= lateral_half)
