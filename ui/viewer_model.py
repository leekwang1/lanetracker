from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ViewerModel:
    xyz: np.ndarray | None = None
    xy: np.ndarray | None = None
    intensity: np.ndarray | None = None
    p0: np.ndarray | None = None
    p1: np.ndarray | None = None
    track_points: np.ndarray | None = None
    current_point: np.ndarray | None = None
    predicted_points: np.ndarray | None = None
    active_cell_box_groups: list[np.ndarray] | None = None
    segment_groups: list[np.ndarray] | None = None
    trajectory_line_points: np.ndarray | None = None
    profile_line_points: np.ndarray | None = None
    stripe_segment_points: np.ndarray | None = None
    stripe_edge_points: np.ndarray | None = None
    search_box_points: np.ndarray | None = None
    profile: object | None = None
    status_text: str = ""
    point_cloud_revision: int = 0
    messages: list[str] = field(default_factory=list)
