from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..core.types import TrackMode, Vec2, Vec3


@dataclass
class LaneState:
    center_xyz: Vec3
    tangent_xy: Vec2
    curvature: float
    lane_width_m: float
    left_edge_m: float
    right_edge_m: float
    stripe_center_m: float
    stripe_strength: float
    profile_quality: float
    center_confidence: float
    identity_confidence: float
    dashed_prob: float
    solid_prob: float
    visible_run_steps: int
    gap_run_steps: int
    total_length_m: float
    mode: TrackMode
    history_centers: list[Vec3] = field(default_factory=list)
    history_widths: list[float] = field(default_factory=list)
    history_tangents: list[Vec2] = field(default_factory=list)
    history_visibility: list[int] = field(default_factory=list)
    history_modes: list[str] = field(default_factory=list)

    def copy_shallow(self) -> "LaneState":
        return LaneState(
            center_xyz=self.center_xyz.copy(),
            tangent_xy=self.tangent_xy.copy(),
            curvature=float(self.curvature),
            lane_width_m=float(self.lane_width_m),
            left_edge_m=float(self.left_edge_m),
            right_edge_m=float(self.right_edge_m),
            stripe_center_m=float(self.stripe_center_m),
            stripe_strength=float(self.stripe_strength),
            profile_quality=float(self.profile_quality),
            center_confidence=float(self.center_confidence),
            identity_confidence=float(self.identity_confidence),
            dashed_prob=float(self.dashed_prob),
            solid_prob=float(self.solid_prob),
            visible_run_steps=int(self.visible_run_steps),
            gap_run_steps=int(self.gap_run_steps),
            total_length_m=float(self.total_length_m),
            mode=self.mode,
            history_centers=list(self.history_centers),
            history_widths=list(self.history_widths),
            history_tangents=list(self.history_tangents),
            history_visibility=list(self.history_visibility),
            history_modes=list(self.history_modes),
        )
