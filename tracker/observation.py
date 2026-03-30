from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.types import SeedProfile, TrackMode
from .cross_section_analyzer import CrossSectionAnalyzerV2
from .cross_section_profile import CrossSectionProfile
from .lane_state import LaneState


@dataclass
class TrackerObservation:
    profile: CrossSectionProfile
    indices: np.ndarray
    along_half_m: float
    lateral_half_m: float
    query_center_xyz: np.ndarray
    query_tangent_xy: np.ndarray


class LaneObservationEngine:
    def __init__(
        self,
        *,
        grid: Any,
        xyz: np.ndarray,
        intensity: np.ndarray,
        analyzer: CrossSectionAnalyzerV2,
        cfg: Any,
    ) -> None:
        self.grid = grid
        self.xyz = xyz
        self.intensity = intensity
        self.analyzer = analyzer
        self.cfg = cfg

    def resolve_search_strip(self, state: LaneState) -> tuple[float, float]:
        along = float(self.cfg.search_along_half_m)
        history_len = len(getattr(state, "history_centers", []))
        lateral = float(self.cfg.init_search_lateral_half_m) if history_len <= 1 else float(self.cfg.step_search_lateral_half_m)
        if state.mode == TrackMode.GAP_BRIDGING or float(state.profile_quality) < float(self.cfg.recovery_quality_threshold):
            along = max(along, float(self.cfg.gap_search_along_half_m))
            lateral = max(lateral, float(self.cfg.gap_search_lateral_half_m))
        return along, lateral

    def observe(
        self,
        *,
        center_xyz: np.ndarray,
        tangent_xy: np.ndarray,
        prev_state: LaneState | None,
        seed_profile: SeedProfile | None,
        along_half_m: float,
        lateral_half_m: float,
        is_gap_mode: bool,
    ) -> TrackerObservation:
        idx = self.grid.query_oriented_strip_xy(
            np.asarray(center_xyz[:2], dtype=np.float64),
            np.asarray(tangent_xy[:2], dtype=np.float64),
            float(along_half_m),
            float(lateral_half_m),
        )
        profile = self.analyzer.analyze(
            self.xyz,
            self.intensity,
            idx,
            center_xyz,
            tangent_xy,
            prev_state,
            seed_profile,
            is_gap_mode,
        )
        return TrackerObservation(
            profile=profile,
            indices=np.asarray(idx, dtype=np.int64),
            along_half_m=float(along_half_m),
            lateral_half_m=float(lateral_half_m),
            query_center_xyz=np.asarray(center_xyz, dtype=np.float64).copy(),
            query_tangent_xy=np.asarray(tangent_xy[:2], dtype=np.float64).copy(),
        )

    def observe_recovery(
        self,
        *,
        center_xyz: np.ndarray,
        tangent_xy: np.ndarray,
        prev_state: LaneState,
        seed_profile: SeedProfile | None,
        anchor_projected_frame: Callable[[LaneState, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        should_refresh_seed_profile: Callable[[LaneState], bool],
        refresh_seed_profile: Callable[[np.ndarray, SeedProfile], SeedProfile],
    ) -> TrackerObservation:
        along_half_m = max(float(self.cfg.search_along_half_m), float(self.cfg.gap_search_along_half_m))
        lateral_half_m = max(float(self.cfg.step_search_lateral_half_m), float(self.cfg.gap_search_lateral_half_m))
        query_center_xyz, query_tangent_xy = anchor_projected_frame(prev_state, center_xyz, tangent_xy)
        temp_seed = seed_profile
        if temp_seed is not None and should_refresh_seed_profile(prev_state):
            temp_seed = refresh_seed_profile(query_center_xyz, temp_seed)
        return self.observe(
            center_xyz=query_center_xyz,
            tangent_xy=query_tangent_xy,
            prev_state=prev_state,
            seed_profile=temp_seed,
            along_half_m=along_half_m,
            lateral_half_m=lateral_half_m,
            is_gap_mode=True,
        )
