from __future__ import annotations

from ..core.types import StopReason
from ..tracker.lane_state import LaneState


class StopPolicy:
    def __init__(
        self,
        max_track_length_m: float = 100.0,
        min_center_confidence: float = 0.10,
        max_gap_steps: int = 14,
        enable_crosswalk_stop: bool = False,
        crosswalk_threshold: float = 0.75,
        min_width_m: float = 0.05,
        max_width_m: float = 0.35,
    ):
        self.max_track_length_m = float(max_track_length_m)
        self.min_center_confidence = float(min_center_confidence)
        self.max_gap_steps = int(max_gap_steps)
        self.enable_crosswalk_stop = bool(enable_crosswalk_stop)
        self.crosswalk_threshold = float(crosswalk_threshold)
        self.min_width_m = float(min_width_m)
        self.max_width_m = float(max_width_m)

    def evaluate(self, state: LaneState, crosswalk_score: float) -> StopReason:
        if state.total_length_m >= self.max_track_length_m:
            return StopReason.MAX_DISTANCE
        if state.center_confidence < self.min_center_confidence:
            return StopReason.LOW_CONFIDENCE
        if state.gap_run_steps > self.max_gap_steps:
            return StopReason.GAP_TOO_LONG
        if self.enable_crosswalk_stop and crosswalk_score >= self.crosswalk_threshold:
            return StopReason.CROSSWALK
        if not (self.min_width_m <= state.lane_width_m <= self.max_width_m):
            return StopReason.WIDTH_INVALID
        return StopReason.NONE
