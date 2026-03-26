from __future__ import annotations

from ..core.types import TrackMode


class LaneStateMachine:
    def __init__(self, dashed_threshold: float = 0.65, solid_threshold: float = 0.75, crosswalk_threshold: float = 0.75):
        self.dashed_threshold = float(dashed_threshold)
        self.solid_threshold = float(solid_threshold)
        self.crosswalk_threshold = float(crosswalk_threshold)

    def update(self, stripe_visible: bool, dashed_prob: float, solid_prob: float, crosswalk_score: float) -> TrackMode:
        if crosswalk_score >= self.crosswalk_threshold:
            return TrackMode.CROSSWALK_CANDIDATE
        if stripe_visible:
            if dashed_prob >= self.dashed_threshold:
                return TrackMode.DASH_VISIBLE
            if solid_prob >= self.solid_threshold:
                return TrackMode.SOLID_VISIBLE
            return TrackMode.SOLID_VISIBLE
        return TrackMode.GAP_BRIDGING
