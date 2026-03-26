from __future__ import annotations

from .tracker.lane_tracker_v2 import TrackerV2Config


def default_config() -> TrackerV2Config:
    return TrackerV2Config()
