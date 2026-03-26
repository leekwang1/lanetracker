from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np

Vec2 = np.ndarray
Vec3 = np.ndarray


class TrackMode(str, Enum):
    SOLID_VISIBLE = "solid_visible"
    DASH_VISIBLE = "dash_visible"
    GAP_BRIDGING = "gap_bridging"
    CROSSWALK_CANDIDATE = "crosswalk_candidate"
    LOST = "lost"
    STOPPED = "stopped"


class StopReason(str, Enum):
    NONE = "none"
    MAX_DISTANCE = "max_distance"
    LOW_CONFIDENCE = "low_confidence"
    GAP_TOO_LONG = "gap_too_long"
    CROSSWALK = "crosswalk"
    WIDTH_INVALID = "width_invalid"
    USER_STOP = "user_stop"


@dataclass
class SeedProfile:
    target_intensity: float
    background_intensity: float
    z_ref: float


@dataclass
class StripeCandidate:
    left_m: float
    right_m: float
    center_m: float
    width_m: float
    peak_value: float
    prominence: float
    support_count: int
    integrated_energy: float
    symmetry_score: float
    strength_score: float = 0.0
    width_consistency: float = 0.0
    center_consistency: float = 0.0
    edge_consistency: float = 0.0
    identity_score: float = 0.0
    signal_consistency: float = 0.0
    switch_penalty: float = 0.0
    final_score: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepScores:
    stripe_fit: float = 0.0
    profile_quality: float = 0.0
    center_continuity: float = 0.0
    width_continuity: float = 0.0
    edge_continuity: float = 0.0
    heading_continuity: float = 0.0
    identity_score: float = 0.0
    visibility_score: float = 0.0
    switch_penalty: float = 0.0
    crosswalk_penalty: float = 0.0
    total: float = 0.0
