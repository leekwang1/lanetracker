from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ProfileStripeCandidate:
    left_m: float
    right_m: float
    center_m: float
    width_m: float


@dataclass
class ProfileData:
    bins_center: np.ndarray
    hist_combined: np.ndarray
    smooth_hist: np.ndarray
    stripe_candidates: list[ProfileStripeCandidate] = field(default_factory=list)
    selected_idx: int | None = None
    quality: float = 0.0
