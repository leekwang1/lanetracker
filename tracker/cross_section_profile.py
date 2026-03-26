from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..core.types import StripeCandidate


@dataclass
class CrossSectionProfile:
    bins_center: np.ndarray
    hist_intensity: np.ndarray
    hist_support: np.ndarray
    hist_combined: np.ndarray
    smooth_hist: np.ndarray
    peak_indices: list[int] = field(default_factory=list)
    stripe_candidates: list[StripeCandidate] = field(default_factory=list)
    selected_idx: int | None = None
    quality: float = 0.0
    neighbor_count: int = 0
    switch_risk: float = 0.0
    crosswalk_like_score: float = 0.0
    debug: dict = field(default_factory=dict)
