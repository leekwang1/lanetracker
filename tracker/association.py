from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .lane_state import LaneState
from .observation import TrackerObservation


@dataclass
class AssociationDecision:
    observation: TrackerObservation
    stripe_visible: bool
    stripe_rejected: bool
    salvage_narrow: bool
    source: str = "primary"
    reliable_checked: bool = False
    reliable_passed: bool = False
    loyalty_value: float | None = None
    rejection_reasons: list[str] = field(default_factory=list)


class LaneAssociator:
    def select_observation(
        self,
        observation: TrackerObservation,
        prev_state: LaneState,
        *,
        salvage_narrow_fn: Callable,
        reliable_fn: Callable,
        reference_history_fn: Callable | None = None,
        reference_tangent_fn: Callable | None = None,
        loyalty_term_fn: Callable | None = None,
        loyalty_gate_min_history: int = 0,
        loyalty_gate_min_visible: float = 0.0,
        loyalty_gate_quality_bypass: float = 1.0,
        source: str = "primary",
    ) -> AssociationDecision:
        profile = observation.profile
        stripe_visible = profile.selected_idx is not None
        stripe_rejected = False
        salvage_narrow = False
        reliable_checked = False
        reliable_passed = False
        loyalty_value: float | None = None
        rejection_reasons: list[str] = []
        if stripe_visible:
            sc = profile.stripe_candidates[profile.selected_idx]
            reliable_checked = True
            salvage_narrow = bool(salvage_narrow_fn(profile, sc, prev_state))
            reliable_passed = bool(reliable_fn(profile, sc, prev_state))
            stripe_visible = reliable_passed
            stripe_rejected = not stripe_visible
            if not reliable_passed:
                rejection_reasons.append("reliable_check_failed")
            if (
                stripe_visible
                and reference_history_fn is not None
                and reference_tangent_fn is not None
                and loyalty_term_fn is not None
                and prev_state.mode != prev_state.mode.GAP_BRIDGING
                and prev_state.gap_run_steps <= 0
            ):
                ref_history = reference_history_fn(prev_state)
                if len(ref_history) >= int(loyalty_gate_min_history) and float(profile.quality) < float(loyalty_gate_quality_bypass):
                    pred_dir = np.asarray(reference_tangent_fn(prev_state)[:2], dtype=np.float64)
                    pred_dir_norm = float(np.linalg.norm(pred_dir))
                    if pred_dir_norm > 1e-9:
                        pred_dir = pred_dir / pred_dir_norm
                        normal = np.array([-pred_dir[1], pred_dir[0]], dtype=np.float64)
                        candidate_center = np.asarray(observation.query_center_xyz, dtype=np.float64).copy()
                        candidate_center[:2] += normal * float(sc.center_m)
                        loyalty_value = float(loyalty_term_fn(ref_history, candidate_center, pred_dir))
                        if loyalty_value < float(loyalty_gate_min_visible):
                            stripe_visible = False
                            stripe_rejected = True
                            rejection_reasons.append("loyalty_gate_failed")
        return AssociationDecision(
            observation=observation,
            stripe_visible=stripe_visible,
            stripe_rejected=stripe_rejected,
            salvage_narrow=salvage_narrow,
            source=source,
            reliable_checked=reliable_checked,
            reliable_passed=reliable_passed,
            loyalty_value=loyalty_value,
            rejection_reasons=rejection_reasons,
        )
