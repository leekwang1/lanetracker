from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..core.spatial_grid import SpatialGrid
from ..core.types import SeedProfile, StopReason, StepScores, TrackMode
from .association import LaneAssociator
from .cross_section_analyzer import CrossSectionAnalyzerV2, CrossSectionAnalyzerV2Config
from .cross_section_profile import CrossSectionProfile
from .lane_hypothesis import LaneHypothesis
from .lane_state import LaneState
from .observation import LaneObservationEngine, TrackerObservation
from .state_machine import LaneStateMachine
from ..detectors.crosswalk_detector import CrosswalkDetector
from ..detectors.line_type_classifier import LineTypeClassifier
from ..detectors.stop_policy import StopPolicy


@dataclass
class TrackerV2Config:
    grid_cell_m: float = 0.10
    step_length_m: float = 0.20
    hypothesis_keep_top_k: int = 3
    search_along_half_m: float = 0.35
    search_lateral_half_m: float = 0.2333333333333333
    init_search_lateral_half_m: float = 0.11666666666666665
    step_search_lateral_half_m: float = 0.15
    lane_width_init_m: float = 0.15
    z_local_radius_m: float = 0.20
    max_heading_change_deg: float = 10.0
    heading_branch_count: int = 3
    lateral_branch_offsets_m: tuple[float, ...] = (-0.03, 0.0, 0.03)
    max_track_length_m: float = 100.0
    seed_radius_m: float = 0.20
    profile_update_radius_m: float = 0.20
    profile_update_alpha: float = 0.12
    profile_update_min_quality: float = 0.72
    profile_update_min_identity: float = 0.70
    profile_update_min_width_m: float = 0.10
    profile_update_max_width_m: float = 0.30
    center_z_fit_radius_m: float = 0.20
    z_fit_window_m: float = 0.18
    save_debug_json: bool = True
    display_candidate_top_k: int = 5
    profile_center_correction_gain: float = 0.20
    enable_center_refinement: bool = True
    center_refine_radius_m: float = 0.23
    center_refine_max_shift_m: float = 0.08
    center_refine_extra_max_shift_m: float = 0.025
    cross_section_radius_m: float = 0.25
    cross_section_forward_window_m: float = 0.15
    cross_section_bin_size_m: float = 0.02
    cross_section_stripe_threshold_ratio: float = 0.35
    cross_section_max_lane_half_width_m: float = 0.14
    cross_section_center_mix: float = 0.65
    enable_motion_tangent_realign: bool = True
    motion_tangent_mix: float = 0.75
    min_motion_tangent_step_m: float = 0.03
    gap_bridge_step_scale: float = 0.50
    gap_search_along_half_m: float = 0.45
    gap_search_lateral_half_m: float = 0.2333333333333333
    recovery_quality_threshold: float = 0.60
    enable_crosswalk_stop: bool = False
    stop_crosswalk_threshold: float = 0.90
    retry_recovery_on_reject: bool = True
    min_profile_quality_visible: float = 0.45
    min_single_candidate_quality_visible: float = 0.45
    min_candidate_width_visible_m: float = 0.08
    max_candidate_center_jump_m: float = 0.10
    recovery_profile_quality_visible: float = 0.28
    recovery_min_candidate_width_m: float = 0.10
    recovery_max_candidate_width_m: float = 0.32
    recovery_max_center_jump_m: float = 0.35
    recovery_min_peak_visible: float = 0.10
    min_single_candidate_identity_visible: float = 0.65
    min_single_candidate_peak_visible: float = 0.35
    single_candidate_salvage_quality: float = 0.24
    single_candidate_salvage_width_m: float = 0.095
    single_candidate_salvage_peak: float = 0.10
    single_candidate_salvage_identity: float = 0.15
    single_candidate_salvage_center_consistency: float = 0.45
    single_candidate_salvage_center_jump_m: float = 0.12
    single_candidate_salvage_max_width_m: float = 0.26
    single_candidate_salvage_force_quality: float = 0.22
    single_candidate_salvage_force_peak: float = 0.05
    single_candidate_salvage_force_center_consistency: float = 0.20
    min_narrow_candidate_center_consistency: float = 0.75
    narrow_candidate_salvage_quality: float = 0.28
    narrow_candidate_salvage_peak: float = 0.20
    narrow_candidate_salvage_center_jump_m: float = 0.08
    narrow_candidate_width_blend: float = 0.85
    normal_mode_min_candidate_width_m: float = 0.14
    normal_mode_single_candidate_min_width_m: float = 0.16
    hard_center_offset_limit_m: float = 0.25
    max_z_step_m: float = 0.12
    hard_max_z_step_m: float = 0.15
    lane_loyalty_history_points: int = 8
    lane_loyalty_tolerance_m: float = 0.135
    lane_loyalty_weight: float = 0.75
    loyalty_gate_min_history: int = 6
    loyalty_gate_min_visible: float = 0.25
    loyalty_gate_quality_bypass: float = 0.92
    trusted_history_min_points: int = 4
    trusted_update_min_quality: float = 0.76
    trusted_update_min_loyalty: float = 0.55
    trusted_update_min_identity: float = 0.62
    trusted_update_min_width_m: float = 0.10
    trusted_update_max_width_m: float = 0.30
    enable_trajectory_fit: bool = True
    trajectory_fit_history_points: int = 8
    trajectory_fit_min_points: int = 4
    trajectory_tangent_blend: float = 0.55
    trajectory_follow_weight: float = 0.90
    trajectory_follow_sigma_m: float = 0.10
    trajectory_debug_half_length_m: float = 0.80


@dataclass
class StepDebugFrame:
    step_index: int
    predicted_centers: list[np.ndarray] = field(default_factory=list)
    selected_center: np.ndarray | None = None
    hypothesis_scores: list[float] = field(default_factory=list)
    selected_idx: int | None = None
    cross_section_profile: CrossSectionProfile | None = None
    dashed_prob: float = 0.0
    solid_prob: float = 0.0
    crosswalk_score: float = 0.0
    mode: str = ""
    stop_reason: str = ""
    switch_risk: float = 0.0
    stripe_rejected: bool = False
    search_along_half_m: float = 0.0
    search_lateral_half_m: float = 0.0
    query_debug: dict = field(default_factory=dict)
    candidate_summaries: list[str] = field(default_factory=list)
    trajectory_line_points: np.ndarray | None = None


@dataclass
class TrackResultV2:
    dense_points: np.ndarray
    output_points: np.ndarray
    stop_reason: str
    debug_frames: list[StepDebugFrame]


class LaneTrackerV2:
    def __init__(self, xyz: np.ndarray, intensity: np.ndarray, cfg: TrackerV2Config):
        self.xyz = np.asarray(xyz, dtype=float)
        self.xy = self.xyz[:, :2]
        self.intensity = np.asarray(intensity, dtype=float)
        self.cfg = cfg
        self.grid = SpatialGrid(self.xy, cell_size=cfg.grid_cell_m)
        cross_cfg = CrossSectionAnalyzerV2Config(along_half_m=min(0.18, cfg.search_along_half_m * 0.55), lateral_half_m=min(0.70, cfg.search_lateral_half_m * 0.95))
        self.cross_analyzer = CrossSectionAnalyzerV2(cross_cfg)
        self.line_classifier = LineTypeClassifier()
        self.crosswalk_detector = CrosswalkDetector()
        self.stop_policy = StopPolicy(
            max_track_length_m=cfg.max_track_length_m,
            enable_crosswalk_stop=cfg.enable_crosswalk_stop,
            crosswalk_threshold=cfg.stop_crosswalk_threshold,
        )
        self.state_machine = LaneStateMachine()
        self.observer = LaneObservationEngine(
            grid=self.grid,
            xyz=self.xyz,
            intensity=self.intensity,
            analyzer=self.cross_analyzer,
            cfg=cfg,
        )
        self.associator = LaneAssociator()
        self.hypotheses: list[LaneHypothesis] = []
        self.debug_frames: list[StepDebugFrame] = []
        self.step_index = 0
        self.initialized = False
        self.stop_reason = StopReason.NONE
        self.seed_profile: SeedProfile | None = None

    def initialize(self, p0: np.ndarray, p1: np.ndarray) -> None:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        tangent = p1[:2] - p0[:2]
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        self.seed_profile = self._estimate_seed_profile(p0)
        init_observation = self.observer.observe(
            center_xyz=p0,
            tangent_xy=tangent,
            prev_state=None,
            seed_profile=self.seed_profile,
            along_half_m=float(self.cfg.search_along_half_m),
            lateral_half_m=float(self.cfg.init_search_lateral_half_m),
            is_gap_mode=False,
        )
        profile = init_observation.profile
        width = self.cfg.lane_width_init_m
        left_edge = -0.5 * width
        right_edge = 0.5 * width
        stripe_center = 0.0
        stripe_strength = 0.0
        quality = profile.quality
        if profile.selected_idx is not None:
            sc = profile.stripe_candidates[profile.selected_idx]
            width = sc.width_m
            left_edge = sc.left_m
            right_edge = sc.right_m
            stripe_center = sc.center_m
            stripe_strength = sc.peak_value
        state = LaneState(
            center_xyz=p0.copy(), tangent_xy=tangent.copy(), curvature=0.0,
            lane_width_m=width, left_edge_m=left_edge, right_edge_m=right_edge,
            stripe_center_m=stripe_center, stripe_strength=stripe_strength,
            profile_quality=quality, center_confidence=max(0.15, quality), identity_confidence=0.5,
            dashed_prob=0.0, solid_prob=0.0,
            visible_run_steps=1 if profile.selected_idx is not None else 0,
            gap_run_steps=0 if profile.selected_idx is not None else 1,
            total_length_m=0.0,
            mode=TrackMode.SOLID_VISIBLE if profile.selected_idx is not None else TrackMode.GAP_BRIDGING,
        )
        if profile.selected_idx is not None:
            correction_gain = float(np.clip(self.cfg.profile_center_correction_gain, 0.0, 1.0))
            self._apply_lateral_center_shift(state, stripe_center * correction_gain)
            if self.cfg.enable_center_refinement:
                refine_shift = self._refine_center_xy_shift(state.center_xyz, state.tangent_xy)
                cross_shift = self._refine_centerline_cross_section_shift(
                    state.center_xyz,
                    state.tangent_xy,
                    max(0.5 * float(state.lane_width_m), 0.05),
                )
                extra_shift = float(
                    np.clip(
                        refine_shift + cross_shift,
                        -float(self.cfg.center_refine_extra_max_shift_m),
                        float(self.cfg.center_refine_extra_max_shift_m),
                    )
                )
                self._apply_lateral_center_shift(state, extra_shift)
        state.center_xyz[2] = self._fit_center_z(state.center_xyz[:2], float(state.center_xyz[2]))
        state.history_centers.append(state.center_xyz.copy())
        state.history_widths.append(width)
        state.history_tangents.append(state.tangent_xy.copy())
        state.history_visibility.append(1 if profile.selected_idx is not None else 0)
        state.history_modes.append(state.mode.value)
        if profile.selected_idx is not None:
            state.trusted_history_centers.append(state.center_xyz.copy())
            state.trusted_history_tangents.append(state.tangent_xy.copy())
        if self.seed_profile is not None and self._should_refresh_seed_profile(state):
            self.seed_profile = self._refresh_seed_profile(state.center_xyz, self.seed_profile)
        self.hypotheses = [LaneHypothesis(state=state, total_score=0.0)]
        self.debug_frames = [
            StepDebugFrame(
                step_index=0,
                predicted_centers=[p0.copy()],
                selected_center=p0.copy(),
                selected_idx=0,
                cross_section_profile=profile,
                dashed_prob=state.dashed_prob,
                solid_prob=state.solid_prob,
                mode=state.mode.value,
                stop_reason="",
                switch_risk=0.0,
                search_along_half_m=float(self.cfg.search_along_half_m),
                search_lateral_half_m=float(self.cfg.init_search_lateral_half_m),
                query_debug=self._build_query_debug(init_observation.indices),
                candidate_summaries=self._build_candidate_summaries(profile),
                trajectory_line_points=self._build_trajectory_line_points(
                    self._reference_history(state),
                    state.center_xyz,
                    self._reference_tangent(state),
                ),
            )
        ]
        self.step_index = 0
        self.initialized = True
        self.stop_reason = StopReason.NONE

    def reset(self) -> None:
        self.hypotheses.clear()
        self.debug_frames.clear()
        self.step_index = 0
        self.initialized = False
        self.stop_reason = StopReason.NONE
        self.seed_profile = None

    def get_current_state(self) -> LaneState | None:
        return self.hypotheses[0].state if self.hypotheses else None

    def run_full(self) -> TrackResultV2:
        if not self.initialized:
            raise RuntimeError("Tracker not initialized")
        guard = 0
        while self.stop_reason == StopReason.NONE and guard < 10000:
            self.step()
            guard += 1
            if self.hypotheses[0].state.mode == TrackMode.STOPPED:
                break
        best = self.hypotheses[0]
        dense = np.asarray(best.state.history_centers, dtype=float)
        output = self._make_output_points(best.state)
        return TrackResultV2(dense_points=dense, output_points=output, stop_reason=self.stop_reason.value, debug_frames=self.debug_frames)

    def step(self) -> StepDebugFrame:
        if not self.initialized or not self.hypotheses:
            raise RuntimeError("Tracker not initialized")
        if self.stop_reason != StopReason.NONE:
            current = self.hypotheses[0].state
            return StepDebugFrame(
                step_index=self.step_index,
                selected_center=current.center_xyz.copy(),
                selected_idx=0,
                cross_section_profile=self.debug_frames[-1].cross_section_profile if self.debug_frames else None,
                dashed_prob=current.dashed_prob,
                solid_prob=current.solid_prob,
                crosswalk_score=self.debug_frames[-1].crosswalk_score if self.debug_frames else 0.0,
                mode=current.mode.value,
                stop_reason=self.stop_reason.value,
                trajectory_line_points=self._build_trajectory_line_points(
                    self._reference_history(current),
                    current.center_xyz,
                    self._reference_tangent(current),
                ),
            )
        expanded: list[LaneHypothesis] = []
        dbg = StepDebugFrame(step_index=self.step_index)
        predicted_debug_points: list[np.ndarray] = []
        best_score = -1e18
        for hyp in self.hypotheses:
            for pred in self._expand_predicted_states(hyp.state):
                predicted_debug_points.append(pred.center_xyz.copy())
                if not self._candidate_is_allowed(pred.center_xyz, hyp.state.center_xyz, hyp.state.tangent_xy):
                    continue
                along_half, lateral_half = self.observer.resolve_search_strip(hyp.state)
                observation = self.observer.observe(
                    center_xyz=pred.center_xyz,
                    tangent_xy=pred.tangent_xy,
                    prev_state=hyp.state,
                    seed_profile=self.seed_profile,
                    along_half_m=along_half,
                    lateral_half_m=lateral_half,
                    is_gap_mode=hyp.state.mode == TrackMode.GAP_BRIDGING,
                )
                new_hyp = self._build_next_hypothesis(hyp, pred, observation)
                expanded.append(new_hyp)
                if new_hyp.total_score > best_score:
                    best_score = float(new_hyp.total_score)
        expanded.sort(key=lambda h: h.total_score, reverse=True)
        keep = max(1, self.cfg.hypothesis_keep_top_k)
        self.hypotheses = expanded[:keep]
        top_k = max(1, int(self.cfg.display_candidate_top_k))
        dbg.predicted_centers = [h.state.center_xyz.copy() for h in expanded[:top_k]]
        if not dbg.predicted_centers:
            dbg.predicted_centers = predicted_debug_points[:top_k]
        best = self.hypotheses[0]
        best.state.center_xyz[2] = self._fit_center_z(best.state.center_xyz[:2], float(best.state.center_xyz[2]))
        if self.seed_profile is not None and self._should_refresh_seed_profile(best.state):
            self.seed_profile = self._refresh_seed_profile(best.state.center_xyz, self.seed_profile)
        dbg.cross_section_profile = best.debug_last.get("profile")
        dbg.hypothesis_scores = [h.total_score for h in self.hypotheses]
        dbg.selected_idx = 0
        dbg.selected_center = best.state.center_xyz.copy()
        dbg.dashed_prob = best.state.dashed_prob
        dbg.solid_prob = best.state.solid_prob
        dbg.mode = best.state.mode.value
        cross_idx = self.grid.query_oriented_strip_xy(best.state.center_xyz[:2], best.state.tangent_xy, 2.5, 2.0)
        crosswalk_score, _ = self.crosswalk_detector.score(self.xyz, self.intensity, cross_idx, best.state.center_xyz, best.state.tangent_xy)
        dbg.crosswalk_score = crosswalk_score
        dbg.switch_risk = float(best.debug_last.get("switch_risk", 0.0))
        dbg.stripe_rejected = bool(best.debug_last.get("stripe_rejected", False))
        dbg.search_along_half_m = float(best.debug_last.get("search_along_half_m", 0.0))
        dbg.search_lateral_half_m = float(best.debug_last.get("search_lateral_half_m", 0.0))
        dbg.query_debug = dict(best.debug_last.get("query_debug", {}))
        dbg.candidate_summaries = list(best.debug_last.get("candidate_summaries", []))
        dbg.trajectory_line_points = self._build_trajectory_line_points(
            self._reference_history(best.state),
            best.state.center_xyz,
            self._reference_tangent(best.state),
        )
        reason = self.stop_policy.evaluate(best.state, crosswalk_score)
        if reason != StopReason.NONE:
            best.state.mode = TrackMode.STOPPED
            self.stop_reason = reason
            dbg.stop_reason = reason.value
        dbg.mode = best.state.mode.value
        self.debug_frames.append(dbg)
        self.step_index += 1
        return dbg

    def _expand_predicted_states(self, state: LaneState) -> list[LaneState]:
        out: list[LaneState] = []
        base_t = self._reference_tangent(state)
        fit_t = self._fit_trajectory_tangent(self._reference_history(state), state.center_xyz, base_t)
        if fit_t is not None:
            blend = float(np.clip(self.cfg.trajectory_tangent_blend, 0.0, 1.0))
            base_t = self._unit2((1.0 - blend) * base_t + blend * fit_t)
        max_delta = np.radians(float(self.cfg.max_heading_change_deg))
        branch_count = max(1, int(self.cfg.heading_branch_count))
        heading_deltas = np.linspace(-max_delta, max_delta, branch_count)
        lateral_offsets = self.cfg.lateral_branch_offsets_m
        step = self.cfg.step_length_m
        for delta in heading_deltas:
            cs = float(np.cos(delta))
            sn = float(np.sin(delta))
            d = np.array([cs * base_t[0] - sn * base_t[1], sn * base_t[0] + cs * base_t[1]], dtype=float)
            n = np.array([-d[1], d[0]], dtype=float)
            for lo in lateral_offsets:
                s = state.copy_shallow()
                s.tangent_xy = d
                s.center_xyz = s.center_xyz.copy()
                s.center_xyz[0] += d[0] * step + n[0] * lo
                s.center_xyz[1] += d[1] * step + n[1] * lo
                s.total_length_m += step
                out.append(s)
        return out

    def _build_next_hypothesis(
        self,
        prev_hyp: LaneHypothesis,
        pred_state: LaneState,
        primary_observation: TrackerObservation,
    ) -> LaneHypothesis:
        state = pred_state.copy_shallow()
        decision = self.associator.select_observation(
            primary_observation,
            prev_hyp.state,
            salvage_narrow_fn=self._can_salvage_narrow_candidate,
            reliable_fn=self._is_selected_stripe_reliable,
            reference_history_fn=self._loyalty_history,
            reference_tangent_fn=self._reference_tangent,
            loyalty_term_fn=self._lane_loyalty_term,
            loyalty_gate_min_history=int(self.cfg.loyalty_gate_min_history),
            loyalty_gate_min_visible=float(self.cfg.loyalty_gate_min_visible),
            loyalty_gate_quality_bypass=float(self.cfg.loyalty_gate_quality_bypass),
            source="primary",
        )
        if (
            bool(self.cfg.retry_recovery_on_reject)
            and prev_hyp.state.mode != TrackMode.GAP_BRIDGING
            and (not decision.stripe_visible or decision.stripe_rejected)
        ):
            retry_observation = self.observer.observe_recovery(
                center_xyz=pred_state.center_xyz,
                tangent_xy=pred_state.tangent_xy,
                prev_state=prev_hyp.state,
                seed_profile=self.seed_profile,
                anchor_projected_frame=self._anchor_projected_frame,
                should_refresh_seed_profile=self._should_refresh_seed_profile,
                refresh_seed_profile=self._refresh_seed_profile,
            )
            retry_decision = self.associator.select_observation(
                retry_observation,
                prev_hyp.state,
                salvage_narrow_fn=self._can_salvage_narrow_candidate,
                reliable_fn=self._is_selected_stripe_reliable,
                reference_history_fn=self._loyalty_history,
                reference_tangent_fn=self._reference_tangent,
                loyalty_term_fn=self._lane_loyalty_term,
                loyalty_gate_min_history=int(self.cfg.loyalty_gate_min_history),
                loyalty_gate_min_visible=float(self.cfg.loyalty_gate_min_visible),
                loyalty_gate_quality_bypass=float(self.cfg.loyalty_gate_quality_bypass),
                source="recovery",
            )
            if retry_decision.stripe_visible:
                decision = retry_decision
                state.center_xyz = decision.observation.query_center_xyz.copy()
                state.tangent_xy = decision.observation.query_tangent_xy.copy()
        active_profile = decision.observation.profile
        active_idx = decision.observation.indices
        active_along_half = float(decision.observation.along_half_m)
        active_lateral_half = float(decision.observation.lateral_half_m)
        stripe_visible = decision.stripe_visible
        stripe_rejected = decision.stripe_rejected
        salvage_narrow = decision.salvage_narrow
        update_profile_shift = 0.0
        update_refine_shift = 0.0
        update_cross_shift = 0.0
        update_tangent_before = state.tangent_xy.copy()
        update_tangent_after = state.tangent_xy.copy()
        update_center_before = state.center_xyz.copy()
        update_center_after = state.center_xyz.copy()
        post_loyalty = decision.loyalty_value
        post_loyalty_failed = False
        if stripe_visible:
            sc = active_profile.stripe_candidates[active_profile.selected_idx]
            correction_gain = float(np.clip(self.cfg.profile_center_correction_gain, 0.0, 1.0))
            stripe_center = float(sc.center_m)
            lane_width = float(sc.width_m)
            left_edge = float(sc.left_m)
            right_edge = float(sc.right_m)
            if salvage_narrow:
                reuse_width = max(
                    lane_width,
                    float(prev_hyp.state.lane_width_m) * float(self.cfg.narrow_candidate_width_blend),
                )
                lane_width = float(
                    np.clip(
                        reuse_width,
                        float(self.cfg.min_candidate_width_visible_m),
                        float(self.cross_analyzer.cfg.lane_width_max_m),
                    )
                )
                left_edge = stripe_center - 0.5 * lane_width
                right_edge = stripe_center + 0.5 * lane_width
            candidate_state = state.copy_shallow()
            update_center_before = candidate_state.center_xyz.copy()
            update_tangent_before = candidate_state.tangent_xy.copy()
            candidate_state.stripe_center_m = stripe_center
            candidate_state.left_edge_m = left_edge
            candidate_state.right_edge_m = right_edge
            candidate_state.lane_width_m = lane_width
            candidate_state.stripe_strength = sc.peak_value
            update_profile_shift = stripe_center * correction_gain
            self._apply_lateral_center_shift(candidate_state, update_profile_shift)
            if self.cfg.enable_center_refinement:
                update_refine_shift = self._refine_center_xy_shift(candidate_state.center_xyz, candidate_state.tangent_xy)
                update_cross_shift = self._refine_centerline_cross_section_shift(
                    candidate_state.center_xyz,
                    candidate_state.tangent_xy,
                    max(0.5 * float(candidate_state.lane_width_m), 0.05),
                )
                extra_shift = float(
                    np.clip(
                        update_refine_shift + update_cross_shift,
                        -float(self.cfg.center_refine_extra_max_shift_m),
                        float(self.cfg.center_refine_extra_max_shift_m),
                    )
                )
                update_cross_shift = extra_shift - update_refine_shift
                self._apply_lateral_center_shift(candidate_state, extra_shift)
            candidate_state.tangent_xy = self._realign_tangent_to_motion(
                prev_hyp.state.center_xyz,
                candidate_state.center_xyz,
                candidate_state.tangent_xy,
            )
            update_center_after = candidate_state.center_xyz.copy()
            update_tangent_after = candidate_state.tangent_xy.copy()
            candidate_state.profile_quality = active_profile.quality
            candidate_state.center_confidence = min(1.0, 0.25 + sc.peak_value + 0.2 * active_profile.quality)
            candidate_state.identity_confidence = max(0.0, min(1.0, sc.identity_score))
            post_loyalty = self._lane_loyalty_term(
                self._loyalty_history(prev_hyp.state),
                candidate_state.center_xyz,
                self._reference_tangent(prev_hyp.state),
            )
            if self._fails_loyalty_gate(prev_hyp.state, candidate_state.center_xyz, candidate_state.profile_quality):
                stripe_visible = False
                stripe_rejected = True
                post_loyalty_failed = True
                if "post_loyalty_gate_failed" not in decision.rejection_reasons:
                    decision.rejection_reasons.append("post_loyalty_gate_failed")
            else:
                state = candidate_state
                state.visible_run_steps += 1
                state.gap_run_steps = 0
        if not stripe_visible:
            anchor_prev_center, prev_tangent = self._anchor_projected_frame(
                prev_hyp.state,
                prev_hyp.state.center_xyz,
                prev_hyp.state.tangent_xy,
            )
            gap_step = float(self.cfg.step_length_m) * float(self.cfg.gap_bridge_step_scale)
            state.tangent_xy = prev_tangent.copy()
            state.center_xyz = anchor_prev_center.copy()
            state.center_xyz[0] += prev_tangent[0] * gap_step
            state.center_xyz[1] += prev_tangent[1] * gap_step
            state.total_length_m = float(prev_hyp.state.total_length_m) + gap_step
            state.profile_quality = active_profile.quality
            state.center_confidence *= 0.90
            state.identity_confidence *= 0.95
            state.visible_run_steps = 0
            state.gap_run_steps += 1
        vis_hist = list(state.history_visibility) + [1 if stripe_visible else 0]
        width_hist = list(state.history_widths) + [state.lane_width_m]
        dashed_prob, solid_prob, _ = self.line_classifier.update(vis_hist, width_hist)
        state.dashed_prob = dashed_prob
        state.solid_prob = solid_prob
        crosswalk_score, _ = self.crosswalk_detector.score(self.xyz, self.intensity, active_idx, state.center_xyz, state.tangent_xy)
        state.mode = self.state_machine.update(stripe_visible, dashed_prob, solid_prob, max(crosswalk_score, active_profile.crosswalk_like_score))
        state.history_centers.append(state.center_xyz.copy())
        state.history_widths.append(state.lane_width_m)
        state.history_tangents.append(state.tangent_xy.copy())
        state.history_visibility.append(1 if stripe_visible else 0)
        state.history_modes.append(state.mode.value)
        scores = self._score_transition(prev_hyp.state, state, active_profile, stripe_visible, crosswalk_score)
        loyalty_term = self._lane_loyalty_term(self._loyalty_history(prev_hyp.state), state.center_xyz, self._reference_tangent(prev_hyp.state))
        loyalty_weight = float(np.clip(self.cfg.lane_loyalty_weight, 0.0, 2.0))
        trajectory_follow = self._trajectory_follow_term(
            self._reference_history(prev_hyp.state),
            state.center_xyz,
            self._reference_tangent(prev_hyp.state),
        )
        if self._should_add_trusted_point(prev_hyp.state, state, stripe_visible, loyalty_term):
            state.trusted_history_centers.append(state.center_xyz.copy())
            state.trusted_history_tangents.append(state.tangent_xy.copy())
        trajectory_weight = float(np.clip(self.cfg.trajectory_follow_weight, 0.0, 2.0))
        total_score = prev_hyp.total_score + scores.total + loyalty_weight * loyalty_term + trajectory_weight * trajectory_follow
        return LaneHypothesis(
            state=state,
            total_score=total_score,
            last_scores=scores,
            age_steps=prev_hyp.age_steps + 1,
            alive=True,
            debug_last={
                "crosswalk_score": crosswalk_score,
                "switch_risk": active_profile.switch_risk,
                "profile": active_profile,
                "query_debug": self._build_query_debug(active_idx),
                "candidate_summaries": self._build_candidate_summaries(active_profile),
                "lane_loyalty": loyalty_term,
                "trajectory_follow": trajectory_follow,
                "stripe_rejected": stripe_rejected,
                "search_along_half_m": active_along_half,
                "search_lateral_half_m": active_lateral_half,
                "obs_debug": {
                    "source": decision.source,
                    "candidate_count": len(active_profile.stripe_candidates),
                    "selected_idx": active_profile.selected_idx,
                    "selected_center_m": float(active_profile.stripe_candidates[active_profile.selected_idx].center_m)
                    if active_profile.selected_idx is not None
                    else None,
                    "selected_width_m": float(active_profile.stripe_candidates[active_profile.selected_idx].width_m)
                    if active_profile.selected_idx is not None
                    else None,
                    "selected_signal": float(active_profile.stripe_candidates[active_profile.selected_idx].signal_consistency)
                    if active_profile.selected_idx is not None
                    else None,
                },
                "asc_debug": {
                    "source": decision.source,
                    "stripe_visible": stripe_visible,
                    "stripe_rejected": stripe_rejected,
                    "reliable_checked": decision.reliable_checked,
                    "reliable_passed": decision.reliable_passed,
                    "loyalty_pre": None if decision.loyalty_value is None else float(decision.loyalty_value),
                    "loyalty_post": None if post_loyalty is None else float(post_loyalty),
                    "post_loyalty_failed": post_loyalty_failed,
                    "reasons": list(decision.rejection_reasons),
                },
                "upd_debug": {
                    "profile_shift_m": float(update_profile_shift),
                    "refine_shift_m": float(update_refine_shift),
                    "cross_shift_m": float(update_cross_shift),
                    "total_shift_m": float(update_profile_shift + update_refine_shift + update_cross_shift),
                    "center_before_xy": [float(update_center_before[0]), float(update_center_before[1])],
                    "center_after_xy": [float(update_center_after[0]), float(update_center_after[1])],
                    "tangent_before_xy": [float(update_tangent_before[0]), float(update_tangent_before[1])],
                    "tangent_after_xy": [float(update_tangent_after[0]), float(update_tangent_after[1])],
                },
            },
        )

    def _score_transition(self, prev_state: LaneState, new_state: LaneState, profile: CrossSectionProfile, stripe_visible: bool, crosswalk_score: float) -> StepScores:
        s = StepScores()
        s.stripe_fit = new_state.stripe_strength if stripe_visible else 0.0
        s.profile_quality = new_state.profile_quality
        s.center_continuity = max(0.0, 1.0 - abs(new_state.stripe_center_m - prev_state.stripe_center_m) / 0.2)
        s.width_continuity = max(0.0, 1.0 - abs(new_state.lane_width_m - prev_state.lane_width_m) / 0.2)
        s.edge_continuity = max(0.0, 1.0 - (abs(new_state.left_edge_m - prev_state.left_edge_m) + abs(new_state.right_edge_m - prev_state.right_edge_m)) / 0.4)
        dot = float(np.clip(np.dot(prev_state.tangent_xy[:2], new_state.tangent_xy[:2]), -1.0, 1.0))
        s.heading_continuity = 0.5 * (dot + 1.0)
        s.identity_score = 0.5 * (s.center_continuity + s.width_continuity)
        s.visibility_score = 1.0 if stripe_visible else 0.0
        s.switch_penalty = max(profile.switch_risk, max(0.0, profile.neighbor_count - 1) * 0.2)
        s.crosswalk_penalty = max(crosswalk_score, profile.crosswalk_like_score) * 0.75
        s.total = (1.25 * s.stripe_fit + 0.6 * s.profile_quality + 1.0 * s.center_continuity + 1.1 * s.width_continuity + 0.9 * s.edge_continuity + 0.6 * s.heading_continuity + 0.8 * s.identity_score + 0.4 * s.visibility_score - 1.0 * s.switch_penalty - 1.0 * s.crosswalk_penalty)
        return s

    def _make_output_points(self, state: LaneState) -> np.ndarray:
        pts = np.asarray(state.history_centers, dtype=float)
        if pts.shape[0] == 0:
            return pts
        if state.dashed_prob > state.solid_prob:
            vis = state.history_visibility
            out = []
            i = 0
            while i < len(vis):
                if vis[i] == 1:
                    start = i
                    while i + 1 < len(vis) and vis[i + 1] == 1:
                        i += 1
                    end = i
                    out.append(pts[start])
                    if end != start:
                        out.append(pts[end])
                i += 1
            return np.asarray(out, dtype=float) if out else np.empty((0, 3), dtype=float)
        return pts

    def _estimate_seed_profile(self, p0: np.ndarray) -> SeedProfile:
        idx = self.grid.query_radius_xy(p0[:2], self.cfg.seed_radius_m)
        return self._estimate_seed_profile_from_indices(idx, float(p0[2]))

    def _estimate_seed_profile_from_indices(self, idx: np.ndarray, fallback_z: float) -> SeedProfile:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return SeedProfile(
                target_intensity=float(np.quantile(self.intensity, 0.90)),
                background_intensity=float(np.quantile(self.intensity, 0.35)),
                z_ref=float(fallback_z),
            )
        local_i = self.intensity[idx]
        local_z = self.xyz[idx, 2]
        return SeedProfile(
            target_intensity=float(np.quantile(local_i, 0.90)),
            background_intensity=float(np.quantile(local_i, 0.35)),
            z_ref=float(np.median(local_z)),
        )

    def _refresh_seed_profile(self, center_xyz: np.ndarray, current: SeedProfile) -> SeedProfile:
        radius = max(float(self.cfg.profile_update_radius_m), 1e-3)
        alpha = float(np.clip(self.cfg.profile_update_alpha, 0.0, 1.0))
        idx = self.grid.query_radius_xy(np.asarray(center_xyz[:2], dtype=np.float64), radius)
        if idx.size == 0:
            return current
        updated = self._estimate_seed_profile_from_indices(idx, float(center_xyz[2]))
        keep = 1.0 - alpha
        return SeedProfile(
            target_intensity=keep * float(current.target_intensity) + alpha * float(updated.target_intensity),
            background_intensity=keep * float(current.background_intensity) + alpha * float(updated.background_intensity),
            z_ref=keep * float(current.z_ref) + alpha * float(updated.z_ref),
        )

    def _should_refresh_seed_profile(self, state: LaneState) -> bool:
        if not self._state_is_currently_trusted(state):
            return False
        if state.mode in (TrackMode.GAP_BRIDGING, TrackMode.CROSSWALK_CANDIDATE, TrackMode.STOPPED):
            return False
        if float(state.profile_quality) < float(self.cfg.profile_update_min_quality):
            return False
        if float(state.identity_confidence) < float(self.cfg.profile_update_min_identity):
            return False
        width = float(state.lane_width_m)
        if width < float(self.cfg.profile_update_min_width_m) or width > float(self.cfg.profile_update_max_width_m):
            return False
        return True

    def _fit_center_z(self, center_xy: np.ndarray, fallback_z: float) -> float:
        radius = max(float(self.cfg.center_z_fit_radius_m), 1e-3)
        idx = self.grid.query_radius_xy(np.asarray(center_xy[:2], dtype=np.float64), radius)
        if idx.size == 0:
            return float(fallback_z)
        local_z = self.xyz[np.asarray(idx, dtype=np.int64), 2]
        z_gate = max(float(self.cfg.z_fit_window_m), float(self.cfg.max_z_step_m))
        mask = np.abs(local_z - float(fallback_z)) <= z_gate
        if np.count_nonzero(mask) >= 3:
            return float(np.median(local_z[mask]))
        return float(np.median(local_z))

    def _candidate_is_allowed(self, candidate_center: np.ndarray, prev_center: np.ndarray, prev_dir: np.ndarray) -> bool:
        step_ref = max(float(self.cfg.step_length_m), 1e-6)
        prev_dir_xy = self._unit2(prev_dir[:2])
        pred_xy = prev_center[:2] + prev_dir_xy * step_ref
        normal_xy = np.array([-prev_dir_xy[1], prev_dir_xy[0]], dtype=np.float64)
        lateral_offset = abs(float(np.dot(candidate_center[:2] - pred_xy, normal_xy)))
        hard_limit = float(self.cfg.hard_center_offset_limit_m)
        if lateral_offset > hard_limit:
            return False
        max_z_step = float(self.cfg.hard_max_z_step_m)
        if abs(float(candidate_center[2] - prev_center[2])) > max_z_step:
            return False
        return True

    def _lane_loyalty_term(self, history_centers: list[np.ndarray], candidate_center: np.ndarray, pred_dir: np.ndarray) -> float:
        history_len = max(1, int(self.cfg.lane_loyalty_history_points))
        if len(history_centers) < 3:
            return 1.0
        hist = np.asarray(history_centers[-history_len:], dtype=np.float64)
        if hist.shape[0] < 2:
            return 1.0
        anchor = hist[-1, :2]
        dir_xy = self._unit2(pred_dir[:2])
        normal_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)
        lateral_hist = (hist[:, :2] - anchor) @ normal_xy
        center_bias = float(np.median(lateral_hist))
        candidate_lateral = float(np.dot(candidate_center[:2] - anchor, normal_xy))
        loyalty_offset = abs(candidate_lateral - center_bias)
        tolerance = max(float(self.cfg.lane_loyalty_tolerance_m), 1e-3)
        return float(np.clip(1.0 - loyalty_offset / tolerance, 0.0, 1.0))

    def _fails_loyalty_gate(self, prev_state: LaneState, candidate_center: np.ndarray, profile_quality: float) -> bool:
        if prev_state.mode == TrackMode.GAP_BRIDGING or prev_state.gap_run_steps > 0:
            return False
        ref_history = self._loyalty_history(prev_state)
        if len(ref_history) < int(self.cfg.loyalty_gate_min_history):
            return False
        if float(profile_quality) >= float(self.cfg.loyalty_gate_quality_bypass):
            return False
        loyalty = self._lane_loyalty_term(
            ref_history,
            candidate_center,
            self._reference_tangent(prev_state),
        )
        return loyalty < float(self.cfg.loyalty_gate_min_visible)

    def _reference_history(self, state: LaneState) -> list[np.ndarray]:
        min_pts = max(1, int(self.cfg.trusted_history_min_points))
        if len(state.trusted_history_centers) >= min_pts:
            return state.trusted_history_centers
        return state.history_centers

    def _loyalty_history(self, state: LaneState) -> list[np.ndarray]:
        gate_min = max(1, int(self.cfg.loyalty_gate_min_history))
        if len(state.trusted_history_centers) >= gate_min:
            return state.trusted_history_centers
        if len(state.history_centers) >= gate_min:
            return state.history_centers
        return self._reference_history(state)

    def _reference_tangent(self, state: LaneState) -> np.ndarray:
        min_pts = max(1, int(self.cfg.trusted_history_min_points))
        if len(state.trusted_history_tangents) >= min_pts:
            return self._unit2(np.asarray(state.trusted_history_tangents[-1][:2], dtype=np.float64))
        return self._unit2(np.asarray(state.tangent_xy[:2], dtype=np.float64))

    def _anchor_projected_frame(
        self,
        state: LaneState,
        reference_center: np.ndarray,
        reference_tangent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(reference_center, dtype=np.float64).copy()
        tangent = self._unit2(np.asarray(reference_tangent[:2], dtype=np.float64))
        if not state.trusted_history_centers or not state.trusted_history_tangents:
            return center, tangent
        anchor_center = np.asarray(state.trusted_history_centers[-1], dtype=np.float64)
        anchor_tangent = self._unit2(np.asarray(state.trusted_history_tangents[-1][:2], dtype=np.float64))
        delta_xy = center[:2] - anchor_center[:2]
        along = max(float(np.dot(delta_xy, anchor_tangent)), 0.0)
        center[:2] = anchor_center[:2] + anchor_tangent * along
        return center, anchor_tangent

    def _should_add_trusted_point(
        self,
        prev_state: LaneState,
        state: LaneState,
        stripe_visible: bool,
        loyalty_term: float,
    ) -> bool:
        if not stripe_visible:
            return False
        if float(state.profile_quality) < float(self.cfg.trusted_update_min_quality):
            return False
        if float(loyalty_term) < float(self.cfg.trusted_update_min_loyalty):
            return False
        if float(state.identity_confidence) < float(self.cfg.trusted_update_min_identity):
            return False
        width = float(state.lane_width_m)
        if width < float(self.cfg.trusted_update_min_width_m) or width > float(self.cfg.trusted_update_max_width_m):
            return False
        return True

    def _state_is_currently_trusted(self, state: LaneState) -> bool:
        if not state.trusted_history_centers:
            return False
        return bool(
            np.linalg.norm(
                np.asarray(state.trusted_history_centers[-1][:2], dtype=np.float64)
                - np.asarray(state.center_xyz[:2], dtype=np.float64)
            )
            <= 1e-6
        )

    def _fit_trajectory_tangent(
        self,
        history_centers: list[np.ndarray],
        center_xyz: np.ndarray,
        fallback_tangent: np.ndarray,
    ) -> np.ndarray | None:
        fit = self._fit_trajectory_basis(history_centers, center_xyz, fallback_tangent, require_min_points=True)
        if fit is None:
            return None
        _, tangent_xy, _ = fit
        return tangent_xy

    def _trajectory_follow_term(
        self,
        history_centers: list[np.ndarray],
        candidate_center: np.ndarray,
        fallback_tangent: np.ndarray,
    ) -> float:
        fit = self._fit_trajectory_basis(history_centers, candidate_center, fallback_tangent, require_min_points=True)
        if fit is None:
            return 0.0
        anchor_xy, tangent_xy, _ = fit
        normal_xy = np.array([-tangent_xy[1], tangent_xy[0]], dtype=np.float64)
        delta_xy = np.asarray(candidate_center[:2], dtype=np.float64) - anchor_xy
        along = float(np.dot(delta_xy, tangent_xy))
        lateral = abs(float(np.dot(delta_xy, normal_xy)))
        sigma = max(float(self.cfg.trajectory_follow_sigma_m), 1e-3)
        lateral_term = float(np.exp(-0.5 * (lateral / sigma) ** 2))
        target_along = float(self.cfg.step_length_m)
        along_sigma = max(float(self.cfg.step_length_m) * 0.85, 1e-3)
        along_term = float(np.exp(-0.5 * ((along - target_along) / along_sigma) ** 2)) if along >= -0.05 else 0.0
        return float(np.clip(0.75 * lateral_term + 0.25 * along_term, 0.0, 1.0))

    def _build_trajectory_line_points(
        self,
        history_centers: list[np.ndarray],
        center_xyz: np.ndarray,
        fallback_tangent: np.ndarray,
    ) -> np.ndarray:
        fit = self._fit_trajectory_basis(history_centers, center_xyz, fallback_tangent, require_min_points=False)
        center = np.asarray(center_xyz, dtype=np.float64)
        if center.shape[0] == 2:
            center = np.array([center[0], center[1], 0.0], dtype=np.float64)
        if fit is None:
            tangent_xy = self._unit2(np.asarray(fallback_tangent[:2], dtype=np.float64))
            pts_xy = np.asarray(history_centers[-2:], dtype=np.float64)[:, :2] if len(history_centers) >= 2 else np.empty((0, 2), dtype=np.float64)
            anchor_xy = center[:2]
        else:
            anchor_xy, tangent_xy, pts_xy = fit
        if pts_xy.shape[0] >= 2:
            span = float(np.linalg.norm(pts_xy[-1] - pts_xy[0]))
        else:
            span = 0.0
        half_len = max(float(self.cfg.trajectory_debug_half_length_m), span * 0.55)
        p0 = np.array([anchor_xy[0] - tangent_xy[0] * half_len, anchor_xy[1] - tangent_xy[1] * half_len, center[2]], dtype=np.float64)
        p1 = np.array([anchor_xy[0] + tangent_xy[0] * half_len, anchor_xy[1] + tangent_xy[1] * half_len, center[2]], dtype=np.float64)
        return np.vstack([p0, p1])

    def _fit_trajectory_basis(
        self,
        history_centers: list[np.ndarray],
        center_xyz: np.ndarray,
        fallback_tangent: np.ndarray,
        require_min_points: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not bool(self.cfg.enable_trajectory_fit):
            return None
        fallback_xy = self._unit2(np.asarray(fallback_tangent[:2], dtype=np.float64))
        anchor_xy = np.asarray(center_xyz[:2], dtype=np.float64)
        history_len = max(2, int(self.cfg.trajectory_fit_history_points))
        if not history_centers:
            return None if require_min_points else (anchor_xy, fallback_xy, np.empty((0, 2), dtype=np.float64))
        pts_xy = np.asarray(history_centers[-history_len:], dtype=np.float64)[:, :2]
        min_pts = max(2, int(self.cfg.trajectory_fit_min_points))
        if pts_xy.shape[0] < min_pts:
            return None if require_min_points else (anchor_xy, fallback_xy, pts_xy)
        weights = np.linspace(1.0, 2.5, pts_xy.shape[0], dtype=np.float64)
        weight_sum = float(np.sum(weights))
        mean_xy = np.sum(pts_xy * weights[:, None], axis=0) / max(weight_sum, 1e-9)
        centered = pts_xy - mean_xy
        cov = (centered * weights[:, None]).T @ centered / max(weight_sum, 1e-9)
        eigvals, eigvecs = np.linalg.eigh(cov)
        tangent_xy = np.asarray(eigvecs[:, int(np.argmax(eigvals))], dtype=np.float64)
        if float(np.dot(tangent_xy, fallback_xy)) < 0.0:
            tangent_xy = -tangent_xy
        if float(np.max(eigvals)) <= 1e-9:
            tangent_xy = fallback_xy
        return anchor_xy, self._unit2(tangent_xy), pts_xy

    def _is_selected_stripe_reliable(self, profile: CrossSectionProfile, sc, prev_state: LaneState) -> bool:
        quality = float(profile.quality)
        width = float(sc.width_m)
        candidate_count = int(len(profile.stripe_candidates))
        identity = float(getattr(sc, "identity_score", 0.0))
        peak = float(getattr(sc, "peak_value", 0.0))
        center_cons = float(getattr(sc, "center_consistency", 0.0))
        center_jump = abs(float(sc.center_m - prev_state.stripe_center_m))
        is_gap_like = prev_state.mode in (TrackMode.GAP_BRIDGING, TrackMode.CROSSWALK_CANDIDATE) or prev_state.gap_run_steps > 0
        salvage_narrow = self._can_salvage_narrow_candidate(profile, sc, prev_state)
        salvage_single = self._can_salvage_single_candidate(profile, sc, prev_state)
        narrow_but_consistent = (
            width >= 0.05
            and center_cons >= float(self.cfg.min_narrow_candidate_center_consistency)
            and identity >= 0.55
        )
        if not is_gap_like:
            if width < float(self.cfg.normal_mode_min_candidate_width_m):
                return False
            if candidate_count <= 1 and width < float(self.cfg.normal_mode_single_candidate_min_width_m):
                return False
        if quality < float(self.cfg.min_profile_quality_visible):
            if not (
                identity >= 0.60
                or peak >= float(self.cfg.min_single_candidate_peak_visible)
                or (is_gap_like and narrow_but_consistent)
                or (is_gap_like and salvage_narrow)
                or (is_gap_like and salvage_single)
            ):
                return False
        if width < float(self.cfg.min_candidate_width_visible_m) and not (is_gap_like and (narrow_but_consistent or salvage_narrow)):
            return False
        if candidate_count <= 1 and quality < float(self.cfg.min_single_candidate_quality_visible):
            if not (
                (
                    identity >= float(self.cfg.min_single_candidate_identity_visible)
                    and peak >= float(self.cfg.min_single_candidate_peak_visible)
                    and center_jump <= float(self.cfg.max_candidate_center_jump_m)
                )
                or (is_gap_like and salvage_narrow)
                or (is_gap_like and salvage_single)
            ):
                return False
        if center_jump > float(self.cfg.max_candidate_center_jump_m) and quality < 0.85:
            return False
        return True

    def _can_salvage_narrow_candidate(self, profile: CrossSectionProfile, sc, prev_state: LaneState) -> bool:
        width = float(sc.width_m)
        if width >= float(self.cfg.min_candidate_width_visible_m):
            return False
        quality = float(profile.quality)
        peak = float(getattr(sc, "peak_value", 0.0))
        identity = float(getattr(sc, "identity_score", 0.0))
        center_cons = float(getattr(sc, "center_consistency", 0.0))
        center_jump = abs(float(sc.center_m - prev_state.stripe_center_m))
        if center_jump > float(self.cfg.narrow_candidate_salvage_center_jump_m):
            return False
        if quality < float(self.cfg.narrow_candidate_salvage_quality) and peak < float(self.cfg.narrow_candidate_salvage_peak):
            return False
        if identity < 0.45 and center_cons < 0.65:
            return False
        return True

    def _can_salvage_single_candidate(self, profile: CrossSectionProfile, sc, prev_state: LaneState) -> bool:
        if int(len(profile.stripe_candidates)) != 1:
            return False
        width = float(sc.width_m)
        quality = float(profile.quality)
        peak = float(getattr(sc, "peak_value", 0.0))
        identity = float(getattr(sc, "identity_score", 0.0))
        center_cons = float(getattr(sc, "center_consistency", 0.0))
        center_jump = abs(float(sc.center_m - prev_state.stripe_center_m))
        if width < float(self.cfg.single_candidate_salvage_width_m):
            return False
        if width > float(self.cfg.single_candidate_salvage_max_width_m):
            return False
        if center_jump > float(self.cfg.single_candidate_salvage_center_jump_m):
            return False
        # Strong recovery path: if it looks like a plausible lane-width stripe and
        # it is close to the previous centerline, prefer keeping the lane over
        # dropping immediately into gap mode.
        if (
            quality >= float(self.cfg.single_candidate_salvage_force_quality)
            and (
                peak >= float(self.cfg.single_candidate_salvage_force_peak)
                or center_cons >= float(self.cfg.single_candidate_salvage_force_center_consistency)
            )
        ):
            return True
        if quality < float(self.cfg.single_candidate_salvage_quality):
            return False
        if peak < float(self.cfg.single_candidate_salvage_peak) and identity < float(self.cfg.single_candidate_salvage_identity):
            return False
        if (
            identity < float(self.cfg.single_candidate_salvage_identity)
            and center_cons < float(self.cfg.single_candidate_salvage_center_consistency)
        ):
            return False
        return True

    def _apply_lateral_center_shift(self, state: LaneState, shift_m: float) -> None:
        shift = float(shift_m)
        if abs(shift) <= 1e-9:
            return
        normal_xy = np.array([-state.tangent_xy[1], state.tangent_xy[0]], dtype=np.float64)
        state.center_xyz = state.center_xyz.copy()
        state.center_xyz[:2] += normal_xy * shift
        state.stripe_center_m -= shift
        state.left_edge_m -= shift
        state.right_edge_m -= shift

    def _realign_tangent_to_motion(self, prev_center: np.ndarray, new_center: np.ndarray, tangent_xy: np.ndarray) -> np.ndarray:
        if not bool(self.cfg.enable_motion_tangent_realign):
            return self._unit2(np.asarray(tangent_xy[:2], dtype=np.float64))
        prev_xy = np.asarray(prev_center[:2], dtype=np.float64)
        new_xy = np.asarray(new_center[:2], dtype=np.float64)
        delta = new_xy - prev_xy
        delta_norm = float(np.linalg.norm(delta))
        base = self._unit2(np.asarray(tangent_xy[:2], dtype=np.float64))
        if delta_norm < float(self.cfg.min_motion_tangent_step_m):
            return base
        normal = np.array([-base[1], base[0]], dtype=np.float64)
        along = float(np.dot(delta, base))
        lateral = float(np.dot(delta, normal))
        if along <= 0.0:
            return base
        # Ignore most of the lateral correction so direction follows the lane,
        # not the centerline refinement side shift.
        lateral_limit = max(float(self.cfg.center_refine_max_shift_m) * 0.35, 0.015)
        limited_lateral = float(np.clip(lateral, -lateral_limit, lateral_limit))
        motion = along * base + limited_lateral * normal
        motion_norm = float(np.linalg.norm(motion))
        if motion_norm <= 1e-9:
            return base
        motion = motion / motion_norm
        mix = float(np.clip(self.cfg.motion_tangent_mix, 0.0, 1.0))
        lateral_ratio = abs(limited_lateral) / max(abs(along), 1e-6)
        effective_mix = mix * max(0.0, 1.0 - min(lateral_ratio / 0.30, 1.0))
        blended = (1.0 - effective_mix) * base + effective_mix * motion
        if float(np.dot(blended, base)) < 0.0:
            blended = -blended
        return self._unit2(blended)

    def _refine_center_xy_shift(self, center: np.ndarray, direction: np.ndarray) -> float:
        radius = max(float(self.cfg.center_refine_radius_m), 1e-3)
        idx = self.grid.query_radius_xy(center[:2], radius)
        if idx.size < 6:
            return 0.0
        local_xyz = self.xyz[idx]
        local_i = self.intensity[idx].astype(np.float64)
        z_mask = np.abs(local_xyz[:, 2] - float(center[2])) <= float(self.cfg.max_z_step_m)
        if np.count_nonzero(z_mask) < 4:
            return 0.0
        pts = local_xyz[z_mask]
        weights = local_i[z_mask]
        if weights.size == 0:
            return 0.0
        dir_xy = self._unit2(direction[:2])
        normal_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)
        deltas = pts[:, :2] - center[:2]
        lateral_offsets = deltas @ normal_xy
        shift_limit = max(float(self.cfg.center_refine_max_shift_m), 1e-3)
        close_term = np.clip(1.0 - np.abs(lateral_offsets) / max(shift_limit * 2.0, 1e-3), 0.0, 1.0)
        weights = np.maximum(weights - float(np.min(weights)), 0.0) + 1e-3
        weights *= 0.35 + 0.65 * close_term
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return 0.0
        shift = float(np.sum(lateral_offsets * weights) / weight_sum)
        return float(np.clip(shift, -shift_limit, shift_limit))

    def _refine_centerline_cross_section_shift(self, center: np.ndarray, direction: np.ndarray, lane_half_width: float) -> float:
        radius = max(float(self.cfg.cross_section_radius_m), 1e-3)
        idx = self.grid.query_radius_xy(center[:2], radius)
        if idx.size < 10:
            return 0.0
        local_xyz = self.xyz[idx]
        local_i = self.intensity[idx].astype(np.float64)
        z_mask = np.abs(local_xyz[:, 2] - float(center[2])) <= float(self.cfg.max_z_step_m)
        if np.count_nonzero(z_mask) < 6:
            return 0.0
        pts = local_xyz[z_mask]
        vals = local_i[z_mask]
        dir_xy = self._unit2(direction[:2])
        normal_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)
        deltas = pts[:, :2] - center[:2]
        along = deltas @ dir_xy
        lateral = deltas @ normal_xy
        along_half = max(float(self.cfg.cross_section_forward_window_m), 1e-3)
        lane_half = max(float(lane_half_width), 0.05)
        mask = (np.abs(along) <= along_half) & (np.abs(lateral) <= lane_half * 2.5)
        if np.count_nonzero(mask) < 6:
            return 0.0
        lateral = lateral[mask]
        vals = vals[mask]
        val_floor = float(np.quantile(vals, 0.35))
        weights = np.clip(vals - val_floor, 0.0, None)
        if float(np.sum(weights)) <= 1e-9:
            return 0.0
        bin_size = max(float(self.cfg.cross_section_bin_size_m), 1e-3)
        bins = np.arange(-lane_half * 2.5, lane_half * 2.5 + bin_size, bin_size, dtype=np.float64)
        if bins.size < 4:
            return 0.0
        hist, edges = np.histogram(lateral, bins=bins, weights=weights)
        if hist.size < 3:
            return 0.0
        smooth_hist = hist.copy()
        smooth_hist[1:-1] = 0.25 * hist[:-2] + 0.5 * hist[1:-1] + 0.25 * hist[2:]
        peak_idx = int(np.argmax(smooth_hist))
        peak_value = float(smooth_hist[peak_idx])
        if peak_value <= 1e-9:
            return 0.0
        active_threshold = peak_value * float(self.cfg.cross_section_stripe_threshold_ratio)
        left_idx = peak_idx
        right_idx = peak_idx
        while left_idx > 0 and smooth_hist[left_idx - 1] >= active_threshold:
            left_idx -= 1
        while right_idx < smooth_hist.size - 1 and smooth_hist[right_idx + 1] >= active_threshold:
            right_idx += 1
        stripe_left = float(edges[left_idx])
        stripe_right = float(edges[right_idx + 1])
        stripe_center = 0.5 * (stripe_left + stripe_right)
        stripe_half_width_limit = max(float(self.cfg.cross_section_max_lane_half_width_m), lane_half)
        stripe_half_width = 0.5 * (stripe_right - stripe_left)
        if stripe_half_width > stripe_half_width_limit:
            stripe_left = stripe_center - stripe_half_width_limit
            stripe_right = stripe_center + stripe_half_width_limit
        in_stripe = (lateral >= stripe_left) & (lateral <= stripe_right)
        if np.count_nonzero(in_stripe) < 4:
            return 0.0
        stripe_weights = weights[in_stripe]
        stripe_lateral = lateral[in_stripe]
        denom = float(np.sum(stripe_weights))
        if denom <= 1e-9:
            return 0.0
        weighted_center = float(np.sum(stripe_lateral * stripe_weights) / denom)
        center_mix = float(np.clip(self.cfg.cross_section_center_mix, 0.0, 1.0))
        refined_shift = center_mix * stripe_center + (1.0 - center_mix) * weighted_center
        shift_limit = max(float(self.cfg.center_refine_max_shift_m), 1e-3)
        return float(np.clip(refined_shift, -shift_limit, shift_limit))

    def _unit2(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v[:2]))
        if n <= 1e-12:
            return np.array([1.0, 0.0], dtype=np.float64)
        return np.array([v[0] / n, v[1] / n], dtype=np.float64)

    def _build_query_debug(self, idx: np.ndarray) -> dict:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return {
                "count": 0,
                "index_head": [],
                "point_head": [],
            }
        pts = self.xyz[idx]
        ii = self.intensity[idx]
        head_n = min(5, idx.size)
        point_head = [
            f"({pts[i, 0]:.3f}, {pts[i, 1]:.3f}, {pts[i, 2]:.3f}; I={ii[i]:.3f})"
            for i in range(head_n)
        ]
        return {
            "count": int(idx.size),
            "index_head": [int(v) for v in idx[: min(8, idx.size)]],
            "point_head": point_head,
            "xy_mean": [float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))],
            "z_mean": float(np.mean(pts[:, 2])),
            "intensity_q50": float(np.quantile(ii, 0.50)),
            "intensity_q90": float(np.quantile(ii, 0.90)),
        }

    def _build_candidate_summaries(self, profile: CrossSectionProfile) -> list[str]:
        if profile is None or not profile.stripe_candidates:
            return []
        rows = []
        scored = list(enumerate(profile.stripe_candidates))
        scored.sort(
            key=lambda item: (
                float(item[1].final_score),
                float(item[1].identity_score),
                float(item[1].peak_value),
            ),
            reverse=True,
        )
        for rank, (idx, cand) in enumerate(scored[:5], start=1):
            marker = "*" if profile.selected_idx == idx else "-"
            rows.append(
                f"{marker}rank={rank}, idx={idx}, center={cand.center_m:.4f}, "
                f"left={cand.left_m:.4f}, right={cand.right_m:.4f}, width={cand.width_m:.4f}, "
                f"peak={cand.peak_value:.4f}, signal={cand.signal_consistency:.4f}, "
                f"identity={cand.identity_score:.4f}, final={cand.final_score:.4f}"
            )
        return rows
