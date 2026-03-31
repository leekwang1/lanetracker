from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..core.oriented_query import make_frame, project_points_xy
from ..core.spatial_grid import SpatialGrid
from ..core.types import StopReason, TrackMode
from .config import TrackerConfig
from .profile_types import ProfileData, ProfileStripeCandidate


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vv = np.asarray(v[:2], dtype=np.float64)
    n = float(np.linalg.norm(vv))
    if n < eps:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vv / n


def _rotate(v: np.ndarray, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c], dtype=np.float64)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(a)
    bb = _normalize(b)
    dot = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(math.acos(dot))


def _smooth_1d(values: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    if values.size == 0:
        return values
    if kernel is None:
        kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel = kernel / np.sum(kernel)
    return np.convolve(values, kernel, mode="same")


def _safe_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return float(np.percentile(values, q))


def _polyfit_predict(points_xy: np.ndarray, step_distance: float) -> tuple[np.ndarray, np.ndarray, float]:
    if len(points_xy) < 2:
        heading = np.array([1.0, 0.0], dtype=np.float64)
        return points_xy[-1] + heading * step_distance, heading, 0.0

    deltas = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(s[-1])
    if total < 1e-6:
        heading = _normalize(points_xy[-1] - points_xy[0])
        return points_xy[-1] + heading * step_distance, heading, 0.0

    degree = 2 if len(points_xy) >= 3 and total >= step_distance * 1.5 else 1
    coeff_x = np.polyfit(s, points_xy[:, 0], degree)
    coeff_y = np.polyfit(s, points_xy[:, 1], degree)
    target_s = s[-1] + step_distance
    pred_xy = np.array([np.polyval(coeff_x, target_s), np.polyval(coeff_y, target_s)], dtype=np.float64)
    deriv_x = np.polyder(coeff_x)
    deriv_y = np.polyder(coeff_y)
    deriv = np.array([np.polyval(deriv_x, target_s), np.polyval(deriv_y, target_s)], dtype=np.float64)
    heading = _normalize(deriv)
    curvature = 0.0
    if degree >= 2:
        d2_x = np.polyder(coeff_x, 2)
        d2_y = np.polyder(coeff_y, 2)
        second = np.array([np.polyval(d2_x, target_s), np.polyval(d2_y, target_s)], dtype=np.float64)
        denom = float(np.linalg.norm(deriv)) ** 3 + 1e-9
        curvature = float((deriv[0] * second[1] - deriv[1] * second[0]) / denom)
    return pred_xy, heading, curvature


def _find_local_positive_peaks(values: np.ndarray, minimum: float) -> np.ndarray:
    if values.size < 3:
        return np.empty((0,), dtype=np.int64)
    idx: list[int] = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1] and values[i] >= minimum:
            idx.append(i)
    return np.asarray(idx, dtype=np.int64)


def _find_local_negative_peaks(values: np.ndarray, minimum: float) -> np.ndarray:
    if values.size < 3:
        return np.empty((0,), dtype=np.int64)
    idx: list[int] = []
    for i in range(1, len(values) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1] and (-values[i]) >= minimum:
            idx.append(i)
    return np.asarray(idx, dtype=np.int64)


def _connected_peak_count(values: np.ndarray, threshold: float) -> int:
    if values.size == 0:
        return 0
    peaks = 0
    active = False
    for v in values:
        if v >= threshold:
            if not active:
                peaks += 1
                active = True
        else:
            active = False
    return peaks


def _autocorr_peak(values: np.ndarray, bin_size_m: float, min_period_m: float, max_period_m: float) -> tuple[float, float]:
    if values.size < 8:
        return 0.0, 0.0
    centered = values.astype(np.float64) - float(np.mean(values))
    denom = float(np.dot(centered, centered)) + 1e-9
    min_lag = max(1, int(round(min_period_m / max(bin_size_m, 1e-6))))
    max_lag = min(len(centered) - 2, int(round(max_period_m / max(bin_size_m, 1e-6))))
    best_corr = 0.0
    best_period = 0.0
    for lag in range(min_lag, max_lag + 1):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        if corr > best_corr:
            best_corr = corr
            best_period = lag * bin_size_m
    return float(np.clip(best_corr, 0.0, 1.0)), float(best_period)


@dataclass
class ObservationCandidate:
    center_xy: np.ndarray
    center_xyz: np.ndarray
    heading_xy: np.ndarray
    angle_offset_deg: float
    stripe_center_m: float
    left_edge_m: float
    right_edge_m: float
    width_m: float
    edge_score: float
    profile_score: float
    autocorr_score: float
    continuity_score: float
    total_score: float
    signal_score: float
    dashed_score: float
    solid_score: float
    crosswalk_score: float
    support_count: int
    z_ref: float
    dominant_period_m: float
    profile: ProfileData | None


@dataclass
class TrackerState:
    center_xyz: np.ndarray
    tangent_xy: np.ndarray
    lane_width_m: float
    left_edge_m: float
    right_edge_m: float
    stripe_center_m: float
    mode: TrackMode
    profile_quality: float
    stripe_strength: float
    center_confidence: float
    identity_confidence: float
    dashed_score: float
    solid_score: float
    crosswalk_score: float
    gap_distance_m: float
    curvature: float
    history_centers: list[np.ndarray] = field(default_factory=list)
    history_headings: list[np.ndarray] = field(default_factory=list)


@dataclass
class DebugFrame:
    step_index: int
    source: str
    candidate_points: np.ndarray | None
    display_search_center_xyz: np.ndarray | None
    display_search_radius_m: float | None
    display_selected_center_xyz: np.ndarray | None
    display_selected_radius_m: float | None
    chosen_candidate: ObservationCandidate | None
    candidate_count: int
    stop_reason: str | None
    profile: ProfileData | None
    trajectory_line_points: np.ndarray | None
    search_box_points: np.ndarray | None
    candidate_summaries: list[str]
    gap_distance_m: float


@dataclass
class TrackerResult:
    dense_points: np.ndarray
    output_points: np.ndarray
    stop_reason: str
    debug_frames: list[DebugFrame]


@dataclass
class BeamHypothesis:
    state: TrackerState
    cumulative_score: float
    first_observation: ObservationCandidate | None
    first_action: str
    depth: int
    path_actions: list[str] = field(default_factory=list)


class FanSearchTracker:
    def __init__(self, xyz: np.ndarray, intensity: np.ndarray, cfg: TrackerConfig):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.xy = self.xyz[:, :2]
        self.z = self.xyz[:, 2]
        self.intensity = np.asarray(intensity, dtype=np.float32)
        self.cfg = cfg
        self.grid = SpatialGrid(self.xy, cell_size=max(cfg.spatial_grid_cell_size_m, cfg.profile_bin_size_m * 4.0, 0.08))
        self._global_q50 = _safe_percentile(self.intensity.astype(np.float64), 50.0, default=0.0)
        self._global_q90 = _safe_percentile(self.intensity.astype(np.float64), 90.0, default=max(self._global_q50 + 1.0, 1.0))
        self.reset()

    def apply_config(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg

    def reset(self) -> None:
        self.state: TrackerState | None = None
        self.debug_frames: list[DebugFrame] = []
        self.stop_reason = StopReason.NONE
        self._distance_travelled_m = 0.0
        self._step_index = 0
        self._p0 = None
        self._p1 = None
        self._init_summary: dict[str, float | int | str] = {}
        self._reacquire_streak = 0
        self._last_reacquire_center_xy: np.ndarray | None = None

    def get_current_state(self) -> TrackerState | None:
        return self.state

    def get_last_debug_frame(self) -> DebugFrame | None:
        return self.debug_frames[-1] if self.debug_frames else None

    def initialize(self, p0_xyz: np.ndarray, p1_xyz: np.ndarray) -> None:
        self.reset()
        p0 = np.asarray(p0_xyz, dtype=np.float64)
        p1 = np.asarray(p1_xyz, dtype=np.float64)
        if p0.shape[0] == 2:
            p0 = np.array([p0[0], p0[1], 0.0], dtype=np.float64)
        if p1.shape[0] == 2:
            p1 = np.array([p1[0], p1[1], 0.0], dtype=np.float64)
        self._p0 = p0
        self._p1 = p1

        heading_xy = _normalize(p1[:2] - p0[:2])
        z_ref = self._estimate_z_ref(p0[:2], float(p0[2]))
        center_xyz = np.array([p0[0], p0[1], z_ref], dtype=np.float64)
        self.state = TrackerState(
            center_xyz=center_xyz,
            tangent_xy=heading_xy,
            lane_width_m=0.15,
            left_edge_m=-0.075,
            right_edge_m=0.075,
            stripe_center_m=0.0,
            mode=TrackMode.DASH_VISIBLE,
            profile_quality=0.0,
            stripe_strength=0.0,
            center_confidence=0.0,
            identity_confidence=1.0,
            dashed_score=0.0,
            solid_score=0.0,
            crosswalk_score=0.0,
            gap_distance_m=0.0,
            curvature=0.0,
            history_centers=[center_xyz.copy()],
            history_headings=[heading_xy.copy()],
        )

        plan = self._plan_beam(self.state)
        candidates, pred_xy, pred_heading, fan_centers = self._observe_candidates()
        best = plan.first_observation if plan.first_observation is not None else (
            self._select_preferred_candidate(candidates, 0.0) if candidates else self._evaluate_seed_pose(p0[:2], heading_xy, z_ref)
        )
        if best is None:
            raise RuntimeError("Failed to build an initial twin-edge candidate.")
        self._apply_observation(best, reset_gap=True)
        self._init_summary = {
            "reason": "beam_seed",
            "candidate_count": len(candidates),
            "best_score": float(plan.cumulative_score if plan.cumulative_score > 0.0 else best.total_score),
            "edge_score": float(best.edge_score),
            "profile_score": float(best.profile_score),
            "autocorr_score": float(best.autocorr_score),
            "continuity": float(best.continuity_score),
            "endpoint_evidence": float(best.total_score),
            "endpoint_distance": float(np.linalg.norm(pred_xy - p1[:2])) if pred_xy is not None else 0.0,
            "endpoint_loyalty": float(np.dot(_normalize(pred_heading), heading_xy)) if pred_heading is not None else 1.0,
        }
        preview_candidates, preview_pred_xy, preview_pred_heading, preview_fan_centers = self._observe_candidates()
        dbg = DebugFrame(
            step_index=0,
            source="seed",
            candidate_points=self._fan_centers_to_xyz(preview_fan_centers, float(self.state.center_xyz[2])),
            display_search_center_xyz=None,
            display_search_radius_m=None,
            display_selected_center_xyz=None,
            display_selected_radius_m=None,
            chosen_candidate=best,
            candidate_count=len(candidates),
            stop_reason=None,
            profile=best.profile,
            trajectory_line_points=self._build_trajectory_preview(preview_pred_xy, preview_pred_heading),
            search_box_points=self._build_search_wedge(self.state.center_xyz[:2], preview_pred_heading),
            candidate_summaries=self._candidate_summaries(candidates),
            gap_distance_m=0.0,
        )
        self.debug_frames.append(dbg)

    def step(self) -> DebugFrame:
        if self.state is None:
            raise RuntimeError("Tracker is not initialized.")
        if self.stop_reason != StopReason.NONE:
            return self.debug_frames[-1]

        plan = self._plan_beam(self.state)
        candidates, pred_xy, pred_heading, preview_centers = self._observe_candidates()
        chosen = plan.first_observation
        prefer_gap = plan.first_action == "gap"
        gap_step = False
        weak_candidate = (
            chosen is None
            or chosen.total_score < self.cfg.candidate_min_score
            or prefer_gap
            or (
                self.state.gap_distance_m <= 0.0
                and chosen.continuity_score < 0.22
                and chosen.total_score < self.cfg.candidate_min_score + 0.12
            )
        )

        if weak_candidate:
            gap_step = True
            next_gap = float(self.state.gap_distance_m + self.cfg.forward_distance_m)
            if next_gap > self.cfg.gap_forward_distance_m:
                self.stop_reason = StopReason.GAP_TOO_LONG
                self.state.mode = TrackMode.STOPPED
            else:
                gap_xy, gap_heading = self._predict_gap_pose()
                z_ref = self._estimate_z_ref(gap_xy, float(self.state.center_xyz[2]))
                self.state.center_xyz = np.array([gap_xy[0], gap_xy[1], z_ref], dtype=np.float64)
                self.state.tangent_xy = _normalize(gap_heading)
                self.state.mode = TrackMode.GAP_BRIDGING
                self.state.profile_quality = 0.0
                self.state.center_confidence = 0.0
                self.state.identity_confidence *= 0.85
                self.state.stripe_strength = 0.0
                self.state.dashed_score = 0.0
                self.state.solid_score = 0.0
                self.state.crosswalk_score = 0.0
                self.state.gap_distance_m = next_gap
                self._append_history(self.state.center_xyz, self.state.tangent_xy)
                self._distance_travelled_m += self.cfg.forward_distance_m
                self._reacquire_streak = 0
                self._last_reacquire_center_xy = None
        else:
            if self.state.gap_distance_m > 0.0:
                if self._is_strict_reacquire(chosen, pred_xy, pred_heading):
                    if self._last_reacquire_center_xy is not None and np.linalg.norm(chosen.center_xy - self._last_reacquire_center_xy) <= self.cfg.forward_distance_m * 1.5:
                        self._reacquire_streak += 1
                    else:
                        self._reacquire_streak = 1
                    self._last_reacquire_center_xy = chosen.center_xy.copy()
                else:
                    self._reacquire_streak = 0
                    self._last_reacquire_center_xy = None

                if self._reacquire_streak >= 2:
                    self._apply_observation(chosen, reset_gap=True)
                    self._distance_travelled_m += self.cfg.forward_distance_m
                    self._reacquire_streak = 0
                    self._last_reacquire_center_xy = None
                    if self.cfg.crosswalk_stop_enabled and chosen.crosswalk_score >= 0.95:
                        self.stop_reason = StopReason.CROSSWALK
                        self.state.mode = TrackMode.STOPPED
                else:
                    gap_step = True
                    next_gap = float(self.state.gap_distance_m + self.cfg.forward_distance_m)
                    if next_gap > self.cfg.gap_forward_distance_m:
                        self.stop_reason = StopReason.GAP_TOO_LONG
                        self.state.mode = TrackMode.STOPPED
                    else:
                        gap_xy, gap_heading = self._predict_gap_pose()
                        z_ref = self._estimate_z_ref(gap_xy, float(self.state.center_xyz[2]))
                        self.state.center_xyz = np.array([gap_xy[0], gap_xy[1], z_ref], dtype=np.float64)
                        self.state.tangent_xy = _normalize(gap_heading)
                        self.state.mode = TrackMode.GAP_BRIDGING
                        self.state.profile_quality = 0.0
                        self.state.center_confidence = 0.0
                        self.state.identity_confidence *= 0.90
                        self.state.stripe_strength = 0.0
                        self.state.dashed_score = 0.0
                        self.state.solid_score = 0.0
                        self.state.crosswalk_score = 0.0
                        self.state.gap_distance_m = next_gap
                        self._append_history(self.state.center_xyz, self.state.tangent_xy)
                        self._distance_travelled_m += self.cfg.forward_distance_m
            else:
                self._apply_observation(chosen, reset_gap=True)
                self._distance_travelled_m += self.cfg.forward_distance_m
                self._reacquire_streak = 0
                self._last_reacquire_center_xy = None
                if self.cfg.crosswalk_stop_enabled and chosen.crosswalk_score >= 0.95:
                    self.stop_reason = StopReason.CROSSWALK
                    self.state.mode = TrackMode.STOPPED

        self._step_index += 1
        if self.stop_reason == StopReason.NONE and self._distance_travelled_m >= self.cfg.max_track_length_m:
            self.stop_reason = StopReason.MAX_DISTANCE
            self.state.mode = TrackMode.STOPPED

        if self.stop_reason == StopReason.NONE:
            next_candidates, next_pred_xy, next_pred_heading, next_fan_centers = self._observe_candidates()
        else:
            next_candidates = []
            next_pred_xy = self.state.center_xyz[:2].copy()
            next_pred_heading = self.state.tangent_xy.copy()
            next_fan_centers = np.empty((0, 2), dtype=np.float64)

        dbg = DebugFrame(
            step_index=self._step_index,
            source="gap" if gap_step else "beam",
            candidate_points=self._fan_centers_to_xyz(next_fan_centers, float(self.state.center_xyz[2])),
            display_search_center_xyz=None,
            display_search_radius_m=None,
            display_selected_center_xyz=None,
            display_selected_radius_m=None,
            chosen_candidate=chosen,
            candidate_count=len(candidates),
            stop_reason=self.stop_reason.value if self.stop_reason != StopReason.NONE else None,
            profile=chosen.profile if chosen is not None else None,
            trajectory_line_points=self._build_trajectory_preview(next_pred_xy, next_pred_heading),
            search_box_points=self._build_search_wedge(self.state.center_xyz[:2], next_pred_heading),
            candidate_summaries=self._candidate_summaries(candidates),
            gap_distance_m=float(self.state.gap_distance_m),
        )
        self.debug_frames.append(dbg)
        return dbg

    def run_full(self) -> TrackerResult:
        while self.stop_reason == StopReason.NONE:
            self.step()
        dense = np.asarray(self.state.history_centers, dtype=np.float64) if self.state is not None else np.empty((0, 3), dtype=np.float64)
        return TrackerResult(dense_points=dense, output_points=dense, stop_reason=self.stop_reason.value, debug_frames=list(self.debug_frames))

    def _apply_observation(self, obs: ObservationCandidate, reset_gap: bool) -> None:
        self._apply_observation_to_state(self.state, obs, reset_gap)

    def _apply_observation_to_state(self, state: TrackerState, obs: ObservationCandidate, reset_gap: bool) -> None:
        state.center_xyz = obs.center_xyz.copy()
        prev_heading = _normalize(state.tangent_xy)
        next_heading = self._limit_heading_change(prev_heading, obs.heading_xy)
        state.tangent_xy = next_heading
        state.curvature = self._signed_angle(prev_heading, next_heading) / max(self.cfg.forward_distance_m, 1e-6)
        state.lane_width_m = float(obs.width_m)
        state.left_edge_m = float(obs.left_edge_m)
        state.right_edge_m = float(obs.right_edge_m)
        state.stripe_center_m = float(obs.stripe_center_m)
        state.mode = self._classify_mode(obs)
        state.profile_quality = float(obs.total_score)
        state.center_confidence = float(obs.total_score)
        state.identity_confidence = float(obs.continuity_score)
        state.stripe_strength = float(obs.edge_score)
        state.dashed_score = float(obs.dashed_score)
        state.solid_score = float(obs.solid_score)
        state.crosswalk_score = float(obs.crosswalk_score)
        if reset_gap:
            state.gap_distance_m = 0.0
        self._append_history_to_state(state, state.center_xyz, state.tangent_xy)

    def _append_history(self, center_xyz: np.ndarray, heading_xy: np.ndarray) -> None:
        self._append_history_to_state(self.state, center_xyz, heading_xy)

    def _append_history_to_state(self, state: TrackerState, center_xyz: np.ndarray, heading_xy: np.ndarray) -> None:
        state.history_centers.append(np.asarray(center_xyz, dtype=np.float64).copy())
        state.history_headings.append(_normalize(heading_xy).copy())

    def _clone_state(self, state: TrackerState) -> TrackerState:
        return TrackerState(
            center_xyz=np.asarray(state.center_xyz, dtype=np.float64).copy(),
            tangent_xy=_normalize(state.tangent_xy).copy(),
            lane_width_m=float(state.lane_width_m),
            left_edge_m=float(state.left_edge_m),
            right_edge_m=float(state.right_edge_m),
            stripe_center_m=float(state.stripe_center_m),
            mode=state.mode,
            profile_quality=float(state.profile_quality),
            stripe_strength=float(state.stripe_strength),
            center_confidence=float(state.center_confidence),
            identity_confidence=float(state.identity_confidence),
            dashed_score=float(state.dashed_score),
            solid_score=float(state.solid_score),
            crosswalk_score=float(state.crosswalk_score),
            gap_distance_m=float(state.gap_distance_m),
            curvature=float(state.curvature),
            history_centers=[np.asarray(c, dtype=np.float64).copy() for c in state.history_centers],
            history_headings=[_normalize(h).copy() for h in state.history_headings],
        )

    def _advance_gap_state(self, state: TrackerState) -> None:
        gap_xy, gap_heading = self._predict_gap_pose_for_state(state)
        z_ref = self._estimate_z_ref(gap_xy, float(state.center_xyz[2]))
        state.center_xyz = np.array([gap_xy[0], gap_xy[1], z_ref], dtype=np.float64)
        state.tangent_xy = _normalize(gap_heading)
        state.mode = TrackMode.GAP_BRIDGING
        state.profile_quality = 0.0
        state.center_confidence = 0.0
        state.identity_confidence *= 0.92
        state.stripe_strength = 0.0
        state.dashed_score = 0.0
        state.solid_score = 0.0
        state.crosswalk_score = 0.0
        state.gap_distance_m = float(state.gap_distance_m + self.cfg.forward_distance_m)
        self._append_history_to_state(state, state.center_xyz, state.tangent_xy)

    def _beam_increment_score(self, prev_state: TrackerState, obs: ObservationCandidate) -> float:
        width_consistency = math.exp(-abs(float(obs.width_m) - float(prev_state.lane_width_m)) / max(float(prev_state.lane_width_m), 0.10))
        period_consistency = 1.0 - min(1.0, abs(float(obs.dashed_score) - float(prev_state.dashed_score)))
        curve_next = self._estimate_heading_curvature_for_state(prev_state, obs.heading_xy)
        curve_consistency = math.exp(-abs(float(curve_next) - float(prev_state.curvature)) / 0.35)
        history_consistency = float(obs.continuity_score)
        edge_pair_score = 0.5 * (float(obs.edge_score) + float(obs.profile_score))
        signal_score = float(obs.signal_score)
        total = (
            self.cfg.beam_edge_weight * edge_pair_score
            + self.cfg.beam_signal_weight * signal_score
            + self.cfg.beam_curve_consistency_weight * curve_consistency
            + self.cfg.beam_width_consistency_weight * width_consistency
            + self.cfg.beam_period_consistency_weight * period_consistency
            + self.cfg.beam_history_weight * history_consistency
            - self.cfg.beam_crosswalk_penalty * float(obs.crosswalk_score)
        )
        return float(total)

    def _beam_gap_score(self, prev_state: TrackerState) -> float:
        gap_ratio = min(1.0, float(prev_state.gap_distance_m + self.cfg.forward_distance_m) / max(self.cfg.gap_forward_distance_m, 1e-6))
        return float(0.16 - 0.28 * gap_ratio)

    def _estimate_z_ref(self, center_xy: np.ndarray, fallback_z: float) -> float:
        idx = self.grid.query_radius_xy(center_xy, radius=max(self.cfg.profile_lateral_half_m, 0.20))
        if idx.size == 0:
            return float(fallback_z)
        z_vals = self.z[idx]
        if self.cfg.use_z_clip:
            lo = fallback_z - self.cfg.z_clip_half_range_m
            hi = fallback_z + self.cfg.z_clip_half_range_m
            z_vals = z_vals[(z_vals >= lo) & (z_vals <= hi)]
        if z_vals.size == 0:
            return float(fallback_z)
        return float(np.median(z_vals))

    def _filter_z(self, idx: np.ndarray, z_ref: float) -> np.ndarray:
        if idx.size == 0 or not self.cfg.use_z_clip:
            return idx
        lo = z_ref - self.cfg.z_clip_half_range_m
        hi = z_ref + self.cfg.z_clip_half_range_m
        return idx[(self.z[idx] >= lo) & (self.z[idx] <= hi)]

    def _observe_candidates(self) -> tuple[list[ObservationCandidate], np.ndarray, np.ndarray, np.ndarray]:
        return self._observe_candidates_for_state(self.state)

    def _observe_candidates_for_state(self, state: TrackerState) -> tuple[list[ObservationCandidate], np.ndarray, np.ndarray, np.ndarray]:
        saved_state = self.state
        self.state = state
        try:
            return self._observe_candidates_current()
        finally:
            self.state = saved_state

    def _observe_candidates_current(self) -> tuple[list[ObservationCandidate], np.ndarray, np.ndarray, np.ndarray]:
        if self.state.gap_distance_m > 0.0:
            pred_xy, pred_heading = self._predict_gap_pose()
            pred_curvature = self.state.curvature
        else:
            history = np.asarray([c[:2] for c in self.state.history_centers[-max(self.cfg.continuity_node_count, 2):]], dtype=np.float64)
            if len(history) < 2:
                pred_heading = _normalize(self.state.tangent_xy)
                pred_xy = self.state.center_xyz[:2] + pred_heading * self.cfg.forward_distance_m
                pred_curvature = self.state.curvature
            else:
                pred_xy, pred_heading, pred_curvature = _polyfit_predict(history, self.cfg.forward_distance_m)

        local_idx = self.grid.query_oriented_strip_xy(
            center_xy=pred_xy,
            tangent_xy=pred_heading,
            along_half=max(self.cfg.along_signal_half_m, self.cfg.forward_distance_m * 1.5),
            lateral_half=max(self.cfg.profile_lateral_half_m, self.cfg.twin_edge_max_width_m * 2.5),
        )
        local_idx = self._filter_z(local_idx, float(self.state.center_xyz[2]))
        local_intensity = self.intensity[local_idx].astype(np.float64)
        q50 = _safe_percentile(local_intensity, 50.0, default=self._global_q50)
        q90 = _safe_percentile(local_intensity, 90.0, default=self._global_q90)
        q90 = max(q90, q50 + 1e-3)

        fan_centers: list[np.ndarray] = []
        candidates: list[ObservationCandidate] = []
        for offset_deg in self._fan_offsets_deg():
            heading = _rotate(pred_heading, math.radians(offset_deg))
            center_xy = self.state.center_xyz[:2] + heading * self.cfg.forward_distance_m
            fan_centers.append(center_xy)
            cand = self._evaluate_candidate(center_xy, heading, offset_deg, pred_xy, pred_heading, pred_curvature, q50, q90)
            if cand is not None:
                candidates.append(cand)
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        return candidates, pred_xy, pred_heading, np.asarray(fan_centers, dtype=np.float64) if fan_centers else np.empty((0, 2), dtype=np.float64)

    def _plan_beam(self, state: TrackerState) -> BeamHypothesis:
        root_state = self._clone_state(state)
        beam = [
            BeamHypothesis(
                state=root_state,
                cumulative_score=0.0,
                first_observation=None,
                first_action="gap",
                depth=0,
                path_actions=[],
            )
        ]
        best_final = beam[0]

        for depth in range(max(1, int(self.cfg.beam_horizon_steps))):
            next_beam: list[BeamHypothesis] = []
            for hyp in beam:
                candidates, pred_xy, pred_heading, _ = self._observe_candidates_for_state(hyp.state)
                top_candidates = candidates[: max(1, int(self.cfg.beam_branching))]
                for cand in top_candidates:
                    next_state = self._clone_state(hyp.state)
                    self._apply_observation_to_state(next_state, cand, reset_gap=True)
                    inc = self._beam_increment_score(hyp.state, cand)
                    next_beam.append(
                        BeamHypothesis(
                            state=next_state,
                            cumulative_score=float(hyp.cumulative_score + inc),
                            first_observation=cand if hyp.first_observation is None else hyp.first_observation,
                            first_action="obs" if hyp.first_observation is None else hyp.first_action,
                            depth=depth + 1,
                            path_actions=hyp.path_actions + [f"obs({cand.angle_offset_deg:.0f})"],
                        )
                    )

                gap_state = self._clone_state(hyp.state)
                self._advance_gap_state(gap_state)
                gap_score = self._beam_gap_score(hyp.state)
                next_beam.append(
                    BeamHypothesis(
                        state=gap_state,
                        cumulative_score=float(hyp.cumulative_score + gap_score),
                        first_observation=None if hyp.first_observation is None else hyp.first_observation,
                        first_action="gap" if hyp.first_observation is None else hyp.first_action,
                        depth=depth + 1,
                        path_actions=hyp.path_actions + ["gap"],
                    )
                )

            if not next_beam:
                break
            next_beam.sort(key=lambda h: h.cumulative_score, reverse=True)
            beam = next_beam[: max(1, int(self.cfg.beam_width))]
            best_final = beam[0]

        return best_final

    def _select_preferred_candidate(
        self,
        candidates: list[ObservationCandidate],
        pred_curvature: float,
    ) -> ObservationCandidate | None:
        if not candidates:
            return None

        best = candidates[0]
        straight = min(
            candidates,
            key=lambda c: (abs(float(c.angle_offset_deg)), -float(c.total_score)),
        )
        if straight is best:
            return best

        if abs(float(pred_curvature)) > self.cfg.straight_hold_curvature_threshold:
            return best

        if abs(float(straight.angle_offset_deg)) > max(10.0, self.cfg.candidate_angle_deg * 0.5):
            return best

        score_close = float(straight.total_score + self.cfg.straight_keep_score_margin) >= float(best.total_score)
        continuity_close = float(straight.continuity_score + 0.10) >= float(best.continuity_score)
        edge_close = float(straight.edge_score + 0.08) >= float(best.edge_score)
        if score_close and continuity_close and edge_close:
            return straight
        if self._should_keep_straight(best, straight, pred_curvature):
            return straight
        return best

    def _should_bridge_gap(
        self,
        chosen: ObservationCandidate | None,
        candidates: list[ObservationCandidate],
    ) -> bool:
        if chosen is None or not candidates or self.state is None:
            return chosen is None
        if self.state.gap_distance_m > 0.0:
            return False

        turning_threshold = max(10.0, self.cfg.candidate_angle_deg / 3.0)
        turning = abs(float(chosen.angle_offset_deg)) >= turning_threshold
        low_continuity = float(chosen.continuity_score) < 0.55
        weak_evidence = float(chosen.edge_score) < 0.60 or float(chosen.profile_score) < 0.60
        low_total = float(chosen.total_score) < max(self.cfg.candidate_min_score + 0.18, 0.58)

        second_score = float(candidates[1].total_score) if len(candidates) > 1 else 0.0
        ambiguous = float(chosen.total_score - second_score) < max(0.03, self.cfg.straight_keep_score_margin)

        straight_candidate = min(
            candidates,
            key=lambda c: (abs(float(c.angle_offset_deg)), -float(c.total_score)),
        )
        straight_close = (
            abs(float(straight_candidate.angle_offset_deg)) <= 1e-6
            and float(straight_candidate.total_score + self.cfg.straight_keep_score_margin) >= float(chosen.total_score)
        )

        if turning and low_continuity and (weak_evidence or ambiguous or straight_close):
            return True
        if low_total and low_continuity:
            return True
        return False

    def _should_keep_straight(
        self,
        best: ObservationCandidate,
        straight: ObservationCandidate,
        pred_curvature: float,
    ) -> bool:
        if abs(float(straight.angle_offset_deg)) > 1e-6:
            return False
        if abs(float(pred_curvature)) > self.cfg.straight_hold_curvature_threshold:
            return False
        if abs(float(best.angle_offset_deg)) <= 1e-6:
            return False
        if float(best.continuity_score) < self.cfg.turn_min_continuity:
            return True
        if float(best.total_score) < float(straight.total_score + self.cfg.turn_commit_score_margin):
            return True
        return False

    def _limit_heading_change(self, prev_heading_xy: np.ndarray, next_heading_xy: np.ndarray) -> np.ndarray:
        prev = _normalize(prev_heading_xy)
        nxt = _normalize(next_heading_xy)
        delta = self._signed_angle(prev, nxt)
        max_delta = math.radians(max(1.0, self.cfg.max_heading_change_deg))
        delta = float(np.clip(delta, -max_delta, max_delta))
        return _normalize(_rotate(prev, delta))

    def _evaluate_seed_pose(self, center_xy: np.ndarray, heading_xy: np.ndarray, z_ref: float) -> ObservationCandidate | None:
        return self._evaluate_candidate(center_xy, heading_xy, 0.0, center_xy, heading_xy, 0.0, self._global_q50, self._global_q90, continuity_override=1.0)

    def _evaluate_candidate(
        self,
        base_center_xy: np.ndarray,
        heading_xy: np.ndarray,
        angle_offset_deg: float,
        pred_xy: np.ndarray,
        pred_heading: np.ndarray,
        pred_curvature: float,
        q50: float,
        q90: float,
        continuity_override: float | None = None,
    ) -> ObservationCandidate | None:
        z_ref = self._estimate_z_ref(base_center_xy, float(self.state.center_xyz[2]))
        profile = self._build_twin_edge_profile(base_center_xy, heading_xy, z_ref, q50, q90)
        if profile is None or profile.selected_idx is None:
            return None

        stripe = profile.stripe_candidates[profile.selected_idx]
        width_m = float(stripe.width_m)
        if width_m < self.cfg.twin_edge_min_width_m or width_m > self.cfg.twin_edge_max_width_m:
            return None

        tangent, normal = make_frame(heading_xy)
        center_xy = base_center_xy + normal * float(stripe.center_m)
        z_ref = self._estimate_z_ref(center_xy, z_ref)
        center_xyz = np.array([center_xy[0], center_xy[1], z_ref], dtype=np.float64)

        signal_score, dashed_score, solid_score, autocorr_score, dominant_period = self._analyze_along_signal(center_xy, tangent, z_ref, width_m, q50, q90)
        crosswalk_score = self._detect_crosswalk(center_xy, tangent, z_ref, q50, q90)

        if continuity_override is None:
            continuity_score = self._compute_continuity_score(center_xy, tangent, width_m, pred_xy, pred_heading, pred_curvature)
        else:
            continuity_score = float(continuity_override)

        edge_score = float(profile.quality)
        total_score = float(
            np.clip(
                0.34 * edge_score
                + 0.18 * profile.quality
                + 0.18 * signal_score
                + 0.12 * autocorr_score
                + 0.18 * continuity_score
                - 0.08 * crosswalk_score,
                0.0,
                1.0,
            )
        )

        support_count = int(np.count_nonzero(profile.hist_combined > 0.0))
        if support_count < self.cfg.candidate_min_support:
            return None

        left_edge_m = float(stripe.left_m - stripe.center_m)
        right_edge_m = float(stripe.right_m - stripe.center_m)
        return ObservationCandidate(
            center_xy=center_xy,
            center_xyz=center_xyz,
            heading_xy=tangent,
            angle_offset_deg=float(angle_offset_deg),
            stripe_center_m=float(stripe.center_m),
            left_edge_m=left_edge_m,
            right_edge_m=right_edge_m,
            width_m=width_m,
            edge_score=edge_score,
            profile_score=float(profile.quality),
            autocorr_score=float(autocorr_score),
            continuity_score=float(continuity_score),
            total_score=total_score,
            signal_score=float(signal_score),
            dashed_score=float(dashed_score),
            solid_score=float(solid_score),
            crosswalk_score=float(crosswalk_score),
            support_count=support_count,
            z_ref=float(z_ref),
            dominant_period_m=float(dominant_period),
            profile=profile,
        )

    def _build_twin_edge_profile(
        self,
        center_xy: np.ndarray,
        tangent_xy: np.ndarray,
        z_ref: float,
        q50: float,
        q90: float,
    ) -> ProfileData | None:
        idx = self.grid.query_oriented_strip_xy(
            center_xy=center_xy,
            tangent_xy=tangent_xy,
            along_half=self.cfg.profile_along_half_m,
            lateral_half=self.cfg.profile_lateral_half_m,
        )
        idx = self._filter_z(idx, z_ref)
        if idx.size < self.cfg.candidate_min_support:
            return None

        along, lateral = project_points_xy(self.xy[idx], center_xy, tangent_xy)
        mask = np.abs(along) <= self.cfg.profile_along_half_m
        if not np.any(mask):
            return None

        lateral = lateral[mask]
        intensity = self.intensity[idx][mask].astype(np.float64)
        signal = np.clip((intensity - q50) / max(q90 - q50, 1e-6), 0.0, 1.0)

        bins = np.arange(
            -self.cfg.profile_lateral_half_m,
            self.cfg.profile_lateral_half_m + self.cfg.profile_bin_size_m,
            self.cfg.profile_bin_size_m,
            dtype=np.float64,
        )
        if bins.size < 4:
            return None
        bin_centers = (bins[:-1] + bins[1:]) * 0.5
        sums = np.zeros(bin_centers.size, dtype=np.float64)
        counts = np.zeros(bin_centers.size, dtype=np.float64)
        bin_idx = np.digitize(lateral, bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < bin_centers.size)
        if not np.any(valid):
            return None
        np.add.at(sums, bin_idx[valid], signal[valid])
        np.add.at(counts, bin_idx[valid], 1.0)
        raw = np.divide(sums, np.maximum(counts, 1.0))
        smooth = _smooth_1d(raw)
        grad = np.gradient(smooth, self.cfg.profile_bin_size_m)

        pos_idx = _find_local_positive_peaks(grad, self.cfg.edge_grad_min)
        neg_idx = _find_local_negative_peaks(grad, self.cfg.edge_grad_min)
        if pos_idx.size == 0 or neg_idx.size == 0:
            return None

        stripes: list[ProfileStripeCandidate] = []
        stripe_scores: list[float] = []
        grad_scale = max(float(np.max(np.abs(grad))), 1e-6)
        for li in pos_idx:
            for ri in neg_idx:
                if ri <= li:
                    continue
                left_m = float(bin_centers[li])
                right_m = float(bin_centers[ri])
                width_m = right_m - left_m
                if width_m < self.cfg.twin_edge_min_width_m or width_m > self.cfg.twin_edge_max_width_m:
                    continue
                inner = smooth[li : ri + 1]
                outer_left = smooth[max(0, li - 3) : li]
                outer_right = smooth[ri + 1 : min(len(smooth), ri + 4)]
                outer = np.concatenate([outer_left, outer_right]) if outer_left.size + outer_right.size > 0 else np.empty((0,), dtype=np.float64)
                interior_mean = float(np.mean(inner)) if inner.size else 0.0
                exterior_mean = float(np.mean(outer)) if outer.size else 0.0
                contrast = max(0.0, interior_mean - exterior_mean)
                edge_strength = (float(grad[li]) + float(-grad[ri])) / (2.0 * grad_scale)
                support = min(1.0, float(np.sum(counts[li : ri + 1])) / 18.0)
                symmetry = 1.0 - min(1.0, abs((left_m + right_m) * 0.5) / max(width_m * 0.75, 1e-6))
                quality = float(np.clip(0.45 * edge_strength + 0.35 * contrast + 0.10 * support + 0.10 * symmetry, 0.0, 1.0))
                stripes.append(
                    ProfileStripeCandidate(
                        left_m=left_m,
                        right_m=right_m,
                        center_m=(left_m + right_m) * 0.5,
                        width_m=width_m,
                    )
                )
                stripe_scores.append(quality)

        if not stripe_scores:
            return None

        selected_idx = int(np.argmax(np.asarray(stripe_scores, dtype=np.float64)))
        return ProfileData(
            bins_center=bin_centers,
            hist_combined=raw,
            smooth_hist=smooth,
            stripe_candidates=stripes,
            selected_idx=selected_idx,
            quality=float(stripe_scores[selected_idx]),
        )

    def _analyze_along_signal(
        self,
        center_xy: np.ndarray,
        tangent_xy: np.ndarray,
        z_ref: float,
        width_m: float,
        q50: float,
        q90: float,
    ) -> tuple[float, float, float, float, float]:
        lateral_half = max(self.cfg.along_signal_lateral_half_m, width_m * 0.8)
        idx = self.grid.query_oriented_strip_xy(
            center_xy=center_xy,
            tangent_xy=tangent_xy,
            along_half=self.cfg.along_signal_half_m,
            lateral_half=lateral_half,
        )
        idx = self._filter_z(idx, z_ref)
        if idx.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        along, lateral = project_points_xy(self.xy[idx], center_xy, tangent_xy)
        mask = np.abs(lateral) <= lateral_half
        if not np.any(mask):
            return 0.0, 0.0, 0.0, 0.0, 0.0

        along = along[mask]
        signal = np.clip((self.intensity[idx][mask].astype(np.float64) - q50) / max(q90 - q50, 1e-6), 0.0, 1.0)
        bins = np.arange(
            -self.cfg.along_signal_half_m,
            self.cfg.along_signal_half_m + self.cfg.along_signal_bin_m,
            self.cfg.along_signal_bin_m,
            dtype=np.float64,
        )
        bin_centers = (bins[:-1] + bins[1:]) * 0.5
        sums = np.zeros(bin_centers.size, dtype=np.float64)
        counts = np.zeros(bin_centers.size, dtype=np.float64)
        bin_idx = np.digitize(along, bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < bin_centers.size)
        if not np.any(valid):
            return 0.0, 0.0, 0.0, 0.0, 0.0
        np.add.at(sums, bin_idx[valid], signal[valid])
        np.add.at(counts, bin_idx[valid], 1.0)
        prof = np.divide(sums, np.maximum(counts, 1.0))
        prof = _smooth_1d(prof)
        max_val = float(np.max(prof)) if prof.size else 0.0
        if max_val > 1e-6:
            prof = prof / max_val

        occupancy = float(np.mean(prof > 0.35))
        autocorr_score, dominant_period = _autocorr_peak(
            prof,
            self.cfg.along_signal_bin_m,
            self.cfg.autocorr_min_period_m,
            self.cfg.autocorr_max_period_m,
        )
        dashed_score = float(np.clip((autocorr_score - self.cfg.dashed_autocorr_min) / max(1.0 - self.cfg.dashed_autocorr_min, 1e-6), 0.0, 1.0))
        solid_score = float(np.clip((occupancy - 0.25) / max(self.cfg.solid_occupancy_min - 0.25, 1e-6), 0.0, 1.0))
        signal_score = float(np.clip(0.45 * occupancy + 0.35 * max_val + 0.20 * max(solid_score, dashed_score), 0.0, 1.0))
        return signal_score, dashed_score, solid_score, autocorr_score, dominant_period

    def _detect_crosswalk(
        self,
        center_xy: np.ndarray,
        tangent_xy: np.ndarray,
        z_ref: float,
        q50: float,
        q90: float,
    ) -> float:
        strip_center = center_xy + tangent_xy * (self.cfg.crosswalk_lookahead_m * 0.5)
        idx = self.grid.query_oriented_strip_xy(
            center_xy=strip_center,
            tangent_xy=tangent_xy,
            along_half=self.cfg.crosswalk_lookahead_m * 0.5,
            lateral_half=self.cfg.crosswalk_lateral_half_m,
        )
        idx = self._filter_z(idx, z_ref)
        if idx.size == 0:
            return 0.0

        along, _ = project_points_xy(self.xy[idx], strip_center, tangent_xy)
        signal = np.clip((self.intensity[idx].astype(np.float64) - q50) / max(q90 - q50, 1e-6), 0.0, 1.0)
        bins = np.arange(
            -self.cfg.crosswalk_lookahead_m * 0.5,
            self.cfg.crosswalk_lookahead_m * 0.5 + 0.05,
            0.05,
            dtype=np.float64,
        )
        bin_centers = (bins[:-1] + bins[1:]) * 0.5
        sums = np.zeros(bin_centers.size, dtype=np.float64)
        counts = np.zeros(bin_centers.size, dtype=np.float64)
        bin_idx = np.digitize(along, bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < bin_centers.size)
        if not np.any(valid):
            return 0.0
        np.add.at(sums, bin_idx[valid], signal[valid])
        np.add.at(counts, bin_idx[valid], 1.0)
        prof = np.divide(sums, np.maximum(counts, 1.0))
        prof = _smooth_1d(prof)
        if np.max(prof) > 1e-6:
            prof = prof / float(np.max(prof))
        peak_count = _connected_peak_count(prof, threshold=0.55)
        return float(np.clip(peak_count / max(self.cfg.crosswalk_min_peaks, 1), 0.0, 1.0))

    def _compute_continuity_score(
        self,
        center_xy: np.ndarray,
        heading_xy: np.ndarray,
        width_m: float,
        pred_xy: np.ndarray,
        pred_heading: np.ndarray,
        pred_curvature: float,
    ) -> float:
        dist_pen = float(np.linalg.norm(center_xy - pred_xy)) / max(self.cfg.forward_distance_m, 1e-6)
        heading_pen = _angle_between(heading_xy, pred_heading) / max(math.radians(self.cfg.candidate_angle_deg), 1e-6)
        width_pen = abs(width_m - float(self.state.lane_width_m)) / max(float(self.state.lane_width_m), 0.10)
        obs_curvature = self._estimate_heading_curvature(heading_xy)
        curve_pen = abs(obs_curvature - pred_curvature) / 0.35
        score = math.exp(
            -self.cfg.continuity_strength
            * (0.55 * dist_pen + 0.25 * heading_pen + 0.12 * width_pen + 0.08 * curve_pen)
        )
        return float(np.clip(score, 0.0, 1.0))

    def _estimate_heading_curvature(self, heading_xy: np.ndarray) -> float:
        return self._estimate_heading_curvature_for_state(self.state, heading_xy)

    def _estimate_heading_curvature_for_state(self, state: TrackerState, heading_xy: np.ndarray) -> float:
        if state is None or not state.history_headings:
            return 0.0
        prev = _normalize(state.history_headings[-1])
        delta = self._signed_angle(prev, heading_xy)
        return float(delta / max(self.cfg.forward_distance_m, 1e-6))

    def _estimate_step_curvature(self, heading_xy: np.ndarray) -> float:
        return self._estimate_heading_curvature(heading_xy)

    def _predict_gap_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._predict_gap_pose_for_state(self.state)

    def _predict_gap_pose_for_state(self, state: TrackerState) -> tuple[np.ndarray, np.ndarray]:
        heading = _normalize(state.tangent_xy)
        gap_ratio = min(1.0, float(state.gap_distance_m) / max(self.cfg.gap_forward_distance_m, 1e-6))
        curvature = float(state.curvature) * max(0.35, 1.0 - 0.65 * gap_ratio)
        next_heading = _rotate(heading, curvature * self.cfg.forward_distance_m)
        next_heading = _normalize(next_heading)
        next_xy = state.center_xyz[:2] + next_heading * self.cfg.forward_distance_m
        return next_xy, next_heading

    def _is_strict_reacquire(self, obs: ObservationCandidate, pred_xy: np.ndarray, pred_heading: np.ndarray) -> bool:
        _, lateral = project_points_xy(obs.center_xy[None, :], pred_xy, pred_heading)
        lateral_err = abs(float(lateral[0]))
        width_err = abs(float(obs.width_m) - float(self.state.lane_width_m))
        heading_err = _angle_between(obs.heading_xy, pred_heading)
        lateral_tol = max(0.08, float(self.state.lane_width_m) * 0.55)
        width_tol = max(0.05, float(self.state.lane_width_m) * 0.45)
        heading_tol = math.radians(max(8.0, self.cfg.candidate_angle_deg * 0.45))
        return (
            obs.total_score >= max(self.cfg.candidate_min_score + 0.08, 0.45)
            and obs.edge_score >= 0.45
            and obs.profile_score >= 0.45
            and obs.continuity_score >= 0.35
            and lateral_err <= lateral_tol
            and width_err <= width_tol
            and heading_err <= heading_tol
        )

    def _signed_angle(self, a: np.ndarray, b: np.ndarray) -> float:
        aa = _normalize(a)
        bb = _normalize(b)
        return float(math.atan2(aa[0] * bb[1] - aa[1] * bb[0], np.dot(aa, bb)))

    def _classify_mode(self, obs: ObservationCandidate) -> TrackMode:
        if obs.crosswalk_score >= 0.75:
            return TrackMode.CROSSWALK_CANDIDATE
        if obs.dashed_score >= obs.solid_score and obs.autocorr_score >= self.cfg.dashed_autocorr_min:
            return TrackMode.DASH_VISIBLE
        return TrackMode.SOLID_VISIBLE

    def _candidate_summaries(self, candidates: list[ObservationCandidate]) -> list[str]:
        return [
            f"a={c.angle_offset_deg:.1f} | edge={c.edge_score:.2f} | prof={c.profile_score:.2f} | ac={c.autocorr_score:.2f} | cont={c.continuity_score:.2f} | total={c.total_score:.2f}"
            for c in candidates
        ]

    def _build_trajectory_preview(self, pred_xy: np.ndarray | None, pred_heading: np.ndarray | None) -> np.ndarray | None:
        if self.state is None or pred_xy is None or pred_heading is None:
            return None
        current = self.state.center_xyz.copy()
        target = np.array([pred_xy[0], pred_xy[1], current[2]], dtype=np.float64)
        return np.vstack([current, target])

    def _build_search_wedge(self, center_xy: np.ndarray, heading_xy: np.ndarray | None) -> np.ndarray | None:
        if heading_xy is None:
            return None
        heading = _normalize(heading_xy)
        origin = np.array([center_xy[0], center_xy[1], float(self.state.center_xyz[2])], dtype=np.float64)
        far = float(self.cfg.forward_distance_m)
        left_dir = _rotate(heading, math.radians(self.cfg.candidate_angle_deg))
        right_dir = _rotate(heading, -math.radians(self.cfg.candidate_angle_deg))
        p_left = np.array([origin[0] + left_dir[0] * far, origin[1] + left_dir[1] * far, origin[2]], dtype=np.float64)
        p_mid = np.array([origin[0] + heading[0] * far, origin[1] + heading[1] * far, origin[2]], dtype=np.float64)
        p_right = np.array([origin[0] + right_dir[0] * far, origin[1] + right_dir[1] * far, origin[2]], dtype=np.float64)
        return np.vstack([origin, p_left, p_mid, p_right, origin])

    def _fan_offsets_deg(self) -> np.ndarray:
        count = max(1, int(self.cfg.side_candidate_count))
        if count == 1:
            return np.array([0.0], dtype=np.float64)
        return np.linspace(-self.cfg.candidate_angle_deg, self.cfg.candidate_angle_deg, count, dtype=np.float64)

    def _fan_centers_to_xyz(self, centers_xy: np.ndarray, z_ref: float) -> np.ndarray:
        if centers_xy.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        z = np.full((len(centers_xy), 1), float(z_ref), dtype=np.float64)
        return np.column_stack([centers_xy, z])
