from __future__ import annotations

import copy
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


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(a)
    bb = _normalize(b)
    dot = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(math.acos(dot))


def _signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(a)
    bb = _normalize(b)
    return float(math.atan2(aa[0] * bb[1] - aa[1] * bb[0], np.dot(aa, bb)))


def _safe_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return float(np.percentile(values, q))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    vv = np.asarray(values, dtype=np.float64)
    if vv.size == 0:
        return 0.0
    ww = np.asarray(weights, dtype=np.float64)
    if ww.size != vv.size:
        return float(np.median(vv))
    ww = np.clip(ww, 0.0, None)
    total = float(np.sum(ww))
    if total <= 1e-9:
        return float(np.median(vv))
    order = np.argsort(vv)
    vv = vv[order]
    ww = ww[order]
    cutoff = 0.5 * total
    idx = int(np.searchsorted(np.cumsum(ww), cutoff, side="left"))
    idx = min(max(idx, 0), len(vv) - 1)
    return float(vv[idx])


def _polyfit_predict(points_xy: np.ndarray, step_distance: float) -> tuple[np.ndarray, np.ndarray, float]:
    pts = np.asarray(points_xy, dtype=np.float64)
    if len(pts) < 2:
        heading = np.array([1.0, 0.0], dtype=np.float64)
        return pts[-1].copy(), heading, 0.0

    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(s[-1])
    if total < 1e-6:
        heading = _normalize(pts[-1] - pts[0])
        return pts[-1] + heading * step_distance, heading, 0.0

    degree = 2 if len(pts) >= 4 and total >= step_distance * 2.0 else 1
    coeff_x = np.polyfit(s, pts[:, 0], degree)
    coeff_y = np.polyfit(s, pts[:, 1], degree)
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
    intensity_score: float
    contrast_score: float
    autocorr_score: float
    continuity_score: float
    total_score: float
    path_score: float
    dashed_score: float
    solid_score: float
    crosswalk_score: float
    support_count: int
    z_ref: float
    dominant_period_m: float
    path_length_m: float
    path_node_count: int
    fill_ratio: float
    mode_consistency_score: float
    identity_score: float
    switch_penalty: float
    center_lateral_m: float
    future_identity_score: float
    future_switch_penalty: float
    endpoint_lateral_m: float
    max_path_lateral_m: float
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
    path_centers: list[np.ndarray] = field(default_factory=list)
    path_headings: list[np.ndarray] = field(default_factory=list)


@dataclass
class DebugFrame:
    step_index: int
    source: str
    candidate_points: np.ndarray | None
    active_cell_box_groups: list[np.ndarray] | None
    segment_groups: list[np.ndarray] | None
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
class TrackerSnapshot:
    state: TrackerState | None
    debug_frames: list[DebugFrame]
    stop_reason: StopReason
    distance_travelled_m: float
    step_index: int
    p0: np.ndarray | None
    p1: np.ndarray | None
    init_summary: dict[str, float | int | str]
    switch_streak: int
    switch_sign: int
    crosswalk_streak: int


@dataclass
class _ComponentCandidate:
    cells: list[tuple[int, int]]
    cell_centers_local: np.ndarray
    cell_weights: np.ndarray
    poly_coeff: np.ndarray
    sample_along: np.ndarray
    sample_lateral: np.ndarray
    unique_along_count: int
    span_m: float
    coverage_ratio: float
    score: float
    start_lateral_m: float
    step_lateral_m: float
    end_lateral_m: float
    max_abs_lateral_m: float
    heading_xy: np.ndarray
    heading_error_rad: float
    sample_left_edge: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    sample_right_edge: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    width_m: float = 0.0
    intensity_score: float = 0.0
    contrast_score: float = 0.0


@dataclass
class _StripePair1D:
    left_idx: int
    right_idx: int
    left_m: float
    right_m: float
    center_m: float
    width_m: float
    edge_strength: float
    inside_mean: float
    contrast_score: float
    score: float


@dataclass
class _RowStripe:
    along_idx: int
    along_m: float
    left_edge_m: float
    right_edge_m: float
    lateral_center_m: float
    width_m: float
    weight: float
    intensity_mean: float
    edge_strength: float
    contrast_score: float
    cells: list[tuple[int, int]]


@dataclass
class _RowCluster:
    along_idx: int
    along_m: float
    lateral_center_m: float
    weight: float
    cells: list[tuple[int, int]]


class SimpleSplineTracker:
    def __init__(self, xyz: np.ndarray, intensity: np.ndarray, cfg: TrackerConfig):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.xy = self.xyz[:, :2]
        self.z = self.xyz[:, 2]
        self.intensity = np.asarray(intensity, dtype=np.float32)
        self.cfg = cfg
        self.grid = SpatialGrid(self.xy, cell_size=max(float(cfg.grid_cell_size_m), 0.08))
        inten = self.intensity.astype(np.float64)
        self._global_q50 = _safe_percentile(inten, 50.0, default=0.0)
        self._global_q95 = _safe_percentile(inten, 95.0, default=max(self._global_q50 + 1.0, 1.0))
        self.reset()

    def apply_config(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg

    def reset(self) -> None:
        self.state: TrackerState | None = None
        self.debug_frames: list[DebugFrame] = []
        self.stop_reason = StopReason.NONE
        self._distance_travelled_m = 0.0
        self._step_index = 0
        self._p0: np.ndarray | None = None
        self._p1: np.ndarray | None = None
        self._init_summary: dict[str, float | int | str] = {}

    def get_current_state(self) -> TrackerState | None:
        return self.state

    def get_last_debug_frame(self) -> DebugFrame | None:
        return self.debug_frames[-1] if self.debug_frames else None

    def make_snapshot(self) -> TrackerSnapshot:
        return TrackerSnapshot(
            state=copy.deepcopy(self.state),
            debug_frames=copy.deepcopy(self.debug_frames),
            stop_reason=self.stop_reason,
            distance_travelled_m=float(self._distance_travelled_m),
            step_index=int(self._step_index),
            p0=None if self._p0 is None else np.asarray(self._p0, dtype=np.float64).copy(),
            p1=None if self._p1 is None else np.asarray(self._p1, dtype=np.float64).copy(),
            init_summary=copy.deepcopy(self._init_summary),
            switch_streak=0,
            switch_sign=0,
            crosswalk_streak=0,
        )

    def restore_snapshot(self, snapshot: TrackerSnapshot) -> None:
        self.state = copy.deepcopy(snapshot.state)
        self.debug_frames = copy.deepcopy(snapshot.debug_frames)
        self.stop_reason = snapshot.stop_reason
        self._distance_travelled_m = float(snapshot.distance_travelled_m)
        self._step_index = int(snapshot.step_index)
        self._p0 = None if snapshot.p0 is None else np.asarray(snapshot.p0, dtype=np.float64).copy()
        self._p1 = None if snapshot.p1 is None else np.asarray(snapshot.p1, dtype=np.float64).copy()
        self._init_summary = copy.deepcopy(snapshot.init_summary)

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
        stripe_width = float(self.cfg.stripe_width_m)
        center_xyz = np.array([p0[0], p0[1], z_ref], dtype=np.float64)
        self.state = TrackerState(
            center_xyz=center_xyz,
            tangent_xy=heading_xy,
            lane_width_m=stripe_width,
            left_edge_m=-0.5 * stripe_width,
            right_edge_m=0.5 * stripe_width,
            stripe_center_m=0.0,
            mode=TrackMode.SOLID_VISIBLE,
            profile_quality=0.0,
            stripe_strength=0.0,
            center_confidence=0.0,
            identity_confidence=1.0,
            dashed_score=0.0,
            solid_score=1.0,
            crosswalk_score=0.0,
            gap_distance_m=0.0,
            curvature=0.0,
            history_centers=[center_xyz.copy()],
            history_headings=[heading_xy.copy()],
            path_centers=[center_xyz.copy()],
            path_headings=[heading_xy.copy()],
        )

        best, dbg = self._observe(self.state, seed_mode=True)
        if best is not None:
            self._apply_observation(best)
            self._init_summary = {
                "reason": "simple_spline",
                "candidate_count": int(dbg.candidate_count),
                "best_score": float(best.total_score),
                "intensity_score": float(best.intensity_score),
                "contrast_score": float(best.contrast_score),
                "autocorr_score": 0.0,
                "continuity": float(best.continuity_score),
                "endpoint_evidence": float(best.path_score),
                "endpoint_distance": float(np.linalg.norm(best.center_xy - p1[:2])),
                "endpoint_loyalty": float(np.dot(_normalize(best.heading_xy), heading_xy)),
            }
            dbg.source = "box_seed"
            dbg.chosen_candidate = best
            dbg.profile = best.profile
        else:
            self._init_summary = {
                "reason": "seed_only",
                "candidate_count": 0,
                "best_score": 0.0,
                "intensity_score": 0.0,
                "contrast_score": 0.0,
                "autocorr_score": 0.0,
                "continuity": 0.0,
                "endpoint_evidence": 0.0,
                "endpoint_distance": float(np.linalg.norm(center_xyz[:2] - p1[:2])),
                "endpoint_loyalty": 1.0,
            }
            dbg.source = "seed_only"
            dbg.chosen_candidate = None
            dbg.profile = None
            if not dbg.candidate_summaries:
                dbg.candidate_summaries = ["seed 기반으로 초기화됨"]
        self.debug_frames.append(dbg)

    def step(self) -> DebugFrame:
        if self.state is None:
            raise RuntimeError("Tracker is not initialized.")
        if self.stop_reason != StopReason.NONE:
            return self.debug_frames[-1]

        best, dbg = self._observe(self.state, seed_mode=False)
        accept = self._accept_candidate(self.state, best)
        if accept and best is not None:
            self._apply_observation(best)
            self._distance_travelled_m += float(self.cfg.forward_distance_m)
            dbg.source = "box_graph"
        else:
            gap_xy, gap_heading = self._predict_gap_pose(self.state)
            z_ref = self._estimate_z_ref(gap_xy, float(self.state.center_xyz[2]))
            self.state.center_xyz = np.array([gap_xy[0], gap_xy[1], z_ref], dtype=np.float64)
            self.state.tangent_xy = gap_heading
            self.state.mode = TrackMode.GAP_BRIDGING
            self.state.profile_quality = 0.0
            self.state.center_confidence = 0.0
            self.state.identity_confidence *= 0.90
            self.state.stripe_strength = 0.0
            self.state.dashed_score *= 0.95
            self.state.solid_score *= 0.95
            self.state.crosswalk_score = 0.0
            self.state.gap_distance_m += float(self.cfg.forward_distance_m)
            self._append_path(self.state.center_xyz, self.state.tangent_xy)
            self._distance_travelled_m += float(self.cfg.forward_distance_m)
            dbg.source = "gap_reject" if best is not None else "gap"
            dbg.chosen_candidate = best
            dbg.profile = best.profile if best is not None else None

            if self.state.gap_distance_m > float(self.cfg.max_gap_distance_m):
                self.stop_reason = StopReason.GAP_TOO_LONG
                self.state.mode = TrackMode.STOPPED

        self._step_index += 1
        if self.stop_reason == StopReason.NONE and self._distance_travelled_m >= float(self.cfg.max_track_length_m):
            self.stop_reason = StopReason.MAX_DISTANCE
            self.state.mode = TrackMode.STOPPED

        dbg.step_index = self._step_index
        dbg.stop_reason = self.stop_reason.value if self.stop_reason != StopReason.NONE else None
        dbg.gap_distance_m = float(self.state.gap_distance_m)
        self.debug_frames.append(dbg)
        return dbg

    def run_full(self) -> TrackerResult:
        while self.stop_reason == StopReason.NONE:
            self.step()
        dense = np.asarray(self.state.path_centers, dtype=np.float64) if self.state is not None else np.empty((0, 3), dtype=np.float64)
        return TrackerResult(
            dense_points=dense,
            output_points=dense,
            stop_reason=self.stop_reason.value,
            debug_frames=list(self.debug_frames),
        )

    def _observe(self, state: TrackerState, seed_mode: bool) -> tuple[ObservationCandidate | None, DebugFrame]:
        pred_xy, pred_heading, _ = self._predict_pose(state)
        cell_view = self._build_active_cells(state, pred_heading)
        search_box = self._build_roi_box(state.center_xyz[:2], pred_heading)

        if cell_view is None:
            dbg = DebugFrame(
                step_index=self._step_index,
                source="box_graph",
                candidate_points=None,
                active_cell_box_groups=None,
                segment_groups=None,
                chosen_candidate=None,
                candidate_count=0,
                stop_reason=None,
                profile=None,
                trajectory_line_points=self._build_trajectory_preview(state, pred_heading, None),
                search_box_points=search_box,
                candidate_summaries=["No lane cells inside ROI"],
                gap_distance_m=float(state.gap_distance_m),
            )
            return None, dbg

        best_record = self._build_best_stripe_record(state, pred_heading, cell_view)
        best = self._record_to_observation(state, pred_heading, best_record) if best_record is not None else None

        candidate_points = self._component_to_world_polyline(state, pred_heading, best_record, float(state.center_xyz[2])) if best_record is not None else None
        segment_groups = self._component_to_world_edge_polylines(state, pred_heading, best_record, float(state.center_xyz[2])) if best_record is not None else None
        active_boxes = self._make_active_cell_boxes(state, pred_heading, cell_view)
        candidate_count = 1 if best is not None else 0
        if best is not None:
            candidate_summaries = [
                f"stripe | score={best.total_score:.2f} | cells={best.path_node_count} | span={best.path_length_m:.2f} | "
                f"width={best.width_m:.3f} | int={best.intensity_score:.3f} | contrast={best.contrast_score:.3f} | "
                f"start_lat={best.center_lateral_m:.2f} | end_lat={best.endpoint_lateral_m:.2f}"
            ]
        else:
            candidate_summaries = ["No valid stripe"]
        dbg = DebugFrame(
            step_index=self._step_index,
            source="box_graph",
            candidate_points=candidate_points,
            active_cell_box_groups=active_boxes,
            segment_groups=segment_groups,
            chosen_candidate=best,
            candidate_count=candidate_count,
            stop_reason=None,
            profile=best.profile if best is not None else None,
            trajectory_line_points=self._build_trajectory_preview(state, pred_heading, best_record),
            search_box_points=search_box,
            candidate_summaries=candidate_summaries or ["No valid spline component"],
            gap_distance_m=float(state.gap_distance_m),
        )
        return best, dbg

    def _accept_candidate(self, state: TrackerState, obs: ObservationCandidate | None) -> bool:
        if obs is None:
            return False
        min_accept_span = max(
            float(self.cfg.component_min_span_m),
            min(float(self.cfg.roi_forward_m) * 0.40, float(self.cfg.forward_distance_m) * 3.5),
        )
        if obs.path_length_m < min_accept_span:
            return False
        score_gate = float(self.cfg.candidate_min_score)
        if state.gap_distance_m > 0.0:
            score_gate += 0.04
        if obs.total_score < score_gate:
            return False
        if abs(obs.endpoint_lateral_m) > max(float(self.cfg.corridor_half_width_m) * 1.6, 0.10):
            return False
        if state.gap_distance_m > 0.0 and obs.future_identity_score < 0.45:
            return False
        return True

    def _predict_pose(self, state: TrackerState) -> tuple[np.ndarray, np.ndarray, float]:
        history_xy = np.asarray([p[:2] for p in state.history_centers[-6:]], dtype=np.float64)
        if len(history_xy) >= 3:
            pred_xy, pred_heading, curvature = _polyfit_predict(history_xy, float(self.cfg.forward_distance_m))
        else:
            pred_heading = _normalize(state.tangent_xy)
            pred_xy = state.center_xyz[:2] + pred_heading * float(self.cfg.forward_distance_m)
            curvature = float(state.curvature)
        pred_heading = self._limit_heading_change(state.tangent_xy, pred_heading)
        return pred_xy, pred_heading, curvature

    def _predict_gap_pose(self, state: TrackerState) -> tuple[np.ndarray, np.ndarray]:
        step_distance = float(self.cfg.forward_distance_m)
        current_heading = _normalize(state.tangent_xy)
        curvature = float(state.curvature)
        if abs(curvature) <= 1e-6:
            next_heading = current_heading
            delta_xy = current_heading * step_distance
        else:
            delta_angle = curvature * step_distance
            half_angle = 0.5 * delta_angle
            c_half = math.cos(half_angle)
            s_half = math.sin(half_angle)
            mid_heading = _normalize(
                np.array(
                    [
                        current_heading[0] * c_half - current_heading[1] * s_half,
                        current_heading[0] * s_half + current_heading[1] * c_half,
                    ],
                    dtype=np.float64,
                )
            )
            c_full = math.cos(delta_angle)
            s_full = math.sin(delta_angle)
            next_heading = _normalize(
                np.array(
                    [
                        current_heading[0] * c_full - current_heading[1] * s_full,
                        current_heading[0] * s_full + current_heading[1] * c_full,
                    ],
                    dtype=np.float64,
                )
            )
            delta_xy = mid_heading * step_distance
        next_xy = state.center_xyz[:2] + delta_xy
        return next_xy, next_heading

    def _build_active_cells(self, state: TrackerState, heading_xy: np.ndarray) -> dict[str, np.ndarray] | None:
        center_xy = state.center_xyz[:2]
        roi_back = float(self.cfg.roi_backward_m)
        roi_forward = float(self.cfg.roi_forward_m)
        roi_lateral = float(self.cfg.roi_lateral_half_m)
        cell_size = float(self.cfg.grid_cell_size_m)

        strip_center = center_xy + heading_xy * ((roi_forward - roi_back) * 0.5)
        idx = self.grid.query_oriented_strip_xy(
            strip_center,
            heading_xy,
            along_half=0.5 * (roi_forward + roi_back),
            lateral_half=roi_lateral,
        )
        if idx.size == 0:
            return None

        pts_xy = self.xy[idx]
        along, lateral = project_points_xy(pts_xy, center_xy, heading_xy)
        mask = (along >= -roi_back) & (along <= roi_forward) & (np.abs(lateral) <= roi_lateral)
        if self.cfg.use_z_clip:
            z_ref = float(state.center_xyz[2])
            lo = z_ref - float(self.cfg.z_clip_half_range_m)
            hi = z_ref + float(self.cfg.z_clip_half_range_m)
            mask &= (self.z[idx] >= lo) & (self.z[idx] <= hi)
        if not np.any(mask):
            return None

        idx = idx[mask]
        along = along[mask]
        lateral = lateral[mask]
        intensity_score = self._normalize_intensity(self.intensity[idx])

        n_along = max(1, int(math.ceil((roi_forward + roi_back) / cell_size)))
        n_lat = max(1, int(math.ceil((2.0 * roi_lateral) / cell_size)))
        count_grid = np.zeros((n_along, n_lat), dtype=np.int32)
        score_grid = np.zeros((n_along, n_lat), dtype=np.float64)

        along_idx = np.floor((along + roi_back) / cell_size).astype(np.int32)
        lat_idx = np.floor((lateral + roi_lateral) / cell_size).astype(np.int32)
        valid = (
            (along_idx >= 0)
            & (along_idx < n_along)
            & (lat_idx >= 0)
            & (lat_idx < n_lat)
        )
        along_idx = along_idx[valid]
        lat_idx = lat_idx[valid]
        intensity_score = intensity_score[valid]
        if along_idx.size == 0:
            return None

        for a_idx, l_idx, score in zip(along_idx, lat_idx, intensity_score):
            count_grid[a_idx, l_idx] += 1
            score_grid[a_idx, l_idx] += float(score)

        mean_grid = np.divide(
            score_grid,
            np.maximum(count_grid, 1),
            out=np.zeros_like(score_grid),
            where=count_grid > 0,
        )
        min_points = max(int(self.cfg.min_points_per_cell), 1)
        valid_scores = mean_grid[count_grid >= min_points]
        adaptive_threshold = float(self.cfg.active_intensity_min)
        if valid_scores.size:
            adaptive_threshold = max(adaptive_threshold, float(np.percentile(valid_scores, 70.0)))

        active_mask = (count_grid >= min_points) & (mean_grid >= adaptive_threshold)
        if not np.any(active_mask):
            return None

        active_mask = self._prune_active_mask(active_mask, count_grid, mean_grid, cell_size)
        if not np.any(active_mask):
            return None

        along_centers = -roi_back + (np.arange(n_along, dtype=np.float64) + 0.5) * cell_size
        lateral_centers = -roi_lateral + (np.arange(n_lat, dtype=np.float64) + 0.5) * cell_size
        return {
            "active_mask": active_mask,
            "count_grid": count_grid,
            "mean_grid": mean_grid,
            "cell_size": np.array([cell_size], dtype=np.float64),
            "along_centers": along_centers,
            "lateral_centers": lateral_centers,
        }

    def _prune_active_mask(
        self,
        active_mask: np.ndarray,
        count_grid: np.ndarray,
        mean_grid: np.ndarray,
        cell_size: float,
    ) -> np.ndarray:
        keep_mask = np.zeros_like(active_mask, dtype=bool)
        peak_radius = max(1, int(round(float(self.cfg.stripe_width_m) / max(cell_size, 1e-6))))
        max_peaks_per_row = 4

        for row_idx in range(active_mask.shape[0]):
            row_active = active_mask[row_idx]
            if not np.any(row_active):
                continue

            row_scores = mean_grid[row_idx]
            row_counts = count_grid[row_idx]
            row_nonzero = row_scores[row_counts > 0]
            if row_nonzero.size == 0:
                continue

            row_baseline = float(np.percentile(row_nonzero, 65.0))
            row_peak = float(np.max(row_scores[row_active]))
            peak_floor = max(float(self.cfg.active_intensity_min), row_baseline + 0.05, row_peak * 0.72)

            peak_candidates: list[tuple[float, int]] = []
            for col_idx in np.where(row_active)[0]:
                left = float(row_scores[col_idx - 1]) if col_idx > 0 else float(row_scores[col_idx])
                right = float(row_scores[col_idx + 1]) if col_idx + 1 < row_scores.size else float(row_scores[col_idx])
                if row_scores[col_idx] + 1e-6 < left or row_scores[col_idx] + 1e-6 < right:
                    continue
                if float(row_scores[col_idx]) < peak_floor:
                    continue
                prominence = float(row_scores[col_idx] - 0.5 * (left + right))
                peak_candidates.append((float(row_scores[col_idx]) + 0.25 * prominence, int(col_idx)))

            peak_candidates.sort(key=lambda item: item[0], reverse=True)
            peak_indices: list[int] = []
            for _, col_idx in peak_candidates:
                if any(abs(col_idx - prev_idx) <= peak_radius for prev_idx in peak_indices):
                    continue
                peak_indices.append(col_idx)
                if len(peak_indices) >= max_peaks_per_row:
                    break

            if not peak_indices:
                continue

            for peak_idx in peak_indices:
                lo = max(0, peak_idx - peak_radius)
                hi = min(row_active.size, peak_idx + peak_radius + 1)
                keep_mask[row_idx, lo:hi] |= row_active[lo:hi]

        return keep_mask

    def _extract_stripe_pairs_1d(
        self,
        signal: np.ndarray,
        valid_mask: np.ndarray,
        lateral_centers: np.ndarray,
        cell_size: float,
    ) -> list[_StripePair1D]:
        smooth = np.convolve(
            np.asarray(signal, dtype=np.float64),
            np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64) / 9.0,
            mode="same",
        )
        valid_values = smooth[np.asarray(valid_mask, dtype=bool)]
        if valid_values.size == 0:
            return []

        baseline = float(np.percentile(valid_values, 40.0))
        peak = float(np.max(valid_values))
        dynamic = max(peak - baseline, 0.0)
        edge_floor = max(0.015, dynamic * 0.18)
        inside_floor = baseline + max(0.02, dynamic * 0.22)
        target_width = max(float(self.cfg.stripe_width_m), cell_size * 1.5)
        width_tol = max(cell_size * 1.0, target_width * 0.35)
        min_width = max(cell_size * 0.8, target_width - width_tol)
        max_width = max(min_width + cell_size * 0.5, target_width + width_tol)

        deriv = np.gradient(smooth)
        rise_indices: list[int] = []
        fall_indices: list[int] = []
        for idx in range(1, len(smooth) - 1):
            if not (valid_mask[idx] or valid_mask[idx - 1] or valid_mask[idx + 1]):
                continue
            if deriv[idx] >= deriv[idx - 1] and deriv[idx] >= deriv[idx + 1] and deriv[idx] >= edge_floor:
                rise_indices.append(idx)
            if deriv[idx] <= deriv[idx - 1] and deriv[idx] <= deriv[idx + 1] and -deriv[idx] >= edge_floor:
                fall_indices.append(idx)

        pairs: list[_StripePair1D] = []
        for rise_idx in rise_indices:
            for fall_idx in fall_indices:
                if fall_idx <= rise_idx:
                    continue
                width_m = float(lateral_centers[fall_idx] - lateral_centers[rise_idx])
                if width_m < min_width:
                    continue
                if width_m > max_width:
                    break
                inside_slice = smooth[rise_idx : fall_idx + 1]
                if inside_slice.size == 0:
                    continue
                inside_peak = float(np.max(inside_slice))
                inside_mean = float(np.mean(inside_slice))
                if inside_peak < inside_floor:
                    continue
                left_outer = baseline if rise_idx < 2 else float(np.mean(smooth[max(0, rise_idx - 2) : rise_idx]))
                right_outer = baseline if fall_idx + 2 >= len(smooth) else float(np.mean(smooth[fall_idx + 1 : min(len(smooth), fall_idx + 3)]))
                contrast = inside_mean - 0.5 * (left_outer + right_outer)
                edge_strength = max(float(deriv[rise_idx]), 0.0) + max(float(-deriv[fall_idx]), 0.0)
                width_score = math.exp(-abs(width_m - target_width) / max(target_width * 0.35, cell_size))
                score = 0.45 * edge_strength + 0.30 * max(contrast, 0.0) + 0.15 * inside_mean + 0.10 * width_score
                if score < 0.03:
                    continue
                pairs.append(
                    _StripePair1D(
                        left_idx=rise_idx,
                        right_idx=fall_idx,
                        left_m=float(lateral_centers[rise_idx]),
                        right_m=float(lateral_centers[fall_idx]),
                        center_m=float(0.5 * (lateral_centers[rise_idx] + lateral_centers[fall_idx])),
                        width_m=width_m,
                        edge_strength=edge_strength,
                        inside_mean=inside_mean,
                        contrast_score=max(contrast, 0.0),
                        score=score,
                    )
                )

        pairs.sort(key=lambda pair: pair.score, reverse=True)
        selected: list[_StripePair1D] = []
        sep_floor = max(target_width * 0.60, cell_size * 2.0)
        for pair in pairs:
            if any(abs(pair.center_m - prev.center_m) < sep_floor for prev in selected):
                continue
            selected.append(pair)
            if len(selected) >= 4:
                break
        return selected

    def _extract_row_stripes(self, cell_view: dict[str, np.ndarray]) -> list[list[_RowStripe]]:
        count_grid = cell_view["count_grid"]
        mean_grid = cell_view["mean_grid"]
        along_centers = cell_view["along_centers"]
        lateral_centers = cell_view["lateral_centers"]
        cell_size = float(cell_view["cell_size"][0])

        row_stripes: list[list[_RowStripe]] = []
        for a_idx in range(count_grid.shape[0]):
            row_counts = np.asarray(count_grid[a_idx], dtype=np.int32)
            row_signal = np.where(row_counts > 0, mean_grid[a_idx], 0.0)
            row_valid = row_counts > 0
            pairs = self._extract_stripe_pairs_1d(row_signal, row_valid, lateral_centers, cell_size)
            stripes: list[_RowStripe] = []
            for pair in pairs:
                cell_cols = [col_idx for col_idx in range(pair.left_idx, pair.right_idx + 1) if row_counts[col_idx] > 0]
                if not cell_cols:
                    continue
                intensity_values = np.asarray([float(mean_grid[a_idx, col_idx]) for col_idx in cell_cols], dtype=np.float64)
                cell_weights = np.asarray(
                    [max(float(mean_grid[a_idx, col_idx]), 0.05) * max(int(row_counts[col_idx]), 1) for col_idx in cell_cols],
                    dtype=np.float64,
                )
                weight_sum = float(np.sum(cell_weights))
                if weight_sum <= 1e-9:
                    continue
                intensity_mean = float(np.sum(intensity_values * cell_weights) / weight_sum)
                stripes.append(
                    _RowStripe(
                        along_idx=a_idx,
                        along_m=float(along_centers[a_idx]),
                        left_edge_m=pair.left_m,
                        right_edge_m=pair.right_m,
                        lateral_center_m=pair.center_m,
                        width_m=pair.width_m,
                        weight=float(weight_sum + 2.0 * pair.edge_strength + pair.contrast_score),
                        intensity_mean=intensity_mean,
                        edge_strength=pair.edge_strength,
                        contrast_score=pair.contrast_score,
                        cells=[(a_idx, col_idx) for col_idx in cell_cols],
                    )
                )
            row_stripes.append(stripes)
        return row_stripes

    def _build_best_stripe_record(
        self,
        state: TrackerState,
        pred_heading: np.ndarray,
        cell_view: dict[str, np.ndarray],
    ) -> _ComponentCandidate | None:
        row_stripes = self._extract_row_stripes(cell_view)
        if not any(row_stripes):
            return None

        cell_size = float(cell_view["cell_size"][0])
        target_width = max(float(self.cfg.stripe_width_m), cell_size * 1.5)
        corridor_half = max(float(self.cfg.corridor_half_width_m), target_width)
        min_span = max(float(self.cfg.component_min_span_m), float(self.cfg.forward_distance_m))
        max_row_gap = 2

        dp_scores: dict[tuple[int, int], float] = {}
        back_ptr: dict[tuple[int, int], tuple[int, int] | None] = {}

        for row_idx, stripes in enumerate(row_stripes):
            for stripe_idx, stripe in enumerate(stripes):
                width_penalty = abs(stripe.width_m - target_width) / max(target_width, 1e-6)
                node_score = (
                    0.45 * stripe.edge_strength
                    + 0.30 * stripe.contrast_score
                    + 0.15 * stripe.intensity_mean
                    + 0.10 * min(stripe.weight / 8.0, 1.0)
                    - 0.25 * width_penalty
                    - 0.35 * (abs(stripe.lateral_center_m) / corridor_half)
                )
                best_total = node_score
                best_prev: tuple[int, int] | None = None

                for prev_gap in range(1, max_row_gap + 2):
                    prev_row = row_idx - prev_gap
                    if prev_row < 0:
                        break
                    for prev_idx, prev_stripe in enumerate(row_stripes[prev_row]):
                        prev_key = (prev_row, prev_idx)
                        if prev_key not in dp_scores:
                            continue
                        lat_jump = abs(stripe.lateral_center_m - prev_stripe.lateral_center_m)
                        width_jump = abs(stripe.width_m - prev_stripe.width_m)
                        trans_penalty = (
                            0.80 * (lat_jump / corridor_half)
                            + 0.35 * (width_jump / max(target_width, 1e-6))
                            + 0.18 * float(prev_gap - 1)
                        )
                        total = dp_scores[prev_key] + node_score - trans_penalty
                        if total > best_total:
                            best_total = total
                            best_prev = prev_key

                dp_scores[(row_idx, stripe_idx)] = best_total
                back_ptr[(row_idx, stripe_idx)] = best_prev

        if not dp_scores:
            return None

        best_key = max(dp_scores.keys(), key=lambda key: dp_scores[key])
        path: list[_RowStripe] = []
        cursor: tuple[int, int] | None = best_key
        while cursor is not None:
            row_idx, stripe_idx = cursor
            path.append(row_stripes[row_idx][stripe_idx])
            cursor = back_ptr.get(cursor)
        path.reverse()

        if not path:
            return None

        along_values = np.asarray([stripe.along_m for stripe in path], dtype=np.float64)
        span_m = float(along_values.max() - along_values.min() + cell_size)
        if span_m < min_span:
            return None

        return self._fit_stripe_record(state, pred_heading, cell_view, path)

    def _fit_stripe_record(
        self,
        state: TrackerState,
        pred_heading: np.ndarray,
        cell_view: dict[str, np.ndarray],
        path: list[_RowStripe],
    ) -> _ComponentCandidate | None:
        if not path:
            return None

        cell_size = float(cell_view["cell_size"][0])
        along_values = np.asarray([stripe.along_m for stripe in path], dtype=np.float64)
        left_values = np.asarray([stripe.left_edge_m for stripe in path], dtype=np.float64)
        right_values = np.asarray([stripe.right_edge_m for stripe in path], dtype=np.float64)
        center_values = 0.5 * (left_values + right_values)
        width_values = right_values - left_values
        weights = np.asarray([stripe.weight for stripe in path], dtype=np.float64)
        support_rows = len(path)
        span_m = float(along_values.max() - along_values.min() + cell_size)
        if support_rows < 2 or span_m < max(float(self.cfg.component_min_span_m), float(self.cfg.forward_distance_m)):
            return None

        degree = min(2, support_rows - 1)
        left_coeff = np.polyfit(along_values, left_values, degree, w=np.maximum(weights, 1e-3))
        right_coeff = np.polyfit(along_values, right_values, degree, w=np.maximum(weights, 1e-3))
        sample_end = max(float(self.cfg.forward_distance_m), min(float(self.cfg.roi_forward_m), along_values.max() + cell_size * 0.5))
        sample_count = max(10, int(math.ceil(sample_end / max(cell_size, 0.05))) * 2)
        sample_along = np.linspace(0.0, sample_end, sample_count, dtype=np.float64)
        sample_left = np.polyval(left_coeff, sample_along)
        sample_right = np.polyval(right_coeff, sample_along)
        sample_center = 0.5 * (sample_left + sample_right)
        sample_width = sample_right - sample_left

        if np.any(sample_width < cell_size * 0.5):
            return None

        start_lateral = float(np.interp(0.0, sample_along, sample_center))
        step_lateral = float(np.interp(float(self.cfg.forward_distance_m), sample_along, sample_center))
        end_along = float(min(sample_end, along_values.max() + cell_size * 0.5))
        end_lateral = float(np.interp(end_along, sample_along, sample_center))
        max_abs_lateral = float(np.max(np.abs(sample_center)))
        width_m = float(np.median(width_values))

        center_coeff = np.polyfit(along_values, center_values, degree, w=np.maximum(weights, 1e-3))
        if center_coeff.size <= 1:
            step_slope = 0.0
        else:
            deriv_coeff = np.polyder(center_coeff)
            step_slope = float(np.polyval(deriv_coeff, min(float(self.cfg.forward_distance_m), sample_end)))
        tangent, normal = make_frame(pred_heading)
        heading_xy = _normalize(tangent + normal * step_slope)
        heading_error = _angle_between(pred_heading, heading_xy)

        lateral_sigma = max(float(self.cfg.candidate_lateral_sigma_m), 1e-3)
        heading_sigma = max(math.radians(float(self.cfg.candidate_heading_sigma_deg)), 1e-3)
        corridor_width = max(float(self.cfg.corridor_half_width_m), 0.01)

        start_score = math.exp(-abs(start_lateral) / lateral_sigma)
        future_score = math.exp(-abs(step_lateral) / lateral_sigma)
        heading_score = math.exp(-heading_error / heading_sigma)

        coverage_ratio = float(np.clip(support_rows / max(int(round(span_m / max(cell_size, 1e-6))), 1), 0.0, 1.0))
        support_score = float(np.clip(span_m / max(float(self.cfg.component_min_span_m), float(self.cfg.forward_distance_m)), 0.0, 1.0))
        edge_score = float(np.clip(np.mean([stripe.edge_strength for stripe in path]) / 0.25, 0.0, 1.0))
        contrast_score = float(np.clip(np.mean([stripe.contrast_score for stripe in path]) / 0.18, 0.0, 1.0))
        support_combo = 0.40 * coverage_ratio + 0.30 * support_score + 0.15 * edge_score + 0.15 * contrast_score
        corridor_penalty = math.exp(-max(0.0, max_abs_lateral - corridor_width) / max(lateral_sigma, 1e-6))
        base_score = 0.42 * start_score + 0.28 * future_score + 0.15 * heading_score + 0.15 * support_combo
        total_score = corridor_penalty * base_score
        if total_score < 0.05:
            return None

        cells = [cell for stripe in path for cell in stripe.cells]
        cell_centers = np.asarray([[stripe.along_m, stripe.lateral_center_m] for stripe in path], dtype=np.float64)
        mean_intensity = float(np.mean([stripe.intensity_mean for stripe in path])) if path else 0.0
        mean_contrast = float(np.mean([stripe.contrast_score for stripe in path])) if path else 0.0
        return _ComponentCandidate(
            cells=cells,
            cell_centers_local=cell_centers,
            cell_weights=weights,
            poly_coeff=np.asarray(center_coeff, dtype=np.float64),
            sample_along=sample_along,
            sample_lateral=sample_center,
            unique_along_count=int(support_rows),
            span_m=span_m,
            coverage_ratio=coverage_ratio,
            score=float(np.clip(total_score, 0.0, 1.0)),
            start_lateral_m=start_lateral,
            step_lateral_m=step_lateral,
            end_lateral_m=end_lateral,
            max_abs_lateral_m=max_abs_lateral,
            heading_xy=heading_xy,
            heading_error_rad=heading_error,
            sample_left_edge=sample_left,
            sample_right_edge=sample_right,
            width_m=width_m,
            intensity_score=mean_intensity,
            contrast_score=mean_contrast,
        )

    def _extract_row_clusters(self, cell_view: dict[str, np.ndarray]) -> list[list[_RowCluster]]:
        active_mask = cell_view["active_mask"]
        count_grid = cell_view["count_grid"]
        mean_grid = cell_view["mean_grid"]
        along_centers = cell_view["along_centers"]
        lateral_centers = cell_view["lateral_centers"]

        row_clusters: list[list[_RowCluster]] = []
        for a_idx in range(active_mask.shape[0]):
            row_mask = active_mask[a_idx]
            clusters: list[_RowCluster] = []
            l_idx = 0
            while l_idx < row_mask.size:
                if not row_mask[l_idx]:
                    l_idx += 1
                    continue
                cells: list[tuple[int, int]] = []
                weighted_lat = 0.0
                weight_sum = 0.0
                while l_idx < row_mask.size and row_mask[l_idx]:
                    cell_weight = float(max(mean_grid[a_idx, l_idx], 0.05) * max(count_grid[a_idx, l_idx], 1))
                    weighted_lat += float(lateral_centers[l_idx]) * cell_weight
                    weight_sum += cell_weight
                    cells.append((a_idx, l_idx))
                    l_idx += 1
                if weight_sum <= 1e-9:
                    continue
                clusters.append(
                    _RowCluster(
                        along_idx=a_idx,
                        along_m=float(along_centers[a_idx]),
                        lateral_center_m=float(weighted_lat / weight_sum),
                        weight=weight_sum,
                        cells=cells,
                    )
                )
            row_clusters.append(clusters)
        return row_clusters

    def _build_ridge_candidates(
        self,
        state: TrackerState,
        pred_heading: np.ndarray,
        cell_view: dict[str, np.ndarray],
    ) -> list[_ComponentCandidate]:
        row_clusters = self._extract_row_clusters(cell_view)
        seed_refs = self._select_ridge_seeds(row_clusters)
        records: list[_ComponentCandidate] = []
        seen_keys: set[tuple[int, int, int]] = set()
        for row_idx, cluster_idx in seed_refs:
            ridge = self._trace_ridge(row_clusters, row_idx, cluster_idx)
            record = self._fit_ridge_candidate(state, pred_heading, cell_view, ridge)
            if record is None:
                continue
            key = (
                int(round(record.start_lateral_m / 0.03)),
                int(round(record.step_lateral_m / 0.03)),
                int(round(record.end_lateral_m / 0.05)),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            records.append(record)
        records.sort(key=lambda rec: rec.score, reverse=True)
        return records[:5]

    def _select_ridge_seeds(self, row_clusters: list[list[_RowCluster]]) -> list[tuple[int, int]]:
        seed_items: list[tuple[float, int, int]] = []
        forward_band = max(float(self.cfg.forward_distance_m) * 0.8, float(self.cfg.grid_cell_size_m) * 2.0)
        for row_idx, clusters in enumerate(row_clusters):
            for cluster_idx, cluster in enumerate(clusters):
                if cluster.along_m < -0.15 or cluster.along_m > forward_band:
                    continue
                priority = abs(cluster.lateral_center_m) - 0.01 * min(cluster.weight, 20.0) + 0.05 * abs(cluster.along_m)
                seed_items.append((priority, row_idx, cluster_idx))
        if not seed_items:
            for row_idx, clusters in enumerate(row_clusters):
                for cluster_idx, cluster in enumerate(clusters):
                    priority = abs(cluster.lateral_center_m) - 0.01 * min(cluster.weight, 20.0) + 0.02 * abs(cluster.along_m)
                    seed_items.append((priority, row_idx, cluster_idx))
        seed_items.sort(key=lambda item: item[0])
        return [(row_idx, cluster_idx) for _, row_idx, cluster_idx in seed_items[:6]]

    def _trace_ridge(
        self,
        row_clusters: list[list[_RowCluster]],
        seed_row_idx: int,
        seed_cluster_idx: int,
    ) -> list[_RowCluster]:
        seed = row_clusters[seed_row_idx][seed_cluster_idx]
        chosen: dict[int, _RowCluster] = {seed_row_idx: seed}
        max_row_gap = 2
        max_jump = max(0.12, float(self.cfg.corridor_half_width_m) * 0.9)

        for direction in (-1, 1):
            prev2: _RowCluster | None = None
            prev1 = seed
            gap_rows = 0
            row_idx = seed_row_idx + direction
            while 0 <= row_idx < len(row_clusters):
                clusters = row_clusters[row_idx]
                if not clusters:
                    gap_rows += 1
                    if gap_rows > max_row_gap:
                        break
                    row_idx += direction
                    continue

                target_along = clusters[0].along_m
                slope = 0.0
                if prev2 is not None:
                    ds = prev1.along_m - prev2.along_m
                    if abs(ds) > 1e-6:
                        slope = (prev1.lateral_center_m - prev2.lateral_center_m) / ds
                pred_lateral = prev1.lateral_center_m + slope * (target_along - prev1.along_m)
                allowed_jump = max_jump * (1.0 + 0.35 * gap_rows)

                best_cluster: _RowCluster | None = None
                best_cost = float("inf")
                for cluster in clusters:
                    lat_err = abs(cluster.lateral_center_m - pred_lateral)
                    delta_lat = abs(cluster.lateral_center_m - prev1.lateral_center_m)
                    corridor_err = abs(cluster.lateral_center_m)
                    if lat_err > allowed_jump and delta_lat > allowed_jump * 1.2:
                        continue
                    cost = lat_err + 0.35 * corridor_err + 0.15 * delta_lat - 0.01 * min(cluster.weight, 20.0)
                    if cost < best_cost:
                        best_cost = cost
                        best_cluster = cluster

                if best_cluster is None:
                    gap_rows += 1
                    if gap_rows > max_row_gap:
                        break
                    row_idx += direction
                    continue

                chosen[row_idx] = best_cluster
                prev2 = prev1
                prev1 = best_cluster
                gap_rows = 0
                row_idx += direction

        return [chosen[idx] for idx in sorted(chosen.keys())]

    def _fit_ridge_candidate(
        self,
        state: TrackerState,
        pred_heading: np.ndarray,
        cell_view: dict[str, np.ndarray],
        ridge: list[_RowCluster],
    ) -> _ComponentCandidate | None:
        if not ridge:
            return None

        cell_size = float(cell_view["cell_size"][0])
        along_values = np.asarray([cluster.along_m for cluster in ridge], dtype=np.float64)
        lateral_values = np.asarray([cluster.lateral_center_m for cluster in ridge], dtype=np.float64)
        weights = np.asarray([cluster.weight for cluster in ridge], dtype=np.float64)
        span_m = float(along_values.max() - along_values.min() + cell_size)
        support_rows = len(ridge)

        if support_rows == 1:
            poly_coeff = np.asarray([lateral_values[0]], dtype=np.float64)
        else:
            degree = min(2, support_rows - 1)
            poly_coeff = np.polyfit(along_values, lateral_values, degree, w=np.maximum(weights, 1e-3))

        sample_end = max(float(self.cfg.forward_distance_m), min(float(self.cfg.roi_forward_m), along_values.max() + cell_size * 0.5))
        sample_count = max(10, int(math.ceil(sample_end / max(cell_size, 0.05))) * 2)
        sample_along = np.linspace(0.0, sample_end, sample_count, dtype=np.float64)
        sample_lateral = np.polyval(poly_coeff, sample_along)
        start_lateral = float(np.polyval(poly_coeff, 0.0))
        step_lateral = float(np.polyval(poly_coeff, float(self.cfg.forward_distance_m)))
        end_along = float(min(sample_end, along_values.max() + cell_size * 0.5))
        end_lateral = float(np.polyval(poly_coeff, end_along))
        max_abs_lateral = float(np.max(np.abs(sample_lateral)))

        if poly_coeff.size <= 1:
            step_slope = 0.0
        else:
            deriv_coeff = np.polyder(poly_coeff)
            step_slope = float(np.polyval(deriv_coeff, min(float(self.cfg.forward_distance_m), sample_end)))
        tangent, normal = make_frame(pred_heading)
        heading_xy = _normalize(tangent + normal * step_slope)
        heading_error = _angle_between(pred_heading, heading_xy)

        lateral_sigma = max(float(self.cfg.candidate_lateral_sigma_m), 1e-3)
        heading_sigma = max(math.radians(float(self.cfg.candidate_heading_sigma_deg)), 1e-3)
        corridor_width = max(float(self.cfg.corridor_half_width_m), 0.01)

        start_score = math.exp(-abs(start_lateral) / lateral_sigma)
        future_score = math.exp(-abs(step_lateral) / lateral_sigma)
        heading_score = math.exp(-heading_error / heading_sigma)

        if support_rows >= 2:
            delta_along = np.diff(along_values)
            delta_along = np.where(np.abs(delta_along) < 1e-6, cell_size, delta_along)
            jumps = np.abs(np.diff(lateral_values) / delta_along)
            smooth_score = math.exp(-float(np.mean(jumps)) * max(cell_size * 6.0, 0.18))
        else:
            smooth_score = 0.45

        expected_bins = max(int(round(span_m / max(cell_size, 1e-6))), 1)
        coverage_ratio = float(np.clip(support_rows / expected_bins, 0.0, 1.0))
        support_span = max(float(self.cfg.component_min_span_m), float(self.cfg.forward_distance_m))
        support_score = float(np.clip(span_m / support_span, 0.0, 1.0))
        support_combo = 0.5 * coverage_ratio + 0.5 * support_score
        corridor_penalty = math.exp(-max(0.0, max_abs_lateral - corridor_width) / max(lateral_sigma, 1e-6))
        base_score = 0.34 * start_score + 0.26 * future_score + 0.20 * heading_score + 0.20 * smooth_score
        total_score = corridor_penalty * base_score * (0.35 + 0.65 * support_combo)
        if total_score < 0.05:
            return None

        cells = [cell for cluster in ridge for cell in cluster.cells]
        cell_centers = np.asarray([[cluster.along_m, cluster.lateral_center_m] for cluster in ridge], dtype=np.float64)
        return _ComponentCandidate(
            cells=cells,
            cell_centers_local=cell_centers,
            cell_weights=weights,
            poly_coeff=np.asarray(poly_coeff, dtype=np.float64),
            sample_along=sample_along,
            sample_lateral=sample_lateral,
            unique_along_count=int(support_rows),
            span_m=span_m,
            coverage_ratio=coverage_ratio,
            score=float(np.clip(total_score, 0.0, 1.0)),
            start_lateral_m=start_lateral,
            step_lateral_m=step_lateral,
            end_lateral_m=end_lateral,
            max_abs_lateral_m=max_abs_lateral,
            heading_xy=heading_xy,
            heading_error_rad=heading_error,
        )

    def _record_to_observation(
        self,
        state: TrackerState,
        pred_heading: np.ndarray,
        record: _ComponentCandidate,
    ) -> ObservationCandidate:
        tangent, normal = make_frame(pred_heading)
        center_xy = state.center_xyz[:2] + tangent * float(self.cfg.forward_distance_m) + normal * record.step_lateral_m
        z_ref = self._estimate_z_ref(center_xy, float(state.center_xyz[2]))
        step_along = float(self.cfg.forward_distance_m)
        step_left_abs = float(np.interp(step_along, record.sample_along, record.sample_left_edge))
        step_right_abs = float(np.interp(step_along, record.sample_along, record.sample_right_edge))
        step_center_abs = float(np.interp(step_along, record.sample_along, record.sample_lateral))
        left_edge_rel = step_left_abs - step_center_abs
        right_edge_rel = step_right_abs - step_center_abs
        profile = self._build_profile(center_xy, record.heading_xy, z_ref, 0.0)
        fill_ratio = float(np.clip(record.coverage_ratio, 0.0, 1.0))
        dashed_score = float(np.clip(1.0 - fill_ratio * 1.2, 0.0, 1.0))
        solid_score = float(np.clip(fill_ratio * 1.15, 0.0, 1.0))
        lateral_sigma = max(float(self.cfg.candidate_lateral_sigma_m), 1e-3)
        identity_score = math.exp(-abs(record.start_lateral_m) / lateral_sigma)
        future_identity = math.exp(-abs(record.step_lateral_m) / lateral_sigma)
        switch_penalty = float(np.clip(abs(record.start_lateral_m) / max(float(self.cfg.corridor_half_width_m), 1e-6), 0.0, 1.0))
        future_switch_penalty = float(np.clip(abs(record.step_lateral_m) / max(float(self.cfg.corridor_half_width_m), 1e-6), 0.0, 1.0))
        continuity = float(np.clip(0.55 * future_identity + 0.45 * solid_score, 0.0, 1.0))
        intensity_score = float(np.clip(record.intensity_score, 0.0, 1.0))
        contrast_score = float(np.clip(record.contrast_score, 0.0, 1.0))
        stripe_width = float(max(step_right_abs - step_left_abs, 0.05))
        return ObservationCandidate(
            center_xy=center_xy,
            center_xyz=np.array([center_xy[0], center_xy[1], z_ref], dtype=np.float64),
            heading_xy=record.heading_xy,
            angle_offset_deg=math.degrees(_signed_angle(pred_heading, record.heading_xy)),
            stripe_center_m=0.0,
            left_edge_m=float(left_edge_rel),
            right_edge_m=float(right_edge_rel),
            width_m=stripe_width,
            intensity_score=intensity_score,
            contrast_score=contrast_score,
            autocorr_score=0.0,
            continuity_score=continuity,
            total_score=float(record.score),
            path_score=float(np.clip(record.span_m / max(float(self.cfg.roi_forward_m), 1e-6), 0.0, 1.0)),
            dashed_score=dashed_score,
            solid_score=solid_score,
            crosswalk_score=0.0,
            support_count=int(len(record.cells)),
            z_ref=z_ref,
            dominant_period_m=0.0,
            path_length_m=float(record.span_m),
            path_node_count=int(len(record.cells)),
            fill_ratio=fill_ratio,
            mode_consistency_score=solid_score,
            identity_score=identity_score,
            switch_penalty=switch_penalty,
            center_lateral_m=record.start_lateral_m,
            future_identity_score=future_identity,
            future_switch_penalty=future_switch_penalty,
            endpoint_lateral_m=record.end_lateral_m,
            max_path_lateral_m=record.max_abs_lateral_m,
            profile=profile,
        )

    def _build_profile(
        self,
        center_xy: np.ndarray,
        heading_xy: np.ndarray,
        z_ref: float,
        stripe_center_m: float,
    ) -> ProfileData | None:
        lateral_half = max(float(self.cfg.roi_lateral_half_m), float(self.cfg.stripe_width_m) * 2.0)
        along_half = max(float(self.cfg.forward_distance_m), 0.12)
        idx = self.grid.query_oriented_strip_xy(center_xy, heading_xy, along_half=along_half, lateral_half=lateral_half)
        if idx.size == 0:
            return None
        if self.cfg.use_z_clip:
            lo = z_ref - float(self.cfg.z_clip_half_range_m)
            hi = z_ref + float(self.cfg.z_clip_half_range_m)
            z_mask = (self.z[idx] >= lo) & (self.z[idx] <= hi)
            idx = idx[z_mask]
            if idx.size == 0:
                return None

        along, lateral = project_points_xy(self.xy[idx], center_xy, heading_xy)
        mask = np.abs(along) <= along_half
        if not np.any(mask):
            return None
        lateral = lateral[mask]
        weights = self._normalize_intensity(self.intensity[idx][mask])
        bin_size = max(float(self.cfg.grid_cell_size_m) * 0.5, 0.01)
        bins = np.arange(-lateral_half, lateral_half + bin_size, bin_size, dtype=np.float64)
        if bins.size < 3:
            return None
        hist, edges = np.histogram(lateral, bins=bins, weights=weights)
        centers = 0.5 * (edges[:-1] + edges[1:])
        smooth = np.convolve(hist, np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64) / 9.0, mode="same")
        stripe_pairs = self._extract_stripe_pairs_1d(smooth, hist > 0.0, centers, bin_size)
        profile_stripes = [
            ProfileStripeCandidate(
                left_m=pair.left_m,
                right_m=pair.right_m,
                center_m=pair.center_m,
                width_m=pair.width_m,
            )
            for pair in stripe_pairs
        ]
        if profile_stripes:
            selected_idx = min(range(len(profile_stripes)), key=lambda idx: abs(profile_stripes[idx].center_m - stripe_center_m))
        else:
            profile_stripes = [
                ProfileStripeCandidate(
                    left_m=stripe_center_m - 0.5 * float(self.cfg.stripe_width_m),
                    right_m=stripe_center_m + 0.5 * float(self.cfg.stripe_width_m),
                    center_m=stripe_center_m,
                    width_m=float(self.cfg.stripe_width_m),
                )
            ]
            selected_idx = 0
        quality = float(np.clip(np.max(smooth) / max(np.sum(smooth), 1e-6) * len(smooth), 0.0, 1.0))
        return ProfileData(
            bins_center=centers,
            hist_combined=hist,
            smooth_hist=smooth,
            stripe_candidates=profile_stripes,
            selected_idx=selected_idx,
            quality=quality,
        )

    def _apply_observation(self, obs: ObservationCandidate) -> None:
        if self.state is None:
            return
        prev_heading = self.state.tangent_xy.copy()
        new_heading = self._blend_heading(prev_heading, obs.heading_xy)
        self.state.center_xyz = np.asarray(obs.center_xyz, dtype=np.float64).copy()
        self.state.tangent_xy = new_heading
        self.state.lane_width_m = float(obs.width_m)
        self.state.left_edge_m = float(obs.left_edge_m)
        self.state.right_edge_m = float(obs.right_edge_m)
        self.state.stripe_center_m = 0.0
        self.state.mode = TrackMode.SOLID_VISIBLE if obs.solid_score >= obs.dashed_score else TrackMode.DASH_VISIBLE
        self.state.profile_quality = float(obs.total_score)
        self.state.stripe_strength = float(obs.intensity_score)
        self.state.center_confidence = float(obs.total_score)
        self.state.identity_confidence = float(obs.identity_score)
        self.state.dashed_score = float(obs.dashed_score)
        self.state.solid_score = float(obs.solid_score)
        self.state.crosswalk_score = 0.0
        self.state.gap_distance_m = 0.0
        self.state.curvature = _signed_angle(prev_heading, new_heading) / max(float(self.cfg.forward_distance_m), 1e-6)
        self._append_history(self.state.center_xyz, self.state.tangent_xy)
        self._append_path(self.state.center_xyz, self.state.tangent_xy)

    def _append_history(self, center_xyz: np.ndarray, heading_xy: np.ndarray) -> None:
        if self.state is None:
            return
        self.state.history_centers.append(np.asarray(center_xyz, dtype=np.float64).copy())
        self.state.history_headings.append(_normalize(heading_xy))

    def _append_path(self, center_xyz: np.ndarray, heading_xy: np.ndarray) -> None:
        if self.state is None:
            return
        self.state.path_centers.append(np.asarray(center_xyz, dtype=np.float64).copy())
        self.state.path_headings.append(_normalize(heading_xy))

    def _blend_heading(self, prev_heading: np.ndarray, candidate_heading: np.ndarray) -> np.ndarray:
        prev = _normalize(prev_heading)
        cand = self._limit_heading_change(prev, candidate_heading)
        alpha = float(np.clip(self.cfg.heading_smoothing_alpha, 0.0, 1.0))
        blended = _normalize((1.0 - alpha) * prev + alpha * cand)
        return self._limit_heading_change(prev, blended)

    def _limit_heading_change(self, prev_heading: np.ndarray, candidate_heading: np.ndarray) -> np.ndarray:
        prev = _normalize(prev_heading)
        cand = _normalize(candidate_heading)
        delta = _signed_angle(prev, cand)
        max_delta = math.radians(max(float(self.cfg.max_heading_change_deg), 0.1))
        if abs(delta) <= max_delta:
            return cand
        clipped = float(np.clip(delta, -max_delta, max_delta))
        c = math.cos(clipped)
        s = math.sin(clipped)
        return _normalize(np.array([prev[0] * c - prev[1] * s, prev[0] * s + prev[1] * c], dtype=np.float64))

    def _estimate_z_ref(self, center_xy: np.ndarray, fallback_z: float) -> float:
        radius = max(float(self.cfg.corridor_half_width_m), float(self.cfg.stripe_width_m) * 2.0, 0.20)
        idx = self.grid.query_radius_xy(center_xy, radius)
        if idx.size == 0:
            return float(fallback_z)
        weights = self._normalize_intensity(self.intensity[idx]) + 0.05
        return _weighted_median(self.z[idx], weights)

    def _normalize_intensity(self, values: np.ndarray) -> np.ndarray:
        vv = np.asarray(values, dtype=np.float64)
        denom = self._global_q95 - self._global_q50
        if denom <= 1e-6:
            return np.ones_like(vv, dtype=np.float64)
        return np.clip((vv - self._global_q50) / denom, 0.0, 1.0)

    def _component_to_world_polyline(
        self,
        state: TrackerState,
        pred_heading: np.ndarray,
        record: _ComponentCandidate | None,
        z_ref: float,
    ) -> np.ndarray | None:
        if record is None:
            return None
        tangent, normal = make_frame(pred_heading)
        pts_xy = (
            state.center_xyz[:2][None, :]
            + record.sample_along[:, None] * tangent[None, :]
            + record.sample_lateral[:, None] * normal[None, :]
        )
        return np.column_stack([pts_xy, np.full(len(pts_xy), z_ref, dtype=np.float64)])

    def _component_to_world_edge_polylines(
        self,
        state: TrackerState,
        pred_heading: np.ndarray,
        record: _ComponentCandidate | None,
        z_ref: float,
    ) -> list[np.ndarray] | None:
        if record is None:
            return None
        tangent, normal = make_frame(pred_heading)
        left_xy = (
            state.center_xyz[:2][None, :]
            + record.sample_along[:, None] * tangent[None, :]
            + record.sample_left_edge[:, None] * normal[None, :]
        )
        right_xy = (
            state.center_xyz[:2][None, :]
            + record.sample_along[:, None] * tangent[None, :]
            + record.sample_right_edge[:, None] * normal[None, :]
        )
        left_poly = np.column_stack([left_xy, np.full(len(left_xy), z_ref, dtype=np.float64)])
        right_poly = np.column_stack([right_xy, np.full(len(right_xy), z_ref, dtype=np.float64)])
        return [left_poly, right_poly]

    def _make_active_cell_boxes(
        self,
        state: TrackerState,
        pred_heading: np.ndarray,
        cell_view: dict[str, np.ndarray],
    ) -> list[np.ndarray] | None:
        active_mask = cell_view["active_mask"]
        along_centers = cell_view["along_centers"]
        lateral_centers = cell_view["lateral_centers"]
        cell_size = float(cell_view["cell_size"][0])
        tangent, normal = make_frame(pred_heading)
        boxes: list[np.ndarray] = []
        active_indices = np.argwhere(active_mask)
        display_limit = max(int(getattr(self.cfg, "active_box_display_limit", 200)), 0)
        if display_limit > 0:
            active_indices = active_indices[:display_limit]
        for a_idx, l_idx in active_indices:
            along_val = float(along_centers[a_idx])
            lateral_val = float(lateral_centers[l_idx])
            center_xy = state.center_xyz[:2] + tangent * along_val + normal * lateral_val
            boxes.append(self._make_box_polyline(center_xy, pred_heading, float(state.center_xyz[2]), cell_size, cell_size))
        return boxes or None

    def _make_box_polyline(
        self,
        center_xy: np.ndarray,
        heading_xy: np.ndarray,
        z_ref: float,
        length_m: float,
        width_m: float,
    ) -> np.ndarray:
        tangent, normal = make_frame(heading_xy)
        half_len = 0.5 * float(length_m)
        half_w = 0.5 * float(width_m)
        corners_xy = np.vstack(
            [
                center_xy - tangent * half_len - normal * half_w,
                center_xy - tangent * half_len + normal * half_w,
                center_xy + tangent * half_len + normal * half_w,
                center_xy + tangent * half_len - normal * half_w,
                center_xy - tangent * half_len - normal * half_w,
            ]
        )
        return np.column_stack([corners_xy, np.full(len(corners_xy), z_ref, dtype=np.float64)])

    def _build_roi_box(self, center_xy: np.ndarray, heading_xy: np.ndarray) -> np.ndarray:
        roi_back = float(self.cfg.roi_backward_m)
        roi_forward = float(self.cfg.roi_forward_m)
        roi_lateral = float(self.cfg.roi_lateral_half_m)
        tangent, normal = make_frame(heading_xy)
        p0 = center_xy - tangent * roi_back - normal * roi_lateral
        p1 = center_xy - tangent * roi_back + normal * roi_lateral
        p2 = center_xy + tangent * roi_forward + normal * roi_lateral
        p3 = center_xy + tangent * roi_forward - normal * roi_lateral
        poly = np.vstack([p0, p1, p2, p3, p0])
        z_ref = float(self.state.center_xyz[2]) if self.state is not None else 0.0
        return np.column_stack([poly, np.full(len(poly), z_ref, dtype=np.float64)])

    def _build_trajectory_preview(
        self,
        state: TrackerState,
        pred_heading: np.ndarray,
        best_record: _ComponentCandidate | None,
    ) -> np.ndarray | None:
        z_ref = float(state.center_xyz[2])
        if best_record is None:
            tangent = _normalize(pred_heading)
            pts_xy = np.vstack(
                [
                    state.center_xyz[:2],
                    state.center_xyz[:2] + tangent * float(self.cfg.roi_forward_m),
                ]
            )
            return np.column_stack([pts_xy, np.full(len(pts_xy), z_ref, dtype=np.float64)])

        poly = self._component_to_world_polyline(state, pred_heading, best_record, z_ref)
        if poly is None:
            return None
        history = np.asarray(state.history_centers[-6:], dtype=np.float64)
        return np.vstack([history, poly])
