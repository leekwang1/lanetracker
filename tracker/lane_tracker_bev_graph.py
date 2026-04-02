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


def _rotate(v: np.ndarray, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c], dtype=np.float64)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(a)
    bb = _normalize(b)
    dot = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(math.acos(dot))


def _signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(a)
    bb = _normalize(b)
    return float(math.atan2(aa[0] * bb[1] - aa[1] * bb[0], np.dot(aa, bb)))


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


def _autocorr_peak(values: np.ndarray, min_lag: int, max_lag: int) -> tuple[float, int]:
    if values.size < 6:
        return 0.0, 0
    centered = values.astype(np.float64) - float(np.mean(values))
    denom = float(np.dot(centered, centered)) + 1e-9
    best_corr = 0.0
    best_lag = 0
    for lag in range(max(1, min_lag), min(len(centered) - 1, max_lag) + 1):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return float(np.clip(best_corr, 0.0, 1.0)), int(best_lag)


def _smooth_grid3x3(grid: np.ndarray) -> np.ndarray:
    if grid.size == 0:
        return grid
    padded = np.pad(grid, 1, mode="edge")
    smoothed = np.zeros_like(grid, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            smoothed += padded[dy : dy + grid.shape[0], dx : dx + grid.shape[1]]
    return smoothed / 9.0


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


@dataclass
class GraphNode:
    node_id: int
    along_idx: int
    lateral_idx: int
    along_m: float
    lateral_m: float
    center_xy: np.ndarray
    heading_xy: np.ndarray
    intensity_score: float
    contrast_score: float
    profile_score: float
    history_score: float
    point_count: int
    length_m: float = 0.0
    width_m: float = 0.0
    component_id: int = -1


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
    active_cell_box_groups: list[np.ndarray] | None
    segment_groups: list[np.ndarray] | None
    graph_edge_groups: list[np.ndarray] | None
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


@dataclass
class PathHypothesis:
    node_ids: list[int]
    cumulative_score: float
    last_angle_rad: float
    total_distance_m: float
    endpoint_along_m: float
    endpoint_lateral_m: float


class BevGraphTracker:
    def __init__(self, xyz: np.ndarray, intensity: np.ndarray, cfg: TrackerConfig):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.xy = self.xyz[:, :2]
        self.z = self.xyz[:, 2]
        self.intensity = np.asarray(intensity, dtype=np.float32)
        self.cfg = cfg
        self.grid = SpatialGrid(self.xy, cell_size=max(cfg.spatial_grid_cell_size_m, cfg.graph_cell_size_m * 2.0, 0.08))
        self._global_q50 = _safe_percentile(self.intensity.astype(np.float64), 50.0, default=0.0)
        self._global_q95 = _safe_percentile(self.intensity.astype(np.float64), 95.0, default=max(self._global_q50 + 1.0, 1.0))
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
        center_xyz = np.array([p0[0], p0[1], z_ref], dtype=np.float64)
        self.state = TrackerState(
            center_xyz=center_xyz,
            tangent_xy=heading_xy,
            lane_width_m=0.18,
            left_edge_m=-0.09,
            right_edge_m=0.09,
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
        )

        best, dbg = self._plan_local_graph(self.state)
        if best is None:
            raise RuntimeError("Failed to build an initial lane-box path.")
        self._apply_observation(best, reset_gap=True)
        self._init_summary = {
            "reason": "box_graph",
            "candidate_count": dbg.candidate_count,
            "best_score": float(best.total_score),
            "intensity_score": float(best.intensity_score),
            "contrast_score": float(best.contrast_score),
            "autocorr_score": float(best.autocorr_score),
            "continuity": float(best.continuity_score),
            "endpoint_evidence": float(best.path_score),
            "endpoint_distance": float(np.linalg.norm(best.center_xy - p1[:2])),
            "endpoint_loyalty": float(np.dot(_normalize(best.heading_xy), heading_xy)),
        }
        dbg.step_index = 0
        dbg.source = "box_seed"
        dbg.chosen_candidate = best
        dbg.profile = best.profile
        self.debug_frames.append(dbg)

    def step(self) -> DebugFrame:
        if self.state is None:
            raise RuntimeError("Tracker is not initialized.")
        if self.stop_reason != StopReason.NONE:
            return self.debug_frames[-1]

        best, dbg = self._plan_local_graph(self.state)
        gap_step = False
        if best is None or best.total_score < self.cfg.candidate_min_score:
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
                self.state.curvature *= 0.85
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
            self._apply_observation(best, reset_gap=True)
            self._distance_travelled_m += self.cfg.forward_distance_m
            if self.cfg.crosswalk_stop_enabled and best.crosswalk_score >= 0.95:
                self.stop_reason = StopReason.CROSSWALK
                self.state.mode = TrackMode.STOPPED

        self._step_index += 1
        if self.stop_reason == StopReason.NONE and self._distance_travelled_m >= self.cfg.max_track_length_m:
            self.stop_reason = StopReason.MAX_DISTANCE
            self.state.mode = TrackMode.STOPPED

        dbg.step_index = self._step_index
        dbg.source = "gap" if gap_step else "box_graph"
        dbg.stop_reason = self.stop_reason.value if self.stop_reason != StopReason.NONE else None
        dbg.chosen_candidate = best
        dbg.profile = best.profile if best is not None else None
        dbg.gap_distance_m = float(self.state.gap_distance_m)
        self.debug_frames.append(dbg)
        return dbg

    def run_full(self) -> TrackerResult:
        while self.stop_reason == StopReason.NONE:
            self.step()
        dense = np.asarray(self.state.history_centers, dtype=np.float64) if self.state is not None else np.empty((0, 3), dtype=np.float64)
        return TrackerResult(dense_points=dense, output_points=dense, stop_reason=self.stop_reason.value, debug_frames=list(self.debug_frames))

    def _plan_local_graph(self, state: TrackerState) -> tuple[ObservationCandidate | None, DebugFrame]:
        roi = self._build_local_graph(state)
        node_points = roi.get("node_points_xyz")
        active_cell_boxes = roi.get("active_cell_box_groups")
        segment_groups = roi.get("lane_box_groups")
        edge_groups = roi.get("graph_edge_groups")
        search_box = self._build_roi_box(state)

        if not roi["nodes"]:
            dbg = DebugFrame(
                step_index=self._step_index,
                source="box_graph",
                candidate_points=node_points,
                active_cell_box_groups=active_cell_boxes,
                segment_groups=segment_groups,
                graph_edge_groups=edge_groups,
                chosen_candidate=None,
                candidate_count=0,
                stop_reason=None,
                profile=None,
                trajectory_line_points=None,
                search_box_points=search_box,
                candidate_summaries=["No lane boxes inside ROI"],
                gap_distance_m=float(state.gap_distance_m),
            )
            return None, dbg

        beams = self._run_node_beam(roi["nodes"], roi["edges"], roi["component_scores"])
        if not beams:
            dbg = DebugFrame(
                step_index=self._step_index,
                source="box_graph",
                candidate_points=node_points,
                active_cell_box_groups=active_cell_boxes,
                segment_groups=segment_groups,
                graph_edge_groups=edge_groups,
                chosen_candidate=None,
                candidate_count=len(roi["nodes"]),
                stop_reason=None,
                profile=None,
                trajectory_line_points=None,
                search_box_points=search_box,
                candidate_summaries=["No antenna-edge path from current ROI"],
                gap_distance_m=float(state.gap_distance_m),
            )
            return None, dbg

        best_beam = beams[0]
        best_obs = self._build_observation_from_beam(state, roi["nodes"], best_beam, roi["q50"], roi["q95"])
        trajectory = self._beam_to_xyz_path(state, roi["nodes"], best_beam)
        summaries = [
            f"score={beam.cumulative_score:.2f} | boxes={len(beam.node_ids)} | dist={beam.total_distance_m:.2f} | end_lat={beam.endpoint_lateral_m:.2f}"
            for beam in beams[:3]
        ]
        dbg = DebugFrame(
            step_index=self._step_index,
            source="box_graph",
            candidate_points=node_points,
            active_cell_box_groups=active_cell_boxes,
            segment_groups=segment_groups,
            graph_edge_groups=edge_groups,
            chosen_candidate=best_obs,
            candidate_count=len(roi["nodes"]),
            stop_reason=None,
            profile=best_obs.profile if best_obs is not None else None,
            trajectory_line_points=trajectory,
            search_box_points=search_box,
            candidate_summaries=summaries,
            gap_distance_m=float(state.gap_distance_m),
        )
        return best_obs, dbg

    def _build_local_graph(self, state: TrackerState) -> dict[str, object]:
        tangent, normal = make_frame(state.tangent_xy)
        strip_center = state.center_xyz[:2] + tangent * (self.cfg.graph_roi_forward_m * 0.5)
        idx = self.grid.query_oriented_strip_xy(
            center_xy=strip_center,
            tangent_xy=tangent,
            along_half=self.cfg.graph_roi_forward_m * 0.5,
            lateral_half=self.cfg.graph_roi_lateral_half_m,
        )
        z_ref = self._estimate_z_ref(state.center_xyz[:2], float(state.center_xyz[2]))
        idx = self._filter_z(idx, z_ref)
        if idx.size == 0:
            return self._empty_roi_result()

        along, lateral = project_points_xy(self.xy[idx], state.center_xyz[:2], tangent)
        mask = (
            (along >= 0.0)
            & (along <= self.cfg.graph_roi_forward_m)
            & (np.abs(lateral) <= self.cfg.graph_roi_lateral_half_m)
        )
        if not np.any(mask):
            return self._empty_roi_result()

        idx = idx[mask]
        along = along[mask]
        lateral = lateral[mask]
        local_intensity = self.intensity[idx].astype(np.float64)
        q50 = _safe_percentile(local_intensity, 50.0, default=self._global_q50)
        q95 = _safe_percentile(local_intensity, 95.0, default=max(q50 + 1e-3, self._global_q95))
        signal = np.clip((local_intensity - q50) / max(q95 - q50, 1e-6), 0.0, 1.0)

        cell_size = float(self.cfg.graph_cell_size_m)
        n_along = max(1, int(math.ceil(self.cfg.graph_roi_forward_m / cell_size)))
        n_lat = max(1, int(math.ceil((2.0 * self.cfg.graph_roi_lateral_half_m) / cell_size)))
        intensity_grid = np.zeros((n_along, n_lat), dtype=np.float64)
        count_grid = np.zeros((n_along, n_lat), dtype=np.int32)
        along_sum_grid = np.zeros((n_along, n_lat), dtype=np.float64)
        lateral_sum_grid = np.zeros((n_along, n_lat), dtype=np.float64)

        along_idx = np.clip(np.floor(along / cell_size).astype(np.int32), 0, n_along - 1)
        lateral_idx = np.clip(
            np.floor((lateral + self.cfg.graph_roi_lateral_half_m) / cell_size).astype(np.int32),
            0,
            n_lat - 1,
        )
        for ia, il, s, a, l in zip(along_idx, lateral_idx, signal, along, lateral):
            if s > intensity_grid[ia, il]:
                intensity_grid[ia, il] = float(s)
            count_grid[ia, il] += 1
            along_sum_grid[ia, il] += float(a)
            lateral_sum_grid[ia, il] += float(l)

        along_center_grid = np.divide(
            along_sum_grid,
            np.maximum(count_grid, 1),
            out=np.zeros_like(along_sum_grid),
            where=count_grid > 0,
        )
        lateral_center_grid = np.divide(
            lateral_sum_grid,
            np.maximum(count_grid, 1),
            out=np.zeros_like(lateral_sum_grid),
            where=count_grid > 0,
        )

        smooth = _smooth_grid3x3(intensity_grid)
        contrast = np.clip(intensity_grid - smooth, 0.0, 1.0)
        contrast_max = float(np.max(contrast)) if contrast.size else 0.0
        contrast_norm = contrast / max(contrast_max, 1e-6) if contrast_max > 1e-6 else contrast

        score_grid = 0.65 * intensity_grid + 0.35 * contrast_norm
        valid_scores = score_grid[count_grid >= int(self.cfg.graph_min_cell_points)]
        adaptive_score_min = _safe_percentile(valid_scores, 84.0, default=self.cfg.graph_active_intensity_min) if valid_scores.size else self.cfg.graph_active_intensity_min
        active = (
            (count_grid >= int(self.cfg.graph_min_cell_points))
            & (
                (score_grid >= adaptive_score_min)
                | ((intensity_grid >= self.cfg.graph_active_intensity_min) & (contrast_norm >= self.cfg.graph_active_contrast_min))
            )
        )
        active = self._denoise_active_grid(active)
        if not np.any(active):
            return self._empty_roi_result()

        active_points_list: list[np.ndarray] = []
        active_cell_box_groups: list[np.ndarray] = []
        active_indices = np.argwhere(active)
        for ia, il in active_indices:
            along_m = float(along_center_grid[ia, il]) if count_grid[ia, il] > 0 else (float(ia) + 0.5) * cell_size
            lateral_m = (
                float(lateral_center_grid[ia, il])
                if count_grid[ia, il] > 0
                else (float(il) + 0.5) * cell_size - self.cfg.graph_roi_lateral_half_m
            )
            center_xy = state.center_xyz[:2] + tangent * along_m + normal * lateral_m
            active_points_list.append(np.array([center_xy[0], center_xy[1], z_ref], dtype=np.float64))
            active_cell_box_groups.append(self._make_box_polyline(center_xy, tangent, z_ref, cell_size))

        selected = self._select_single_lane_box(
            state=state,
            tangent=tangent,
            normal=normal,
            z_ref=z_ref,
            q50=q50,
            q95=q95,
            active_mask=active,
            active_indices=active_indices,
            along_center_grid=along_center_grid,
            lateral_center_grid=lateral_center_grid,
            intensity_grid=intensity_grid,
            contrast_norm=contrast_norm,
            cell_size=cell_size,
        )
        if selected is None:
            return self._empty_roi_result(active_points_list=active_points_list, active_cell_box_groups=active_cell_box_groups)

        node, lane_box, antenna_groups, component_score = selected
        nodes = [node]
        node_points_xyz = self._nodes_to_xyz(nodes, z_ref)
        return {
            "nodes": nodes,
            "edges": {node.node_id: []},
            "q50": q50,
            "q95": q95,
            "component_scores": {node.component_id: float(component_score)},
            "node_points_xyz": node_points_xyz,
            "active_cell_box_groups": active_cell_box_groups if active_cell_box_groups else None,
            "lane_box_groups": [lane_box],
            "graph_edge_groups": antenna_groups,
        }

    def _empty_roi_result(
        self,
        active_points_list: list[np.ndarray] | None = None,
        active_cell_box_groups: list[np.ndarray] | None = None,
    ) -> dict[str, object]:
        node_points_xyz = np.asarray(active_points_list, dtype=np.float64) if active_points_list else None
        return {
            "nodes": [],
            "edges": {},
            "q50": self._global_q50,
            "q95": self._global_q95,
            "component_scores": {},
            "node_points_xyz": node_points_xyz,
            "active_cell_box_groups": active_cell_box_groups if active_cell_box_groups else None,
            "lane_box_groups": None,
            "graph_edge_groups": None,
        }

    def _select_single_lane_box(
        self,
        state: TrackerState,
        tangent: np.ndarray,
        normal: np.ndarray,
        z_ref: float,
        q50: float,
        q95: float,
        active_mask: np.ndarray,
        active_indices: np.ndarray,
        along_center_grid: np.ndarray,
        lateral_center_grid: np.ndarray,
        intensity_grid: np.ndarray,
        contrast_norm: np.ndarray,
        cell_size: float,
    ) -> tuple[GraphNode, np.ndarray, list[np.ndarray], float] | None:
        if active_indices.size == 0 or not np.any(active_mask):
            return None

        box_length = max(float(self.cfg.lane_box_length_m), cell_size)
        box_width = max(float(self.cfg.lane_box_width_m), cell_size)
        half_box_length = box_length * 0.5
        half_box_width = box_width * 0.5
        pred_xy, _ = self._predict_pose(state, self.cfg.forward_distance_m)
        pred_along, pred_lateral = project_points_xy(pred_xy[None, :], state.center_xyz[:2], tangent)
        target_along = float(pred_along[0])
        target_lateral = float(pred_lateral[0])
        pred_heading_xy = self._predict_pose(state, max(self.cfg.forward_distance_m, box_length))[1]
        pred_heading_local = _normalize(
            np.array(
                [
                    float(np.dot(pred_heading_xy, tangent)),
                    float(np.dot(pred_heading_xy, normal)),
                ],
                dtype=np.float64,
            )
        )
        desired_cells = max((box_length * box_width) / max(cell_size * cell_size, 1e-6) * 0.35, 1.0)

        best_node: GraphNode | None = None
        best_lane_box: np.ndarray | None = None
        best_antenna_groups: list[np.ndarray] | None = None
        best_component_score = -1.0
        best_score = -1.0
        components = self._extract_active_components(active_mask)
        min_cells = max(int(self.cfg.lane_box_min_active_cells), 1)
        for component_id, component in enumerate(components):
            if len(component) < min_cells:
                continue

            rows = component[:, 0]
            cols = component[:, 1]
            local_cluster = np.column_stack(
                [
                    along_center_grid[rows, cols].astype(np.float64),
                    lateral_center_grid[rows, cols].astype(np.float64),
                ]
            )
            intensity_values = intensity_grid[rows, cols].astype(np.float64)
            contrast_values = contrast_norm[rows, cols].astype(np.float64)
            weights = np.clip(0.72 * intensity_values + 0.28 * np.maximum(contrast_values, 0.05), 1e-3, None)
            cluster_center = np.average(local_cluster, axis=0, weights=weights)
            centered = local_cluster - cluster_center[None, :]
            if len(local_cluster) >= 2:
                cov = (centered * weights[:, None]).T @ centered / max(float(np.sum(weights)), 1e-9)
                eigvals, eigvecs = np.linalg.eigh(cov)
                axis_local = eigvecs[:, int(np.argmax(eigvals))]
                axis_local = _normalize(axis_local)
                if np.dot(axis_local, pred_heading_local) < 0.0:
                    axis_local *= -1.0
                major = float(np.max(eigvals))
                minor = float(np.min(eigvals))
                elongation = major / max(minor, 1e-9)
                if not np.isfinite(elongation) or elongation < 1.15:
                    axis_local = pred_heading_local.copy()
            else:
                axis_local = pred_heading_local.copy()
            axis_normal = np.array([-axis_local[1], axis_local[0]], dtype=np.float64)
            s = centered @ axis_local
            t = centered @ axis_normal
            if s.size == 0:
                continue

            pred_local = np.array([target_along, target_lateral], dtype=np.float64)
            pred_centered = pred_local - cluster_center
            pred_s = float(np.dot(pred_centered, axis_local))
            pred_t = float(np.dot(pred_centered, axis_normal))
            candidate_s = np.unique(
                np.round(
                    np.concatenate(
                        [
                            s,
                            np.array([np.clip(pred_s, np.min(s), np.max(s)), float(np.median(s))], dtype=np.float64),
                        ]
                    )
                    / max(cell_size, 1e-6)
                )
                * cell_size
            )
            if candidate_s.size == 0:
                candidate_s = np.asarray([pred_s], dtype=np.float64)

            best_mask: np.ndarray | None = None
            best_axis_center_s = 0.0
            best_axis_center_t = 0.0
            best_window_score = -1.0
            for center_s in candidate_s:
                along_mask = np.abs(s - center_s) <= half_box_length
                along_count = int(np.count_nonzero(along_mask))
                if along_count < min_cells:
                    continue
                center_t = _weighted_median(t[along_mask], weights[along_mask])
                mask = along_mask & (np.abs(t - center_t) <= half_box_width)
                count = int(np.count_nonzero(mask))
                if count < min_cells:
                    continue
                refined_weights = weights[mask]
                refined_t = t[mask]
                occupancy_score = min(1.0, float(count) / desired_cells)
                density_score = float(np.mean(refined_weights))
                progress_score = math.exp(-abs(center_s - pred_s) / max(box_length * 0.70, self.cfg.forward_distance_m * 0.60, 1e-6))
                lateral_score = math.exp(-abs(center_t - pred_t) / max(half_box_width, self.cfg.antenna_half_width_m, 1e-6))
                if refined_t.size >= 2:
                    compactness_width = float(np.percentile(refined_t, 90.0) - np.percentile(refined_t, 10.0))
                else:
                    compactness_width = 0.0
                compactness_score = math.exp(-compactness_width / max(box_width * 0.55, cell_size, 1e-6))
                support_score = min(1.0, float(count) / max(len(local_cluster) * 0.65, 1.0))
                window_score = float(
                    np.clip(
                        0.34 * occupancy_score
                        + 0.22 * density_score
                        + 0.18 * progress_score
                        + 0.14 * lateral_score
                        + 0.12 * compactness_score
                        + 0.06 * support_score,
                        0.0,
                        1.0,
                    )
                )
                if window_score <= best_window_score:
                    continue
                best_mask = mask
                best_axis_center_s = float(center_s)
                best_axis_center_t = float(center_t)
                best_window_score = window_score

            if best_mask is None:
                continue

            refined_cluster = local_cluster[best_mask]
            refined_intensity = intensity_values[best_mask]
            refined_contrast = contrast_values[best_mask]
            refined_weights = weights[best_mask]
            refined_centered = refined_cluster - cluster_center[None, :]
            refined_s = refined_centered @ axis_local
            refined_t = refined_centered @ axis_normal
            median_s = _weighted_median(refined_s, refined_weights)
            median_t = _weighted_median(refined_t, refined_weights)
            center_s = 0.65 * best_axis_center_s + 0.35 * median_s
            center_t = 0.35 * best_axis_center_t + 0.65 * median_t
            center_local = cluster_center + axis_local * center_s + axis_normal * center_t
            along_m = float(center_local[0])
            lateral_m = float(center_local[1])
            center_xy = state.center_xyz[:2] + tangent * along_m + normal * lateral_m
            heading_xy = _normalize(tangent * axis_local[0] + normal * axis_local[1])
            if np.dot(heading_xy, pred_heading_xy) < 0.0:
                heading_xy *= -1.0

            z_ref_node = self._estimate_z_ref(center_xy, z_ref)
            profile = self._build_twin_edge_profile(center_xy, heading_xy, z_ref_node, q50, q95)
            if profile is not None and profile.selected_idx is not None:
                stripe = profile.stripe_candidates[profile.selected_idx]
                width_m = float(stripe.width_m)
                profile_score = float(profile.quality)
            else:
                width_m = float(state.lane_width_m)
                profile_score = 0.0

            history_score = self._history_alignment_score(state, center_xy, along_m)
            intensity_score = float(np.average(refined_intensity, weights=refined_weights))
            contrast_score = float(np.average(refined_contrast, weights=refined_weights))
            occupancy_score = min(1.0, float(np.count_nonzero(best_mask)) / desired_cells)
            progress_score = math.exp(-abs(along_m - target_along) / max(box_length, self.cfg.forward_distance_m * 0.75, 1e-6))
            lateral_score = math.exp(-abs(lateral_m - target_lateral) / max(self.cfg.antenna_half_width_m, half_box_width, 1e-6))
            heading_score = math.exp(-_angle_between(heading_xy, pred_heading_xy) / math.radians(16.0))
            if refined_t.size >= 2:
                compactness_width = float(np.percentile(refined_t, 90.0) - np.percentile(refined_t, 10.0))
            else:
                compactness_width = 0.0
            compactness_score = math.exp(-compactness_width / max(box_width * 0.55, cell_size, 1e-6))
            total_score = float(
                np.clip(
                    0.28 * intensity_score
                    + 0.14 * contrast_score
                    + 0.22 * history_score
                    + 0.14 * progress_score
                    + 0.08 * lateral_score
                    + 0.08 * occupancy_score
                    + 0.04 * compactness_score
                    + 0.06 * profile_score,
                    0.0,
                    1.0,
                )
            )
            total_score = float(
                np.clip(
                    0.84 * total_score
                    + 0.16 * heading_score,
                    0.0,
                    1.0,
                )
            )
            if total_score <= best_score:
                continue

            node = GraphNode(
                node_id=0,
                along_idx=int(round(along_m / max(cell_size, 1e-6))),
                lateral_idx=int(round((lateral_m + self.cfg.graph_roi_lateral_half_m) / max(cell_size, 1e-6))),
                along_m=along_m,
                lateral_m=lateral_m,
                center_xy=center_xy,
                heading_xy=heading_xy,
                intensity_score=intensity_score,
                contrast_score=contrast_score,
                profile_score=profile_score,
                history_score=history_score,
                point_count=int(np.count_nonzero(best_mask)),
                length_m=box_length,
                width_m=width_m,
                component_id=component_id,
            )
            best_node = node
            best_lane_box = self._make_box_polyline(center_xy, heading_xy, z_ref_node, box_length, box_width)
            best_antenna_groups = self._make_antenna_visuals(center_xy, heading_xy, z_ref_node)
            best_component_score = float(total_score)
            best_score = total_score

        if best_node is None or best_lane_box is None or best_antenna_groups is None or best_component_score < 0.0:
            return None
        return best_node, best_lane_box, best_antenna_groups, best_component_score

    def _extract_active_components(self, active_mask: np.ndarray) -> list[np.ndarray]:
        active = np.asarray(active_mask, dtype=bool)
        if active.size == 0 or not np.any(active):
            return []
        visited = np.zeros_like(active, dtype=bool)
        components: list[np.ndarray] = []
        for row in range(active.shape[0]):
            for col in range(active.shape[1]):
                if not active[row, col] or visited[row, col]:
                    continue
                stack = [(row, col)]
                visited[row, col] = True
                component: list[tuple[int, int]] = []
                while stack:
                    cur_row, cur_col = stack.pop()
                    component.append((cur_row, cur_col))
                    for da in (-1, 0, 1):
                        for dl in (-1, 0, 1):
                            if da == 0 and dl == 0:
                                continue
                            next_row = cur_row + da
                            next_col = cur_col + dl
                            if (
                                0 <= next_row < active.shape[0]
                                and 0 <= next_col < active.shape[1]
                                and active[next_row, next_col]
                                and not visited[next_row, next_col]
                            ):
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))
                components.append(np.asarray(component, dtype=np.int32))
        components.sort(key=len, reverse=True)
        return components

    def _denoise_active_grid(self, active: np.ndarray) -> np.ndarray:
        clean = np.asarray(active, dtype=bool).copy()
        if clean.size == 0:
            return clean
        min_neighbors = max(int(self.cfg.graph_noise_min_neighbors), 0)
        if min_neighbors > 0:
            padded = np.pad(clean.astype(np.int32), 1, mode="constant")
            neighbor_count = np.zeros_like(clean, dtype=np.int32)
            for dy in range(3):
                for dx in range(3):
                    if dy == 1 and dx == 1:
                        continue
                    neighbor_count += padded[dy : dy + clean.shape[0], dx : dx + clean.shape[1]]
            clean &= neighbor_count >= min_neighbors
        if not np.any(clean):
            return clean

        min_component = max(int(self.cfg.graph_noise_min_component_cells), 1)
        if min_component <= 1:
            return clean
        kept = np.zeros_like(clean, dtype=bool)
        visited = np.zeros_like(clean, dtype=bool)
        for row in range(clean.shape[0]):
            for col in range(clean.shape[1]):
                if not clean[row, col] or visited[row, col]:
                    continue
                stack = [(row, col)]
                visited[row, col] = True
                component: list[tuple[int, int]] = []
                while stack:
                    cur_row, cur_col = stack.pop()
                    component.append((cur_row, cur_col))
                    for da in (-1, 0, 1):
                        for dl in (-1, 0, 1):
                            if da == 0 and dl == 0:
                                continue
                            next_row = cur_row + da
                            next_col = cur_col + dl
                            if (
                                0 <= next_row < clean.shape[0]
                                and 0 <= next_col < clean.shape[1]
                                and clean[next_row, next_col]
                                and not visited[next_row, next_col]
                            ):
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))
                if len(component) >= min_component:
                    for cur_row, cur_col in component:
                        kept[cur_row, cur_col] = True
        return kept

    def _run_node_beam(
        self,
        nodes: list[GraphNode],
        edges: dict[int, list[tuple[int, float, float]]],
        component_scores: dict[int, float],
    ) -> list[PathHypothesis]:
        nodes_by_id = {n.node_id: n for n in nodes}
        state_lateral_center = float(self.state.stripe_center_m) if self.state is not None else 0.0
        lane_width_m = float(self.state.lane_width_m) if self.state is not None else 0.18
        start_along_limit_m = min(
            float(self.cfg.graph_roi_forward_m),
            max(float(self.cfg.forward_distance_m) * 2.5, float(self.cfg.antenna_length_m) * 0.75),
        )
        start_corridor_half_m = max(
            float(self.cfg.antenna_half_width_m),
            lane_width_m * 0.90,
            0.18,
        )
        ranked_components = [
            component_id
            for component_id, _score in sorted(component_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        allowed_components = set(ranked_components[:3]) if ranked_components else set()

        start_nodes = [
            n
            for n in nodes
            if n.along_m <= start_along_limit_m
            and abs(n.lateral_m - state_lateral_center) <= start_corridor_half_m
            and (not allowed_components or n.component_id in allowed_components)
        ]
        if not start_nodes:
            start_nodes = [n for n in nodes if not allowed_components or n.component_id in allowed_components]
        if not start_nodes:
            start_nodes = list(nodes)
            allowed_components = set()

        beam: list[PathHypothesis] = []
        for node in sorted(
            start_nodes,
            key=lambda n: (
                -(
                    component_scores.get(n.component_id, 0.0)
                    + 0.55 * n.intensity_score
                    + 0.45 * n.history_score
                    + 0.20 * n.profile_score
                ),
                abs(n.lateral_m - state_lateral_center),
            ),
        )[: max(1, self.cfg.graph_beam_width)]:
            first_angle = float(math.atan2(node.lateral_m - state_lateral_center, max(node.along_m, 1e-6)))
            first_score = (
                self._node_base_score(node)
                + self._start_transition_score(node, first_angle)
                + 0.15 * component_scores.get(node.component_id, 0.0)
            )
            beam.append(
                PathHypothesis(
                    node_ids=[node.node_id],
                    cumulative_score=float(first_score),
                    last_angle_rad=first_angle,
                    total_distance_m=float(math.hypot(node.along_m, node.lateral_m - state_lateral_center)),
                    endpoint_along_m=float(node.along_m),
                    endpoint_lateral_m=float(node.lateral_m),
                )
            )
        if not beam:
            return []

        for _ in range(max(1, int(self.cfg.graph_beam_horizon_nodes)) - 1):
            next_beam: list[PathHypothesis] = []
            for hyp in beam:
                last_id = hyp.node_ids[-1]
                last_node = nodes_by_id[last_id]
                candidates = edges.get(last_id, [])
                if not candidates:
                    next_beam.append(hyp)
                    continue
                scored_edges: list[tuple[float, int, float, float]] = []
                for next_id, dist, angle in candidates:
                    if next_id in hyp.node_ids:
                        continue
                    next_node = nodes_by_id[next_id]
                    if allowed_components and next_node.component_id not in allowed_components:
                        continue
                    edge_score = self._transition_score(last_node, next_node, hyp.last_angle_rad, dist, angle)
                    total = self._node_base_score(next_node) + edge_score + 0.10 * component_scores.get(next_node.component_id, 0.0)
                    scored_edges.append((total, next_id, dist, angle))
                scored_edges.sort(key=lambda item: item[0], reverse=True)
                if not scored_edges:
                    next_beam.append(hyp)
                    continue
                for total, next_id, dist, angle in scored_edges[: max(1, int(self.cfg.graph_beam_branching))]:
                    next_node = nodes_by_id[next_id]
                    next_beam.append(
                        PathHypothesis(
                            node_ids=hyp.node_ids + [next_id],
                            cumulative_score=float(hyp.cumulative_score + total),
                            last_angle_rad=float(angle),
                            total_distance_m=float(hyp.total_distance_m + dist),
                            endpoint_along_m=float(next_node.along_m),
                            endpoint_lateral_m=float(next_node.lateral_m),
                        )
                    )
            next_beam.sort(key=self._beam_rank_key, reverse=True)
            beam = next_beam[: max(1, int(self.cfg.graph_beam_width))]
            if not beam:
                break
        beam.sort(key=self._beam_rank_key, reverse=True)
        return beam

    def _beam_rank_key(self, hyp: PathHypothesis) -> float:
        forward_bonus = 0.10 * min(1.0, hyp.endpoint_along_m / max(self.cfg.graph_roi_forward_m, 1e-6))
        lateral_penalty = 0.06 * min(1.0, abs(hyp.endpoint_lateral_m) / max(self.cfg.antenna_half_width_m, 1e-6))
        return float(hyp.cumulative_score + forward_bonus - lateral_penalty)

    def _node_base_score(self, node: GraphNode) -> float:
        support = min(1.0, float(node.point_count) / max(float(self.cfg.lane_box_min_active_cells) + 2.0, 1.0))
        return float(
            self.cfg.graph_intensity_weight * node.intensity_score
            + self.cfg.graph_contrast_weight * node.contrast_score
            + self.cfg.graph_history_weight * node.history_score
            + 0.12 * node.profile_score
            + 0.08 * support
        )

    def _start_transition_score(self, node: GraphNode, angle: float) -> float:
        heading_score = math.exp(-_angle_between(node.heading_xy, self.state.tangent_xy) / math.radians(18.0)) if self.state is not None else 1.0
        direction_score = math.exp(-abs(angle) / math.radians(18.0))
        distance_score = math.exp(-abs(node.along_m - self.cfg.forward_distance_m) / max(self.cfg.antenna_length_m * 0.45, 1e-6))
        lateral_center = self.state.stripe_center_m if self.state is not None else 0.0
        lateral_score = math.exp(-abs(node.lateral_m - lateral_center) / max(self.cfg.antenna_half_width_m, 1e-6))
        return float(
            self.cfg.graph_direction_weight * (0.45 * direction_score + 0.55 * heading_score)
            + self.cfg.graph_distance_weight * distance_score
            + 0.10 * lateral_score
        )

    def _transition_score(self, prev_node: GraphNode, next_node: GraphNode, prev_angle: float, dist: float, angle_abs: float) -> float:
        along, lateral = project_points_xy(next_node.center_xy[None, :], prev_node.center_xy, prev_node.heading_xy)
        along_m = float(along[0])
        lateral_m = float(lateral[0])
        target_spacing = max(float(self.cfg.lane_box_length_m), float(self.cfg.forward_distance_m) * 0.9)
        edge_angle = math.atan2(lateral_m, max(along_m, 1e-6))
        turn_continuity = math.exp(-abs(edge_angle - prev_angle) / math.radians(16.0))
        direction_score = math.exp(-abs(angle_abs) / math.radians(18.0))
        distance_score = math.exp(-abs(dist - target_spacing) / max(self.cfg.antenna_length_m * 0.35, 1e-6))
        progress_score = math.exp(-abs(along_m - target_spacing) / max(self.cfg.antenna_length_m * 0.30, 1e-6))
        lateral_score = math.exp(-abs(next_node.lateral_m - prev_node.lateral_m) / max(self.cfg.antenna_half_width_m, 1e-6))
        heading_pair_score = math.exp(-_angle_between(prev_node.heading_xy, next_node.heading_xy) / math.radians(18.0))
        component_bonus = 0.08 if next_node.component_id == prev_node.component_id else 0.0
        return float(
            self.cfg.graph_direction_weight * (0.30 * direction_score + 0.30 * turn_continuity + 0.40 * heading_pair_score)
            + self.cfg.graph_distance_weight * (0.50 * distance_score + 0.50 * progress_score)
            + 0.08 * lateral_score
            + component_bonus
        )

    def _build_observation_from_beam(
        self,
        state: TrackerState,
        nodes: list[GraphNode],
        beam: PathHypothesis,
        q50: float,
        q95: float,
    ) -> ObservationCandidate | None:
        if not beam.node_ids:
            return None
        node_by_id = {n.node_id: n for n in nodes}
        path_nodes = [node_by_id[i] for i in beam.node_ids]
        polyline_xy = [state.center_xyz[:2].copy()] + [node.center_xy for node in path_nodes]
        cumulative: list[float] = [0.0]
        total = 0.0
        for i in range(1, len(polyline_xy)):
            total += float(np.linalg.norm(polyline_xy[i] - polyline_xy[i - 1]))
            cumulative.append(total)
        if total < 1e-6:
            return None

        if len(path_nodes) == 1:
            center_xy = path_nodes[0].center_xy.copy()
            heading_xy = _normalize(path_nodes[0].heading_xy)
        else:
            target_s = min(float(self.cfg.forward_distance_m), total)
            center_xy = polyline_xy[-1].copy()
            heading_xy = _normalize(path_nodes[0].heading_xy if path_nodes else state.tangent_xy)
            for i in range(1, len(polyline_xy)):
                s0 = cumulative[i - 1]
                s1 = cumulative[i]
                if target_s <= s1 + 1e-9:
                    seg = polyline_xy[i] - polyline_xy[i - 1]
                    seg_len = float(np.linalg.norm(seg))
                    if seg_len > 1e-9:
                        alpha = np.clip((target_s - s0) / seg_len, 0.0, 1.0)
                        center_xy = polyline_xy[i - 1] + seg * alpha
                        heading_xy = _normalize(seg)
                    else:
                        center_xy = polyline_xy[i].copy()
                    break

        target_node = min(path_nodes, key=lambda node: float(np.linalg.norm(node.center_xy - center_xy)))
        z_ref = self._estimate_z_ref(center_xy, float(state.center_xyz[2]))
        center_xyz = np.array([center_xy[0], center_xy[1], z_ref], dtype=np.float64)
        profile = self._build_twin_edge_profile(center_xy, heading_xy, z_ref, q50, q95)
        if profile is not None and profile.selected_idx is not None:
            stripe = profile.stripe_candidates[profile.selected_idx]
            width_m = float(stripe.width_m)
            stripe_center_m = float(stripe.center_m)
            left_edge_m = float(stripe.left_m - stripe.center_m)
            right_edge_m = float(stripe.right_m - stripe.center_m)
            profile_quality = float(profile.quality)
        else:
            width_m = float(target_node.width_m if target_node.width_m > 1e-6 else state.lane_width_m)
            stripe_center_m = 0.0
            left_edge_m = -0.5 * width_m
            right_edge_m = 0.5 * width_m
            profile_quality = float(target_node.profile_score)

        node_signal = np.asarray(
            [0.70 * n.intensity_score + 0.30 * max(n.contrast_score, n.profile_score) for n in path_nodes],
            dtype=np.float64,
        )
        ac_score, lag = _autocorr_peak(node_signal, 1, min(6, len(node_signal) - 1))
        dominant_period = lag * max(float(self.cfg.lane_box_length_m), float(self.cfg.graph_cell_size_m))
        dashed_score = float(np.clip((ac_score - self.cfg.dashed_autocorr_min) / max(1.0 - self.cfg.dashed_autocorr_min, 1e-6), 0.0, 1.0))
        solid_score = float(np.clip(np.mean(node_signal) + 0.25 * (1.0 - dashed_score), 0.0, 1.0))
        continuity_score = float(np.clip(np.mean([n.history_score for n in path_nodes]), 0.0, 1.0))
        intensity_score = float(np.mean([n.intensity_score for n in path_nodes]))
        contrast_score = float(np.mean([max(n.contrast_score, n.profile_score) for n in path_nodes]))
        crosswalk_score = self._detect_crosswalk(center_xy, heading_xy, z_ref, q50, q95)
        avg_raw = beam.cumulative_score / max(len(path_nodes), 1)
        path_score = float(np.clip(avg_raw, 0.0, 1.0))
        _, center_lateral = project_points_xy(center_xy[None, :], state.center_xyz[:2], state.tangent_xy)
        center_penalty = math.exp(
            -abs(float(center_lateral[0]) - state.stripe_center_m) / max(state.lane_width_m * 0.65, self.cfg.antenna_half_width_m, 0.12)
        )
        total_score = float(
            np.clip(
                0.70 * path_score
                + 0.16 * center_penalty
                + 0.08 * profile_quality
                + self.cfg.graph_period_weight * ac_score
                - self.cfg.graph_crosswalk_penalty * crosswalk_score,
                0.0,
                1.0,
            )
        )
        angle_offset_deg = math.degrees(_signed_angle(state.tangent_xy, heading_xy))
        return ObservationCandidate(
            center_xy=center_xy.copy(),
            center_xyz=center_xyz,
            heading_xy=heading_xy,
            angle_offset_deg=float(angle_offset_deg),
            stripe_center_m=float(stripe_center_m),
            left_edge_m=float(left_edge_m),
            right_edge_m=float(right_edge_m),
            width_m=float(width_m),
            intensity_score=float(intensity_score),
            contrast_score=float(contrast_score),
            autocorr_score=float(ac_score),
            continuity_score=float(continuity_score),
            total_score=float(total_score),
            path_score=float(path_score),
            dashed_score=float(dashed_score),
            solid_score=float(solid_score),
            crosswalk_score=float(crosswalk_score),
            support_count=len(path_nodes),
            z_ref=float(z_ref),
            dominant_period_m=float(dominant_period),
            path_length_m=float(beam.total_distance_m),
            path_node_count=len(path_nodes),
            profile=profile,
        )

    def _beam_to_xyz_path(self, state: TrackerState, nodes: list[GraphNode], beam: PathHypothesis) -> np.ndarray | None:
        if not beam.node_ids:
            return None
        node_by_id = {n.node_id: n for n in nodes}
        pts = [state.center_xyz.copy()]
        for node_id in beam.node_ids:
            xy = node_by_id[node_id].center_xy
            z_ref = self._estimate_z_ref(xy, float(state.center_xyz[2]))
            pts.append(np.array([xy[0], xy[1], z_ref], dtype=np.float64))
        return np.asarray(pts, dtype=np.float64)

    def _history_alignment_score(self, state: TrackerState, center_xy: np.ndarray, along_m: float) -> float:
        pred_xy, pred_heading = self._predict_pose(state, along_m)
        dist = float(np.linalg.norm(center_xy - pred_xy))
        _, lateral = project_points_xy(center_xy[None, :], pred_xy, pred_heading)
        lateral_err = abs(float(lateral[0]) - float(state.stripe_center_m))
        heading_score = math.exp(-_angle_between(pred_heading, state.tangent_xy) / math.radians(20.0))
        dist_score = math.exp(-dist / 0.22)
        lateral_score = math.exp(-lateral_err / max(state.lane_width_m * 0.55, 0.10))
        return float(np.clip(0.45 * lateral_score + 0.35 * dist_score + 0.20 * heading_score, 0.0, 1.0))

    def _predict_pose(self, state: TrackerState, distance_m: float) -> tuple[np.ndarray, np.ndarray]:
        heading = _normalize(state.tangent_xy)
        curvature = float(state.curvature)
        if abs(curvature) < 1e-6:
            return state.center_xyz[:2] + heading * distance_m, heading
        delta = curvature * distance_m
        next_heading = _normalize(_rotate(heading, delta))
        return state.center_xyz[:2] + next_heading * distance_m, next_heading

    def _predict_gap_pose(self) -> tuple[np.ndarray, np.ndarray]:
        heading = _normalize(self.state.tangent_xy)
        gap_ratio = min(1.0, float(self.state.gap_distance_m) / max(self.cfg.gap_forward_distance_m, 1e-6))
        curvature = float(self.state.curvature) * max(0.35, 1.0 - 0.65 * gap_ratio)
        next_heading = _normalize(_rotate(heading, curvature * self.cfg.forward_distance_m))
        next_xy = self.state.center_xyz[:2] + next_heading * self.cfg.forward_distance_m
        return next_xy, next_heading

    def _apply_observation(self, obs: ObservationCandidate, reset_gap: bool) -> None:
        prev_heading = _normalize(self.state.tangent_xy)
        self.state.center_xyz = obs.center_xyz.copy()
        self.state.tangent_xy = _normalize(obs.heading_xy)
        self.state.curvature = _signed_angle(prev_heading, self.state.tangent_xy) / max(self.cfg.forward_distance_m, 1e-6)
        self.state.lane_width_m = float(obs.width_m)
        self.state.left_edge_m = float(obs.left_edge_m)
        self.state.right_edge_m = float(obs.right_edge_m)
        self.state.stripe_center_m = float(obs.stripe_center_m)
        self.state.mode = self._classify_mode(obs)
        self.state.profile_quality = float(obs.total_score)
        self.state.center_confidence = float(obs.total_score)
        self.state.identity_confidence = float(obs.continuity_score)
        self.state.stripe_strength = float(obs.intensity_score)
        self.state.dashed_score = float(obs.dashed_score)
        self.state.solid_score = float(obs.solid_score)
        self.state.crosswalk_score = float(obs.crosswalk_score)
        if reset_gap:
            self.state.gap_distance_m = 0.0
        self._append_history(self.state.center_xyz, self.state.tangent_xy)

    def _append_history(self, center_xyz: np.ndarray, heading_xy: np.ndarray) -> None:
        self.state.history_centers.append(np.asarray(center_xyz, dtype=np.float64).copy())
        self.state.history_headings.append(_normalize(heading_xy).copy())

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

    def _build_twin_edge_profile(
        self,
        center_xy: np.ndarray,
        tangent_xy: np.ndarray,
        z_ref: float,
        q50: float,
        q95: float,
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
        signal = np.clip((intensity - q50) / max(q95 - q50, 1e-6), 0.0, 1.0)

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

    def _detect_crosswalk(
        self,
        center_xy: np.ndarray,
        tangent_xy: np.ndarray,
        z_ref: float,
        q50: float,
        q95: float,
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
        signal = np.clip((self.intensity[idx].astype(np.float64) - q50) / max(q95 - q50, 1e-6), 0.0, 1.0)
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
        max_val = float(np.max(prof)) if prof.size else 0.0
        if max_val > 1e-6:
            prof = prof / max_val
        peak_count = 0
        active = False
        for value in prof:
            if value >= 0.55:
                if not active:
                    peak_count += 1
                    active = True
            else:
                active = False
        return float(np.clip(peak_count / max(self.cfg.crosswalk_min_peaks, 1), 0.0, 1.0))

    def _classify_mode(self, obs: ObservationCandidate) -> TrackMode:
        if obs.crosswalk_score >= 0.75:
            return TrackMode.CROSSWALK_CANDIDATE
        if obs.dashed_score >= obs.solid_score and obs.autocorr_score >= self.cfg.dashed_autocorr_min:
            return TrackMode.DASH_VISIBLE
        return TrackMode.SOLID_VISIBLE

    def _nodes_to_xyz(self, nodes: list[GraphNode], fallback_z: float) -> np.ndarray | None:
        if not nodes:
            return None
        pts = np.empty((len(nodes), 3), dtype=np.float64)
        for i, node in enumerate(nodes):
            pts[i] = np.array([node.center_xy[0], node.center_xy[1], self._estimate_z_ref(node.center_xy, fallback_z)], dtype=np.float64)
        return pts

    def _edges_to_world_lines(
        self,
        nodes: list[GraphNode],
        edges: dict[int, list[tuple[int, float, float]]],
        fallback_z: float,
    ) -> list[np.ndarray] | None:
        if not nodes or not edges:
            return None
        node_by_id = {n.node_id: n for n in nodes}
        lines: list[np.ndarray] = []
        for node_id, neigh in edges.items():
            src = node_by_id.get(node_id)
            if src is None:
                continue
            src_z = self._estimate_z_ref(src.center_xy, fallback_z)
            src_pt = np.array([src.center_xy[0], src.center_xy[1], src_z], dtype=np.float64)
            for next_id, _, _ in neigh:
                dst = node_by_id.get(next_id)
                if dst is None:
                    continue
                dst_z = self._estimate_z_ref(dst.center_xy, fallback_z)
                dst_pt = np.array([dst.center_xy[0], dst.center_xy[1], dst_z], dtype=np.float64)
                lines.append(np.vstack([src_pt, dst_pt]))
        return lines if lines else None

    def _segments_to_world_lines(self, nodes: list[GraphNode], fallback_z: float) -> list[np.ndarray] | None:
        if not nodes:
            return None
        lines: list[np.ndarray] = []
        for node in nodes:
            z_ref = self._estimate_z_ref(node.center_xy, fallback_z)
            box_length = max(float(node.length_m), float(self.cfg.lane_box_length_m), float(self.cfg.graph_cell_size_m))
            box_width = max(float(self.cfg.lane_box_width_m), float(self.cfg.graph_cell_size_m))
            lines.append(self._make_box_polyline(node.center_xy, node.heading_xy, z_ref, box_length, box_width))
        return lines if lines else None

    def _make_box_polyline(
        self,
        center_xy: np.ndarray,
        heading_xy: np.ndarray,
        z_ref: float,
        along_size_m: float,
        lateral_size_m: float | None = None,
    ) -> np.ndarray:
        tangent = _normalize(heading_xy)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        along_half = float(along_size_m) * 0.5
        lateral_half = float(along_size_m if lateral_size_m is None else lateral_size_m) * 0.5
        corners_xy = np.asarray(
            [
                center_xy - tangent * along_half - normal * lateral_half,
                center_xy + tangent * along_half - normal * lateral_half,
                center_xy + tangent * along_half + normal * lateral_half,
                center_xy - tangent * along_half + normal * lateral_half,
                center_xy - tangent * along_half - normal * lateral_half,
            ],
            dtype=np.float64,
        )
        return np.column_stack([corners_xy, np.full(5, z_ref, dtype=np.float64)])

    def _make_antenna_visuals(self, center_xy: np.ndarray, heading_xy: np.ndarray, z_ref: float) -> list[np.ndarray]:
        tangent = _normalize(heading_xy)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        start_xy = np.asarray(center_xy, dtype=np.float64)
        end_xy = start_xy + tangent * float(self.cfg.antenna_length_m)
        half_width = float(self.cfg.antenna_half_width_m)
        corridor = np.vstack(
            [
                np.array([start_xy[0] - normal[0] * half_width, start_xy[1] - normal[1] * half_width, z_ref], dtype=np.float64),
                np.array([end_xy[0] - normal[0] * half_width, end_xy[1] - normal[1] * half_width, z_ref], dtype=np.float64),
                np.array([end_xy[0] + normal[0] * half_width, end_xy[1] + normal[1] * half_width, z_ref], dtype=np.float64),
                np.array([start_xy[0] + normal[0] * half_width, start_xy[1] + normal[1] * half_width, z_ref], dtype=np.float64),
                np.array([start_xy[0] - normal[0] * half_width, start_xy[1] - normal[1] * half_width, z_ref], dtype=np.float64),
            ]
        )
        center_line = np.vstack(
            [
                np.array([start_xy[0], start_xy[1], z_ref], dtype=np.float64),
                np.array([end_xy[0], end_xy[1], z_ref], dtype=np.float64),
            ]
        )
        arrow_len = min(max(float(self.cfg.antenna_length_m) * 0.10, 0.08), 0.30)
        arrow_half_width = min(max(half_width * 0.40, 0.04), max(half_width, 0.04))
        arrow_base = end_xy - tangent * arrow_len
        arrow_left = arrow_base + normal * arrow_half_width
        arrow_right = arrow_base - normal * arrow_half_width
        arrow_head = np.vstack(
            [
                np.array([arrow_left[0], arrow_left[1], z_ref], dtype=np.float64),
                np.array([end_xy[0], end_xy[1], z_ref], dtype=np.float64),
                np.array([arrow_right[0], arrow_right[1], z_ref], dtype=np.float64),
            ]
        )
        return [corridor, center_line, arrow_head]

    def _build_roi_box(self, state: TrackerState) -> np.ndarray:
        tangent, normal = make_frame(state.tangent_xy)
        origin = np.asarray(state.center_xyz, dtype=np.float64)
        far_center = origin[:2] + tangent * self.cfg.graph_roi_forward_m
        left0 = origin[:2] + normal * self.cfg.graph_roi_lateral_half_m
        right0 = origin[:2] - normal * self.cfg.graph_roi_lateral_half_m
        left1 = far_center + normal * self.cfg.graph_roi_lateral_half_m
        right1 = far_center - normal * self.cfg.graph_roi_lateral_half_m
        z = float(origin[2])
        return np.vstack(
            [
                np.array([left0[0], left0[1], z], dtype=np.float64),
                np.array([left1[0], left1[1], z], dtype=np.float64),
                np.array([right1[0], right1[1], z], dtype=np.float64),
                np.array([right0[0], right0[1], z], dtype=np.float64),
                np.array([left0[0], left0[1], z], dtype=np.float64),
            ]
        )
