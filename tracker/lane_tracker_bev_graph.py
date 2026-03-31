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
            raise RuntimeError("Failed to build an initial BEV graph path.")
        self._apply_observation(best, reset_gap=True)
        self._init_summary = {
            "reason": "segment_graph",
            "candidate_count": dbg.candidate_count,
            "best_score": float(best.total_score),
            "intensity_score": float(best.intensity_score),
            "contrast_score": float(best.contrast_score),
            "autocorr_score": float(best.autocorr_score),
            "continuity": float(best.continuity_score),
            "endpoint_evidence": float(best.total_score),
            "endpoint_distance": float(np.linalg.norm(best.center_xy - p1[:2])),
            "endpoint_loyalty": float(np.dot(_normalize(best.heading_xy), heading_xy)),
        }
        dbg.step_index = 0
        dbg.source = "segment_seed"
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
        dbg.source = "gap" if gap_step else "segment"
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
        active_points = roi.get("active_points_xyz")
        edge_groups = self._edges_to_world_lines(roi["nodes"], roi["edges"], float(state.center_xyz[2]))
        search_box = self._build_roi_box(state)

        if not roi["nodes"]:
            dbg = DebugFrame(
                step_index=self._step_index,
                source="segment",
                candidate_points=active_points,
                graph_edge_groups=edge_groups,
                chosen_candidate=None,
                candidate_count=0,
                stop_reason=None,
                profile=None,
                trajectory_line_points=None,
                search_box_points=search_box,
                candidate_summaries=[],
                gap_distance_m=float(state.gap_distance_m),
            )
            return None, dbg

        beams = self._run_node_beam(roi["nodes"], roi["edges"], roi["component_scores"])
        if not beams:
            dbg = DebugFrame(
                step_index=self._step_index,
                source="segment",
                candidate_points=active_points,
                graph_edge_groups=edge_groups,
                chosen_candidate=None,
                candidate_count=len(roi["nodes"]),
                stop_reason=None,
                profile=None,
                trajectory_line_points=None,
                search_box_points=search_box,
                candidate_summaries=[],
                gap_distance_m=float(state.gap_distance_m),
            )
            return None, dbg

        best_beam = beams[0]
        best_obs = self._build_observation_from_beam(state, roi["nodes"], best_beam, roi["q50"], roi["q95"])
        trajectory = self._beam_to_xyz_path(state, roi["nodes"], best_beam)
        summaries = [
            f"score={beam.cumulative_score:.2f} | nodes={len(beam.node_ids)} | dist={beam.total_distance_m:.2f} | end_lat={beam.endpoint_lateral_m:.2f}"
            for beam in beams[:3]
        ]
        dbg = DebugFrame(
            step_index=self._step_index,
            source="segment",
            candidate_points=active_points,
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
            return {"nodes": [], "edges": {}, "q50": self._global_q50, "q95": self._global_q95}

        along, lateral = project_points_xy(self.xy[idx], state.center_xyz[:2], tangent)
        mask = (
            (along >= 0.0)
            & (along <= self.cfg.graph_roi_forward_m)
            & (np.abs(lateral) <= self.cfg.graph_roi_lateral_half_m)
        )
        if not np.any(mask):
            return {"nodes": [], "edges": {}, "q50": self._global_q50, "q95": self._global_q95}

        idx = idx[mask]
        along = along[mask]
        lateral = lateral[mask]
        local_intensity = self.intensity[idx].astype(np.float64)
        q50 = _safe_percentile(local_intensity, 50.0, default=self._global_q50)
        q95 = _safe_percentile(local_intensity, 95.0, default=max(q50 + 1e-3, self._global_q95))
        signal = np.clip((local_intensity - q50) / max(q95 - q50, 1e-6), 0.0, 1.0)

        n_along = max(1, int(math.ceil(self.cfg.graph_roi_forward_m / self.cfg.graph_cell_size_m)))
        n_lat = max(1, int(math.ceil((2.0 * self.cfg.graph_roi_lateral_half_m) / self.cfg.graph_cell_size_m)))
        intensity_grid = np.zeros((n_along, n_lat), dtype=np.float64)
        count_grid = np.zeros((n_along, n_lat), dtype=np.int32)

        along_idx = np.clip(np.floor(along / self.cfg.graph_cell_size_m).astype(np.int32), 0, n_along - 1)
        lateral_idx = np.clip(
            np.floor((lateral + self.cfg.graph_roi_lateral_half_m) / self.cfg.graph_cell_size_m).astype(np.int32),
            0,
            n_lat - 1,
        )
        for ia, il, s in zip(along_idx, lateral_idx, signal):
            if s > intensity_grid[ia, il]:
                intensity_grid[ia, il] = float(s)
            count_grid[ia, il] += 1

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
        active_points_list: list[np.ndarray] = []
        for ia in range(n_along):
            for il in range(n_lat):
                if not active[ia, il]:
                    continue
                along_m = (ia + 0.5) * self.cfg.graph_cell_size_m
                lateral_m = (il + 0.5) * self.cfg.graph_cell_size_m - self.cfg.graph_roi_lateral_half_m
                center_xy = state.center_xyz[:2] + tangent * along_m + normal * lateral_m
                active_points_list.append(np.array([center_xy[0], center_xy[1], z_ref], dtype=np.float64))

        min_length_m = float(getattr(self.cfg, "segment_min_length_m", 0.20))
        target_length_m = float(getattr(self.cfg, "segment_target_length_m", 0.50))
        max_length_m = float(getattr(self.cfg, "segment_max_length_m", 1.00))
        heading_gate_deg = float(getattr(self.cfg, "segment_heading_gate_deg", 28.0))
        visited_cells = np.zeros_like(active, dtype=bool)
        nodes: list[GraphNode] = []
        node_id = 0
        component_id = 0
        for ia0 in range(n_along):
            for il0 in range(n_lat):
                if not active[ia0, il0] or visited_cells[ia0, il0]:
                    continue
                stack = [(ia0, il0)]
                visited_cells[ia0, il0] = True
                comp_cells: list[tuple[int, int]] = []
                while stack:
                    ia, il = stack.pop()
                    comp_cells.append((ia, il))
                    for da in (-1, 0, 1):
                        for dl in (-1, 0, 1):
                            if da == 0 and dl == 0:
                                continue
                            na = ia + da
                            nl = il + dl
                            if 0 <= na < n_along and 0 <= nl < n_lat and active[na, nl] and not visited_cells[na, nl]:
                                visited_cells[na, nl] = True
                                stack.append((na, nl))

                comp_pts_local = np.asarray(
                    [
                        [
                            (ia + 0.5) * self.cfg.graph_cell_size_m,
                            (il + 0.5) * self.cfg.graph_cell_size_m - self.cfg.graph_roi_lateral_half_m,
                        ]
                        for ia, il in comp_cells
                    ],
                    dtype=np.float64,
                )
                if comp_pts_local.size == 0:
                    continue
                comp_intensity = np.asarray([intensity_grid[ia, il] for ia, il in comp_cells], dtype=np.float64)
                comp_contrast = np.asarray([contrast_norm[ia, il] for ia, il in comp_cells], dtype=np.float64)
                comp_counts = np.asarray([count_grid[ia, il] for ia, il in comp_cells], dtype=np.float64)

                comp_center = np.mean(comp_pts_local, axis=0)
                centered = comp_pts_local - comp_center[None, :]
                if len(comp_pts_local) >= 2:
                    cov = (centered.T @ centered) / max(len(comp_pts_local) - 1, 1)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    axis_local = eigvecs[:, int(np.argmax(eigvals))]
                    eigvals = np.maximum(eigvals, 1e-9)
                    elongation = float(np.max(eigvals) / np.min(eigvals))
                else:
                    axis_local = np.array([1.0, 0.0], dtype=np.float64)
                    elongation = 999.0
                axis_local = _normalize(axis_local)
                if axis_local[0] < 0.0:
                    axis_local *= -1.0
                ortho_local = np.array([-axis_local[1], axis_local[0]], dtype=np.float64)
                proj_u = centered @ axis_local
                proj_v = centered @ ortho_local
                component_length = float(np.ptp(proj_u) + self.cfg.graph_cell_size_m)
                if component_length < min_length_m or elongation < 2.2:
                    continue

                u_min = float(np.min(proj_u))
                u_max = float(np.max(proj_u))
                chunk_ranges: list[tuple[float, float]] = []
                if component_length <= max_length_m:
                    chunk_ranges.append((u_min, u_max))
                else:
                    cursor = u_min
                    while cursor < u_max + 1e-6:
                        chunk_ranges.append((cursor, min(cursor + max_length_m, u_max)))
                        cursor += max(target_length_m, min_length_m)

                for start_u, end_u in chunk_ranges:
                    chunk_mask = (proj_u >= start_u - 0.10 * target_length_m) & (proj_u <= end_u + 0.10 * target_length_m)
                    if int(np.count_nonzero(chunk_mask)) < max(2, int(self.cfg.graph_min_cell_points)):
                        continue
                    chunk_pts_local = comp_pts_local[chunk_mask]
                    chunk_intensity = comp_intensity[chunk_mask]
                    chunk_contrast = comp_contrast[chunk_mask]
                    chunk_counts = comp_counts[chunk_mask]
                    chunk_center = np.mean(chunk_pts_local, axis=0)
                    chunk_centered = chunk_pts_local - chunk_center[None, :]
                    chunk_u = chunk_centered @ axis_local
                    chunk_v = chunk_centered @ ortho_local
                    seg_length = float(np.ptp(chunk_u) + self.cfg.graph_cell_size_m)
                    if seg_length < min_length_m:
                        continue
                    seg_width_guess = float(max(self.cfg.graph_cell_size_m, np.ptp(chunk_v) + self.cfg.graph_cell_size_m))
                    center_xy = state.center_xyz[:2] + tangent * chunk_center[0] + normal * chunk_center[1]
                    heading_xy = _normalize(tangent * axis_local[0] + normal * axis_local[1])
                    if np.dot(heading_xy, state.tangent_xy) < 0.0:
                        heading_xy *= -1.0
                    z_ref_node = self._estimate_z_ref(center_xy, z_ref)
                    profile = self._build_twin_edge_profile(center_xy, heading_xy, z_ref_node, q50, q95)
                    if profile is not None and profile.selected_idx is not None:
                        stripe = profile.stripe_candidates[profile.selected_idx]
                        seg_width = float(stripe.width_m)
                        profile_score = float(profile.quality)
                    else:
                        seg_width = seg_width_guess
                        profile_score = 0.0
                    history_score = self._history_alignment_score(state, center_xy, float(chunk_center[0]))
                    node = GraphNode(
                        node_id=node_id,
                        along_idx=int(np.clip(round(chunk_center[0] / self.cfg.graph_cell_size_m), 0, n_along - 1)),
                        lateral_idx=int(np.clip(round((chunk_center[1] + self.cfg.graph_roi_lateral_half_m) / self.cfg.graph_cell_size_m), 0, n_lat - 1)),
                        along_m=float(chunk_center[0]),
                        lateral_m=float(chunk_center[1]),
                        center_xy=center_xy,
                        heading_xy=heading_xy,
                        intensity_score=float(np.mean(chunk_intensity)),
                        contrast_score=float(np.mean(chunk_contrast)),
                        profile_score=float(profile_score),
                        history_score=float(history_score),
                        point_count=int(np.sum(chunk_counts)),
                        length_m=float(seg_length),
                        width_m=float(seg_width),
                        component_id=component_id,
                    )
                    nodes.append(node)
                    node_id += 1
                component_id += 1

        neighbor_max_gap_m = max(self.cfg.graph_neighbor_max_distance_m, target_length_m * 1.2)
        lateral_limit_m = max(self.cfg.graph_neighbor_lateral_limit_m, 0.30)
        heading_limit_rad = math.radians(heading_gate_deg)
        edges: dict[int, list[tuple[int, float, float]]] = {}
        nodes_sorted = sorted(nodes, key=lambda n: (n.along_m, n.lateral_m))
        for node in nodes_sorted:
            neigh: list[tuple[int, float, float]] = []
            for other in nodes_sorted:
                if other.node_id == node.node_id:
                    continue
                if other.along_m <= node.along_m + self.cfg.graph_cell_size_m * 0.5:
                    continue
                delta_xy = other.center_xy - node.center_xy
                dist = float(np.linalg.norm(delta_xy))
                if dist > neighbor_max_gap_m:
                    continue
                lateral_delta = abs(other.lateral_m - node.lateral_m)
                if lateral_delta > lateral_limit_m:
                    continue
                center_dir = _normalize(delta_xy)
                heading_diff = _angle_between(node.heading_xy, other.heading_xy)
                link_diff = _angle_between(node.heading_xy, center_dir)
                width_diff = abs(other.width_m - node.width_m)
                if heading_diff > heading_limit_rad or link_diff > heading_limit_rad:
                    continue
                if width_diff > max(0.14, 0.55 * max(node.width_m, other.width_m)):
                    continue
                angle = max(heading_diff, link_diff)
                neigh.append((other.node_id, dist, float(angle)))
            neigh.sort(key=lambda item: item[1])
            edges[node.node_id] = neigh

        nodes_by_id = {node.node_id: node for node in nodes}
        undirected: dict[int, set[int]] = {node.node_id: set() for node in nodes}
        for src_id, neigh in edges.items():
            for dst_id, _, _ in neigh:
                undirected[src_id].add(dst_id)
                undirected[dst_id].add(src_id)

        component_scores: dict[int, float] = {}
        visited: set[int] = set()
        component_id = 0
        for node in nodes:
            if node.node_id in visited:
                continue
            stack = [node.node_id]
            comp_nodes: list[GraphNode] = []
            visited.add(node.node_id)
            while stack:
                cur = stack.pop()
                cur_node = nodes_by_id[cur]
                cur_node.component_id = component_id
                comp_nodes.append(cur_node)
                for nxt in undirected.get(cur, ()):
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)

            along_span = max(n.along_m for n in comp_nodes) - min(n.along_m for n in comp_nodes)
            mean_intensity = float(np.mean([n.intensity_score for n in comp_nodes]))
            mean_contrast = float(np.mean([n.contrast_score for n in comp_nodes]))
            mean_history = float(np.mean([n.history_score for n in comp_nodes]))
            length_score = min(1.0, along_span / max(self.cfg.forward_distance_m * 1.25, 1e-6))
            size_score = min(1.0, len(comp_nodes) / 8.0)
            component_scores[component_id] = float(
                0.28 * mean_intensity
                + 0.14 * mean_contrast
                + 0.22 * mean_history
                + 0.22 * length_score
                + 0.14 * size_score
            )
            component_id += 1

        active_points_xyz = np.asarray(active_points_list, dtype=np.float64) if active_points_list else None
        return {
            "nodes": nodes,
            "edges": edges,
            "q50": q50,
            "q95": q95,
            "component_scores": component_scores,
            "active_points_xyz": active_points_xyz,
        }

    def _run_node_beam(
        self,
        nodes: list[GraphNode],
        edges: dict[int, list[tuple[int, float, float]]],
        component_scores: dict[int, float],
    ) -> list[PathHypothesis]:
        nodes_by_id = {n.node_id: n for n in nodes}
        if component_scores:
            ordered_components = sorted(component_scores.items(), key=lambda item: item[1], reverse=True)
            best_comp_score = ordered_components[0][1]
            allowed_components = {
                comp_id
                for comp_id, score in ordered_components[:2]
                if score >= best_comp_score - 0.08
            }
        else:
            allowed_components = set()
        start_nodes = [
            n
            for n in nodes
            if n.along_m <= max(self.cfg.forward_distance_m * 1.8, self.cfg.graph_cell_size_m * 2.0)
            and abs(n.lateral_m) <= self.cfg.graph_neighbor_lateral_limit_m * 1.5
            and (not allowed_components or n.component_id in allowed_components)
        ]
        if not start_nodes:
            fallback_nodes = [n for n in nodes if not allowed_components or n.component_id in allowed_components]
            start_nodes = sorted(
                fallback_nodes if fallback_nodes else nodes,
                key=lambda n: (-component_scores.get(n.component_id, 0.0), abs(n.lateral_m), n.along_m),
            )[: max(1, self.cfg.graph_beam_branching)]

        beam: list[PathHypothesis] = []
        for node in sorted(
            start_nodes,
            key=lambda n: (-(component_scores.get(n.component_id, 0.0) + 0.6 * n.intensity_score + 0.4 * n.contrast_score), abs(n.lateral_m)),
        )[: max(1, self.cfg.graph_beam_width)]:
            first_angle = float(math.atan2(node.lateral_m, max(node.along_m, 1e-6)))
            first_score = self._node_base_score(node) + self._start_transition_score(node, first_angle) + 0.18 * component_scores.get(node.component_id, 0.0)
            beam.append(
                PathHypothesis(
                    node_ids=[node.node_id],
                    cumulative_score=float(first_score),
                    last_angle_rad=first_angle,
                    total_distance_m=float(math.hypot(node.along_m, node.lateral_m)),
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
                scored_edges: list[tuple[float, int, float]] = []
                for next_id, dist, angle in candidates:
                    next_node = nodes_by_id[next_id]
                    if allowed_components and next_node.component_id not in allowed_components:
                        continue
                    edge_score = self._transition_score(last_node, next_node, hyp.last_angle_rad, dist, angle)
                    total = self._node_base_score(next_node) + edge_score + 0.10 * component_scores.get(next_node.component_id, 0.0)
                    scored_edges.append((total, next_id, dist))
                scored_edges.sort(key=lambda item: item[0], reverse=True)
                for total, next_id, dist in scored_edges[: max(1, int(self.cfg.graph_beam_branching))]:
                    next_node = nodes_by_id[next_id]
                    next_angle = math.atan2(next_node.lateral_m - last_node.lateral_m, max(next_node.along_m - last_node.along_m, 1e-6))
                    next_beam.append(
                        PathHypothesis(
                            node_ids=hyp.node_ids + [next_id],
                            cumulative_score=float(hyp.cumulative_score + total),
                            last_angle_rad=float(next_angle),
                            total_distance_m=float(hyp.total_distance_m + dist),
                            endpoint_along_m=float(next_node.along_m),
                            endpoint_lateral_m=float(next_node.lateral_m),
                        )
                    )
            next_beam.sort(
                key=lambda h: (h.cumulative_score + 0.10 * min(1.0, h.endpoint_along_m / max(self.cfg.graph_roi_forward_m, 1e-6))),
                reverse=True,
            )
            beam = next_beam[: max(1, int(self.cfg.graph_beam_width))]
            if not beam:
                break
        beam.sort(
            key=lambda h: (h.cumulative_score + 0.10 * min(1.0, h.endpoint_along_m / max(self.cfg.graph_roi_forward_m, 1e-6))),
            reverse=True,
        )
        return beam

    def _node_base_score(self, node: GraphNode) -> float:
        support = min(1.0, float(node.point_count) / 4.0)
        length_score = min(1.0, node.length_m / max(getattr(self.cfg, "segment_target_length_m", 0.50), 1e-6))
        return float(
            self.cfg.graph_intensity_weight * node.intensity_score
            + self.cfg.graph_contrast_weight * node.contrast_score
            + self.cfg.graph_history_weight * node.history_score
            + 0.10 * node.profile_score
            + 0.08 * length_score
            + 0.06 * support
        )

    def _start_transition_score(self, node: GraphNode, angle: float) -> float:
        direction_score = math.exp(-abs(angle) / math.radians(18.0))
        distance_score = math.exp(-abs(node.along_m - self.cfg.forward_distance_m) / max(self.cfg.forward_distance_m, 1e-6))
        lateral_score = math.exp(-abs(node.lateral_m) / max(self.cfg.graph_neighbor_lateral_limit_m, 1e-6))
        heading_score = math.exp(-_angle_between(node.heading_xy, self.state.tangent_xy) / math.radians(20.0)) if self.state is not None else 1.0
        return float(
            self.cfg.graph_direction_weight * (0.55 * direction_score + 0.45 * heading_score)
            + self.cfg.graph_distance_weight * distance_score
            + 0.08 * lateral_score
        )

    def _transition_score(self, prev_node: GraphNode, next_node: GraphNode, prev_angle: float, dist: float, angle_abs: float) -> float:
        edge_angle = math.atan2(next_node.lateral_m - prev_node.lateral_m, max(next_node.along_m - prev_node.along_m, 1e-6))
        turn_continuity = math.exp(-abs(edge_angle - prev_angle) / math.radians(18.0))
        direction_score = math.exp(-angle_abs / math.radians(22.0))
        distance_score = math.exp(-abs(dist - max(getattr(self.cfg, "segment_target_length_m", 0.50), self.cfg.forward_distance_m)) / max(self.cfg.graph_neighbor_max_distance_m, 1e-6))
        lateral_score = math.exp(-abs(next_node.lateral_m - prev_node.lateral_m) / max(self.cfg.graph_neighbor_lateral_limit_m, 1e-6))
        heading_pair_score = math.exp(-_angle_between(prev_node.heading_xy, next_node.heading_xy) / math.radians(18.0))
        width_pair_score = math.exp(-abs(next_node.width_m - prev_node.width_m) / max(max(prev_node.width_m, next_node.width_m), 1e-6))
        return float(
            self.cfg.graph_direction_weight * (0.35 * direction_score + 0.30 * turn_continuity + 0.35 * heading_pair_score)
            + self.cfg.graph_distance_weight * distance_score
            + 0.08 * width_pair_score
            + 0.06 * lateral_score
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
        center_xy = target_node.center_xy.copy()
        heading_xy = _normalize(target_node.heading_xy)
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
            width_m = float(state.lane_width_m)
            stripe_center_m = 0.0
            left_edge_m = -0.5 * width_m
            right_edge_m = 0.5 * width_m
            profile_quality = 0.0

        node_signal = np.asarray([n.intensity_score for n in path_nodes], dtype=np.float64)
        ac_score, lag = _autocorr_peak(node_signal, 1, min(8, len(node_signal) - 1))
        dominant_period = lag * self.cfg.graph_cell_size_m
        dashed_score = float(np.clip((ac_score - self.cfg.dashed_autocorr_min) / max(1.0 - self.cfg.dashed_autocorr_min, 1e-6), 0.0, 1.0))
        solid_score = float(np.clip(np.mean(node_signal) + 0.25 * (1.0 - dashed_score), 0.0, 1.0))
        continuity_score = float(np.clip(np.mean([n.history_score for n in path_nodes]), 0.0, 1.0))
        intensity_score = float(np.mean([n.intensity_score for n in path_nodes]))
        contrast_score = float(np.mean([n.contrast_score for n in path_nodes]))
        crosswalk_score = self._detect_crosswalk(target_node.center_xy, heading_xy, z_ref, q50, q95)
        avg_raw = beam.cumulative_score / max(len(path_nodes), 1)
        path_score = float(np.clip(avg_raw, 0.0, 1.0))
        center_penalty = math.exp(-abs(target_node.lateral_m - state.stripe_center_m) / max(state.lane_width_m * 0.65, 0.12))
        total_score = float(
            np.clip(
                0.72 * path_score
                + 0.18 * center_penalty
                + self.cfg.graph_period_weight * ac_score
                + 0.06 * profile_quality
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
            contrast_score=float(max(contrast_score, profile_quality)),
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
