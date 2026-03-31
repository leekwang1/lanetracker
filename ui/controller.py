from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore
import numpy as np

from ..core.types import StopReason, TrackMode
from ..io.las_io import load_las_xyz_intensity
from ..tracker.config import DEFAULT_CONFIG_PATH, TrackerConfig, ensure_config_file, load_tracker_config, save_tracker_config
from ..tracker.lane_tracker_bev_graph import BevGraphTracker, DebugFrame, TrackerResult, TrackerState
from .viewer_model import ViewerModel


class TrackerController(QtCore.QObject):
    changed = QtCore.Signal()
    log_message = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.model = ViewerModel()
        self.las = None
        self.tracker: BevGraphTracker | None = None
        self._config_path = ensure_config_file(DEFAULT_CONFIG_PATH)
        self._tracker_cfg = load_tracker_config(self._config_path)

    @property
    def config_path(self) -> Path:
        return Path(self._config_path)

    def get_config(self) -> TrackerConfig:
        return TrackerConfig(**vars(self._tracker_cfg))

    def set_config_path(self, path: str | Path) -> None:
        self._config_path = ensure_config_file(path)

    def load_config(self, path: str | Path | None = None) -> TrackerConfig:
        if path is not None:
            self.set_config_path(path)
        self._tracker_cfg = load_tracker_config(self._config_path)
        if self.tracker is not None:
            self.tracker.apply_config(self._tracker_cfg)
        self.log_message.emit(f"Config loaded: {self._config_path}")
        self.changed.emit()
        return self.get_config()

    def save_config(self, cfg: TrackerConfig, path: str | Path | None = None) -> Path:
        if path is not None:
            self.set_config_path(path)
        self._tracker_cfg = cfg
        saved = save_tracker_config(cfg, self._config_path)
        if self.tracker is not None:
            self.tracker.apply_config(cfg)
        self.log_message.emit(f"Config saved: {saved}")
        return saved

    def apply_config(self, cfg: TrackerConfig) -> None:
        self._tracker_cfg = cfg
        if self.tracker is not None:
            self.tracker.apply_config(cfg)
        self.log_message.emit("Config applied")

    def load_las(self, path: str) -> None:
        self.las = load_las_xyz_intensity(path)
        self.tracker = None
        self.model.xyz = self.las.xyz
        self.model.xy = self.las.xyz[:, :2]
        self.model.intensity = self.las.intensity
        self.model.point_cloud_revision += 1
        self.model.profile = None
        self.model.track_points = None
        self.model.current_point = None
        self.model.predicted_points = None
        self.model.trajectory_line_points = None
        self.model.profile_line_points = None
        self.model.stripe_segment_points = None
        self.model.stripe_edge_points = None
        self.model.search_box_points = None
        self.model.candidate_circle_groups = None
        self.model.selected_circle_points = None
        self.model.status_text = f"Loaded {len(self.las.xyz):,} points"
        self.log_message.emit(f"Loaded LAS: {path}, points={len(self.las.xyz):,}")
        self.changed.emit()

    def set_p0(self, x: float, y: float, z: float) -> None:
        self.model.p0 = np.array([x, y, z], dtype=float)
        self.changed.emit()

    def set_p1(self, x: float, y: float, z: float) -> None:
        self.model.p1 = np.array([x, y, z], dtype=float)
        self.changed.emit()

    def initialize_tracker(self) -> None:
        if self.las is None or self.model.p0 is None or self.model.p1 is None:
            raise RuntimeError("LAS, P0, P1 are required.")
        reuse_tracker = (
            self.tracker is not None
            and self.tracker.xyz is self.las.xyz
            and self.tracker.intensity is self.las.intensity
        )
        if reuse_tracker:
            self.tracker.apply_config(self._tracker_cfg)
            self.tracker.reset()
        else:
            self.tracker = BevGraphTracker(self.las.xyz, self.las.intensity, self._tracker_cfg)

        self.tracker.initialize(self.model.p0, self.model.p1)
        self._update_model_from_tracker(self.tracker.get_current_state(), self.tracker.get_last_debug_frame())
        self.model.status_text = f"Initialized | mode={self.tracker.get_current_state().mode.value}"
        self.log_message.emit(
            "Tracker initialized" + (" | reused segment tracker" if reuse_tracker else "")
        )
        if self.tracker._init_summary:
            s = self.tracker._init_summary
            self.log_message.emit(
                "SEGMENT_INIT | "
                f"reason={s.get('reason', 'na')} | "
                f"cand={s.get('candidate_count', 'na')} | "
                f"best={s.get('best_score', 'na')} | "
                f"intensity={s.get('intensity_score', 'na')} | "
                f"contrast={s.get('contrast_score', 'na')} | "
                f"autocorr={s.get('autocorr_score', 'na')} | "
                f"continuity={s.get('continuity', 'na')} | "
                f"endq={s.get('endpoint_evidence', 'na')} | "
                f"endd={s.get('endpoint_distance', 'na')} | "
                f"endl={s.get('endpoint_loyalty', 'na')}"
            )
        self._emit_state_log("INIT", self.tracker.get_current_state(), self.tracker.get_last_debug_frame())
        self.changed.emit()

    def run_step(self) -> DebugFrame:
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized.")
        if self.tracker.stop_reason != StopReason.NONE:
            stop_text = self.tracker.stop_reason.value
            self.model.status_text = f"Stopped | reason={stop_text}"
            self.log_message.emit(f"Step blocked: already stopped ({stop_text})")
            self.changed.emit()
            return self.tracker.get_last_debug_frame()
        dbg = self.tracker.step()
        self._update_model_from_tracker(self.tracker.get_current_state(), dbg)
        self.model.status_text = f"Step {dbg.step_index} | mode={self.tracker.get_current_state().mode.value} | stop={dbg.stop_reason or 'none'}"
        self.log_message.emit(f"Step {dbg.step_index}: mode={self.tracker.get_current_state().mode.value}, stop={dbg.stop_reason or 'none'}")
        self._emit_state_log(f"STEP {dbg.step_index}", self.tracker.get_current_state(), dbg)
        self.changed.emit()
        return dbg

    def run_full(self) -> TrackerResult:
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized.")
        result = self.tracker.run_full()
        self._update_model_from_tracker(self.tracker.get_current_state(), self.tracker.get_last_debug_frame())
        self.model.track_points = result.dense_points
        self.model.current_point = result.dense_points[-1] if len(result.dense_points) else None
        self.model.status_text = f"Run full done | stop={result.stop_reason} | output={len(result.output_points)}"
        self.log_message.emit(f"Run full done: stop_reason={result.stop_reason}, output_points={len(result.output_points)}")
        self.changed.emit()
        return result

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()
        self.model.track_points = None
        self.model.current_point = None
        self.model.predicted_points = None
        self.model.trajectory_line_points = None
        self.model.profile_line_points = None
        self.model.stripe_segment_points = None
        self.model.stripe_edge_points = None
        self.model.search_box_points = None
        self.model.candidate_circle_groups = None
        self.model.selected_circle_points = None
        self.model.profile = None
        self.model.status_text = "Reset"
        self.log_message.emit("Reset")
        self.changed.emit()

    def _update_model_from_tracker(self, state: TrackerState | None, dbg: DebugFrame | None) -> None:
        self.model.current_point = state.center_xyz.copy() if state is not None else None
        self.model.track_points = np.asarray(state.history_centers, dtype=float) if state is not None else None
        self.model.predicted_points = dbg.candidate_points if dbg is not None else None
        self.model.trajectory_line_points = dbg.trajectory_line_points if dbg is not None else None
        self.model.search_box_points = dbg.search_box_points if dbg is not None else None
        self.model.profile = dbg.profile if dbg is not None else None
        self.model.candidate_circle_groups = dbg.graph_edge_groups if dbg is not None else None
        self.model.selected_circle_points = None
        self._update_profile_overlay(state)

    def _update_profile_overlay(self, state: TrackerState | None) -> None:
        self.model.profile_line_points = None
        self.model.stripe_segment_points = None
        self.model.stripe_edge_points = None
        if state is None:
            return
        tangent = _safe_normalize(state.tangent_xy)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        center = np.asarray(state.center_xyz, dtype=np.float64)
        half_len = max(abs(float(state.left_edge_m)), abs(float(state.right_edge_m)), abs(float(state.lane_width_m)) * 0.75, 0.35)
        profile_pts = np.vstack(
            [
                center + np.array([normal[0] * -half_len, normal[1] * -half_len, 0.0], dtype=np.float64),
                center + np.array([normal[0] * half_len, normal[1] * half_len, 0.0], dtype=np.float64),
            ]
        )
        left_pt = center + np.array([normal[0] * float(state.left_edge_m), normal[1] * float(state.left_edge_m), 0.0], dtype=np.float64)
        right_pt = center + np.array([normal[0] * float(state.right_edge_m), normal[1] * float(state.right_edge_m), 0.0], dtype=np.float64)
        self.model.profile_line_points = profile_pts
        self.model.stripe_segment_points = np.vstack([left_pt, right_pt])
        self.model.stripe_edge_points = np.vstack([left_pt, right_pt])

    def _emit_state_log(self, label: str, state: TrackerState | None, dbg: DebugFrame | None) -> None:
        if state is None:
            return
        tan = _safe_normalize(state.tangent_xy)
        center = np.asarray(state.center_xyz, dtype=float)
        chosen = dbg.chosen_candidate if dbg is not None else None
        candidate_count = dbg.candidate_count if dbg is not None else 0
        sel_center = chosen.stripe_center_m if chosen is not None else None
        width = chosen.width_m if chosen is not None else state.lane_width_m
        q = chosen.total_score if chosen is not None else state.profile_quality
        continuity = chosen.continuity_score if chosen is not None else state.identity_confidence
        dash = state.dashed_score
        solid = state.solid_score
        crosswalk = state.crosswalk_score
        source = dbg.source if dbg is not None else "seed"
        self.log_message.emit(
            f"{label} | center=({center[0]:.3f}, {center[1]:.3f}) | tan=({tan[0]:.3f}, {tan[1]:.3f}) | "
            f"mode={state.mode.value} | cand={candidate_count} | sel_center={sel_center} | width={width} | "
            f"q={q:.3f} | continuity={continuity:.3f} | dash={dash:.3f} | solid={solid:.3f} | crosswalk={crosswalk:.3f}"
        )
        if dbg is not None:
            if chosen is not None:
                self.log_message.emit(
                    "OBS | "
                    f"src={source} | cand={candidate_count} | angle={chosen.angle_offset_deg:.1f} | "
                    f"intensity={chosen.intensity_score:.3f} | contrast={chosen.contrast_score:.3f} | "
                    f"ac={chosen.autocorr_score:.3f} | cont={chosen.continuity_score:.3f} | path={chosen.path_score:.3f}"
                )
            else:
                self.log_message.emit(
                    "OBS | "
                    f"src={source} | cand={candidate_count} | gap={state.gap_distance_m:.2f} | "
                    f"stop={dbg.stop_reason or 'none'}"
                )
            if dbg.candidate_summaries:
                self.log_message.emit("CAND | " + " || ".join(dbg.candidate_summaries[:3]))

def _safe_normalize(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v[:2], dtype=np.float64)
    n = float(np.linalg.norm(vv))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vv / n
