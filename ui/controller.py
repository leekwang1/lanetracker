from __future__ import annotations

from PySide6 import QtCore
import numpy as np

from ..io.las_io import load_las_xyz_intensity
from ..tracker.lane_tracker_v2 import LaneTrackerV2, TrackerV2Config
from ..core.types import StopReason
from .viewer_model import ViewerModel


class TrackerController(QtCore.QObject):
    changed = QtCore.Signal()
    log_message = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.model = ViewerModel()
        self.las = None
        self.tracker = None
        self._tracker_cfg = TrackerV2Config()
        self._detail_log_enabled = False

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
        self.model.status_text = f"Loaded {len(self.las.xyz):,} points"
        self.log_message.emit(f"Loaded LAS: {path}, points={len(self.las.xyz)}")
        self.changed.emit()

    def set_p0(self, x: float, y: float, z: float) -> None:
        self.model.p0 = np.array([x, y, z], dtype=float)
        self.changed.emit()

    def set_p1(self, x: float, y: float, z: float) -> None:
        self.model.p1 = np.array([x, y, z], dtype=float)
        self.changed.emit()

    def initialize_tracker(self) -> None:
        if self.las is None or self.model.p0 is None or self.model.p1 is None:
            raise RuntimeError("LAS, P0, P1 are required")
        reuse_tracker = (
            self.tracker is not None
            and self.tracker.xyz is self.las.xyz
            and self.tracker.intensity is self.las.intensity
        )
        if reuse_tracker:
            self.tracker.reset()
        else:
            self.tracker = LaneTrackerV2(self.las.xyz, self.las.intensity, self._tracker_cfg)
        self.tracker.initialize(self.model.p0, self.model.p1)
        state = self.tracker.get_current_state()
        self.model.current_point = state.center_xyz.copy() if state is not None else None
        self.model.track_points = np.asarray(state.history_centers, dtype=float) if state is not None else None
        self.model.predicted_points = None
        self.model.trajectory_line_points = (
            np.asarray(self.tracker.debug_frames[-1].trajectory_line_points, dtype=float)
            if self.tracker.debug_frames and self.tracker.debug_frames[-1].trajectory_line_points is not None
            else None
        )
        self.model.profile = self.tracker.debug_frames[-1].cross_section_profile if self.tracker.debug_frames else None
        self._update_profile_overlay(state, self.model.profile)
        self._update_search_box_overlay(state, self.tracker.debug_frames[-1] if self.tracker.debug_frames else None)
        self.model.status_text = f"Initialized | mode={state.mode.value if state is not None else 'unknown'}"
        self.log_message.emit(
            "Tracker initialized"
            + (" | reused spatial grid" if reuse_tracker else "")
        )
        if state is not None:
            self._emit_state_debug_log("INIT", state, self.model.profile, self.tracker.debug_frames[-1] if self.tracker.debug_frames else None)
        self.changed.emit()

    def run_step(self):
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized")
        if self.tracker.stop_reason != StopReason.NONE:
            state = self.tracker.get_current_state()
            stop_text = self.tracker.stop_reason.value
            self.model.status_text = f"Stopped | reason={stop_text}"
            self.log_message.emit(f"Step blocked: already stopped ({stop_text})")
            self.changed.emit()
            return self.tracker.debug_frames[-1] if self.tracker.debug_frames else None
        dbg = self.tracker.step()
        state = self.tracker.get_current_state()
        self.model.current_point = state.center_xyz.copy() if state is not None else None
        self.model.track_points = np.asarray(state.history_centers, dtype=float) if state is not None else None
        self.model.predicted_points = np.asarray(dbg.predicted_centers, dtype=float) if dbg.predicted_centers else None
        self.model.trajectory_line_points = (
            np.asarray(dbg.trajectory_line_points, dtype=float)
            if dbg.trajectory_line_points is not None
            else None
        )
        self.model.profile = dbg.cross_section_profile
        self._update_profile_overlay(state, dbg.cross_section_profile)
        self._update_search_box_overlay(state, dbg)
        self.model.status_text = f"Step {dbg.step_index} | mode={dbg.mode} | stop={dbg.stop_reason or 'none'}"
        self.log_message.emit(f"Step {dbg.step_index}: mode={dbg.mode}, stop={dbg.stop_reason or 'none'}")
        if state is not None:
            self._emit_state_debug_log(f"STEP {dbg.step_index}", state, dbg.cross_section_profile, dbg)
        self.changed.emit()
        return dbg

    def run_full(self):
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized")
        result = self.tracker.run_full()
        self.model.track_points = result.dense_points
        self.model.current_point = result.dense_points[-1] if len(result.dense_points) else None
        self.model.predicted_points = None
        self.model.trajectory_line_points = (
            np.asarray(result.debug_frames[-1].trajectory_line_points, dtype=float)
            if result.debug_frames and result.debug_frames[-1].trajectory_line_points is not None
            else None
        )
        self.model.profile = result.debug_frames[-1].cross_section_profile if result.debug_frames else None
        self._update_profile_overlay(self.tracker.get_current_state(), self.model.profile)
        self._update_search_box_overlay(self.tracker.get_current_state(), result.debug_frames[-1] if result.debug_frames else None)
        self.model.status_text = f"Run full done | stop={result.stop_reason} | output={len(result.output_points)}"
        self.log_message.emit(f"Run full done: stop_reason={result.stop_reason}, output_points={len(result.output_points)}")
        self.changed.emit()
        return result

    def reset(self):
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
        self.model.profile = None
        self.model.status_text = "Reset"
        self.log_message.emit("Reset")
        self.changed.emit()

    def set_detail_log_enabled(self, enabled: bool) -> None:
        self._detail_log_enabled = bool(enabled)
        self.log_message.emit(f"Detail log {'enabled' if enabled else 'disabled'}")

    def _emit_state_debug_log(self, label: str, state, profile, dbg) -> None:
        def _fmt_optional(value, digits: int = 3) -> str:
            if value is None:
                return "na"
            try:
                return f"{float(value):.{digits}f}"
            except Exception:
                return str(value)

        tangent = np.asarray(state.tangent_xy, dtype=float)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-9:
            tangent = tangent / tangent_norm
        center = np.asarray(state.center_xyz, dtype=float)

        selected_line = "selected_stripe=None"
        candidate_count = 0
        selected_idx = None
        selected_center_m = None
        selected_width_m = None
        selected_final = None
        if profile is not None:
            candidate_count = len(profile.stripe_candidates)
            selected_idx = profile.selected_idx
            if profile.selected_idx is not None and 0 <= profile.selected_idx < candidate_count:
                sc = profile.stripe_candidates[profile.selected_idx]
                selected_center_m = float(sc.center_m)
                selected_width_m = float(sc.width_m)
                selected_final = float(sc.final_score)
                selected_line = (
                    "selected_stripe="
                    f"idx={profile.selected_idx}, "
                    f"center={float(sc.center_m):.4f}, "
                    f"left={float(sc.left_m):.4f}, "
                    f"right={float(sc.right_m):.4f}, "
                    f"width={float(sc.width_m):.4f}, "
                    f"peak={float(sc.peak_value):.4f}, "
                    f"identity={float(sc.identity_score):.4f}, "
                    f"final={float(sc.final_score):.4f}"
                )

        predicted_count = len(dbg.predicted_centers) if dbg is not None and getattr(dbg, "predicted_centers", None) else 0
        score_text = "[]"
        if dbg is not None and getattr(dbg, "hypothesis_scores", None):
            score_text = "[" + ", ".join(f"{float(v):.3f}" for v in dbg.hypothesis_scores[:5]) + "]"

        history_len = len(getattr(state, "history_centers", []))
        lateral_half = (
            float(self.tracker.cfg.init_search_lateral_half_m)
            if history_len <= 1
            else float(self.tracker.cfg.step_search_lateral_half_m)
        ) if self.tracker is not None else 0.0
        along_half = float(self.tracker.cfg.search_along_half_m) if self.tracker is not None else 0.0
        if dbg is not None:
            along_half = float(getattr(dbg, "search_along_half_m", along_half) or along_half)
            lateral_half = float(getattr(dbg, "search_lateral_half_m", lateral_half) or lateral_half)

        lane_loyalty = 0.0
        debug_last = {}
        if dbg is not None:
            debug_last = getattr(self.tracker.hypotheses[0], "debug_last", {}) if self.tracker and self.tracker.hypotheses else {}
            lane_loyalty = float(debug_last.get("lane_loyalty", 0.0))

        stripe_status = "none"
        if selected_center_m is not None:
            stripe_status = "rejected" if (dbg is not None and getattr(dbg, "stripe_rejected", False)) else "used"

        lines = [
            (
                f"{label} | "
                f"center=({center[0]:.3f}, {center[1]:.3f}) | "
                f"tan=({tangent[0]:.3f}, {tangent[1]:.3f}) | "
                f"mode={state.mode.value} | "
                f"cand={candidate_count} | "
                f"sel_center={selected_center_m if selected_center_m is not None else 'None'} | "
                f"width={selected_width_m if selected_width_m is not None else 'None'} | "
                f"q={float(state.profile_quality):.3f} | "
                f"loyalty={lane_loyalty:.3f} | "
                f"switch={float(profile.switch_risk if profile is not None else 0.0):.3f} | "
                f"stripe={stripe_status}"
            ),
        ]
        if self._detail_log_enabled:
            lines.append(
                "scores="
                f"{score_text} | "
                f"conf=({float(state.center_confidence):.3f}, {float(state.identity_confidence):.3f}) | "
                f"strength={float(state.stripe_strength):.3f} | "
                f"pred={predicted_count} | "
                f"sel_idx={selected_idx} | "
                f"state_center_m={float(state.stripe_center_m):.4f} | "
                f"strip=({along_half:.2f}, {lateral_half:.2f})"
            )
        if dbg is not None:
            obs_debug = debug_last.get("obs_debug", {}) if isinstance(debug_last, dict) else {}
            asc_debug = debug_last.get("asc_debug", {}) if isinstance(debug_last, dict) else {}
            upd_debug = debug_last.get("upd_debug", {}) if isinstance(debug_last, dict) else {}
            if obs_debug:
                lines.append(
                    "OBS | "
                    f"src={obs_debug.get('source', 'primary')} | "
                    f"cand={int(obs_debug.get('candidate_count', 0))} | "
                    f"sel_idx={obs_debug.get('selected_idx', 'None')} | "
                    f"center={obs_debug.get('selected_center_m', 'None')} | "
                    f"width={obs_debug.get('selected_width_m', 'None')} | "
                    f"signal={obs_debug.get('selected_signal', 'None')}"
                )
            if asc_debug:
                reasons = asc_debug.get("reasons", []) or []
                reason_text = ",".join(str(v) for v in reasons) if reasons else "accepted"
                lines.append(
                    "ASC | "
                    f"src={asc_debug.get('source', 'primary')} | "
                    f"used={'yes' if stripe_status == 'used' else 'no'} | "
                    f"reliable={asc_debug.get('reliable_passed', False)} | "
                    f"loyalty_pre={_fmt_optional(asc_debug.get('loyalty_pre'))} | "
                    f"loyalty_post={_fmt_optional(asc_debug.get('loyalty_post'))} | "
                    f"reasons={reason_text}"
                )
            if upd_debug:
                tan_before = upd_debug.get("tangent_before_xy", [0.0, 0.0])
                tan_after = upd_debug.get("tangent_after_xy", [0.0, 0.0])
                lines.append(
                    "UPD | "
                    f"shift=({float(upd_debug.get('profile_shift_m', 0.0)):.4f}, "
                    f"{float(upd_debug.get('refine_shift_m', 0.0)):.4f}, "
                    f"{float(upd_debug.get('cross_shift_m', 0.0)):.4f}) | "
                    f"total={float(upd_debug.get('total_shift_m', 0.0)):.4f} | "
                    f"tan=({float(tan_before[0]):.3f}, {float(tan_before[1]):.3f})"
                    f"->({float(tan_after[0]):.3f}, {float(tan_after[1]):.3f})"
                )
            lines.append(
                "debug="
                f"stop={dbg.stop_reason or 'none'}, "
                f"crosswalk={float(dbg.crosswalk_score):.3f}, "
                f"dbg_switch={float(dbg.switch_risk):.3f}"
            )
            if self._detail_log_enabled:
                qd = getattr(dbg, "query_debug", {}) or {}
                lines.append(
                    "query="
                    f"count={int(qd.get('count', 0))}, "
                    f"I50={float(qd.get('intensity_q50', 0.0)):.1f}, "
                    f"I90={float(qd.get('intensity_q90', 0.0)):.1f}"
                )
                cand_rows = getattr(dbg, "candidate_summaries", []) or []
                if cand_rows:
                    lines.append("top_candidates=" + " || ".join(cand_rows[:3]))
        self.log_message.emit("\n".join(lines))

    def _update_profile_overlay(self, state, profile) -> None:
        self.model.profile_line_points = None
        self.model.stripe_segment_points = None
        self.model.stripe_edge_points = None
        if state is None:
            return

        tangent = np.asarray(state.tangent_xy, dtype=float)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-9:
            return
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        center = np.asarray(state.center_xyz, dtype=float)
        center3 = center.copy() if center.shape[0] == 3 else np.array([center[0], center[1], 0.0], dtype=float)

        half_len = max(abs(float(state.left_edge_m)), abs(float(state.right_edge_m)), abs(float(state.lane_width_m)) * 0.75, 0.5)
        if profile is not None and profile.bins_center.size:
            half_len = max(half_len, float(np.max(np.abs(profile.bins_center))) + 0.05)

        profile_pts = np.vstack(
            [
                center3 + np.array([normal[0] * -half_len, normal[1] * -half_len, 0.0], dtype=float),
                center3 + np.array([normal[0] * half_len, normal[1] * half_len, 0.0], dtype=float),
            ]
        )
        left_pt = center3 + np.array([normal[0] * float(state.left_edge_m), normal[1] * float(state.left_edge_m), 0.0], dtype=float)
        right_pt = center3 + np.array([normal[0] * float(state.right_edge_m), normal[1] * float(state.right_edge_m), 0.0], dtype=float)

        self.model.profile_line_points = profile_pts
        self.model.stripe_segment_points = np.vstack([left_pt, right_pt])
        self.model.stripe_edge_points = np.vstack([left_pt, right_pt])

    def _update_search_box_overlay(self, state, dbg=None) -> None:
        self.model.search_box_points = None
        if state is None or self.tracker is None:
            return
        tangent = np.asarray(state.tangent_xy, dtype=float)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-9:
            return
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        center = np.asarray(state.center_xyz, dtype=float)
        center3 = center.copy() if center.shape[0] == 3 else np.array([center[0], center[1], 0.0], dtype=float)
        along = float(self.tracker.cfg.search_along_half_m)
        history_len = len(getattr(state, "history_centers", []))
        if history_len <= 1:
            lateral = float(self.tracker.cfg.init_search_lateral_half_m)
        else:
            lateral = float(self.tracker.cfg.step_search_lateral_half_m)
        if dbg is not None:
            along = float(getattr(dbg, "search_along_half_m", along) or along)
            lateral = float(getattr(dbg, "search_lateral_half_m", lateral) or lateral)
        t3 = np.array([tangent[0], tangent[1], 0.0], dtype=float)
        n3 = np.array([normal[0], normal[1], 0.0], dtype=float)
        corners = np.vstack(
            [
                center3 + t3 * along + n3 * lateral,
                center3 + t3 * along - n3 * lateral,
                center3 - t3 * along - n3 * lateral,
                center3 - t3 * along + n3 * lateral,
                center3 + t3 * along + n3 * lateral,
            ]
        )
        self.model.search_box_points = corners
