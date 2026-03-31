from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..tracker.config import DEFAULT_CONFIG_PATH, TrackerConfig
from .controller import TrackerController
from .pointcloud_view_widget import PointCloudViewWidget
from .profile_plot_widget import ProfilePlotWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        las_path: str | None = None,
        p0: list[float] | tuple[float, ...] | None = None,
        p1: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Lane Tracker V2")
        self.controller = TrackerController()
        self.view = PointCloudViewWidget()
        self.profile = ProfilePlotWidget()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)

        self.las_edit = QtWidgets.QLineEdit()
        self.las_edit.setPlaceholderText("Select a .las file")
        self.p0_edit = QtWidgets.QLineEdit("0 0 0")
        self.p1_edit = QtWidgets.QLineEdit("1 0 0")
        self.config_path_edit = QtWidgets.QLineEdit(str(self.controller.config_path))
        self.status = QtWidgets.QLabel("Ready")
        self.view_log_check = QtWidgets.QCheckBox("View Log")
        self.view_log_check.setChecked(False)
        self._config_collapsed = True

        self._config_widgets: dict[str, QtWidgets.QWidget] = {}

        central = QtWidgets.QSplitter()
        central.addWidget(self.view)
        central.addWidget(self._build_side_panel())
        central.setStretchFactor(0, 3)
        central.setStretchFactor(1, 2)
        central.setSizes([1000, 640])
        self.setCentralWidget(central)

        self._set_default_las_path(las_path)
        if p0 is not None:
            self.p0_edit.setText(" ".join(f"{float(v):.10f}" for v in p0))
        if p1 is not None:
            self.p1_edit.setText(" ".join(f"{float(v):.10f}" for v in p1))

        self._connect_signals()
        self._load_config_into_ui(self.controller.get_config())
        self._set_config_collapsed(True)

    def _build_side_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_data_group())
        layout.addWidget(self._build_config_group())
        layout.addWidget(self.status)
        layout.addWidget(self.profile, 1)
        layout.addWidget(self.log, 2)

        log_opts = QtWidgets.QHBoxLayout()
        log_opts.addStretch(1)
        log_opts.addWidget(self.view_log_check)
        layout.addLayout(log_opts)
        return panel

    def _build_data_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Inputs")
        form = QtWidgets.QFormLayout(group)

        btn_browse_las = QtWidgets.QPushButton("Browse...")
        las_row = QtWidgets.QHBoxLayout()
        las_row.addWidget(self.las_edit)
        las_row.addWidget(btn_browse_las)
        form.addRow("LAS", las_row)
        form.addRow("P0", self.p0_edit)
        form.addRow("P1", self.p1_edit)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load LAS")
        self.btn_init = QtWidgets.QPushButton("Initialize")
        self.btn_step = QtWidgets.QPushButton("Run One Step")
        self.btn_full = QtWidgets.QPushButton("Run Full")
        self.btn_reset = QtWidgets.QPushButton("Reset")
        for btn in [self.btn_load, self.btn_init, self.btn_step, self.btn_full, self.btn_reset]:
            btn_row.addWidget(btn)
        form.addRow(btn_row)

        self.btn_browse_las = btn_browse_las
        return group

    def _build_config_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("YAML Config")
        outer = QtWidgets.QVBoxLayout(group)
        outer.setContentsMargins(8, 8, 8, 8)

        self.btn_toggle_cfg = QtWidgets.QToolButton()
        self.btn_toggle_cfg.setText("YAML Config")
        self.btn_toggle_cfg.setCheckable(True)
        self.btn_toggle_cfg.setChecked(False)
        self.btn_toggle_cfg.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_toggle_cfg.setArrowType(QtCore.Qt.RightArrow)
        self.btn_toggle_cfg.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        outer.addWidget(self.btn_toggle_cfg)

        self.config_body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(self.config_body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        path_row = QtWidgets.QHBoxLayout()
        self.btn_browse_cfg = QtWidgets.QPushButton("Browse...")
        self.btn_reload_cfg = QtWidgets.QPushButton("Reload")
        self.btn_apply_cfg = QtWidgets.QPushButton("Apply")
        self.btn_save_cfg = QtWidgets.QPushButton("Save")
        path_row.addWidget(self.config_path_edit)
        path_row.addWidget(self.btn_browse_cfg)
        body_layout.addLayout(path_row)

        form = QtWidgets.QFormLayout()
        self._config_widgets["forward_distance_m"] = self._make_double_spin(0.05, 1.0, 0.01)
        self._config_widgets["graph_roi_forward_m"] = self._make_double_spin(0.5, 8.0, 0.1)
        self._config_widgets["graph_roi_lateral_half_m"] = self._make_double_spin(0.2, 3.0, 0.05)
        self._config_widgets["graph_cell_size_m"] = self._make_double_spin(0.01, 0.20, 0.01)
        self._config_widgets["graph_active_intensity_min"] = self._make_double_spin(0.0, 1.0, 0.01)
        self._config_widgets["graph_active_contrast_min"] = self._make_double_spin(0.0, 1.0, 0.01)
        self._config_widgets["graph_min_cell_points"] = self._make_int_spin(1, 20)
        self._config_widgets["graph_neighbor_max_distance_m"] = self._make_double_spin(0.05, 1.0, 0.01)
        self._config_widgets["graph_neighbor_lateral_limit_m"] = self._make_double_spin(0.05, 1.0, 0.01)
        self._config_widgets["segment_min_length_m"] = self._make_double_spin(0.05, 2.0, 0.01)
        self._config_widgets["segment_target_length_m"] = self._make_double_spin(0.05, 2.0, 0.01)
        self._config_widgets["segment_max_length_m"] = self._make_double_spin(0.05, 3.0, 0.01)
        self._config_widgets["segment_heading_gate_deg"] = self._make_double_spin(1.0, 90.0, 1.0)
        self._config_widgets["graph_beam_width"] = self._make_int_spin(1, 24)
        self._config_widgets["graph_beam_horizon_nodes"] = self._make_int_spin(1, 24)
        self._config_widgets["graph_beam_branching"] = self._make_int_spin(1, 16)
        self._config_widgets["graph_intensity_weight"] = self._make_double_spin(0.0, 1.0, 0.01)
        self._config_widgets["graph_contrast_weight"] = self._make_double_spin(0.0, 1.0, 0.01)
        self._config_widgets["graph_direction_weight"] = self._make_double_spin(0.0, 1.0, 0.01)
        self._config_widgets["graph_distance_weight"] = self._make_double_spin(0.0, 1.0, 0.01)
        self._config_widgets["graph_history_weight"] = self._make_double_spin(0.0, 1.0, 0.01)
        self._config_widgets["graph_period_weight"] = self._make_double_spin(0.0, 1.0, 0.01)
        self._config_widgets["graph_crosswalk_penalty"] = self._make_double_spin(0.0, 1.0, 0.01)
        self._config_widgets["use_z_clip"] = QtWidgets.QCheckBox("Enable")
        self._config_widgets["z_clip_half_range_m"] = self._make_double_spin(0.05, 1.0, 0.01)
        self._config_widgets["gap_forward_distance_m"] = self._make_double_spin(0.5, 20.0, 0.1)
        self._config_widgets["continuity_node_count"] = self._make_int_spin(2, 20)
        self._config_widgets["continuity_strength"] = self._make_double_spin(0.0, 10.0, 0.1)
        self._config_widgets["crosswalk_stop_enabled"] = QtWidgets.QCheckBox("Enable")

        form.addRow("Forward Distance (m)", self._config_widgets["forward_distance_m"])
        form.addRow("Segment ROI Forward (m)", self._config_widgets["graph_roi_forward_m"])
        form.addRow("Segment ROI Half Width (m)", self._config_widgets["graph_roi_lateral_half_m"])
        form.addRow("BEV Cell Size (m)", self._config_widgets["graph_cell_size_m"])
        form.addRow("Active Intensity Min", self._config_widgets["graph_active_intensity_min"])
        form.addRow("Active Contrast Min", self._config_widgets["graph_active_contrast_min"])
        form.addRow("Min Cell Points", self._config_widgets["graph_min_cell_points"])
        form.addRow("Segment Min Length (m)", self._config_widgets["segment_min_length_m"])
        form.addRow("Segment Target Length (m)", self._config_widgets["segment_target_length_m"])
        form.addRow("Segment Max Length (m)", self._config_widgets["segment_max_length_m"])
        form.addRow("Segment Heading Gate (deg)", self._config_widgets["segment_heading_gate_deg"])
        form.addRow("Segment Max Gap (m)", self._config_widgets["graph_neighbor_max_distance_m"])
        form.addRow("Segment Lateral Limit (m)", self._config_widgets["graph_neighbor_lateral_limit_m"])
        form.addRow("Beam Width", self._config_widgets["graph_beam_width"])
        form.addRow("Beam Horizon Nodes", self._config_widgets["graph_beam_horizon_nodes"])
        form.addRow("Beam Branching", self._config_widgets["graph_beam_branching"])
        form.addRow("Intensity Weight", self._config_widgets["graph_intensity_weight"])
        form.addRow("Contrast Weight", self._config_widgets["graph_contrast_weight"])
        form.addRow("Direction Weight", self._config_widgets["graph_direction_weight"])
        form.addRow("Distance Weight", self._config_widgets["graph_distance_weight"])
        form.addRow("History Weight", self._config_widgets["graph_history_weight"])
        form.addRow("Period Weight", self._config_widgets["graph_period_weight"])
        form.addRow("Crosswalk Penalty", self._config_widgets["graph_crosswalk_penalty"])
        form.addRow("Use Z Clip", self._config_widgets["use_z_clip"])
        form.addRow("Z Clip Range (+/- m)", self._config_widgets["z_clip_half_range_m"])
        form.addRow("Gap Distance Limit (m)", self._config_widgets["gap_forward_distance_m"])
        form.addRow("Continuity Nodes", self._config_widgets["continuity_node_count"])
        form.addRow("Continuity Strength", self._config_widgets["continuity_strength"])
        form.addRow("Crosswalk Stop", self._config_widgets["crosswalk_stop_enabled"])
        body_layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btn_reload_cfg)
        btn_row.addWidget(self.btn_apply_cfg)
        btn_row.addWidget(self.btn_save_cfg)
        body_layout.addLayout(btn_row)
        outer.addWidget(self.config_body)
        return group

    def _connect_signals(self) -> None:
        self.btn_toggle_cfg.toggled.connect(self._set_config_expanded)
        self.btn_browse_las.clicked.connect(self.on_browse_las)
        self.btn_browse_cfg.clicked.connect(self.on_browse_config)
        self.btn_load.clicked.connect(self.on_load)
        self.btn_init.clicked.connect(self.on_init)
        self.btn_step.clicked.connect(self.on_step)
        self.btn_full.clicked.connect(self.on_full)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_reload_cfg.clicked.connect(self.on_reload_config)
        self.btn_apply_cfg.clicked.connect(self.on_apply_config)
        self.btn_save_cfg.clicked.connect(self.on_save_config)

        self.controller.changed.connect(self.refresh)
        self.controller.log_message.connect(self.log.appendPlainText)
        self.view.point_context_menu_requested.connect(self._open_point_context_menu)
        self.view.debug_message.connect(lambda msg: self.log.appendPlainText(f"VIEW: {msg}"))
        self.view_log_check.toggled.connect(self.view.set_view_log_enabled)

    def _set_default_las_path(self, las_path: str | None) -> None:
        if las_path:
            self.las_edit.setText(las_path)
            return
        if self.las_edit.text().strip():
            return
        data_dir = Path(__file__).resolve().parents[1] / "data"
        candidates = sorted(data_dir.glob("*.las"))
        if candidates:
            self.las_edit.setText(str(candidates[0]))

    def _make_double_spin(self, minimum: float, maximum: float, step: float) -> QtWidgets.QDoubleSpinBox:
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(minimum, maximum)
        w.setSingleStep(step)
        w.setDecimals(3)
        return w

    def _make_int_spin(self, minimum: int, maximum: int) -> QtWidgets.QSpinBox:
        w = QtWidgets.QSpinBox()
        w.setRange(minimum, maximum)
        return w

    def _config_from_ui(self) -> TrackerConfig:
        cfg = TrackerConfig()
        cfg.forward_distance_m = self._config_widgets["forward_distance_m"].value()
        cfg.graph_roi_forward_m = self._config_widgets["graph_roi_forward_m"].value()
        cfg.graph_roi_lateral_half_m = self._config_widgets["graph_roi_lateral_half_m"].value()
        cfg.graph_cell_size_m = self._config_widgets["graph_cell_size_m"].value()
        cfg.graph_active_intensity_min = self._config_widgets["graph_active_intensity_min"].value()
        cfg.graph_active_contrast_min = self._config_widgets["graph_active_contrast_min"].value()
        cfg.graph_min_cell_points = self._config_widgets["graph_min_cell_points"].value()
        cfg.graph_neighbor_max_distance_m = self._config_widgets["graph_neighbor_max_distance_m"].value()
        cfg.graph_neighbor_lateral_limit_m = self._config_widgets["graph_neighbor_lateral_limit_m"].value()
        cfg.segment_min_length_m = self._config_widgets["segment_min_length_m"].value()
        cfg.segment_target_length_m = self._config_widgets["segment_target_length_m"].value()
        cfg.segment_max_length_m = self._config_widgets["segment_max_length_m"].value()
        cfg.segment_heading_gate_deg = self._config_widgets["segment_heading_gate_deg"].value()
        cfg.graph_beam_width = self._config_widgets["graph_beam_width"].value()
        cfg.graph_beam_horizon_nodes = self._config_widgets["graph_beam_horizon_nodes"].value()
        cfg.graph_beam_branching = self._config_widgets["graph_beam_branching"].value()
        cfg.graph_intensity_weight = self._config_widgets["graph_intensity_weight"].value()
        cfg.graph_contrast_weight = self._config_widgets["graph_contrast_weight"].value()
        cfg.graph_direction_weight = self._config_widgets["graph_direction_weight"].value()
        cfg.graph_distance_weight = self._config_widgets["graph_distance_weight"].value()
        cfg.graph_history_weight = self._config_widgets["graph_history_weight"].value()
        cfg.graph_period_weight = self._config_widgets["graph_period_weight"].value()
        cfg.graph_crosswalk_penalty = self._config_widgets["graph_crosswalk_penalty"].value()
        cfg.use_z_clip = self._config_widgets["use_z_clip"].isChecked()
        cfg.z_clip_half_range_m = self._config_widgets["z_clip_half_range_m"].value()
        cfg.gap_forward_distance_m = self._config_widgets["gap_forward_distance_m"].value()
        cfg.continuity_node_count = self._config_widgets["continuity_node_count"].value()
        cfg.continuity_strength = self._config_widgets["continuity_strength"].value()
        cfg.crosswalk_stop_enabled = self._config_widgets["crosswalk_stop_enabled"].isChecked()
        return cfg

    def _load_config_into_ui(self, cfg: TrackerConfig) -> None:
        self._config_widgets["forward_distance_m"].setValue(cfg.forward_distance_m)
        self._config_widgets["graph_roi_forward_m"].setValue(cfg.graph_roi_forward_m)
        self._config_widgets["graph_roi_lateral_half_m"].setValue(cfg.graph_roi_lateral_half_m)
        self._config_widgets["graph_cell_size_m"].setValue(cfg.graph_cell_size_m)
        self._config_widgets["graph_active_intensity_min"].setValue(cfg.graph_active_intensity_min)
        self._config_widgets["graph_active_contrast_min"].setValue(cfg.graph_active_contrast_min)
        self._config_widgets["graph_min_cell_points"].setValue(cfg.graph_min_cell_points)
        self._config_widgets["graph_neighbor_max_distance_m"].setValue(cfg.graph_neighbor_max_distance_m)
        self._config_widgets["graph_neighbor_lateral_limit_m"].setValue(cfg.graph_neighbor_lateral_limit_m)
        self._config_widgets["segment_min_length_m"].setValue(cfg.segment_min_length_m)
        self._config_widgets["segment_target_length_m"].setValue(cfg.segment_target_length_m)
        self._config_widgets["segment_max_length_m"].setValue(cfg.segment_max_length_m)
        self._config_widgets["segment_heading_gate_deg"].setValue(cfg.segment_heading_gate_deg)
        self._config_widgets["graph_beam_width"].setValue(cfg.graph_beam_width)
        self._config_widgets["graph_beam_horizon_nodes"].setValue(cfg.graph_beam_horizon_nodes)
        self._config_widgets["graph_beam_branching"].setValue(cfg.graph_beam_branching)
        self._config_widgets["graph_intensity_weight"].setValue(cfg.graph_intensity_weight)
        self._config_widgets["graph_contrast_weight"].setValue(cfg.graph_contrast_weight)
        self._config_widgets["graph_direction_weight"].setValue(cfg.graph_direction_weight)
        self._config_widgets["graph_distance_weight"].setValue(cfg.graph_distance_weight)
        self._config_widgets["graph_history_weight"].setValue(cfg.graph_history_weight)
        self._config_widgets["graph_period_weight"].setValue(cfg.graph_period_weight)
        self._config_widgets["graph_crosswalk_penalty"].setValue(cfg.graph_crosswalk_penalty)
        self._config_widgets["use_z_clip"].setChecked(cfg.use_z_clip)
        self._config_widgets["z_clip_half_range_m"].setValue(cfg.z_clip_half_range_m)
        self._config_widgets["gap_forward_distance_m"].setValue(cfg.gap_forward_distance_m)
        self._config_widgets["continuity_node_count"].setValue(cfg.continuity_node_count)
        self._config_widgets["continuity_strength"].setValue(cfg.continuity_strength)
        self._config_widgets["crosswalk_stop_enabled"].setChecked(cfg.crosswalk_stop_enabled)

    def _set_config_expanded(self, expanded: bool) -> None:
        self._set_config_collapsed(not expanded)

    def _set_config_collapsed(self, collapsed: bool) -> None:
        self._config_collapsed = bool(collapsed)
        if hasattr(self, "config_body"):
            self.config_body.setVisible(not collapsed)
        if hasattr(self, "btn_toggle_cfg"):
            self.btn_toggle_cfg.blockSignals(True)
            self.btn_toggle_cfg.setChecked(not collapsed)
            self.btn_toggle_cfg.setArrowType(QtCore.Qt.RightArrow if collapsed else QtCore.Qt.DownArrow)
            self.btn_toggle_cfg.blockSignals(False)

    def on_browse_las(self) -> None:
        start_dir = str((Path(self.las_edit.text()).parent if self.las_edit.text().strip() else Path(__file__).resolve().parents[1] / "data"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select LAS", start_dir, "LAS Files (*.las);;All Files (*)")
        if path:
            self.las_edit.setText(path)

    def on_browse_config(self) -> None:
        start_path = self.config_path_edit.text().strip() or str(DEFAULT_CONFIG_PATH)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select YAML Config", start_path, "YAML Files (*.yaml *.yml)")
        if path:
            self.config_path_edit.setText(path)

    def on_load(self) -> None:
        path = self.las_edit.text().strip()
        if not path:
            self._show_error("LAS path is empty.")
            return
        if not Path(path).exists():
            self._show_error(f"LAS file not found:\n{path}")
            return
        self._run_action("Loading LAS...", lambda: self.controller.load_las(path))

    def on_init(self) -> None:
        try:
            p0 = self._parse_xyz_text(self.p0_edit.text())
            p1 = self._parse_xyz_text(self.p1_edit.text())
        except ValueError as exc:
            self._show_error(f"Invalid P0/P1 format.\nUse: x y z\n\n{exc}")
            return
        self.log.clear()
        if self._run_action("Initializing tracker...", lambda: self._initialize_with_points(p0, p1)):
            self.view.focus_on_point(self.controller.model.current_point)

    def on_step(self) -> None:
        if self._run_action("Running one step...", self.controller.run_step):
            self.view.focus_on_point(self.controller.model.current_point)

    def on_full(self) -> None:
        self._run_action("Running full tracker...", self.controller.run_full)

    def on_reset(self) -> None:
        self.log.clear()
        self.controller.reset()

    def on_reload_config(self) -> None:
        path = self.config_path_edit.text().strip() or str(DEFAULT_CONFIG_PATH)
        if self._run_action("Reloading config...", lambda: self._reload_config(path)):
            self.status.setText("Config reloaded")

    def on_apply_config(self) -> None:
        cfg = self._config_from_ui()
        self.controller.apply_config(cfg)
        self.status.setText("Config applied")

    def on_save_config(self) -> None:
        cfg = self._config_from_ui()
        path = self.config_path_edit.text().strip() or str(DEFAULT_CONFIG_PATH)
        if self._run_action("Saving config...", lambda: self.controller.save_config(cfg, path)):
            self.status.setText("Config saved")

    def _reload_config(self, path: str) -> None:
        cfg = self.controller.load_config(path)
        self.config_path_edit.setText(str(self.controller.config_path))
        self._load_config_into_ui(cfg)

    def _initialize_with_points(self, p0: list[float], p1: list[float]) -> None:
        self.controller.set_p0(*p0)
        self.controller.set_p1(*p1)
        self.controller.apply_config(self._config_from_ui())
        self.controller.initialize_tracker()

    def _parse_xyz_text(self, text: str) -> list[float]:
        parts = text.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError("Need exactly 3 numbers.")
        return [float(x) for x in parts]

    def _run_action(self, status_text: str, fn) -> bool:
        self.status.setText(status_text)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        try:
            QtWidgets.QApplication.processEvents()
            fn()
            return True
        except Exception as exc:
            self._show_error(str(exc))
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _show_error(self, message: str) -> None:
        self.log.appendPlainText(f"ERROR: {message}")
        self.status.setText("Error")
        QtWidgets.QMessageBox.critical(self, "Lane Tracker V2", message)

    def _open_point_context_menu(self, point_xyz, global_pos) -> None:
        menu = QtWidgets.QMenu(self)
        if point_xyz is not None:
            xyz_text = ", ".join(f"{float(v):.6f}" for v in point_xyz)
            title = QtGui.QAction(f"Point: {xyz_text}", menu)
            title.setEnabled(False)
            menu.addAction(title)
            menu.addSeparator()
            act_set_p0 = menu.addAction("Set P0 Here")
            act_set_p1 = menu.addAction("Set P1 Here")
            act_copy_xyz = menu.addAction("Copy XYZ")
            menu.addSeparator()
            act_init = menu.addAction("Initialize From P0 / P1")
        else:
            disabled = QtGui.QAction("No nearby point", menu)
            disabled.setEnabled(False)
            menu.addAction(disabled)
            menu.addSeparator()
            act_set_p0 = None
            act_set_p1 = None
            act_copy_xyz = None
            act_init = menu.addAction("Initialize From P0 / P1")

        menu.addSeparator()
        act_reset_view = menu.addAction("Reset Camera / View")

        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen == act_set_p0 and point_xyz is not None:
            self._apply_seed_point(self.p0_edit, point_xyz, "P0")
        elif chosen == act_set_p1 and point_xyz is not None:
            self._apply_seed_point(self.p1_edit, point_xyz, "P1")
        elif chosen == act_copy_xyz and point_xyz is not None:
            QtWidgets.QApplication.clipboard().setText(" ".join(f"{float(v):.10f}" for v in point_xyz))
            self.status.setText("Copied point XYZ")
        elif chosen == act_init:
            self.on_init()
        elif chosen == act_reset_view:
            self.view.reset_view()

    def _apply_seed_point(self, target: QtWidgets.QLineEdit, point_xyz, label: str) -> None:
        vals = [float(v) for v in point_xyz]
        target.setText(" ".join(f"{v:.10f}" for v in vals))
        if label == "P0":
            self.controller.set_p0(*vals)
        elif label == "P1":
            self.controller.set_p1(*vals)
        self.status.setText(f"{label} updated from point cloud")
        self.log.appendPlainText(f"{label} <- {vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}")

    def refresh(self) -> None:
        m = self.controller.model
        if m.xyz is not None:
            self.view.set_point_cloud(m.xyz, m.intensity, revision=m.point_cloud_revision)
        self.view.set_seed_points(m.p0, m.p1, render=False)
        self.view.set_track(m.track_points, render=False)
        self.view.set_current(m.current_point, render=False)
        self.view.set_predicted(m.predicted_points, render=False)
        self.view.set_trajectory_line(m.trajectory_line_points, render=False)
        self.view.set_profile_overlay(m.profile_line_points, m.stripe_segment_points, m.stripe_edge_points, render=False)
        self.view.set_search_box(m.search_box_points, render=False)
        self.view.set_candidate_circles(m.candidate_circle_groups, m.selected_circle_points, render=False)
        self.view.render()
        self.profile.update_profile(m.profile)
        self.status.setText(m.status_text or "Ready")
