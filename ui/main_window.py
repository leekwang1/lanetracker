from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..tracker.config import CONFIG_GROUPS, DEFAULT_CONFIG_PATH, TrackerConfig
from .controller import TrackerController
from .pointcloud_view_widget import PointCloudViewWidget
from .profile_plot_widget import ProfilePlotWidget


CONFIG_WIDGET_SPECS: dict[str, tuple] = {
    "forward_distance_m": ("float", 0.05, 1.0, 0.01),
    "max_track_length_m": ("float", 1.0, 500.0, 1.0),
    "max_gap_distance_m": ("float", 0.1, 20.0, 0.1),
    "roi_forward_m": ("float", 0.5, 8.0, 0.1),
    "roi_backward_m": ("float", 0.0, 2.0, 0.05),
    "roi_lateral_half_m": ("float", 0.2, 3.0, 0.05),
    "corridor_half_width_m": ("float", 0.05, 1.0, 0.01),
    "grid_cell_size_m": ("float", 0.01, 0.20, 0.01),
    "active_intensity_min": ("float", 0.0, 1.0, 0.01),
    "min_points_per_cell": ("int", 1, 20),
    "component_min_cells": ("int", 1, 100),
    "component_min_span_m": ("float", 0.05, 3.0, 0.01),
    "stripe_width_m": ("float", 0.03, 1.0, 0.01),
    "candidate_lateral_sigma_m": ("float", 0.01, 0.50, 0.01),
    "candidate_heading_sigma_deg": ("float", 1.0, 45.0, 1.0),
    "candidate_min_score": ("float", 0.0, 1.0, 0.01),
    "heading_smoothing_alpha": ("float", 0.0, 1.0, 0.01),
    "max_heading_change_deg": ("float", 0.1, 45.0, 0.1),
    "use_z_clip": ("bool",),
    "z_clip_half_range_m": ("float", 0.05, 1.0, 0.01),
}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        las_path: str | None = None,
        p0: list[float] | tuple[float, ...] | None = None,
        p1: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.setWindowTitle("차선 추적기 V2")
        self.controller = TrackerController()
        self.view = PointCloudViewWidget()
        self.profile = ProfilePlotWidget()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)

        self.las_edit = QtWidgets.QLineEdit()
        self.las_edit.setPlaceholderText(".las 파일을 선택하세요")
        self.p0_edit = QtWidgets.QLineEdit("0 0 0")
        self.p1_edit = QtWidgets.QLineEdit("1 0 0")
        self.config_path_edit = QtWidgets.QLineEdit(str(self.controller.config_path))
        self.status = QtWidgets.QLabel("준비됨")
        self.view_log_check = QtWidgets.QCheckBox("뷰 로그 보기")
        self.view_log_check.setChecked(False)
        self._config_collapsed = True

        self._config_widgets: dict[str, QtWidgets.QWidget] = {}
        self._config_group_titles: list[str] = []
        self._config_group_rows: dict[int, list[tuple[QtWidgets.QWidget, str, str, str]]] = {}
        
        central = QtWidgets.QSplitter()
        central.addWidget(self.view)
        central.addWidget(self._build_side_panel())
        central.setStretchFactor(0, 3)
        central.setStretchFactor(1, 2)
        central.setSizes([1000, 800])
        self.setCentralWidget(central)

        self._set_default_las_path(las_path)
        if p0 is not None:
            self.p0_edit.setText(" ".join(f"{float(v):.10f}" for v in p0))
        if p1 is not None:
            self.p1_edit.setText(" ".join(f"{float(v):.10f}" for v in p1))

        self._connect_signals()
        self._install_shortcuts()
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
        group = QtWidgets.QGroupBox("입력")
        form = QtWidgets.QFormLayout(group)

        btn_browse_las = QtWidgets.QPushButton("찾아보기...")
        las_row = QtWidgets.QHBoxLayout()
        las_row.addWidget(self.las_edit)
        las_row.addWidget(btn_browse_las)
        form.addRow("LAS 파일", las_row)
        form.addRow("P0", self.p0_edit)
        form.addRow("P1", self.p1_edit)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("LAS 불러오기")
        self.btn_init = QtWidgets.QPushButton("초기화")
        self.btn_back = QtWidgets.QPushButton("한 스텝 뒤로")
        self.btn_step = QtWidgets.QPushButton("한 스텝 실행")
        self.btn_full = QtWidgets.QPushButton("전체 실행")
        self.btn_reset = QtWidgets.QPushButton("리셋")
        for btn in [self.btn_load, self.btn_init, self.btn_back, self.btn_step, self.btn_full, self.btn_reset]:
            btn_row.addWidget(btn)
        form.addRow(btn_row)

        self.btn_browse_las = btn_browse_las
        return group

    def _build_config_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("설정")
        outer = QtWidgets.QVBoxLayout(group)
        outer.setContentsMargins(8, 8, 8, 8)

        self.btn_toggle_cfg = QtWidgets.QToolButton()
        self.btn_toggle_cfg.setText("YAML 설정")
        self.btn_toggle_cfg.setCheckable(True)
        self.btn_toggle_cfg.setChecked(False)
        self.btn_toggle_cfg.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_toggle_cfg.setArrowType(QtCore.Qt.RightArrow)
        self.btn_toggle_cfg.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        outer.addWidget(self.btn_toggle_cfg)

        self.config_body = QtWidgets.QWidget()
        self.config_body.setMaximumHeight(430)
        body_layout = QtWidgets.QVBoxLayout(self.config_body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        path_row = QtWidgets.QHBoxLayout()
        self.btn_browse_cfg = QtWidgets.QPushButton("찾아보기...")
        self.btn_reload_cfg = QtWidgets.QPushButton("다시 불러오기")
        self.btn_apply_cfg = QtWidgets.QPushButton("적용")
        self.btn_save_cfg = QtWidgets.QPushButton("저장")
        path_row.addWidget(self.config_path_edit)
        path_row.addWidget(self.btn_browse_cfg)
        body_layout.addLayout(path_row)

        self.config_search_edit = QtWidgets.QLineEdit()
        self.config_search_edit.setPlaceholderText("설정 이름 또는 설명 검색")
        body_layout.addWidget(self.config_search_edit)

        self._create_config_widgets()
        self.config_group_titles = [group_title for group_title, _ in CONFIG_GROUPS]
        self.config_group_rows = {}
        self.config_group_list = QtWidgets.QListWidget()
        self.config_group_list.setMaximumWidth(220)
        self.config_group_list.setMinimumWidth(180)
        self.config_group_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.config_group_list.setSpacing(2)

        self.config_pages = QtWidgets.QStackedWidget()

        for group_index, (group_title, items) in enumerate(CONFIG_GROUPS):
            self.config_group_list.addItem(group_title)

            page_scroll = QtWidgets.QScrollArea()
            page_scroll.setWidgetResizable(True)
            page_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            page_layout.setContentsMargins(8, 8, 8, 8)
            page_layout.setSpacing(6)

            group_rows: list[tuple[QtWidgets.QWidget, str, str, str]] = []
            for key, label_text, comment in items:
                widget = self._config_widgets[key]
                label = QtWidgets.QLabel(label_text)
                label.setToolTip(comment)
                label.setMinimumWidth(180)
                widget.setToolTip(comment)

                row = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)
                row_layout.addWidget(label, 0)
                row_layout.addWidget(widget, 1)
                page_layout.addWidget(row)
                group_rows.append((row, key, label_text, comment))

            page_layout.addStretch(1)
            page_scroll.setWidget(page)
            self.config_pages.addWidget(page_scroll)
            self.config_group_rows[group_index] = group_rows

        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addWidget(self.config_group_list, 0)
        nav_layout.addWidget(self.config_pages, 1)
        body_layout.addLayout(nav_layout)

        if self.config_group_list.count() > 0:
            self.config_group_list.setCurrentRow(0)

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
        self.btn_back.clicked.connect(self.on_back)
        self.btn_step.clicked.connect(self.on_step)
        self.btn_full.clicked.connect(self.on_full)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_reload_cfg.clicked.connect(self.on_reload_config)
        self.btn_apply_cfg.clicked.connect(self.on_apply_config)
        self.btn_save_cfg.clicked.connect(self.on_save_config)
        self.config_group_list.currentRowChanged.connect(self._on_config_group_changed)
        self.config_search_edit.textChanged.connect(self._filter_config_options)

        self.controller.changed.connect(self.refresh)
        self.controller.log_message.connect(self.log.appendPlainText)
        self.view.point_context_menu_requested.connect(self._open_point_context_menu)
        self.view.debug_message.connect(lambda msg: self.log.appendPlainText(f"VIEW: {msg}"))
        self.view_log_check.toggled.connect(self.view.set_view_log_enabled)

    def _install_shortcuts(self) -> None:
        self._step_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        self._step_shortcut.setContext(QtCore.Qt.WindowShortcut)
        self._step_shortcut.activated.connect(self._on_space_step)
        self._undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Undo), self)
        self._undo_shortcut.setContext(QtCore.Qt.WindowShortcut)
        self._undo_shortcut.activated.connect(self._on_back_shortcut)
        self._undo_backspace_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self)
        self._undo_backspace_shortcut.setContext(QtCore.Qt.WindowShortcut)
        self._undo_backspace_shortcut.activated.connect(self._on_back_shortcut)

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

    def _create_config_widgets(self) -> None:
        if self._config_widgets:
            return
        for key, spec in CONFIG_WIDGET_SPECS.items():
            kind = spec[0]
            if kind == "float":
                _, minimum, maximum, step = spec
                widget = self._make_double_spin(minimum, maximum, step)
            elif kind == "int":
                _, minimum, maximum = spec
                widget = self._make_int_spin(minimum, maximum)
            elif kind == "bool":
                widget = QtWidgets.QCheckBox()
            else:
                raise ValueError(f"Unknown config widget kind: {kind}")
            self._config_widgets[key] = widget

    def _config_from_ui(self) -> TrackerConfig:
        cfg = TrackerConfig()
        for key, widget in self._config_widgets.items():
            if isinstance(widget, QtWidgets.QCheckBox):
                setattr(cfg, key, widget.isChecked())
            else:
                setattr(cfg, key, widget.value())
        return cfg

    def _load_config_into_ui(self, cfg: TrackerConfig) -> None:
        for key, widget in self._config_widgets.items():
            value = getattr(cfg, key)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            else:
                widget.setValue(value)

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
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "LAS 파일 선택", start_dir, "LAS Files (*.las);;All Files (*)")
        if path:
            self.las_edit.setText(path)

    def on_browse_config(self) -> None:
        start_path = self.config_path_edit.text().strip() or str(DEFAULT_CONFIG_PATH)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "YAML 설정 파일 선택", start_path, "YAML Files (*.yaml *.yml)")
        if path:
            self.config_path_edit.setText(path)

    def on_load(self) -> None:
        path = self.las_edit.text().strip()
        if not path:
            self._show_error("LAS 경로가 비어 있습니다.")
            return
        if not Path(path).exists():
            self._show_error(f"LAS 파일을 찾을 수 없습니다.\n{path}")
            return
        self._run_action("LAS 불러오는 중...", lambda: self.controller.load_las(path))

    def on_init(self) -> None:
        try:
            p0 = self._parse_xyz_text(self.p0_edit.text())
            p1 = self._parse_xyz_text(self.p1_edit.text())
        except ValueError as exc:
            self._show_error(f"P0/P1 형식이 올바르지 않습니다.\n형식: x y z\n\n{exc}")
            return
        self.log.clear()
        if self._run_action("추적기 초기화 중...", lambda: self._initialize_with_points(p0, p1)):
            self.view.focus_on_point(self.controller.model.current_point)

    def on_step(self) -> None:
        if self._run_action("한 스텝 실행 중...", self.controller.run_step):
            self.view.focus_on_point(self.controller.model.current_point)

    def on_back(self) -> None:
        if self._run_action("이전 스텝 복원 중...", self.controller.undo_step):
            self.view.focus_on_point(self.controller.model.current_point)

    def _on_space_step(self) -> None:
        if not self._can_trigger_global_shortcut():
            return
        self.on_step()

    def _on_back_shortcut(self) -> None:
        if not self._can_trigger_global_shortcut():
            return
        self.on_back()

    def _can_trigger_global_shortcut(self) -> bool:
        focus = QtWidgets.QApplication.focusWidget()
        if focus is None:
            return True
        if isinstance(
            focus,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
                QtWidgets.QAbstractSpinBox,
                QtWidgets.QComboBox,
                QtWidgets.QAbstractButton,
                QtWidgets.QAbstractItemView,
                QtWidgets.QSlider,
            ),
        ):
            return False
        return True

    def on_full(self) -> None:
        self._run_action("전체 추적 실행 중...", self.controller.run_full)

    def on_reset(self) -> None:
        self.log.clear()
        self.controller.reset()

    def on_reload_config(self) -> None:
        path = self.config_path_edit.text().strip() or str(DEFAULT_CONFIG_PATH)
        if self._run_action("설정 다시 불러오는 중...", lambda: self._reload_config(path)):
            self.status.setText("설정을 다시 불러왔습니다")

    def on_apply_config(self) -> None:
        cfg = self._config_from_ui()
        self.controller.apply_config(cfg)
        self.status.setText("설정을 적용했습니다")

    def on_save_config(self) -> None:
        cfg = self._config_from_ui()
        path = self.config_path_edit.text().strip() or str(DEFAULT_CONFIG_PATH)
        if self._run_action("설정 저장 중...", lambda: self.controller.save_config(cfg, path)):
            self.status.setText("설정을 저장했습니다")

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
            raise ValueError("숫자 3개가 필요합니다.")
        return [float(x) for x in parts]

    def _on_config_group_changed(self, row: int) -> None:
        if row < 0:
            return
        if hasattr(self, "config_pages"):
            self.config_pages.setCurrentIndex(row)

    def _filter_config_options(self, text: str) -> None:
        query = text.strip().lower()
        visible_groups: list[int] = []
        for group_index, group_title in enumerate(self.config_group_titles):
            rows = self.config_group_rows.get(group_index, [])
            group_match = bool(query) and query in group_title.lower()
            visible_count = 0
            for row_widget, key, label_text, comment in rows:
                visible = (
                    not query
                    or group_match
                    or query in key.lower()
                    or query in label_text.lower()
                    or query in comment.lower()
                )
                row_widget.setVisible(visible)
                if visible:
                    visible_count += 1
            item = self.config_group_list.item(group_index)
            if item is not None:
                item.setHidden(visible_count == 0)
            if visible_count > 0:
                visible_groups.append(group_index)

        if visible_groups:
            current = self.config_group_list.currentRow()
            if current not in visible_groups:
                self.config_group_list.setCurrentRow(visible_groups[0])

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
            self.status.setText("점 좌표를 복사했습니다")
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
        self.status.setText(f"{label}를 포인트클라우드에서 가져왔습니다")
        self.log.appendPlainText(f"{label} <- {vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}")

    def refresh(self) -> None:
        m = self.controller.model
        if m.xyz is not None:
            self.view.set_point_cloud(m.xyz, m.intensity, revision=m.point_cloud_revision)
        self.view.set_seed_points(m.p0, m.p1, render=False)
        self.view.set_track(m.track_points, render=False)
        self.view.set_current(m.current_point, render=False)
        self.view.set_predicted(m.predicted_points, render=False)
        self.view.set_active_cell_boxes(m.active_cell_box_groups, render=False)
        self.view.set_segments(m.segment_groups, render=False)
        self.view.set_trajectory_line(m.trajectory_line_points, render=False)
        self.view.set_profile_overlay(m.profile_line_points, m.stripe_segment_points, m.stripe_edge_points, render=False)
        self.view.set_search_box(m.search_box_points, render=False)
        self.view.render()
        self.profile.update_profile(m.profile)
        self.status.setText(m.status_text or "준비됨")
