from __future__ import annotations

import csv
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .controller import TrackerController
from .pointcloud_view_widget import PointCloudViewWidget
from .profile_plot_widget import ProfilePlotWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, las_path: str | None = None, p0: list[float] | tuple[float, ...] | None = None, p1: list[float] | tuple[float, ...] | None = None):
        super().__init__()
        self.setWindowTitle("Lane Tracker V2 Debugger")
        self.controller = TrackerController()
        self.view = PointCloudViewWidget()
        self.profile = ProfilePlotWidget()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.view_log_check = QtWidgets.QCheckBox("View Log")
        self.view_log_check.setChecked(False)
        self.detail_log_check = QtWidgets.QCheckBox("Detail Log")
        self.detail_log_check.setChecked(False)
        self.legend = QtWidgets.QLabel(
            "\n".join(
                [
                    "범례",
                    "",
                    "[메인 화면]",
                    "P0: 초록, P1: 파랑, 시드선: 연회색",
                    "현재 중심: 빨강", "프로파일 단면: 하늘색 선",
                    "선택 stripe: 분홍 선분", "stripe 좌우 에지: 분홍 점",
                    "예측 후보점: 주황 점", "검색 박스: 보라색 박스",
                    "",
                    "[그래프]",
                    "프로파일 원본: 회청색 곡선",
                    "스무딩 프로파일: 주황 곡선",
                    "좌우 경계: 빨간 세로선",
                    "중심: 초록 세로선",
                ]
            )
        )
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet(
            "QLabel {"
            "background: #111827;"
            "color: #e5e7eb;"
            "border: 1px solid #374151;"
            "border-radius: 6px;"
            "padding: 8px;"
            "}"
        )
        self.status = QtWidgets.QLabel("Ready")
        self.las_edit = QtWidgets.QLineEdit()
        self.las_edit.setPlaceholderText("Select a .las file")
        self.p0_edit = QtWidgets.QLineEdit("0 0 0")
        self.p1_edit = QtWidgets.QLineEdit("1 0 0")
        btn_load = QtWidgets.QPushButton("Load LAS")
        btn_init = QtWidgets.QPushButton("Initialize")
        btn_step = QtWidgets.QPushButton("Run One Step")
        btn_full = QtWidgets.QPushButton("Run Full")
        btn_reset = QtWidgets.QPushButton("Reset")
        btn_browse = QtWidgets.QPushButton("Browse...")
        form = QtWidgets.QFormLayout()
        las_row = QtWidgets.QHBoxLayout()
        las_row.addWidget(self.las_edit)
        las_row.addWidget(btn_browse)
        form.addRow("LAS", las_row)
        form.addRow("P0", self.p0_edit)
        form.addRow("P1", self.p1_edit)
        btns = QtWidgets.QHBoxLayout()
        for w in [btn_load, btn_init, btn_step, btn_full, btn_reset]:
            btns.addWidget(w)
        side = QtWidgets.QVBoxLayout()
        side.addLayout(form)
        side.addLayout(btns)
        side.addWidget(self.status)
        side.addWidget(self.legend)
        side.addWidget(self.profile)
        side.addWidget(self.log)
        log_opts = QtWidgets.QHBoxLayout()
        log_opts.addStretch(1)
        log_opts.addWidget(self.view_log_check)
        log_opts.addWidget(self.detail_log_check)
        side.addLayout(log_opts)
        right = QtWidgets.QWidget()
        right.setLayout(side)
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([960, 640])
        self.setCentralWidget(splitter)
        self._set_default_las_path(las_path)
        if p0 is not None:
            self.p0_edit.setText(" ".join(f"{float(v):.10f}" for v in p0))
        if p1 is not None:
            self.p1_edit.setText(" ".join(f"{float(v):.10f}" for v in p1))
        btn_load.clicked.connect(self.on_load)
        btn_browse.clicked.connect(self.on_browse)
        btn_init.clicked.connect(self.on_init)
        btn_step.clicked.connect(self.on_step)
        btn_full.clicked.connect(self.on_full)
        btn_reset.clicked.connect(self.on_reset)
        self.controller.changed.connect(self.refresh)
        self.controller.log_message.connect(self.log.appendPlainText)
        self.view.point_context_menu_requested.connect(self._open_point_context_menu)
        self.view.debug_message.connect(lambda msg: self.log.appendPlainText(f"VIEW: {msg}"))
        self.view_log_check.toggled.connect(self.view.set_view_log_enabled)
        self.detail_log_check.toggled.connect(self.controller.set_detail_log_enabled)

    def _set_default_las_path(self, las_path: str | None = None):
        if las_path:
            self.las_edit.setText(las_path)
            return
        if self.las_edit.text().strip():
            return
        data_dir = Path(__file__).resolve().parents[1] / "data"
        candidates = sorted(data_dir.glob("*.las"))
        if candidates:
            self.las_edit.setText(str(candidates[0]))

    def on_browse(self):
        start_dir = str((Path(self.las_edit.text()).parent if self.las_edit.text().strip() else Path(__file__).resolve().parents[1] / "data"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select LAS", start_dir, "LAS Files (*.las);;All Files (*)")
        if path:
            self.las_edit.setText(path)

    def on_load(self):
        path = self.las_edit.text().strip()
        if not path:
            self._show_error("LAS path is empty. Choose a .las file first.")
            return
        if not Path(path).exists():
            self._show_error(f"LAS file not found:\n{path}")
            return
        if self._run_action("Loading LAS...", lambda: self.controller.load_las(path)):
            self._populate_seed_points_from_csv(Path(path))

    def on_init(self):
        try:
            p0 = self._parse_xyz_text(self.p0_edit.text())
            p1 = self._parse_xyz_text(self.p1_edit.text())
        except ValueError as exc:
            self._show_error(f"Invalid P0/P1 format.\nUse: x y z\n\n{exc}")
            return
        if len(p0) != 3 or len(p1) != 3:
            self._show_error("P0 and P1 must each contain exactly 3 numbers: x y z")
            return
        self.log.clear()
        if self._run_action(
            "Initializing tracker...",
            lambda: self._initialize_with_points(p0, p1),
        ):
            self.view.focus_on_point(self.controller.model.current_point)

    def on_step(self):
        if self._run_action("Running one step...", self.controller.run_step):
            self.view.focus_on_point(self.controller.model.current_point)

    def on_full(self):
        self._run_action("Running full tracker...", self.controller.run_full)

    def on_reset(self):
        self.log.clear()
        self.controller.reset()

    def _initialize_with_points(self, p0, p1):
        self.controller.set_p0(*p0)
        self.controller.set_p1(*p1)
        self.controller.initialize_tracker()

    def _parse_xyz_text(self, text: str) -> list[float]:
        parts = text.replace(",", " ").split()
        return [float(x) for x in parts]

    def _populate_seed_points_from_csv(self, las_path: Path):
        csv_path = las_path.with_suffix(".csv")
        if not csv_path.exists():
            return
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for row in reader:
                    rows.append(row)
                    if len(rows) >= 2:
                        break
            if len(rows) < 2:
                return
            p0 = [float(rows[0][k]) for k in ("x", "y", "z")]
            p1 = [float(rows[1][k]) for k in ("x", "y", "z")]
        except Exception as exc:
            self.log.appendPlainText(f"WARN: failed to read seed CSV {csv_path}: {exc}")
            return
        self.p0_edit.setText(" ".join(f"{v:.6f}" for v in p0))
        self.p1_edit.setText(" ".join(f"{v:.6f}" for v in p1))
        self.log.appendPlainText(f"Loaded seed points from {csv_path.name}")

    def _run_action(self, status_text: str, fn):
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

    def _show_error(self, message: str):
        self.log.appendPlainText(f"ERROR: {message}")
        self.status.setText("Error")
        QtWidgets.QMessageBox.critical(self, "Lane Tracker V2", message)

    def _open_point_context_menu(self, point_xyz, global_pos):
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

    def refresh(self):
        m = self.controller.model
        self.view.set_point_cloud(m.xyz, m.intensity, revision=m.point_cloud_revision) if m.xyz is not None else None
        self.view.set_seed_points(m.p0, m.p1, render=False)
        self.view.set_track(m.track_points, render=False)
        self.view.set_current(m.current_point, render=False)
        self.view.set_predicted(m.predicted_points, render=False)
        self.view.set_trajectory_line(m.trajectory_line_points, render=False)
        self.view.set_profile_overlay(m.profile_line_points, m.stripe_segment_points, m.stripe_edge_points, render=False)
        self.view.set_search_box(m.search_box_points, render=False)
        self.view.render()
        self.profile.update_profile(m.profile)
        status = m.status_text or "Ready"
        if m.xyz is not None:
            status = f"{status} | viewer fast-load LOD"
        self.status.setText(status)
