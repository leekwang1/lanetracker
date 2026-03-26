from __future__ import annotations

from PySide6 import QtWidgets
import pyqtgraph as pg
import numpy as np


class ProfilePlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve_raw = self.plot.plot(pen=pg.mkPen("#94a3b8", width=1))
        self.curve_smooth = self.plot.plot(pen=pg.mkPen("#f97316", width=2))
        self.vline_left = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ef4444", width=1))
        self.vline_right = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ef4444", width=1))
        self.vline_center = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#22c55e", width=1.5))
        self.plot.addItem(self.vline_left)
        self.plot.addItem(self.vline_right)
        self.plot.addItem(self.vline_center)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.plot)

    def update_profile(self, profile):
        if profile is None or profile.bins_center.size == 0:
            self.curve_raw.setData([], [])
            self.curve_smooth.setData([], [])
            self.vline_left.hide()
            self.vline_right.hide()
            self.vline_center.hide()
            return
        self.curve_raw.setData(profile.bins_center, profile.hist_combined)
        self.curve_smooth.setData(profile.bins_center, profile.smooth_hist)
        if profile.selected_idx is not None:
            sc = profile.stripe_candidates[profile.selected_idx]
            self.vline_left.setValue(sc.left_m)
            self.vline_right.setValue(sc.right_m)
            self.vline_center.setValue(sc.center_m)
            self.vline_left.show()
            self.vline_right.show()
            self.vline_center.show()
        else:
            self.vline_left.hide()
            self.vline_right.hide()
            self.vline_center.hide()
