from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lane Tracker V2 GUI")
    parser.add_argument("--las", default="")
    parser.add_argument("--p0", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--p1", nargs=3, type=float, metavar=("X", "Y", "Z"))
    return parser


def main() -> None:
    try:
        from PySide6 import QtCore, QtWidgets
        from ..ui.main_window import MainWindow

        args = build_parser().parse_args()
        app = QtWidgets.QApplication(sys.argv)
        w = MainWindow(
            las_path=args.las or None,
            p0=args.p0,
            p1=args.p1,
        )
        w.resize(1600, 900)
        w.show()
        w.showNormal()
        w.raise_()
        w.activateWindow()
        QtCore.QTimer.singleShot(150, w.raise_)
        QtCore.QTimer.singleShot(150, w.activateWindow)
        sys.exit(app.exec())
    except Exception:
        log_path = Path(__file__).resolve().parents[1] / "app_startup_error.log"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
