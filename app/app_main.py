from __future__ import annotations

import argparse
import sys
from PySide6 import QtWidgets

from ..ui.main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lane Tracker V2 GUI")
    parser.add_argument("--las", default="")
    parser.add_argument("--p0", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--p1", nargs=3, type=float, metavar=("X", "Y", "Z"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(
        las_path=args.las or None,
        p0=args.p0,
        p1=args.p1,
    )
    w.resize(1600, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
