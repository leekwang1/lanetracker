from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from ..io.csv_io import save_xyz_csv
from ..io.debug_io import save_debug_json
from ..io.las_io import load_las_xyz_intensity
from ..tracker.lane_tracker_v2 import LaneTrackerV2, TrackerV2Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lane tracker V2")
    p.add_argument("--las", required=True)
    p.add_argument("--p0", nargs=3, required=True, type=float, metavar=("X", "Y", "Z"))
    p.add_argument("--p1", nargs=3, required=True, type=float, metavar=("X", "Y", "Z"))
    p.add_argument("--output", required=True)
    p.add_argument("--debug", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    las = load_las_xyz_intensity(args.las)
    tracker = LaneTrackerV2(las.xyz, las.intensity, TrackerV2Config())
    p0 = np.asarray(args.p0, dtype=float)
    p1 = np.asarray(args.p1, dtype=float)
    tracker.initialize(p0, p1)
    result = tracker.run_full()
    out = Path(args.output)
    save_xyz_csv(out, result.output_points)
    save_xyz_csv(out.with_name(f"{out.stem}_dense{out.suffix}"), result.dense_points)
    debug_path = args.debug or str(out.with_suffix(out.suffix + ".debug.json"))
    save_debug_json(debug_path, result)
    print(f"Saved output: {out}")
    print(f"Saved dense: {out.with_name(f'{out.stem}_dense{out.suffix}')}")
    print(f"Saved debug: {debug_path}")
    print(f"stop_reason = {result.stop_reason}")
    print(f"num_dense_points = {len(result.dense_points)}")
    print(f"num_output_points = {len(result.output_points)}")


if __name__ == "__main__":
    main()
