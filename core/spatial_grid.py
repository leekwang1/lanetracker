from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from .oriented_query import filter_oriented_strip


class SpatialGrid:
    def __init__(self, xy: np.ndarray, cell_size: float):
        self.xy = np.asarray(xy, dtype=float)
        self.cell_size = float(cell_size)
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, p in enumerate(self.xy):
            cx = int(math.floor(p[0] / self.cell_size))
            cy = int(math.floor(p[1] / self.cell_size))
            self.cells[(cx, cy)].append(i)

    def _cell_of(self, p: np.ndarray) -> tuple[int, int]:
        return (int(math.floor(p[0] / self.cell_size)), int(math.floor(p[1] / self.cell_size)))

    def query_radius_xy(self, center_xy: np.ndarray, radius: float) -> np.ndarray:
        r = float(radius)
        c0x, c0y = self._cell_of(center_xy)
        cr = int(math.ceil(r / self.cell_size))
        out: list[int] = []
        r2 = r * r
        for cy in range(c0y - cr, c0y + cr + 1):
            for cx in range(c0x - cr, c0x + cr + 1):
                ids = self.cells.get((cx, cy))
                if not ids:
                    continue
                pts = self.xy[np.asarray(ids, dtype=int)]
                d2 = np.sum((pts - center_xy[None, :]) ** 2, axis=1)
                keep = np.asarray(ids, dtype=int)[d2 <= r2]
                if keep.size > 0:
                    out.extend(keep.tolist())
        return np.asarray(out, dtype=int) if out else np.empty((0,), dtype=int)

    def query_bbox_xy(self, min_xy: np.ndarray, max_xy: np.ndarray) -> np.ndarray:
        min_cell = self._cell_of(min_xy)
        max_cell = self._cell_of(max_xy)
        out: list[int] = []
        for cy in range(min_cell[1], max_cell[1] + 1):
            for cx in range(min_cell[0], max_cell[0] + 1):
                ids = self.cells.get((cx, cy))
                if ids:
                    out.extend(ids)
        if not out:
            return np.empty((0,), dtype=int)
        idx = np.asarray(out, dtype=int)
        pts = self.xy[idx]
        mask = ((pts[:, 0] >= min_xy[0]) & (pts[:, 0] <= max_xy[0]) &
                (pts[:, 1] >= min_xy[1]) & (pts[:, 1] <= max_xy[1]))
        return idx[mask]

    def query_oriented_strip_xy(self, center_xy: np.ndarray, tangent_xy: np.ndarray, along_half: float, lateral_half: float) -> np.ndarray:
        t = np.asarray(tangent_xy[:2], dtype=float)
        n = np.array([-t[1], t[0]], dtype=float)
        p1 = center_xy + t * along_half + n * lateral_half
        p2 = center_xy + t * along_half - n * lateral_half
        p3 = center_xy - t * along_half + n * lateral_half
        p4 = center_xy - t * along_half - n * lateral_half
        corners = np.vstack([p1, p2, p3, p4])
        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)
        idx = self.query_bbox_xy(min_xy, max_xy)
        if idx.size == 0:
            return idx
        pts = self.xy[idx]
        mask = filter_oriented_strip(pts, center_xy, tangent_xy, along_half, lateral_half)
        return idx[mask]
