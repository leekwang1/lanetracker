from __future__ import annotations

import numpy as np


class LineTypeClassifier:
    def __init__(self, history_len: int = 20):
        self.history_len = int(history_len)

    def update(self, visibility_history: list[int], width_history: list[float]) -> tuple[float, float, dict]:
        vis = visibility_history[-self.history_len:]
        if not vis:
            return 0.0, 0.0, {"reason": "empty_history"}
        runs = self._runs(vis)
        visible_runs = [ln for val, ln in runs if val == 1]
        gap_runs = [ln for val, ln in runs if val == 0]
        dashed_prob = 0.0
        solid_prob = 0.0
        if visible_runs and gap_runs:
            if len(visible_runs) >= 2 and len(gap_runs) >= 1:
                dashed_prob += 0.5
            if np.mean(gap_runs) >= 1.0:
                dashed_prob += 0.3
            if width_history and np.std(width_history[-min(len(width_history), 10):]) < 0.05:
                dashed_prob += 0.2
        visible_ratio = float(np.mean(vis))
        if visible_ratio > 0.8 and (not gap_runs or np.mean(gap_runs) < 1.0):
            solid_prob = 0.8 + 0.2 * visible_ratio
        return float(np.clip(dashed_prob, 0.0, 1.0)), float(np.clip(solid_prob, 0.0, 1.0)), {"runs": runs, "visible_ratio": visible_ratio}

    def _runs(self, seq: list[int]) -> list[tuple[int, int]]:
        if not seq:
            return []
        out = []
        cur = seq[0]
        cnt = 1
        for x in seq[1:]:
            if x == cur:
                cnt += 1
            else:
                out.append((cur, cnt))
                cur = x
                cnt = 1
        out.append((cur, cnt))
        return out
