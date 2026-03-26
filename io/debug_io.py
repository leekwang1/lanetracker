from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np


def _convert(obj):
    if is_dataclass(obj):
        return _convert(asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


def save_debug_json(path: str | Path, payload) -> None:
    Path(path).write_text(json.dumps(_convert(payload), indent=2, ensure_ascii=False), encoding="utf-8")
