from __future__ import annotations

from .tracker.config import DEFAULT_CONFIG_PATH, TrackerConfig, load_tracker_config


def default_config(path: str | None = None) -> TrackerConfig:
    return load_tracker_config(path or DEFAULT_CONFIG_PATH)
