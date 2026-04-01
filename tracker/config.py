from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tracker.yaml"


CONFIG_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "1. Basic Tracking",
        [
            ("forward_distance_m", "Distance moved per tracking step in meters"),
            ("max_track_length_m", "Maximum total tracking distance for run_full in meters"),
        ],
    ),
    (
        "2. Local ROI And BEV Cells",
        [
            ("graph_roi_forward_m", "Forward ROI length in meters"),
            ("graph_roi_lateral_half_m", "Half ROI width to left and right in meters"),
            ("graph_cell_size_m", "BEV cell size in meters; smaller is finer but slower"),
            ("graph_active_intensity_min", "Minimum normalized intensity for active cells"),
            ("graph_active_contrast_min", "Minimum local contrast for active cells"),
            ("graph_min_cell_points", "Minimum number of points required in a cell"),
        ],
    ),
    (
        "3. Segment Extraction",
        [
            ("segment_min_length_m", "Minimum valid segment length in meters"),
            ("segment_target_length_m", "Target segment length when splitting components"),
            ("segment_max_length_m", "Maximum segment length in meters"),
            ("segment_heading_gate_deg", "Allowed segment heading difference in degrees"),
        ],
    ),
    (
        "4. Graph Connection",
        [
            ("graph_neighbor_max_distance_m", "Maximum distance to connect neighbor nodes"),
            ("graph_neighbor_lateral_limit_m", "Maximum lateral difference when linking neighbors"),
        ],
    ),
    (
        "5. Beam Search",
        [
            ("graph_beam_width", "Number of path hypotheses kept per step"),
            ("graph_beam_horizon_nodes", "How many nodes ahead to plan"),
            ("graph_beam_branching", "Maximum branch count expanded from each node"),
        ],
    ),
    (
        "6. Score Weights",
        [
            ("graph_intensity_weight", "Weight for strong intensity"),
            ("graph_contrast_weight", "Weight for local contrast"),
            ("graph_direction_weight", "Weight for direction and curvature continuity"),
            ("graph_distance_weight", "Weight against large distance jumps"),
            ("graph_history_weight", "Weight for consistency with recent history"),
            ("graph_period_weight", "Weight for dashed-period consistency"),
            ("graph_crosswalk_penalty", "Penalty for crosswalk-like patterns"),
        ],
    ),
    (
        "7. Profile And Stripe Geometry",
        [
            ("profile_lateral_half_m", "Half lateral width used for cross-section profiles"),
            ("profile_along_half_m", "Half along-track window used for profiles"),
            ("profile_bin_size_m", "Profile lateral bin size in meters"),
            ("twin_edge_min_width_m", "Minimum stripe width for twin-edge candidates"),
            ("twin_edge_max_width_m", "Maximum stripe width for twin-edge candidates"),
            ("edge_grad_min", "Minimum gradient used to detect twin edges"),
            ("candidate_min_support", "Minimum support points for a profile candidate"),
            ("candidate_min_score", "Minimum total score required to accept a candidate"),
        ],
    ),
    (
        "8. Along Signal And Dashed Pattern",
        [
            ("along_signal_half_m", "Half length used for 1D along-track signal"),
            ("along_signal_bin_m", "Bin size for along-track signal"),
            ("along_signal_lateral_half_m", "Half lateral width used to gather along-track signal"),
            ("autocorr_min_period_m", "Minimum dashed period considered valid"),
            ("autocorr_max_period_m", "Maximum dashed period considered valid"),
            ("dashed_autocorr_min", "Minimum autocorrelation score for dashed classification"),
            ("solid_occupancy_min", "Minimum occupancy used for solid classification"),
        ],
    ),
    (
        "9. Gap And Continuity",
        [
            ("gap_forward_distance_m", "Maximum distance bridged without valid observations"),
            ("continuity_node_count", "How many recent nodes are used for continuity"),
            ("continuity_strength", "Strength of continuity penalties"),
        ],
    ),
    (
        "10. Z Filtering",
        [
            ("use_z_clip", "If true, use only points near the current lane height"),
            ("z_clip_half_range_m", "Half Z range used when clipping points, in meters"),
        ],
    ),
    (
        "11. Crosswalk Handling",
        [
            ("crosswalk_stop_enabled", "If true, stop when crosswalk pattern is detected"),
            ("crosswalk_lookahead_m", "Forward distance used for crosswalk checks"),
            ("crosswalk_lateral_half_m", "Half lateral width used for crosswalk checks"),
            ("crosswalk_min_peaks", "Minimum repeated peaks to classify a crosswalk"),
        ],
    ),
    (
        "12. Internal Grid",
        [
            ("spatial_grid_cell_size_m", "Internal spatial-grid cell size for neighbor queries"),
        ],
    ),
]


@dataclass
class TrackerConfig:
    forward_distance_m: float = 0.30
    graph_roi_forward_m: float = 3.0
    graph_roi_lateral_half_m: float = 1.2
    graph_cell_size_m: float = 0.05
    graph_active_intensity_min: float = 0.62
    graph_active_contrast_min: float = 0.10
    graph_min_cell_points: int = 2
    graph_neighbor_max_distance_m: float = 0.18
    graph_neighbor_lateral_limit_m: float = 0.18
    segment_min_length_m: float = 0.20
    segment_target_length_m: float = 0.50
    segment_max_length_m: float = 1.00
    segment_heading_gate_deg: float = 28.0
    graph_beam_width: int = 6
    graph_beam_horizon_nodes: int = 8
    graph_beam_branching: int = 5
    graph_intensity_weight: float = 0.30
    graph_contrast_weight: float = 0.18
    graph_direction_weight: float = 0.16
    graph_distance_weight: float = 0.10
    graph_history_weight: float = 0.16
    graph_period_weight: float = 0.10
    graph_crosswalk_penalty: float = 0.08
    z_clip_half_range_m: float = 0.30
    use_z_clip: bool = True
    gap_forward_distance_m: float = 10.0
    continuity_node_count: int = 6
    continuity_strength: float = 1.50
    crosswalk_stop_enabled: bool = True

    spatial_grid_cell_size_m: float = 0.10
    max_track_length_m: float = 120.0
    profile_lateral_half_m: float = 0.45
    profile_along_half_m: float = 0.10
    profile_bin_size_m: float = 0.01
    twin_edge_min_width_m: float = 0.08
    twin_edge_max_width_m: float = 0.24
    edge_grad_min: float = 0.10
    candidate_min_support: int = 5
    candidate_min_score: float = 0.30
    along_signal_half_m: float = 1.20
    along_signal_bin_m: float = 0.05
    along_signal_lateral_half_m: float = 0.12
    autocorr_min_period_m: float = 0.25
    autocorr_max_period_m: float = 1.20
    dashed_autocorr_min: float = 0.20
    solid_occupancy_min: float = 0.58
    crosswalk_lookahead_m: float = 1.60
    crosswalk_lateral_half_m: float = 1.20
    crosswalk_min_peaks: int = 4

    @property
    def init_search_lateral_half_m(self) -> float:
        return max(self.graph_roi_lateral_half_m, self.profile_lateral_half_m)

    @property
    def step_search_lateral_half_m(self) -> float:
        return max(self.graph_roi_lateral_half_m, self.profile_lateral_half_m)

    @property
    def search_along_half_m(self) -> float:
        return max(self.graph_roi_forward_m * 0.5, self.along_signal_half_m, self.forward_distance_m * 1.5)


def tracker_config_to_dict(cfg: TrackerConfig) -> dict[str, Any]:
    return asdict(cfg)


def tracker_config_from_dict(data: dict[str, Any] | None) -> TrackerConfig:
    if not data:
        return TrackerConfig()
    cfg = TrackerConfig()
    for field_name in asdict(cfg).keys():
        if field_name in data:
            setattr(cfg, field_name, data[field_name])
    return cfg


def ensure_config_file(path: str | Path | None = None) -> Path:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        save_tracker_config(TrackerConfig(), config_path)
    return config_path


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data


def _dump_simple_yaml(data: dict[str, Any]) -> str:
    def render(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    remaining = dict(data)
    lines: list[str] = []
    for title, items in CONFIG_GROUPS:
        lines.append(f"# [{title}]")
        for key, comment in items:
            if key not in remaining:
                continue
            lines.append(f"# {comment}")
            lines.append(f"{key}: {render(remaining.pop(key))}")
        lines.append("")

    if remaining:
        lines.append("# [Etc]")
        for key, value in remaining.items():
            lines.append(f"{key}: {render(value)}")
    return "\n".join(lines) + "\n"


def load_tracker_config(path: str | Path | None = None) -> TrackerConfig:
    config_path = ensure_config_file(path)
    data = _load_simple_yaml(config_path)
    return tracker_config_from_dict(data)


def save_tracker_config(cfg: TrackerConfig, path: str | Path | None = None) -> Path:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_dump_simple_yaml(tracker_config_to_dict(cfg)), encoding="utf-8")
    return config_path
