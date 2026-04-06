from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tracker.yaml"


CONFIG_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "1. 추적",
        [
            ("forward_distance_m", "스텝 전진 거리", "한 스텝마다 앞으로 전진하는 거리"),
            ("max_track_length_m", "최대 추적 길이", "전체 실행에서 허용할 최대 추적 거리"),
            ("max_gap_distance_m", "최대 갭 길이", "관측이 끊겨도 버틸 최대 갭 거리"),
        ],
    ),
    (
        "2. 탐색 영역",
        [
            ("roi_forward_m", "전방 ROI 길이", "앞쪽으로 보는 ROI 길이"),
            ("roi_backward_m", "후방 ROI 길이", "뒤쪽으로 포함하는 ROI 길이"),
            ("roi_lateral_half_m", "좌우 ROI 반폭", "좌우로 보는 ROI 반폭"),
            ("corridor_half_width_m", "차선 복도 반폭", "같은 차선으로 보는 선호 복도 반폭"),
        ],
    ),
    (
        "3. 그리드",
        [
            ("grid_cell_size_m", "그리드 셀 크기", "로컬 BEV 그리드 셀 크기"),
            ("active_intensity_min", "활성 강도 최소값", "활성 셀로 인정할 최소 정규화 강도"),
            ("min_points_per_cell", "셀 최소 포인트 수", "활성 셀로 인정할 최소 포인트 수"),
            ("component_min_cells", "최소 셀 수", "후보 ridge가 가져야 하는 최소 셀 수"),
            ("component_min_span_m", "최소 길이", "후보 ridge가 가져야 하는 최소 진행 길이"),
            ("active_box_display_limit", "박스 표시 개수", "활성 셀 박스를 최대 몇 개까지 표시할지 (0=무제한)"),
        ],
    ),
    (
        "4. 후보",
        [
            ("stripe_width_m", "차선 폭 표시값", "오버레이와 프로파일에 쓰는 차선 폭"),
            ("candidate_lateral_sigma_m", "횡방향 오차 스케일", "후보 점수의 횡방향 오차 스케일"),
            ("candidate_heading_sigma_deg", "방향 오차 스케일", "후보 점수의 방향 오차 스케일"),
            ("candidate_min_score", "후보 최소 점수", "후보를 채택할 최소 점수"),
        ],
    ),
    (
        "5. 진행 방향",
        [
            ("heading_smoothing_alpha", "방향 보간 비율", "이전 방향과 새 방향을 섞는 비율"),
            ("max_heading_change_deg", "스텝 최대 회전각", "한 스텝에서 허용할 최대 방향 변화"),
        ],
    ),
    (
        "6. 높이",
        [
            ("use_z_clip", "로컬 Z 클립 사용", "추적 차선 주변 높이만 남길지 여부"),
            ("z_clip_half_range_m", "Z 클립 반범위", "로컬 Z 클립에 쓰는 높이 반범위"),
        ],
    ),
]


@dataclass
class TrackerConfig:
    forward_distance_m: float = 0.30
    max_track_length_m: float = 120.0
    max_gap_distance_m: float = 2.4

    roi_forward_m: float = 3.0
    roi_backward_m: float = 0.4
    roi_lateral_half_m: float = 0.9
    corridor_half_width_m: float = 0.18

    grid_cell_size_m: float = 0.05
    active_intensity_min: float = 0.45
    min_points_per_cell: int = 1
    component_min_cells: int = 4
    component_min_span_m: float = 0.35
    active_box_display_limit: int = 200

    stripe_width_m: float = 0.15
    candidate_lateral_sigma_m: float = 0.08
    candidate_heading_sigma_deg: float = 12.0
    candidate_min_score: float = 0.72

    heading_smoothing_alpha: float = 0.35
    max_heading_change_deg: float = 8.0

    use_z_clip: bool = True
    z_clip_half_range_m: float = 0.30


def tracker_config_to_dict(cfg: TrackerConfig) -> dict[str, Any]:
    return asdict(cfg)


def tracker_config_from_dict(data: dict[str, Any] | None) -> TrackerConfig:
    if not data:
        return TrackerConfig()

    data = dict(data)
    legacy_box_size = data.pop("lane_box_size_m", None)
    if legacy_box_size is not None:
        data.setdefault("stripe_width_m", legacy_box_size)

    legacy_map = {
        "graph_roi_forward_m": "roi_forward_m",
        "graph_roi_lateral_half_m": "roi_lateral_half_m",
        "graph_cell_size_m": "grid_cell_size_m",
        "spatial_grid_cell_size_m": "grid_cell_size_m",
        "graph_active_intensity_min": "active_intensity_min",
        "graph_min_cell_points": "min_points_per_cell",
        "graph_noise_min_component_cells": "component_min_cells",
        "lane_heading_min_span_m": "component_min_span_m",
        "segment_min_length_m": "component_min_span_m",
        "antenna_half_width_m": "corridor_half_width_m",
        "lane_box_width_m": "stripe_width_m",
        "heading_smoothing_alpha": "heading_smoothing_alpha",
        "heading_max_turn_deg": "max_heading_change_deg",
        "candidate_min_score": "candidate_min_score",
        "gap_forward_distance_m": "max_gap_distance_m",
        "use_z_clip": "use_z_clip",
        "z_clip_half_range_m": "z_clip_half_range_m",
    }
    for old_key, new_key in legacy_map.items():
        if new_key not in data and old_key in data:
            data[new_key] = data[old_key]

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
        for key, _label, comment in items:
            if key not in remaining:
                continue
            lines.append(f"# {comment}")
            lines.append(f"{key}: {render(remaining.pop(key))}")
        lines.append("")

    if remaining:
        lines.append("# [기타]")
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
