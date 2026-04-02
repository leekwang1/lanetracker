from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tracker.yaml"


CONFIG_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "1. 기본 추적",
        [
            ("forward_distance_m", "트래킹 한 스텝에서 전진하는 거리(m)"),
            ("max_track_length_m", "run_full 실행 시 허용되는 최대 누적 추적 거리(m)"),
        ],
    ),
    (
        "2. 로컬 ROI 및 활성 셀",
        [
            ("graph_roi_forward_m", "전방 ROI 길이(m)"),
            ("graph_roi_lateral_half_m", "좌우 ROI 반폭(m)"),
            ("graph_cell_size_m", "BEV 셀 크기(m). 작을수록 더 촘촘하지만 느려짐"),
            ("graph_active_intensity_min", "활성 셀로 인정할 최소 정규화 intensity"),
            ("graph_active_contrast_min", "활성 셀로 인정할 최소 로컬 contrast"),
            ("graph_min_cell_points", "셀 하나를 유효하게 볼 최소 포인트 수"),
        ],
    ),
    (
        "3. 활성 셀 노이즈 제거",
        [
            ("graph_noise_min_neighbors", "고립 셀을 유지하기 위해 필요한 최소 활성 이웃 수"),
            ("graph_noise_min_component_cells", "노이즈 제거 후 유지할 최소 연결 활성 셀 개수"),
        ],
    ),
    (
        "4. 차선 박스",
        [
            ("lane_box_length_m", "검출할 차선 박스의 종방향 길이(m)"),
            ("lane_box_width_m", "검출할 차선 박스의 횡방향 폭(m)"),
            ("lane_box_min_active_cells", "차선 박스 1개를 만들기 위한 최소 활성 셀 수"),
        ],
    ),
    (
        "5. 안테나 엣지",
        [
            ("antenna_length_m", "차선 박스를 연결할 때 쓰는 전방 안테나 corridor 길이(m)"),
            ("antenna_half_width_m", "다음 차선 박스를 허용할 안테나 corridor 반폭(m)"),
            ("antenna_heading_tolerance_deg", "차선 박스 연결 시 허용할 최대 진행방향 차이(도)"),
        ],
    ),
    (
        "6. 빔 탐색",
        [
            ("graph_beam_width", "스텝마다 유지할 경로 가설 수"),
            ("graph_beam_horizon_nodes", "앞으로 탐색할 노드 깊이"),
            ("graph_beam_branching", "노드 하나에서 확장할 최대 분기 수"),
        ],
    ),
    (
        "7. 점수 가중치",
        [
            ("graph_intensity_weight", "강한 intensity에 줄 가중치"),
            ("graph_contrast_weight", "로컬 contrast에 줄 가중치"),
            ("graph_direction_weight", "진행방향 및 곡률 연속성에 줄 가중치"),
            ("graph_distance_weight", "과도한 거리 점프를 억제하는 가중치"),
            ("graph_history_weight", "최근 추적 이력과의 일관성 가중치"),
            ("graph_period_weight", "점선 주기 일관성 가중치"),
            ("graph_crosswalk_penalty", "횡단보도처럼 보이는 패턴에 대한 패널티"),
        ],
    ),
    (
        "8. 프로파일 및 차선 형상",
        [
            ("profile_lateral_half_m", "단면 프로파일에 사용할 좌우 반폭(m)"),
            ("profile_along_half_m", "프로파일 계산에 사용할 종방향 반길이(m)"),
            ("profile_bin_size_m", "프로파일의 좌우 bin 크기(m)"),
            ("twin_edge_min_width_m", "쌍 에지 후보로 인정할 최소 차선 폭(m)"),
            ("twin_edge_max_width_m", "쌍 에지 후보로 인정할 최대 차선 폭(m)"),
            ("edge_grad_min", "쌍 에지를 검출할 최소 gradient"),
            ("candidate_min_support", "프로파일 후보를 인정할 최소 지지 포인트 수"),
            ("candidate_min_score", "후보를 채택할 최소 총점"),
        ],
    ),
    (
        "9. 종방향 신호 및 점선 패턴",
        [
            ("along_signal_half_m", "1D 종방향 신호를 만들 때 사용할 반길이(m)"),
            ("along_signal_bin_m", "종방향 신호 bin 크기(m)"),
            ("along_signal_lateral_half_m", "종방향 신호를 수집할 좌우 반폭(m)"),
            ("autocorr_min_period_m", "유효한 점선 주기로 볼 최소 거리(m)"),
            ("autocorr_max_period_m", "유효한 점선 주기로 볼 최대 거리(m)"),
            ("dashed_autocorr_min", "점선으로 분류할 최소 자기상관 점수"),
            ("solid_occupancy_min", "실선으로 분류할 최소 점유율"),
        ],
    ),
    (
        "10. 끊김 보정 및 연속성",
        [
            ("gap_forward_distance_m", "유효 관측 없이 보정 이동할 최대 거리(m)"),
            ("continuity_node_count", "연속성 계산에 사용할 최근 노드 수"),
            ("continuity_strength", "연속성 패널티 강도"),
        ],
    ),
    (
        "11. Z 필터링",
        [
            ("use_z_clip", "true이면 현재 차선 높이 근처 포인트만 사용"),
            ("z_clip_half_range_m", "포인트 Z 클리핑에 사용할 반범위(m)"),
        ],
    ),
    (
        "12. 횡단보도 처리",
        [
            ("crosswalk_stop_enabled", "true이면 횡단보도 패턴 검출 시 정지"),
            ("crosswalk_lookahead_m", "횡단보도 검사용 전방 거리(m)"),
            ("crosswalk_lateral_half_m", "횡단보도 검사용 좌우 반폭(m)"),
            ("crosswalk_min_peaks", "횡단보도로 분류할 최소 반복 피크 수"),
        ],
    ),
    (
        "13. 내부 그리드",
        [
            ("spatial_grid_cell_size_m", "이웃 질의에 사용할 내부 spatial-grid 셀 크기(m)"),
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
    graph_noise_min_neighbors: int = 2
    graph_noise_min_component_cells: int = 3
    lane_box_length_m: float = 0.15
    lane_box_width_m: float = 0.15
    lane_box_min_active_cells: int = 2
    antenna_length_m: float = 2.0
    antenna_half_width_m: float = 0.30
    antenna_heading_tolerance_deg: float = 24.0
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

    data = dict(data)
    legacy_box_size = data.pop("lane_box_size_m", None)
    if legacy_box_size is not None:
        data.setdefault("lane_box_length_m", legacy_box_size)
        data.setdefault("lane_box_width_m", legacy_box_size)

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
