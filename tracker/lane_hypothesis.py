from __future__ import annotations

from dataclasses import dataclass, field

from ..core.types import StepScores
from .lane_state import LaneState


@dataclass
class LaneHypothesis:
    state: LaneState
    total_score: float
    last_scores: StepScores = field(default_factory=StepScores)
    age_steps: int = 0
    alive: bool = True
    debug_last: dict = field(default_factory=dict)
