"""Evaluation harness.

`PositionSet` is a labeled bag of game positions (state + oracle value +
optimal-move mask). Metrics in `metrics.py` are pure functions of
`(agent, position_set)`. Sets are persisted to `data/eval/<name>/` and
versioned with the codebase (a held-out set is committed once and never
regenerated, so we cannot accidentally train on its specs).
"""

from .metrics import per_spec_agreement, score_agent
from .positions import PositionSet, build_position_set

__all__ = ["PositionSet", "build_position_set", "score_agent", "per_spec_agreement"]
