"""Tests for the eval harness.

Three checks: PositionSet roundtrips through disk; alphabeta hits exactly
1.0 oracle agreement (it *is* the oracle); random scores both below 1.0 and
greater than zero on a non-trivial set.
"""

from __future__ import annotations

import numpy as np

from main_street.agents import AlphaBetaAgentSpec, RandomAgentSpec, build
from main_street.core import GameSpec
from main_street.eval.metrics import score_agent
from main_street.eval.positions import (
    PositionSet,
    SourceSpec,
    assert_valid,
    build_position_set,
)


def _small_sources() -> list[SourceSpec]:
    return [
        SourceSpec(spec=GameSpec(n=5, schedule=(2, 3)), mode="all_reachable"),
        SourceSpec(spec=GameSpec(n=6, schedule=(1, 2, 2)), mode="all_reachable"),
    ]


def test_build_and_roundtrip(tmp_path):
    ps = build_position_set("small", _small_sources())
    assert_valid(ps)
    assert len(ps) > 0
    ps.save(root=tmp_path)
    loaded = PositionSet.load("small", root=tmp_path)
    assert_valid(loaded)
    assert len(loaded) == len(ps)
    assert np.array_equal(loaded.board, ps.board)
    assert np.array_equal(loaded.optimal_mask, ps.optimal_mask)
    assert np.array_equal(loaded.value, ps.value)


def test_alphabeta_is_perfect():
    ps = build_position_set("small", _small_sources())
    agent = build(AlphaBetaAgentSpec())
    score = score_agent(agent, ps, agent_label="alphabeta")
    assert score.oracle_agreement == 1.0


def test_random_is_imperfect_but_not_zero():
    ps = build_position_set("small", _small_sources())
    agent = build(RandomAgentSpec(seed=0))
    score = score_agent(agent, ps, agent_label="random")
    # On enough positions, random will sometimes land on an optimal cell by
    # chance (often the optimal set has several members), and will sometimes
    # miss. Both endpoints should hold.
    assert 0.0 < score.oracle_agreement < 1.0


def test_diagnostics_labels_propagate():
    sources = [
        SourceSpec(
            spec=GameSpec(n=5, schedule=(2, 3)),
            mode="initial_only",
            label="diag_a",
        ),
        SourceSpec(
            spec=GameSpec(n=6, schedule=(1, 2, 2)),
            mode="initial_only",
            label="diag_b",
        ),
    ]
    ps = build_position_set("diag", sources)
    assert ps.labels == ("diag_a", "diag_b")
    agent = build(AlphaBetaAgentSpec())
    score = score_agent(agent, ps, agent_label="alphabeta")
    assert score.per_label == {"diag_a": True, "diag_b": True}
