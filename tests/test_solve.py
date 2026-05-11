"""Tests for the exact solver.

The solver is the project's ground truth, so the test suite has to actually
verify it. We rely on three angles:

1. Manual small cases with hand-computable values.
2. Self-consistency: the value of `state` equals the value of the best move
   from `state` (applied to its child).
3. Random rollouts: a "best move" picked by the solver vs. any other legal
   move never decreases the solving player's value.
"""

from __future__ import annotations

import numpy as np
import pytest

from main_street.core import GameSpec, GameState, X, legal_actions, outcome, step
from main_street.solve import (
    Solver,
    build_table,
    load_table,
    save_table,
    search_with_depth,
    solve,
)


def test_trivial_one_cell_x_wins():
    # X places once on a 1-cell board: X wins (longest run 1, O has 0).
    spec = GameSpec(n=1, schedule=(1,))
    state = GameState.initial(spec)
    result = solve(state)
    assert result.value == 1
    assert result.best_cell == 0


def test_two_cell_one_each_x_takes_rightmost():
    # N=2, schedule=(1,1). X plays first and can take cell 1; O must take cell 0.
    # Both have 1-runs; X is rightmost, so X wins.
    spec = GameSpec(n=2, schedule=(1, 1))
    result = solve(GameState.initial(spec))
    assert result.value == 1
    assert result.best_cell == 1


def test_three_cell_one_each_two_each():
    spec = GameSpec(n=3, schedule=(1, 2))
    state = GameState.initial(spec)
    # O has two placements on a 3-cell board. Whatever X does, O can take the
    # two remaining cells and get a 2-run.
    result = solve(state)
    assert result.value == -1


def test_terminal_value_matches_outcome():
    spec = GameSpec(n=3, schedule=(1, 1, 1))
    state = step(step(step(GameState.initial(spec), 0), 1), 2)
    assert state.is_terminal
    result = solve(state)
    assert result.value == outcome(state)
    assert result.best_cell == -1


def test_per_cell_values_populated_for_every_legal_move():
    spec = GameSpec(n=4, schedule=(1, 1, 1, 1))
    result = solve(GameState.initial(spec))
    assert set(result.per_cell_values.keys()) == {0, 1, 2, 3}
    # All values are ±1 at terminal-reachable states.
    assert all(v in (-1, 1) for v in result.per_cell_values.values())


@pytest.mark.parametrize(
    ("n", "schedule"),
    [
        (3, (1, 1)),
        (4, (1, 1, 1)),
        (4, (2, 2)),
        (5, (1, 2, 1)),
        (6, (2, 2, 1)),
    ],
)
def test_root_value_matches_best_child_value(n: int, schedule: tuple[int, ...]):
    spec = GameSpec(n=n, schedule=schedule)
    solver = Solver()
    root = GameState.initial(spec)
    result = solver.solve(root)
    child = step(root, result.best_cell)
    child_result = solver.solve(child)
    assert result.value == child_result.value


@pytest.mark.parametrize("seed", range(20))
def test_best_move_is_not_dominated(seed: int):
    """For random small games, no legal move strictly beats the solver's choice."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 6))
    # Schedule with sum <= n.
    rem, sched = n, []
    while rem > 0 and len(sched) < 4:
        k = int(rng.integers(1, min(rem, 2) + 1))
        sched.append(k)
        rem -= k
        if rng.random() < 0.4:
            break
    spec = GameSpec(n=n, schedule=tuple(sched))
    state = GameState.initial(spec)
    # Take a random walk a few steps in, then solve.
    while not state.is_terminal and rng.random() < 0.5:
        cells = legal_actions(state)
        state = step(state, int(rng.choice(cells)))
    if state.is_terminal:
        return  # nothing to check

    solver = Solver()
    result = solver.solve(state)
    maximizing = state.current_player == X
    best_value = result.per_cell_values[result.best_cell]
    for cell, val in result.per_cell_values.items():
        if maximizing:
            assert val <= best_value, f"cell {cell} value {val} > best {best_value}"
        else:
            assert val >= best_value, f"cell {cell} value {val} < best {best_value}"


def test_table_roundtrip(tmp_path):
    spec = GameSpec(n=4, schedule=(1, 2))
    table = build_table(spec)
    save_table(table, root=tmp_path)
    loaded = load_table(spec, root=tmp_path)
    assert loaded is not None
    assert loaded.value == table.value
    assert loaded.entries == table.entries


def test_table_lookup_matches_solver(tmp_path):
    spec = GameSpec(n=4, schedule=(1, 1, 1))
    table = build_table(spec)
    solver = Solver()
    # Spot-check a few reachable states.
    s = GameState.initial(spec)
    for _ in range(3):
        if s.is_terminal:
            break
        lookup = table.lookup(s)
        assert lookup is not None
        v, c = lookup
        assert solver.solve(s).value == v
        if c >= 0:
            s = step(s, c)


def test_search_with_depth_none_equals_solve():
    spec = GameSpec(n=4, schedule=(1, 1, 1))
    state = GameState.initial(spec)
    assert search_with_depth(state, None).value == solve(state).value


def test_search_with_depth_terminal():
    spec = GameSpec(n=2, schedule=(1, 1))
    state = step(step(GameState.initial(spec), 0), 1)
    assert state.is_terminal
    r = search_with_depth(state, 3)
    assert r.value == outcome(state)
    assert r.best_cell == -1
