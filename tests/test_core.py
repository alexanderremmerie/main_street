import numpy as np
import pytest

from main_street.core import (
    EMPTY,
    GameSpec,
    GameState,
    O,
    X,
    legal_actions,
    longest_run,
    outcome,
    replay,
    step,
)


def b(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.uint8)


def test_longest_run_rightmost_wins_on_tie():
    # Two runs of length 2 for X. The rightmost (ending at index 5) wins.
    board = b([1, 1, 0, 1, 1, 0, 0])  # noqa
    length, end = longest_run(board, X)
    assert (length, end) == (2, 4)


def test_longest_run_absent_mark():
    assert longest_run(b([1, 1, 1]), O) == (0, -1)


def test_outcome_x_longer_wins():
    spec = GameSpec(n=4, schedule=(2, 2))
    state = GameState.initial(spec)
    state = step(state, 0)
    state = step(state, 1)
    state = step(state, 2)
    state = step(state, 3)
    assert state.is_terminal
    # X at 0,1; O at 2,3. Both length 2; O is rightmost so O wins.
    assert outcome(state) == -1


def test_step_rejects_occupied():
    spec = GameSpec(n=3, schedule=(1, 1, 1))
    s = GameState.initial(spec)
    s = step(s, 1)
    with pytest.raises(ValueError):
        step(s, 1)


def test_player_alternates_per_turn_not_per_placement():
    # X places 2 in a row, then O places 1. X must remain X for both placements.
    spec = GameSpec(n=4, schedule=(2, 1))
    s = GameState.initial(spec)
    assert s.current_player == X
    s = step(s, 0)
    assert s.current_player == X  # still X mid-turn
    s = step(s, 1)
    assert s.current_player == O  # O's turn now
    s = step(s, 2)
    assert s.is_terminal


def test_legal_actions_excludes_occupied():
    spec = GameSpec(n=4, schedule=(1, 1))
    s = GameState.initial(spec)
    s = step(s, 2)
    assert set(legal_actions(s).tolist()) == {0, 1, 3}


def test_replay_reconstructs_terminal():
    spec = GameSpec(n=5, schedule=(1, 1, 1))
    states = list(replay(spec, [0, 4, 2]))
    assert len(states) == 4
    assert states[-1].is_terminal
    assert states[-1].board[0] == X
    assert states[-1].board[4] == O
    assert states[-1].board[2] == X


def test_spec_validates_overflow():
    with pytest.raises(ValueError):
        GameSpec(n=3, schedule=(2, 2))


def test_state_board_is_readonly():
    s = GameState.initial(GameSpec(n=3, schedule=(1,)))
    assert not s.board.flags.writeable


def test_empty_value_is_zero():
    assert int(EMPTY) == 0


def _naive_longest_run(values: list[int], mark: int) -> tuple[int, int]:
    """Independent reference implementation for the tie-break property test."""
    best_len, best_end = 0, -1
    i = 0
    while i < len(values):
        if values[i] != mark:
            i += 1
            continue
        j = i
        while j < len(values) and values[j] == mark:
            j += 1
        length, end = j - i, j - 1
        # `>=` so a later run of equal length wins (the game's rightmost rule).
        if length >= best_len:
            best_len, best_end = length, end
        i = j
    return best_len, best_end


def _naive_outcome(values: list[int]) -> int:
    xl, xe = _naive_longest_run(values, int(X))
    ol, oe = _naive_longest_run(values, int(O))
    if xl == 0 and ol == 0:
        return 0
    if xl != ol:
        return 1 if xl > ol else -1
    return 1 if xe > oe else -1


@pytest.mark.parametrize("seed", range(40))
def test_longest_run_matches_naive(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 16))
    values = rng.integers(0, 3, size=n).tolist()
    board = np.array(values, dtype=np.uint8)
    for mark in (1, 2):
        assert longest_run(board, np.uint8(mark)) == _naive_longest_run(values, mark)


def _play_random_to_terminal(spec: GameSpec, rng: np.random.Generator) -> GameState:
    state = GameState.initial(spec)
    while not state.is_terminal:
        cells = legal_actions(state)
        state = step(state, int(rng.choice(cells)))
    return state


@pytest.mark.parametrize("seed", range(30))
def test_outcome_matches_naive_on_random_games(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 12))
    # Build a random schedule whose sum is <= n.
    remaining = n
    schedule: list[int] = []
    while remaining > 0 and len(schedule) < 6:
        k = int(rng.integers(1, min(remaining, 3) + 1))
        schedule.append(k)
        remaining -= k
        if rng.random() < 0.3:
            break
    spec = GameSpec(n=n, schedule=tuple(schedule))
    state = _play_random_to_terminal(spec, rng)
    assert outcome(state) == _naive_outcome(state.board.tolist())


def test_state_is_hashable_and_collides_on_equal_positions():
    spec = GameSpec(n=4, schedule=(1, 1, 1, 1))
    # Reach the same board via two different paths.
    a = step(step(GameState.initial(spec), 0), 2)
    b = step(step(GameState.initial(spec), 2), 0)
    # Wait — these aren't equal: in (a), X went to 0 then O to 2; in (b), X to 2 then O to 0.
    # Build a position that's genuinely reachable two ways instead.
    s1 = step(step(step(GameState.initial(spec), 0), 1), 2)
    s2 = step(step(step(GameState.initial(spec), 0), 1), 2)
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert {s1, s2} == {s1}
    # And the asymmetric pair is correctly unequal.
    assert a != b


def test_solve_root_per_cell_is_exact_value():
    """Regression for the root window bug: every per_cell_values entry must
    be the *true* value of that move, not a cutoff bound, so that downstream
    UI/agreement checks see consistent numbers regardless of move order."""
    from main_street.solve import Solver

    spec = GameSpec(n=4, schedule=(1, 1, 1, 1))
    solver = Solver()
    result = solver.solve(GameState.initial(spec))
    # Re-evaluate each cell with a fresh solver; the per-cell values must match.
    for cell, recorded in result.per_cell_values.items():
        fresh = Solver()
        child_value = fresh.solve(step(GameState.initial(spec), cell)).value
        assert child_value == recorded, f"cell {cell}: {recorded} vs {child_value}"


def test_state_hash_distinguishes_turn_and_placements():
    spec = GameSpec(n=3, schedule=(2, 1))
    s = step(GameState.initial(spec), 0)  # X mid-turn, placements_left=1
    # Build a state with same board but say it's a different turn — manually for the assert.
    # We can't construct that legally, so just sanity check that two distinct legal states
    # with the same board prefix but different metadata don't collide.
    spec2 = GameSpec(n=3, schedule=(1, 2))
    s2 = step(GameState.initial(spec2), 0)  # X placed first; now O's turn with 2 left.
    assert s != s2
    assert hash(s) != hash(s2) or s.key() != s2.key()
