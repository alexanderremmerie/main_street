"""Exact solver for the placement game.

Negamax with alpha-beta and a transposition table. Values are from X's
perspective: +1 if X wins under best play, -1 if O wins. Ties are impossible
at terminal in a valid game.

This is the project's ground truth: cheap on small `(N, schedule)`, and every
downstream component is validated against it.

Internal representation. The hot inner loop works on two Python `int`
bitboards `xs` (cells X has played) and `os_` (cells O has played) — one bit
per cell. This avoids the per-lookup `board.tobytes()` allocation the naïve
`GameState`-keyed TT pays. `Solver.solve(state)` is the public boundary; it
packs once on entry and unpacks back to cell indices on exit.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .core import GameSpec, GameState, O, X, legal_actions, longest_run, outcome, step

# Sentinel values stored at terminals. The (alpha, beta) window opens wider
# than this so a search never returns a value at the window boundary except
# at a true terminal.
_WIN: Final[int] = 1
_LOSS: Final[int] = -1

_EXACT: Final[int] = 0
_LOWER: Final[int] = 1
_UPPER: Final[int] = 2


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Game-theoretic value and the best move from the side-to-move's view.

    `value` is +1 if X wins under perfect play from `state`, -1 if O wins.
    `best_cell` is one optimal move for the side to move (X or O). With ties
    between equally-good cells, the rightmost cell index wins, matching the
    game's tie-break direction.
    `per_cell_values` maps each legal cell to the value that results from
    playing it (still from X's perspective). Useful for UI overlays.
    """

    value: int
    best_cell: int
    per_cell_values: dict[int, int]


# ---------- Bit-packed helpers (hot path) -----------------------------------


def _pack(board: np.ndarray) -> tuple[int, int]:
    """Convert a board array into (xs, os) bitboards.

    Called once per public `Solver.solve` entry. Not on the hot path.
    """
    xs = 0
    os_ = 0
    for i in range(int(board.shape[0])):
        v = int(board[i])
        if v == 1:
            xs |= 1 << i
        elif v == 2:
            os_ |= 1 << i
    return xs, os_


def _longest_run_bits(bits: int, n: int) -> tuple[int, int]:
    """Length and rightmost-end index of the longest run of set bits in
    `bits`, scanning positions 0..n-1. Ties resolve to the rightmost run
    (matching the game's tie-break)."""
    best_len = 0
    best_end = -1
    cur = 0
    for i in range(n):
        if (bits >> i) & 1:
            cur += 1
            if cur >= best_len:
                best_len, best_end = cur, i
        else:
            cur = 0
    return best_len, best_end


def _terminal_value(xs: int, os_: int, n: int) -> int:
    xl, xe = _longest_run_bits(xs, n)
    ol, oe = _longest_run_bits(os_, n)
    if xl == 0 and ol == 0:
        return 0
    if xl != ol:
        return 1 if xl > ol else -1
    return 1 if xe > oe else -1


def _empty_cells_rtl(empty: int, n: int) -> list[int]:
    """Empty cell indices in right-to-left order. Right-to-left matches the
    game's tie-break: when two moves are equally optimal, we want to discover
    the rightmost one first."""
    return [i for i in range(n - 1, -1, -1) if (empty >> i) & 1]


# ---------- Solver ----------------------------------------------------------


class Solver:
    """A solver instance carries one transposition table; reuse it across
    queries on the same spec for big speedups. The TT key omits the spec, so
    the table is cleared automatically if `solve` is called with a different
    spec than the previous one."""

    __slots__ = ("_tt", "_schedule", "_n", "_full")

    def __init__(self) -> None:
        # key: (turn_idx, placements_left, xs, os_) -> (value, flag, best_cell)
        self._tt: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}
        self._schedule: tuple[int, ...] = ()
        self._n: int = 0
        self._full: int = 0

    @property
    def tt_size(self) -> int:
        return len(self._tt)

    def _configure(self, spec: GameSpec) -> None:
        if spec.n == self._n and spec.schedule == self._schedule:
            return
        self._tt.clear()
        self._n = spec.n
        self._schedule = spec.schedule
        self._full = (1 << spec.n) - 1

    def solve(self, state: GameState) -> SolveResult:
        if state.is_terminal:
            return SolveResult(value=outcome(state), best_cell=-1, per_cell_values={})

        self._configure(state.spec)
        xs, os_ = _pack(state.board)
        turn_idx = state.turn_idx
        placements_left = state.placements_left
        empty = (~(xs | os_)) & self._full
        player_x = (turn_idx % 2) == 0

        # Evaluate every legal root move with a full (alpha, beta) window so
        # per-cell values are exact, not cutoff bounds. Right-to-left order
        # makes the rightmost optimal cell win ties.
        per_cell: dict[int, int] = {}
        cells = self._ordered_root_cells(turn_idx, placements_left, xs, os_, empty)

        best_val = _LOSS - 1 if player_x else _WIN + 1
        best_cell = cells[0]
        new_turn, new_left = _advance(turn_idx, placements_left, self._schedule)
        for c in cells:
            if player_x:
                xs2, os2 = xs | (1 << c), os_
            else:
                xs2, os2 = xs, os_ | (1 << c)
            v = self._negamax(xs2, os2, new_turn, new_left, _LOSS - 1, _WIN + 1)
            per_cell[c] = v
            if player_x:
                if v > best_val:
                    best_val, best_cell = v, c
            else:
                if v < best_val:
                    best_val, best_cell = v, c

        return SolveResult(value=best_val, best_cell=best_cell, per_cell_values=per_cell)

    def _ordered_root_cells(
        self, turn_idx: int, placements_left: int, xs: int, os_: int, empty: int
    ) -> list[int]:
        cells = _empty_cells_rtl(empty, self._n)
        tt = self._tt.get((turn_idx, placements_left, xs, os_))
        if tt is not None and tt[2] >= 0 and tt[2] in cells:
            cells.remove(tt[2])
            cells.insert(0, tt[2])
        return cells

    def _negamax(
        self,
        xs: int,
        os_: int,
        turn_idx: int,
        placements_left: int,
        alpha: int,
        beta: int,
    ) -> int:
        if turn_idx >= len(self._schedule):
            return _terminal_value(xs, os_, self._n)

        key = (turn_idx, placements_left, xs, os_)
        tt = self._tt.get(key)
        if tt is not None:
            v, flag, _ = tt
            if flag == _EXACT:
                return v
            if flag == _LOWER and v >= beta:
                return v
            if flag == _UPPER and v <= alpha:
                return v

        original_alpha, original_beta = alpha, beta
        player_x = (turn_idx % 2) == 0
        empty = (~(xs | os_)) & self._full
        cells = _empty_cells_rtl(empty, self._n)
        if tt is not None and tt[2] >= 0 and tt[2] in cells:
            cells.remove(tt[2])
            cells.insert(0, tt[2])

        best_val = _LOSS - 1 if player_x else _WIN + 1
        best_cell = cells[0]
        new_turn, new_left = _advance(turn_idx, placements_left, self._schedule)
        for c in cells:
            if player_x:
                xs2, os2 = xs | (1 << c), os_
            else:
                xs2, os2 = xs, os_ | (1 << c)
            v = self._negamax(xs2, os2, new_turn, new_left, alpha, beta)
            if player_x:
                if v > best_val:
                    best_val, best_cell = v, c
                if best_val > alpha:
                    alpha = best_val
            else:
                if v < best_val:
                    best_val, best_cell = v, c
                if best_val < beta:
                    beta = best_val
            if alpha >= beta:
                break

        if best_val <= original_alpha:
            flag = _UPPER
        elif best_val >= original_beta:
            flag = _LOWER
        else:
            flag = _EXACT
        self._tt[key] = (best_val, flag, best_cell)
        return best_val


def _advance(
    turn_idx: int, placements_left: int, schedule: tuple[int, ...]
) -> tuple[int, int]:
    placements_left -= 1
    if placements_left == 0:
        turn_idx += 1
        if turn_idx < len(schedule):
            placements_left = schedule[turn_idx]
    return turn_idx, placements_left


def solve(state: GameState) -> SolveResult:
    """Convenience for one-off queries. For repeated queries on the same spec,
    instantiate `Solver` once and reuse it."""
    return Solver().solve(state)


# ---------- Persisted policy tables -----------------------------------------
#
# A solved table enumerates the game-theoretic value at every reachable state
# for a given spec, plus one optimal move per non-terminal state. Persisted as
# JSON to `data/solved/` so the dashboard can answer "what's the truth here?"
# in O(1) without re-running the solver.

DEFAULT_TABLE_DIR: Final[Path] = Path("data") / "solved"

StateKey = tuple[int, int, int, bytes]


def _state_key(state: GameState) -> StateKey:
    return (
        state.spec.n,
        state.turn_idx,
        state.placements_left,
        state.board.tobytes(),
    )


@dataclass(slots=True)
class SolvedTable:
    spec: GameSpec
    value: int  # value from the initial state
    # state-key -> (value, best_cell). best_cell == -1 at terminals.
    entries: dict[StateKey, tuple[int, int]]

    def lookup(self, state: GameState) -> tuple[int, int] | None:
        return self.entries.get(_state_key(state))


def reachable_states(spec: GameSpec) -> Iterator[GameState]:
    """Yield every reachable state under `spec` exactly once, terminals included.

    DFS traversal, deduplicated by state identity. Used by `build_table` and by
    the eval position-set builder.
    """
    seen: set[GameState] = set()
    stack: list[GameState] = [GameState.initial(spec)]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        yield s
        if not s.is_terminal:
            for c in legal_actions(s):
                stack.append(step(s, int(c)))


def build_table(spec: GameSpec) -> SolvedTable:
    """Solve every reachable state under `spec`. Returns a table that can be
    persisted and queried in O(1) per state."""
    solver = Solver()
    entries: dict[StateKey, tuple[int, int]] = {}
    root_val = solver.solve(GameState.initial(spec)).value

    for s in reachable_states(spec):
        if s.is_terminal:
            entries[_state_key(s)] = (outcome(s), -1)
        else:
            result = solver.solve(s)
            entries[_state_key(s)] = (result.value, result.best_cell)
    return SolvedTable(spec=spec, value=root_val, entries=entries)


def table_path(spec: GameSpec, root: Path = DEFAULT_TABLE_DIR) -> Path:
    sched = "-".join(str(k) for k in spec.schedule)
    return root / f"n{spec.n}_s{sched}.json"


def save_table(table: SolvedTable, root: Path = DEFAULT_TABLE_DIR) -> Path:
    p = table_path(table.spec, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": {"n": table.spec.n, "schedule": list(table.spec.schedule)},
        "value": table.value,
        "entries": [
            [k[0], k[1], k[2], k[3].hex(), v[0], v[1]]
            for k, v in table.entries.items()
        ],
    }
    p.write_text(json.dumps(payload))
    return p


def load_table(spec: GameSpec, root: Path = DEFAULT_TABLE_DIR) -> SolvedTable | None:
    p = table_path(spec, root)
    try:
        payload = json.loads(p.read_text())
    except FileNotFoundError:
        return None
    entries: dict[StateKey, tuple[int, int]] = {}
    for n, turn, left, board_hex, value, best in payload["entries"]:
        entries[(int(n), int(turn), int(left), bytes.fromhex(board_hex))] = (
            int(value),
            int(best),
        )
    return SolvedTable(spec=spec, value=int(payload["value"]), entries=entries)


# ---------- Heuristic for partial-depth search ------------------------------


def heuristic(state: GameState) -> float:
    """Linear position score from X's perspective. Used by the agent only when
    depth-limited; the exact solver never calls this. Bounded to (-1, 1)."""
    xl, xe = longest_run(state.board, X)
    ol, oe = longest_run(state.board, O)
    if xl == 0 and ol == 0:
        return 0.0
    if xl != ol:
        return 0.5 * float(np.tanh(0.5 * (xl - ol)))
    return 0.05 if xe > oe else -0.05


# ---------- Depth-limited search for the dashboard agent --------------------


def search_with_depth(state: GameState, depth: int | None) -> SolveResult:
    """Exact solve when `depth is None`; depth-limited negamax with a
    heuristic otherwise (used by the `alphabeta` agent on large boards)."""
    if depth is None:
        return solve(state)
    return _depth_limited(state, depth)


def _depth_limited(state: GameState, depth: int) -> SolveResult:
    if state.is_terminal:
        return SolveResult(value=outcome(state), best_cell=-1, per_cell_values={})

    per_cell: dict[int, float] = {}
    cells = sorted(legal_actions(state).tolist(), reverse=True)
    maximizing = state.current_player == X
    alpha, beta = -2.0, 2.0
    best_cell = int(cells[0])
    best_val = -2.0 if maximizing else 2.0
    for c in cells:
        v = _negamax_h(step(state, int(c)), depth - 1, alpha, beta)
        per_cell[int(c)] = v
        if maximizing:
            if v > best_val:
                best_val, best_cell = v, int(c)
            if best_val > alpha:
                alpha = best_val
        else:
            if v < best_val:
                best_val, best_cell = v, int(c)
            if best_val < beta:
                beta = best_val

    return SolveResult(
        value=int(np.sign(best_val)) if best_val != 0 else 0,
        best_cell=best_cell,
        per_cell_values={k: int(np.sign(v)) if v != 0 else 0 for k, v in per_cell.items()},
    )


def _negamax_h(state: GameState, depth: int, alpha: float, beta: float) -> float:
    if state.is_terminal:
        return float(outcome(state))
    if depth == 0:
        return heuristic(state)
    cells = sorted(legal_actions(state).tolist(), reverse=True)
    maximizing = state.current_player == X
    best = -2.0 if maximizing else 2.0
    for c in cells:
        v = _negamax_h(step(state, c), depth - 1, alpha, beta)
        if maximizing:
            if v > best:
                best = v
            if best > alpha:
                alpha = best
        else:
            if v < best:
                best = v
            if best < beta:
                beta = best
        if alpha >= beta:
            break
    return best
