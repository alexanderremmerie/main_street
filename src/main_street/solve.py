"""Exact solver for the placement game.

Negamax with alpha-beta and a transposition table. Values are from X's
perspective: +1 if X wins under best play, -1 if O wins. Ties are impossible
at terminal in a valid game.

This is the project's ground truth: cheap on small `(N, schedule)`, and every
downstream component is validated against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .core import GameSpec, GameState, O, X, legal_actions, longest_run, outcome, step

# Stored as terminal sentinel values; we use a wide window so any non-terminal
# heuristic stays inside it.
_WIN: Final[int] = 1
_LOSS: Final[int] = -1

_EXACT: Final[int] = 0
_LOWER: Final[int] = 1
_UPPER: Final[int] = 2


@dataclass(slots=True)
class _TTEntry:
    value: int
    flag: int  # _EXACT | _LOWER | _UPPER
    best_cell: int  # -1 if none (terminal entry)


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Game-theoretic value and the best move from X's perspective.

    `value` is +1 if X wins under perfect play from `state`, -1 if O wins.
    `best_cell` is one optimal move for the side to move (X or O).
    `per_cell_values` maps each legal cell to the value that results from
    playing it (still from X's perspective). Useful for UI overlays.
    """

    value: int
    best_cell: int
    per_cell_values: dict[int, int]


class Solver:
    """A solver instance carries one transposition table; reuse it across
    queries on the same spec for big speedups."""

    __slots__ = ("_tt",)

    def __init__(self) -> None:
        self._tt: dict[GameState, _TTEntry] = {}

    def solve(self, state: GameState) -> SolveResult:
        if state.is_terminal:
            v = outcome(state)
            return SolveResult(value=v, best_cell=-1, per_cell_values={})

        # Evaluate every root move with a full (alpha, beta) window so each
        # per-cell value is exact (not a cutoff bound). Right-to-left order
        # means the rightmost optimal cell wins ties deterministically.
        per_cell: dict[int, int] = {}
        cells = self._ordered_root_moves(state)
        for c in cells:
            per_cell[int(c)] = self._negamax(step(state, int(c)), _LOSS - 1, _WIN + 1)

        maximizing = state.current_player == X
        ordered = sorted(per_cell.items(), key=lambda kv: kv[0], reverse=True)
        best_cell, best_val = ordered[0]
        for c, v in ordered:
            if (maximizing and v > best_val) or (not maximizing and v < best_val):
                best_val, best_cell = v, c
        return SolveResult(value=best_val, best_cell=best_cell, per_cell_values=per_cell)

    def _ordered_root_moves(self, state: GameState) -> list[int]:
        # Right-to-left ordering matches the game's tie-break; the TT-best
        # move (if any) takes priority.
        cells = sorted(legal_actions(state).tolist(), reverse=True)
        entry = self._tt.get(state)
        if entry is not None and entry.best_cell >= 0 and entry.best_cell in cells:
            cells.remove(entry.best_cell)
            cells.insert(0, entry.best_cell)
        return cells

    def _negamax(self, state: GameState, alpha: int, beta: int) -> int:
        if state.is_terminal:
            return outcome(state)

        tt = self._tt.get(state)
        if tt is not None:
            if tt.flag == _EXACT:
                return tt.value
            if tt.flag == _LOWER and tt.value >= beta:
                return tt.value
            if tt.flag == _UPPER and tt.value <= alpha:
                return tt.value

        original_alpha, original_beta = alpha, beta
        cells = sorted(legal_actions(state).tolist(), reverse=True)
        if tt is not None and tt.best_cell >= 0 and tt.best_cell in cells:
            cells.remove(tt.best_cell)
            cells.insert(0, tt.best_cell)

        maximizing = state.current_player == X
        best_val = _LOSS - 1 if maximizing else _WIN + 1
        best_cell = cells[0]

        for c in cells:
            v = self._negamax(step(state, c), alpha, beta)
            if maximizing:
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
        self._tt[state] = _TTEntry(value=best_val, flag=flag, best_cell=best_cell)
        return best_val


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


def build_table(spec: GameSpec) -> SolvedTable:
    """Solve every reachable state under `spec`. Returns a table that can be
    persisted and queried in O(1) per state."""
    solver = Solver()
    entries: dict[StateKey, tuple[int, int]] = {}
    initial = GameState.initial(spec)
    root_val = solver.solve(initial).value

    # Walk every reachable state and store its solved value/best move. The
    # solver's TT keeps each unique state O(1) after first evaluation.
    seen: set[GameState] = set()
    frontier: list[GameState] = [initial]
    while frontier:
        s = frontier.pop()
        if s in seen:
            continue
        seen.add(s)
        if s.is_terminal:
            entries[_state_key(s)] = (outcome(s), -1)
            continue
        result = solver.solve(s)
        entries[_state_key(s)] = (result.value, result.best_cell)
        for c in legal_actions(s):
            frontier.append(step(s, int(c)))
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

    # Per-cell values are signs of the heuristic so callers can rank moves
    # but shouldn't treat them as exact win/loss.
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


