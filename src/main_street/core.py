"""Game core: spec, state, transitions, scoring.

The single hot-path module. `GameSpec` is pydantic (it crosses the wire and is
created once per game), but `GameState` is a frozen dataclass wrapping a numpy
array so steps are cheap.

Player encoding: 0 empty, 1 X, 2 O. The current player is derived from
`turn_idx` (X on even turns, O on odd), never stored.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

EMPTY: Final[np.uint8] = np.uint8(0)
X: Final[np.uint8] = np.uint8(1)
O: Final[np.uint8] = np.uint8(2)  # noqa: E741 — game uses X/O as canonical mark names


class GameSpec(BaseModel):
    """Immutable parameters of a game: line length and per-turn placement counts."""

    model_config = ConfigDict(frozen=True)

    n: int = Field(gt=0)
    schedule: tuple[int, ...] = Field(description="Marks per turn, X starts.")

    @field_validator("schedule")
    @classmethod
    def _check_schedule(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("schedule must be non-empty")
        if any(k <= 0 for k in v):
            raise ValueError("schedule entries must be positive")
        return v

    def model_post_init(self, _: object) -> None:
        if sum(self.schedule) > self.n:
            raise ValueError(f"schedule sum {sum(self.schedule)} exceeds N={self.n}")


@dataclass(frozen=True, slots=True, eq=False)
class GameState:
    """A frozen game position. Equality and hashing are by (spec, turn_idx,
    placements_left, board contents), so identical positions reached by different
    paths are interchangeable as transposition-table keys."""

    spec: GameSpec
    board: np.ndarray
    turn_idx: int
    placements_left: int

    @staticmethod
    def initial(spec: GameSpec) -> GameState:
        board = np.zeros(spec.n, dtype=np.uint8)
        board.flags.writeable = False
        return GameState(spec, board, 0, spec.schedule[0])

    @property
    def is_terminal(self) -> bool:
        return self.turn_idx >= len(self.spec.schedule)

    @property
    def current_player(self) -> np.uint8:
        return X if self.turn_idx % 2 == 0 else O

    def key(self) -> tuple[GameSpec, int, int, bytes]:
        """Identity for hashing/equality. `board.tobytes()` is the only stable,
        cheap way to get a hashable view of the array."""
        return (self.spec, self.turn_idx, self.placements_left, self.board.tobytes())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return NotImplemented
        return self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())


def legal_mask(state: GameState) -> np.ndarray:
    if state.is_terminal:
        return np.zeros(state.spec.n, dtype=bool)
    return state.board == EMPTY


def legal_actions(state: GameState) -> np.ndarray:
    return np.flatnonzero(legal_mask(state))


def step(state: GameState, cell: int) -> GameState:
    if state.is_terminal:
        raise ValueError("cannot step a terminal state")
    if not (0 <= cell < state.spec.n):
        raise ValueError(f"cell {cell} out of range")
    if state.board[cell] != EMPTY:
        raise ValueError(f"cell {cell} not empty")

    new_board = state.board.copy()
    new_board[cell] = state.current_player
    new_board.flags.writeable = False

    placements_left = state.placements_left - 1
    turn_idx = state.turn_idx
    if placements_left == 0:
        turn_idx += 1
        if turn_idx < len(state.spec.schedule):
            placements_left = state.spec.schedule[turn_idx]
    return replace(state, board=new_board, turn_idx=turn_idx, placements_left=placements_left)


def longest_run(board: np.ndarray, mark: np.uint8) -> tuple[int, int]:
    """Return (length, rightmost_end_index) of the longest run of `mark`.

    With ties, the run whose rightmost cell has the larger index wins (the game's
    tie-breaker), so we use `>=` to update.
    """
    best_len = 0
    best_end = -1
    cur = 0
    for i in range(board.shape[0]):
        if board[i] == mark:
            cur += 1
            if cur >= best_len:
                best_len, best_end = cur, i
        else:
            cur = 0
    return best_len, best_end


def outcome(state: GameState) -> int:
    """+1 X wins, -1 O wins, 0 only if the board is empty (impossible at terminal)."""
    if not state.is_terminal:
        raise ValueError("outcome only defined at terminal")
    x_len, x_end = longest_run(state.board, X)
    o_len, o_end = longest_run(state.board, O)
    if x_len == 0 and o_len == 0:
        return 0
    if x_len != o_len:
        return 1 if x_len > o_len else -1
    return 1 if x_end > o_end else -1


def replay(spec: GameSpec, actions: Iterable[int]) -> Iterator[GameState]:
    state = GameState.initial(spec)
    yield state
    for a in actions:
        state = step(state, int(a))
        yield state


def final_state(spec: GameSpec, actions: Iterable[int]) -> GameState:
    """Apply a sequence of actions and return the final state. Raises
    ValueError on the first illegal action."""
    state = GameState.initial(spec)
    for a in actions:
        state = step(state, int(a))
    return state
