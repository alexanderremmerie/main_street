"""Labeled position sets.

A `PositionSet` is a parallel-array bag of `(state, oracle_value, optimal_mask)`
triples drawn from one or more `GameSpec`s. The oracle is the exact solver in
`solve.py`; "optimal" means "any cell that achieves the side-to-move's best
value from this position". Top-1 oracle agreement is "did the agent pick a
cell in this mask".

Storage layout: `data/eval/<name>/` with
  - `manifest.json` — spec list, labels (optional), counts.
  - `positions.npz` — parallel arrays, padded to `max_n` over the spec list.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

from ..core import EMPTY, GameSpec, GameState, legal_actions, step
from ..solve import Solver

SourceMode = Literal["all_reachable", "initial_only"]

DEFAULT_ROOT: Final[Path] = Path("data") / "eval"


@dataclass(frozen=True, slots=True)
class PositionSet:
    """Padded parallel arrays over M labeled positions.

    Per row, `n[i]` is the actual board length; `board[i, :n[i]]` and
    `optimal_mask[i, :n[i]]` are the valid slices. Slots past `n[i]` are 0
    / False.
    """

    name: str
    specs: tuple[GameSpec, ...]
    spec_idx: np.ndarray  # (M,) int32 — index into `specs`
    n: np.ndarray  # (M,) int32 — actual board length
    turn_idx: np.ndarray  # (M,) int32
    placements_left: np.ndarray  # (M,) int32
    board: np.ndarray  # (M, max_n) uint8
    value: np.ndarray  # (M,) int8, from X's perspective
    optimal_mask: np.ndarray  # (M, max_n) bool
    labels: tuple[str, ...]  # length M, "" if unlabeled

    def __len__(self) -> int:
        return int(self.spec_idx.shape[0])

    @property
    def max_n(self) -> int:
        return int(self.board.shape[1]) if len(self) > 0 else 0

    def state(self, i: int) -> GameState:
        spec = self.specs[int(self.spec_idx[i])]
        ni = int(self.n[i])
        board = np.ascontiguousarray(self.board[i, :ni], dtype=np.uint8)
        board.flags.writeable = False
        return GameState(
            spec=spec,
            board=board,
            turn_idx=int(self.turn_idx[i]),
            placements_left=int(self.placements_left[i]),
        )

    def optimal_cells(self, i: int) -> np.ndarray:
        ni = int(self.n[i])
        return np.flatnonzero(self.optimal_mask[i, :ni])

    def save(self, root: Path | None = None) -> Path:
        if root is None:
            root = DEFAULT_ROOT
        d = root / self.name
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "positions.npz",
            spec_idx=self.spec_idx,
            n=self.n,
            turn_idx=self.turn_idx,
            placements_left=self.placements_left,
            board=self.board,
            value=self.value,
            optimal_mask=self.optimal_mask,
        )
        manifest = {
            "name": self.name,
            "count": len(self),
            "max_n": self.max_n,
            "specs": [
                {"n": s.n, "schedule": list(s.schedule)} for s in self.specs
            ],
            "labels": list(self.labels),
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return d

    @staticmethod
    def load(name: str, root: Path | None = None) -> PositionSet:
        if root is None:
            root = DEFAULT_ROOT
        d = root / name
        manifest = json.loads((d / "manifest.json").read_text())
        specs = tuple(
            GameSpec(n=s["n"], schedule=tuple(s["schedule"])) for s in manifest["specs"]
        )
        z = np.load(d / "positions.npz")
        labels = tuple(manifest.get("labels", []))
        if not labels:
            labels = ("",) * int(z["spec_idx"].shape[0])
        return PositionSet(
            name=name,
            specs=specs,
            spec_idx=z["spec_idx"],
            n=z["n"],
            turn_idx=z["turn_idx"],
            placements_left=z["placements_left"],
            board=z["board"],
            value=z["value"],
            optimal_mask=z["optimal_mask"],
            labels=labels,
        )


# ---------- Construction ----------------------------------------------------


def _label(solver: Solver, state: GameState) -> tuple[int, np.ndarray]:
    """Compute (X-perspective value, optimal-cell mask) for a non-terminal state.

    Mask cells are those whose per-cell value equals the position's value
    (i.e. equally optimal under the side-to-move's objective). The mask spans
    `state.spec.n`; padding is the caller's job.
    """
    result = solver.solve(state)
    n = state.spec.n
    mask = np.zeros(n, dtype=bool)
    best = result.value
    for cell, v in result.per_cell_values.items():
        if v == best:
            mask[cell] = True
    return result.value, mask


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One spec to draw positions from, plus what to draw."""

    spec: GameSpec
    mode: SourceMode = "all_reachable"
    prefix_actions: tuple[int, ...] = ()
    label: str = ""  # optional tag attached to every position from this source


def _root_state(src: SourceSpec) -> GameState:
    """Apply an optional prefix and return the rooted state for this source."""
    state = GameState.initial(src.spec)
    for action in src.prefix_actions:
        state = step(state, int(action))
    if state.is_terminal:
        raise ValueError(
            f"source {src.label or src.spec} prefix reaches terminal state: "
            f"{src.prefix_actions}"
        )
    return state


def _reachable_from_state(root: GameState) -> Iterable[GameState]:
    """DFS traversal from an arbitrary rooted state, deduplicated by state."""
    seen: set[GameState] = set()
    stack: list[GameState] = [root]
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        yield state
        if not state.is_terminal:
            for cell in legal_actions(state):
                stack.append(step(state, int(cell)))


def build_position_set(name: str, sources: Iterable[SourceSpec]) -> PositionSet:
    """Solve and pack positions from a list of `SourceSpec`s.

    `mode="all_reachable"` walks every reachable non-terminal state under the
    spec (use sparingly — full tree). `mode="initial_only"` yields just the
    spec's initial state (used for diagnostics).
    """
    sources = list(sources)
    specs = tuple(s.spec for s in sources)
    max_n = max((s.n for s in specs), default=0)

    spec_idx_l: list[int] = []
    n_l: list[int] = []
    turn_l: list[int] = []
    plc_l: list[int] = []
    boards_l: list[np.ndarray] = []
    masks_l: list[np.ndarray] = []
    values_l: list[int] = []
    labels_l: list[str] = []

    for si, src in enumerate(sources):
        solver = Solver()
        root = _root_state(src)
        if src.mode == "initial_only":
            states: Iterable[GameState] = [root]
        else:
            states = (s for s in _reachable_from_state(root) if not s.is_terminal)
        for s in states:
            value, mask = _label(solver, s)
            spec_idx_l.append(si)
            n_l.append(src.spec.n)
            turn_l.append(s.turn_idx)
            plc_l.append(s.placements_left)
            # Pad board and mask to max_n.
            b = np.zeros(max_n, dtype=np.uint8)
            b[: src.spec.n] = s.board
            boards_l.append(b)
            m = np.zeros(max_n, dtype=bool)
            m[: src.spec.n] = mask
            masks_l.append(m)
            values_l.append(value)
            labels_l.append(src.label)

    return PositionSet(
        name=name,
        specs=specs,
        spec_idx=np.array(spec_idx_l, dtype=np.int32),
        n=np.array(n_l, dtype=np.int32),
        turn_idx=np.array(turn_l, dtype=np.int32),
        placements_left=np.array(plc_l, dtype=np.int32),
        board=np.stack(boards_l) if boards_l else np.zeros((0, max_n), dtype=np.uint8),
        value=np.array(values_l, dtype=np.int8),
        optimal_mask=(
            np.stack(masks_l) if masks_l else np.zeros((0, max_n), dtype=bool)
        ),
        labels=tuple(labels_l),
    )


# ---------- Sanity check ----------------------------------------------------


def assert_valid(ps: PositionSet) -> None:
    """Cheap structural checks. Raises on inconsistency."""
    m = len(ps)
    for arr, exp in [
        (ps.spec_idx, (m,)),
        (ps.n, (m,)),
        (ps.turn_idx, (m,)),
        (ps.placements_left, (m,)),
        (ps.value, (m,)),
    ]:
        if arr.shape != exp:
            raise ValueError(f"shape mismatch: {arr.shape} != {exp}")
    if ps.board.shape != (m, ps.max_n):
        raise ValueError("board shape mismatch")
    if ps.optimal_mask.shape != (m, ps.max_n):
        raise ValueError("mask shape mismatch")
    if len(ps.labels) != m:
        raise ValueError("labels length mismatch")
    # Padding regions must be EMPTY / False.
    for i in range(m):
        ni = int(ps.n[i])
        if ni < ps.max_n:
            if (ps.board[i, ni:] != EMPTY).any():
                raise ValueError(f"row {i} has non-empty padding in board")
            if ps.optimal_mask[i, ni:].any():
                raise ValueError(f"row {i} has non-empty padding in mask")
        # At least one optimal cell at every non-terminal position.
        if not ps.optimal_mask[i, :ni].any():
            raise ValueError(f"row {i} has empty optimal mask")
