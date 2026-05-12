"""Baked-in spec lists for the eval + training pools.

Four lists, with strict disjointness:

  `DIAGNOSTIC_SPECS` — named hard cases from the notes.
  `HOLDOUT_SPECS`    — out-of-distribution generalization probes.
  `TRAIN_SMALL`      — broad pool used to generate supervised training data.
  `STARTER_SPECS`    — a subset of `TRAIN_SMALL` (in-distribution eval).

The `_disjointness_check` at import time enforces that no spec appears in
both training and held-out lists.
"""

from __future__ import annotations

from ..core import GameSpec
from .positions import SourceSpec

# ---------- DIAGNOSTICS ----------------------------------------------------

_DIAGNOSTIC_SPECS: list[tuple[str, int, tuple[int, ...]]] = [
    ("T2_boundary_n5_2-3", 5, (2, 3)),
    ("T3_boundary_n8_3-4", 8, (3, 4)),
    ("T4_boundary_n12_4-5", 12, (4, 5)),
    ("T5_boundary_n16_5-6", 16, (5, 6)),
    ("T6_boundary_n21_6-7", 21, (6, 7)),
    ("order_n10_3-4-2-1", 10, (3, 4, 2, 1)),
    ("order_n10_2-4-3-1", 10, (2, 4, 3, 1)),
    ("sandwich_n5_1-2-1", 5, (1, 2, 1)),
    ("sandwich_n6_1-3-1", 6, (1, 3, 1)),
    ("sandwich_n7_1-4-1", 7, (1, 4, 1)),
    ("more_marks_loses_n9_1-4-4", 9, (1, 4, 4)),
    ("more_marks_loses_n10_1-4-4", 10, (1, 4, 4)),
    ("more_marks_loses_n11_1-4-4", 11, (1, 4, 4)),
    ("local_sens_n8_singletons", 8, (1, 1, 1, 1, 1, 1, 1, 1)),
    ("local_sens_n8_one-double", 8, (1, 1, 1, 2, 1, 1, 1)),
]


# ---------- HOLDOUT --------------------------------------------------------

_HOLDOUT_SPECS: list[tuple[int, tuple[int, ...]]] = [
    (10, (2, 3, 3)),
    (10, (3, 2, 3)),
    (8, (1, 2, 1, 2, 2)),
    (11, (3, 4)),
    (13, (3, 4)),
]


# ---------- TRAIN_SMALL (training pool) ------------------------------------
#
# Selected to span the regime (N ≤ 12, sum(schedule) ≤ 10) with variety in
# turn counts, schedule shapes, and parities. Strictly disjoint from
# diagnostics + holdout.

_TRAIN_SMALL_SPECS: list[tuple[int, tuple[int, ...]]] = [
    # Two-turn.
    (4, (1, 2)),
    (5, (1, 3)),
    (6, (2, 3)),
    (7, (3, 3)),
    (9, (3, 4)),
    (10, (4, 4)),
    (10, (3, 5)),
    # Three-turn.
    (6, (1, 2, 2)),
    (7, (2, 2, 2)),
    (7, (1, 3, 2)),
    (8, (2, 3, 2)),
    (8, (1, 3, 3)),
    (9, (3, 3, 3)),
    (10, (3, 3, 3)),
    (10, (2, 4, 3)),
    # Four-turn.
    (8, (2, 2, 2, 2)),
    (9, (2, 2, 2, 2)),
    (10, (2, 2, 3, 2)),
    # Equal singletons.
    (6, (1, 1, 1, 1, 1, 1)),
    (7, (1, 1, 1, 1, 1, 1, 1)),
    (9, (1, 1, 1, 1, 1, 1, 1, 1, 1)),
]


# ---------- STARTER (in-distribution eval; subset of TRAIN_SMALL) -----------

_STARTER_SPECS: list[tuple[int, tuple[int, ...]]] = [
    (6, (2, 3)),
    (9, (3, 4)),
    (10, (4, 4)),
    (6, (1, 2, 2)),
    (8, (2, 3, 2)),
    (9, (3, 3, 3)),
    (10, (3, 3, 3)),
    (8, (2, 2, 2, 2)),
    (7, (1, 1, 1, 1, 1, 1, 1)),
]


# ---------- Wire into SourceSpec lists -------------------------------------


def _make_sources(
    specs: list[tuple[int, tuple[int, ...]]], mode: str = "all_reachable"
) -> list[SourceSpec]:
    return [SourceSpec(spec=GameSpec(n=n, schedule=s), mode=mode) for n, s in specs]


def _diagnostic_sources() -> list[SourceSpec]:
    return [
        SourceSpec(
            spec=GameSpec(n=n, schedule=s), mode="initial_only", label=label
        )
        for label, n, s in _DIAGNOSTIC_SPECS
    ]


STARTER: list[SourceSpec] = _make_sources(_STARTER_SPECS)
HOLDOUT: list[SourceSpec] = _make_sources(_HOLDOUT_SPECS)
TRAIN_SMALL: list[SourceSpec] = _make_sources(_TRAIN_SMALL_SPECS)
DIAGNOSTICS: list[SourceSpec] = _diagnostic_sources()


PRESETS: dict[str, list[SourceSpec]] = {
    "starter": STARTER,
    "holdout": HOLDOUT,
    "diagnostics": DIAGNOSTICS,
    "train_small": TRAIN_SMALL,
}


def _disjointness_check() -> None:
    diag_keys = {(n, s) for _, n, s in _DIAGNOSTIC_SPECS}
    hold_keys = set(_HOLDOUT_SPECS)
    train_keys = set(_TRAIN_SMALL_SPECS)
    starter_keys = set(_STARTER_SPECS)

    if not starter_keys.issubset(train_keys):
        raise AssertionError("STARTER must be a subset of TRAIN_SMALL")
    overlap = train_keys & (diag_keys | hold_keys)
    if overlap:
        raise AssertionError(
            f"TRAIN_SMALL overlaps with held-out specs: {sorted(overlap)}"
        )


_disjointness_check()
