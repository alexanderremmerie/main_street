"""Baked-in spec lists for the eval + training pools.

Four lists, with strict disjointness:

  `DIAGNOSTIC_SPECS` — named hard cases from the notes.
  `HOLDOUT_SPECS`    — out-of-distribution generalization probes.
  `TRAIN_SMALL`      — broad pool used to generate supervised training data.
  `STARTER_SPECS`    — a subset of `TRAIN_SMALL` (in-distribution eval).

Additional probe lists capture specific strategic skills on a small number of
initial or prefix-rooted positions. These are intentionally tiny and labeled so
paper analysis can report "which strategic probe failed", not just one global
held-out score.

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


# ---------- STRATEGY PROBES -------------------------------------------------

_PROBE_PARTITION: list[tuple[str, int, tuple[int, ...]]] = [
    ("partition_T2_n5_2-3", 5, (2, 3)),
    ("partition_T3_n8_3-4", 8, (3, 4)),
    ("partition_T4_n12_4-5", 12, (4, 5)),
    ("partition_T5_n16_5-6", 16, (5, 6)),
    ("partition_T6_n21_6-7", 21, (6, 7)),
]

_PROBE_ORDER: list[tuple[str, int, tuple[int, ...]]] = [
    ("order_n10_1-2-3-4", 10, (1, 2, 3, 4)),
    ("order_n10_3-4-2-1", 10, (3, 4, 2, 1)),
    ("order_n10_2-4-3-1", 10, (2, 4, 3, 1)),
]

# Paper §3.3.1 Case 2 — threshold sweep for (γ₁, γ₂) = (5, 6) and (4, 5).
# T(5) = floor((5+4)^2/4) - 4 = 16; T(4) = 12.
# Labels encode whether X wins (oracle value +1) or O wins (-1) at that N.
_PROBE_THRESHOLD: list[tuple[str, int, tuple[int, ...]]] = [
    ("threshold_5-6_n12_xwins", 12, (5, 6)),
    ("threshold_5-6_n14_xwins", 14, (5, 6)),
    ("threshold_5-6_n16_xwins", 16, (5, 6)),   # exactly at T(5)
    ("threshold_5-6_n18_owins", 18, (5, 6)),
    ("threshold_5-6_n20_owins", 20, (5, 6)),
    ("threshold_4-5_n10_xwins", 10, (4, 5)),
    ("threshold_4-5_n12_xwins", 12, (4, 5)),   # exactly at T(4)
    ("threshold_4-5_n14_owins", 14, (4, 5)),
    ("threshold_4-5_n16_owins", 16, (4, 5)),
]

# Paper §3.3.1 Case 1 — γ₁ ≥ γ₂: X wins unconditionally; dominant strategy is
# rightmost γ₁ cells.  "equal" tests tie-break awareness; "adv" tests run dominance.
_PROBE_RIGHTMOST: list[tuple[str, int, tuple[int, ...]]] = [
    ("rightmost_equal_n6_3-3", 6, (3, 3)),
    ("rightmost_equal_n8_4-4", 8, (4, 4)),
    ("rightmost_equal_n10_5-5", 10, (5, 5)),
    ("rightmost_adv_n6_3-2", 6, (3, 2)),
    ("rightmost_adv_n8_4-3", 8, (4, 3)),
]

# Paper §3.3.3 — unit schedule perturbation flips the winner.
# (8, (2,3,2)) ∈ TRAIN_SMALL; included as in-distribution correctness check.
_PROBE_PERTURBATION: list[tuple[str, int, tuple[int, ...]]] = [
    ("perturb_n8_2-3-2_xwins", 8, (2, 3, 2)),
    ("perturb_n8_2-4-2_owins", 8, (2, 4, 2)),
]

_PROBE_PREFIX: list[tuple[str, int, tuple[int, ...], tuple[int, ...]]] = [
    (
        "tiebreak_n6_1-2-2_prefix_1-2-3-5",
        6,
        (1, 2, 2),
        (1, 2, 3, 5),
    ),
    (
        "tiebreak_n10_2-3-3_prefix_1-2-4-5-6-8-9",
        10,
        (2, 3, 3),
        (1, 2, 4, 5, 6, 8, 9),
    ),
    (
        "fork_n8_2-3-2_prefix_6-7-3-4-5",
        8,
        (2, 3, 2),
        (6, 7, 3, 4, 5),
    ),
    (
        "potential_n9_1-4-4_prefix_0-4-5-6-7-2-8",
        9,
        (1, 4, 4),
        (0, 4, 5, 6, 7, 2, 8),
    ),
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


def _probe_sources(
    specs: list[tuple[str, int, tuple[int, ...]]],
) -> list[SourceSpec]:
    return [
        SourceSpec(
            spec=GameSpec(n=n, schedule=s),
            mode="initial_only",
            label=label,
        )
        for label, n, s in specs
    ]


def _probe_prefix_sources(
    specs: list[tuple[str, int, tuple[int, ...], tuple[int, ...]]],
) -> list[SourceSpec]:
    return [
        SourceSpec(
            spec=GameSpec(n=n, schedule=s),
            mode="initial_only",
            prefix_actions=prefix,
            label=label,
        )
        for label, n, s, prefix in specs
    ]


STARTER: list[SourceSpec] = _make_sources(_STARTER_SPECS)
HOLDOUT: list[SourceSpec] = _make_sources(_HOLDOUT_SPECS)
TRAIN_SMALL: list[SourceSpec] = _make_sources(_TRAIN_SMALL_SPECS)
DIAGNOSTICS: list[SourceSpec] = _diagnostic_sources()
PROBE_PARTITION: list[SourceSpec] = _probe_sources(_PROBE_PARTITION)
PROBE_ORDER: list[SourceSpec] = _probe_sources(_PROBE_ORDER)
PROBE_THRESHOLD: list[SourceSpec] = _probe_sources(_PROBE_THRESHOLD)
PROBE_RIGHTMOST: list[SourceSpec] = _probe_sources(_PROBE_RIGHTMOST)
PROBE_PERTURBATION: list[SourceSpec] = _probe_sources(_PROBE_PERTURBATION)
PROBE_TIEBREAK: list[SourceSpec] = _probe_prefix_sources(_PROBE_PREFIX[:2])
PROBE_FORK: list[SourceSpec] = _probe_prefix_sources(_PROBE_PREFIX[2:3])
PROBE_POTENTIAL: list[SourceSpec] = _probe_prefix_sources(_PROBE_PREFIX[3:])


PRESETS: dict[str, list[SourceSpec]] = {
    "starter": STARTER,
    "holdout": HOLDOUT,
    "diagnostics": DIAGNOSTICS,
    "train_small": TRAIN_SMALL,
    "probe_partition": PROBE_PARTITION,
    "probe_order": PROBE_ORDER,
    "probe_threshold": PROBE_THRESHOLD,
    "probe_rightmost": PROBE_RIGHTMOST,
    "probe_perturbation": PROBE_PERTURBATION,
    "probe_tiebreak": PROBE_TIEBREAK,
    "probe_fork": PROBE_FORK,
    "probe_potential": PROBE_POTENTIAL,
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
