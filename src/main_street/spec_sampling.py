"""Structured sampling of GameSpec values.

Used by self-play training and sampled comparison arenas. The sampler is
deliberately structured rather than fully random: it mixes schedule families
that stress different game shapes while still producing many concrete specs.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .core import GameSpec


class SpecSamplerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["mixture"] = "mixture"
    n_min: int = Field(default=12, gt=0)
    n_max: int = Field(default=24, gt=0)
    turns_min: int = Field(default=3, gt=0)
    turns_max: int = Field(default=8, gt=0)
    fill_min: float = Field(default=0.60, ge=0.0, le=1.0)
    fill_max: float = Field(default=0.95, ge=0.0, le=1.0)
    max_marks_per_turn: int = Field(default=6, gt=0)
    random_weight: float = Field(default=0.40, ge=0.0)
    arc_weight: float = Field(default=0.25, ge=0.0)
    few_big_weight: float = Field(default=0.20, ge=0.0)
    many_small_weight: float = Field(default=0.15, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> SpecSamplerConfig:
        if self.n_min > self.n_max:
            raise ValueError("n_min must be <= n_max")
        if self.turns_min > self.turns_max:
            raise ValueError("turns_min must be <= turns_max")
        if self.fill_min > self.fill_max:
            raise ValueError("fill_min must be <= fill_max")
        total_weight = (
            self.random_weight
            + self.arc_weight
            + self.few_big_weight
            + self.many_small_weight
        )
        if total_weight <= 0:
            raise ValueError("at least one family weight must be positive")
        return self

    def sample(self, rng: np.random.Generator) -> GameSpec:
        n = int(rng.integers(self.n_min, self.n_max + 1))
        weights = np.array(
            [
                self.random_weight,
                self.arc_weight,
                self.few_big_weight,
                self.many_small_weight,
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        family = int(rng.choice(4, p=weights))
        if family == 1:
            schedule = self._arc_schedule(n, rng)
        elif family == 2:
            schedule = self._few_big_schedule(n, rng)
        elif family == 3:
            schedule = self._many_small_schedule(n, rng)
        else:
            schedule = self._random_schedule(n, rng)
        return GameSpec(n=n, schedule=tuple(schedule))

    def _turns(
        self,
        n: int,
        rng: np.random.Generator,
        lo: int | None = None,
        hi: int | None = None,
    ) -> int:
        lo = self.turns_min if lo is None else max(self.turns_min, lo)
        hi = self.turns_max if hi is None else min(self.turns_max, hi)
        # Schedules have strictly positive entries, so turns > N can never
        # produce a valid GameSpec. If caller constraints exceed N, clamp to N.
        hi = min(hi, n)
        lo = min(lo, hi)
        if lo > hi:
            lo = hi
        return int(rng.integers(lo, hi + 1))

    def _total_marks(
        self, n: int, turns: int, max_marks: int, rng: np.random.Generator
    ) -> int:
        feasible_lo = turns
        feasible_hi = min(n, turns * max_marks)
        if feasible_lo > feasible_hi:
            raise ValueError(
                f"cannot sample valid schedule: n={n}, turns={turns}, max_marks={max_marks}"
            )

        lo = max(feasible_lo, int(math.ceil(self.fill_min * n)))
        hi = min(feasible_hi, int(math.floor(self.fill_max * n)))
        if lo > hi:
            # The fill range is a preference, not worth producing invalid specs
            # over. Fall back to the full feasible mark-count range.
            lo = feasible_lo
            hi = feasible_hi
        return int(rng.integers(lo, hi + 1))

    def _random_schedule(self, n: int, rng: np.random.Generator) -> list[int]:
        turns = self._turns(n, rng)
        total = self._total_marks(n, turns, self.max_marks_per_turn, rng)
        return _bounded_composition(total, turns, self.max_marks_per_turn, rng)

    def _arc_schedule(self, n: int, rng: np.random.Generator) -> list[int]:
        turns = self._turns(n, rng, lo=4)
        total = self._total_marks(n, turns, self.max_marks_per_turn, rng)
        mid = (turns - 1) / 2.0
        weights = [1.0 + (1.0 - abs(i - mid) / max(mid, 1.0)) for i in range(turns)]
        return _weighted_bounded_composition(total, weights, self.max_marks_per_turn, rng)

    def _few_big_schedule(self, n: int, rng: np.random.Generator) -> list[int]:
        turns = self._turns(n, rng, lo=2, hi=4)
        total = self._total_marks(n, turns, self.max_marks_per_turn, rng)
        weights = rng.uniform(0.75, 1.25, size=turns).tolist()
        return _weighted_bounded_composition(total, weights, self.max_marks_per_turn, rng)

    def _many_small_schedule(self, n: int, rng: np.random.Generator) -> list[int]:
        turns = self._turns(n, rng, lo=6)
        max_marks = min(3, self.max_marks_per_turn)
        total = self._total_marks(n, turns, max_marks, rng)
        return _bounded_composition(total, turns, max_marks, rng)


def sample_unique_specs(
    cfg: SpecSamplerConfig, count: int, seed: int
) -> tuple[GameSpec, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    specs: list[GameSpec] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    attempts = 0
    max_attempts = max(1000, count * 100)
    while len(specs) < count and attempts < max_attempts:
        attempts += 1
        spec = cfg.sample(rng)
        key = (spec.n, spec.schedule)
        if key in seen:
            continue
        seen.add(key)
        specs.append(spec)
    if len(specs) < count:
        raise ValueError(f"could only sample {len(specs)} unique specs after {attempts} attempts")
    return tuple(specs)


def _bounded_composition(
    total: int, parts: int, max_part: int, rng: np.random.Generator
) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if max_part <= 0:
        raise ValueError("max_part must be positive")
    if total < parts or total > parts * max_part:
        raise ValueError(
            f"cannot compose total={total} into {parts} positive parts capped at {max_part}"
        )
    out: list[int] = []
    remaining = total
    for i in range(parts):
        slots_after = parts - i - 1
        lo = max(1, remaining - max_part * slots_after)
        hi = min(max_part, remaining - slots_after)
        value = int(rng.integers(lo, hi + 1))
        out.append(value)
        remaining -= value
    return out


def _weighted_bounded_composition(
    total: int,
    weights: list[float],
    max_part: int,
    rng: np.random.Generator,
) -> list[int]:
    if not weights:
        raise ValueError("weights must be non-empty")
    if max_part <= 0:
        raise ValueError("max_part must be positive")
    if total < len(weights) or total > len(weights) * max_part:
        raise ValueError(
            f"cannot compose total={total} into {len(weights)} positive parts capped at {max_part}"
        )
    out = [1] * len(weights)
    extras = total - len(out)
    w = np.array(weights, dtype=np.float64)
    for _ in range(extras):
        eligible = np.array([v < max_part for v in out], dtype=bool)
        probs = np.where(eligible, w, 0.0)
        probs /= probs.sum()
        idx = int(rng.choice(len(out), p=probs))
        out[idx] += 1
    return out
