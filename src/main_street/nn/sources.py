"""Concrete `SampleSource` implementations.

`SourceConfig` is a discriminated union over `SupervisedSourceConfig` and
`SelfPlaySourceConfig`; each config builds its own source via `cfg.build()`.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..agents import AgentSpec
from ..agents import build as build_agent
from ..core import GameSpec, O, X
from ..eval.positions import PositionSet
from ..spec_sampling import SpecSamplerConfig
from .buffer import CyclicBuffer, Sample, SampleSource
from .encode import Encoder
from .models import Model
from .selfplay import self_play_game


class _SourceBase(BaseModel):
    model_config = ConfigDict(frozen=True)
    weight: float = Field(default=1.0, ge=0.0)


class SupervisedSourceConfig(_SourceBase):
    kind: Literal["supervised"] = "supervised"
    set: str

    def build(self) -> SupervisedFromSet:
        return SupervisedFromSet(self)


class SelfPlaySourceConfig(_SourceBase):
    """One self-play stream. Each game samples either uniformly from `specs`,
    or from `spec_sampler` when a broader distribution is configured.
    `opponent` is `"self"` (current model plays both sides; every move
    becomes a sample) or an `AgentSpec` (learner plays one side, alternating
    per game; only learner moves become samples)."""

    kind: Literal["selfplay"] = "selfplay"
    specs: list[tuple[int, list[int]]] | None = Field(
        default=None,
        description="Each entry is [n, [schedule...]]."
    )
    spec_sampler: SpecSamplerConfig | None = None
    games_per_iter: int = 16
    n_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature_moves: int = 5
    capacity: int = 100_000
    opponent: Literal["self"] | AgentSpec = "self"

    @model_validator(mode="after")
    def _check_specs(self) -> SelfPlaySourceConfig:
        if (self.specs is None) == (self.spec_sampler is None):
            raise ValueError("provide exactly one of specs or spec_sampler")
        return self

    def build(self) -> SelfPlay:
        return SelfPlay(self)


SourceConfig = Annotated[
    SupervisedSourceConfig | SelfPlaySourceConfig,
    Field(discriminator="kind"),
]


class SupervisedFromSet(SampleSource):
    """Loads a labeled `PositionSet` once. Samples uniformly with replacement."""

    def __init__(self, cfg: SupervisedSourceConfig) -> None:
        self.weight = cfg.weight
        ps = PositionSet.load(cfg.set)
        self._pool: list[Sample] = []
        for i in range(len(ps)):
            cells = ps.optimal_cells(i)
            if cells.size == 0:
                continue
            p = 1.0 / float(cells.size)
            pi = {int(c): p for c in cells}
            self._pool.append(
                Sample(state=ps.state(i), pi=pi, z=float(ps.value[i]))
            )

    def populate(self, model, encoder, rng) -> int:
        return 0

    def sample(self, n: int, rng: np.random.Generator) -> list[Sample]:
        if n <= 0 or not self._pool:
            return []
        idx = rng.integers(0, len(self._pool), size=n)
        return [self._pool[int(i)] for i in idx]

    @property
    def size(self) -> int:
        return len(self._pool)


class SelfPlay(SampleSource):
    """Generates `games_per_iter` games per `populate`. Cyclic FIFO at `capacity`."""

    def __init__(self, cfg: SelfPlaySourceConfig) -> None:
        self.weight = cfg.weight
        self.cfg = cfg
        self._specs = (
            [GameSpec(n=n, schedule=tuple(s)) for n, s in cfg.specs]
            if cfg.specs is not None
            else None
        )
        self._buffer = CyclicBuffer(cfg.capacity)
        self._opponent = (
            build_agent(cfg.opponent) if cfg.opponent != "self" else None
        )

    def _sample_spec(self, rng: np.random.Generator) -> GameSpec:
        if self._specs is not None:
            return self._specs[int(rng.integers(0, len(self._specs)))]
        assert self.cfg.spec_sampler is not None
        return self.cfg.spec_sampler.sample(rng)

    def populate(
        self, model: Model, encoder: Encoder, rng: np.random.Generator
    ) -> int:
        added = 0
        for _ in range(self.cfg.games_per_iter):
            spec = self._sample_spec(rng)
            learner_side = (
                None
                if self._opponent is None
                else (int(X) if rng.random() < 0.5 else int(O))
            )
            samples = self_play_game(
                spec=spec,
                model=model,
                encoder=encoder,
                n_simulations=self.cfg.n_simulations,
                c_puct=self.cfg.c_puct,
                dirichlet_alpha=self.cfg.dirichlet_alpha,
                dirichlet_eps=self.cfg.dirichlet_eps,
                temperature_moves=self.cfg.temperature_moves,
                rng=rng,
                opponent=self._opponent,
                learner_side=learner_side,
            )
            self._buffer.extend(samples)
            added += len(samples)
        return added

    def sample(self, n: int, rng: np.random.Generator) -> list[Sample]:
        if n <= 0 or len(self._buffer) == 0:
            return []
        idx = rng.integers(0, len(self._buffer), size=n)
        return [self._buffer[int(i)] for i in idx]

    @property
    def size(self) -> int:
        return len(self._buffer)
