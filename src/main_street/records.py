"""Persisted records (pydantic). What goes on the wire and into the database."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agents import AgentSpec
from .core import GameSpec
from .spec_sampling import SpecSamplerConfig

Status = Literal["running", "done", "failed", "cancelling", "cancelled"]


def _now() -> datetime:
    return datetime.now(UTC)


class GameRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    spec: GameSpec
    x_agent: AgentSpec
    o_agent: AgentSpec
    actions: tuple[int, ...]
    outcome: int = Field(ge=-1, le=1)
    created_at: datetime = Field(default_factory=_now)
    eval_id: str | None = None
    seed: int | None = None


class EvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_a: AgentSpec
    agent_b: AgentSpec
    specs: tuple[GameSpec, ...]
    n_games_per_spec: int = Field(gt=0)
    swap_sides: bool = True
    seed: int = 0


class EvalSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_games: int
    a_wins: int
    b_wins: int
    ties: int
    a_winrate: float


class EvalRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    config: EvalConfig
    status: Status
    summary: EvalSummary | None = None
    created_at: datetime = Field(default_factory=_now)


class PairResult(BaseModel):
    """One unordered pair's outcome in a comparison. Winrate is from `a`'s
    perspective across every spec and side-swap in the underlying eval."""

    model_config = ConfigDict(frozen=True)

    a_player_id: str
    b_player_id: str
    eval_id: str
    a_winrate: float
    n_games: int


class ComparisonSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    pairs: tuple[PairResult, ...]
    n_total_games: int


class ComparisonConfig(BaseModel):
    """An N-player × M-spec comparison. The server runs every unordered pair
    of players as one tournament per spec, then aggregates. We keep parameters
    here (rather than implicit in PairResult) so failed comparisons remain
    interpretable."""

    model_config = ConfigDict(frozen=True)

    player_ids: tuple[str, ...] = Field(min_length=2)
    specs: tuple[GameSpec, ...] = ()
    spec_sampler: SpecSamplerConfig | None = None
    n_sampled_specs: int | None = Field(default=None, gt=0)
    n_games_per_spec: int = Field(gt=0)
    swap_sides: bool = True
    seed: int = 0

    @model_validator(mode="after")
    def _check_specs(self) -> ComparisonConfig:
        has_fixed = len(self.specs) > 0
        has_sampler = self.spec_sampler is not None or self.n_sampled_specs is not None
        if not has_fixed and not has_sampler:
            raise ValueError("provide fixed specs or both spec_sampler and n_sampled_specs")
        if has_sampler and (self.spec_sampler is None or self.n_sampled_specs is None):
            raise ValueError("spec_sampler and n_sampled_specs must be provided together")
        return self


class ComparisonRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    config: ComparisonConfig
    status: Status
    summary: ComparisonSummary | None = None
    progress_done: int = Field(default=0, ge=0)
    progress_total: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_now)


class PlayerRecord(BaseModel):
    """A named, persistent identity for an agent.

    Defaults (classical baselines) are seeded on first connect and protected
    from deletion. Customs are user-created and may be deleted. Trained-model
    checkpoints will eventually live here too.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    agent_spec: AgentSpec
    is_default: bool = False
    created_at: datetime = Field(default_factory=_now)
