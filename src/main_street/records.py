"""Persisted records (pydantic). What goes on the wire and into the database."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .agents import AgentSpec
from .core import GameSpec

Status = Literal["running", "done", "failed"]


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
    specs: tuple[GameSpec, ...] = Field(min_length=1)
    n_games_per_spec: int = Field(gt=0)
    swap_sides: bool = True
    seed: int = 0


class ComparisonRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    config: ComparisonConfig
    status: Status
    summary: ComparisonSummary | None = None
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
