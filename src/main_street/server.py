"""FastAPI dashboard server. All routes live here, deliberately."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store
from .agents import KINDS, LABELS, SPEC_TYPES, AgentSpec, HumanAgentSpec, build
from .core import GameSpec, final_state
from .records import (
    ComparisonConfig,
    ComparisonRecord,
    EvalConfig,
    EvalRecord,
    GameRecord,
    PlayerRecord,
)
from .runner import record_from_actions, run_comparison, run_tournament
from .solve import solve

# Cap the state space the oracle is willing to solve. The branching factor is
# at most N, so this bounds the number of leaves at roughly N**sum(schedule).
# Above this, the API politely declines rather than locking up the server.
ORACLE_MAX_LEAVES = 200_000


def _db(request: Request) -> Iterator[sqlite3.Connection]:
    conn = store.connect(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()


Conn = Annotated[sqlite3.Connection, Depends(_db)]


class AgentKind(BaseModel):
    kind: str
    label: str
    schema_: dict


class GameList(BaseModel):
    games: list[GameRecord]
    total: int


class SaveGameRequest(BaseModel):
    """Persist an externally-played (e.g. interactive) game."""

    spec: GameSpec
    x_agent: AgentSpec
    o_agent: AgentSpec
    actions: tuple[int, ...]


class MoveRequest(BaseModel):
    """Compute one move for a bot agent from the current position."""

    spec: GameSpec
    actions: tuple[int, ...]
    agent: AgentSpec


class MoveResponse(BaseModel):
    cell: int


class OracleRequest(BaseModel):
    """Ask the exact solver for the value of the current position. The solver
    is cheap on small `(N, schedule)` and refused on configurations whose
    state space is too large to enumerate quickly."""

    spec: GameSpec
    actions: tuple[int, ...]


class OracleResponse(BaseModel):
    """`value` and `per_cell_values` are from X's perspective (+1 X wins,
    -1 O wins). `best_cell` is the optimal cell for the side to move at the
    current position; `-1` if terminal."""

    value: int
    best_cell: int
    per_cell_values: dict[int, int]
    is_terminal: bool


class SpecSummary(BaseModel):
    """Per-spec aggregate over saved games. Used by the Specs index to surface
    every config that's been played and a quick win-rate snapshot."""

    spec: GameSpec
    n_games: int
    x_wins: int
    o_wins: int
    ties: int
    last_game_at: str | None = None


class CreatePlayerRequest(BaseModel):
    """Create a custom player. The label must be unique; we don't auto-suffix
    because silent collisions in a research database are worse than a 409."""

    label: str
    agent_spec: AgentSpec


class AnalyzeRequest(BaseModel):
    """Ask each of `player_ids` what it would play at `(spec, actions)`.

    The position must not be terminal — there's no move to recommend there.
    If the spec is solver-eligible, the response also carries the oracle's
    per-cell evaluations so the caller can score each player against truth.
    """

    spec: GameSpec
    actions: tuple[int, ...]
    player_ids: tuple[str, ...]


class PlayerVerdict(BaseModel):
    """One player's response at a single position.

    `cell` is the cell the player would play. `agrees_with_oracle` is `None`
    when no oracle is available (large board) and a bool otherwise. `error`
    is set instead of `cell` if the player can't act (e.g. a human player was
    passed by mistake)."""

    player_id: str
    label: str
    cell: int | None = None
    agrees_with_oracle: bool | None = None
    error: str | None = None


class AnalyzeResponse(BaseModel):
    """Result of an analysis. `oracle` is present iff the spec was solver-
    eligible and the position is non-terminal."""

    verdicts: list[PlayerVerdict]
    oracle_value: int | None = None
    oracle_best_cell: int | None = None
    oracle_per_cell_values: dict[int, int] | None = None


def _oracle_state_budget(spec: GameSpec) -> int:
    """Cheap upper bound on the search tree size. Used to refuse oversized
    queries before they hang the server."""
    leaves = 1
    available = spec.n
    for k in spec.schedule:
        for _ in range(k):
            if available <= 0:
                return leaves
            leaves *= available
            available -= 1
            if leaves > ORACLE_MAX_LEAVES:
                return leaves
    return leaves


def create_app(db_path: Path | None = None) -> FastAPI:
    resolved_path = db_path if db_path is not None else store.DEFAULT_DB_PATH
    store.bootstrap(resolved_path)

    app = FastAPI(title="main-street", version="0.1.0")
    app.state.db_path = resolved_path
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/agents", response_model=list[AgentKind])
    def list_agent_kinds() -> list[AgentKind]:
        return [
            AgentKind(kind=k, label=LABELS[k], schema_=SPEC_TYPES[k].model_json_schema())
            for k in KINDS
        ]

    @app.get("/api/games", response_model=GameList)
    def list_games(
        conn: Conn,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        eval_id: str | None = None,
        x_kind: str | None = None,
        o_kind: str | None = None,
    ) -> GameList:
        items, total = store.list_games(
            conn, limit=limit, offset=offset, eval_id=eval_id, x_kind=x_kind, o_kind=o_kind
        )
        return GameList(games=items, total=total)

    @app.get("/api/games/{game_id}", response_model=GameRecord)
    def get_game(game_id: str, conn: Conn) -> GameRecord:
        rec = store.get_game(conn, game_id)
        if rec is None:
            raise HTTPException(404, "game not found")
        return rec

    @app.post("/api/games", response_model=GameRecord)
    def save_game(req: SaveGameRequest, conn: Conn) -> GameRecord:
        try:
            record = record_from_actions(req.spec, req.x_agent, req.o_agent, req.actions)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        with conn:
            store.insert_game(conn, record)
        return record

    @app.post("/api/move", response_model=MoveResponse)
    def compute_move(req: MoveRequest) -> MoveResponse:
        if isinstance(req.agent, HumanAgentSpec):
            raise HTTPException(400, "cannot ask a human agent to act on the server")
        try:
            state = final_state(req.spec, req.actions)
        except ValueError as e:
            raise HTTPException(400, f"invalid action sequence: {e}") from e
        if state.is_terminal:
            raise HTTPException(400, "game is already terminal")
        agent = build(req.agent)
        cell = int(agent.act(state))
        return MoveResponse(cell=cell)

    @app.get("/api/specs", response_model=list[SpecSummary])
    def list_specs(conn: Conn) -> list[SpecSummary]:
        rows = store.list_spec_summaries(conn)
        return [SpecSummary.model_validate(r) for r in rows]

    @app.get("/api/players", response_model=list[PlayerRecord])
    def list_players(conn: Conn) -> list[PlayerRecord]:
        return store.list_players(conn)

    @app.post("/api/players", response_model=PlayerRecord)
    def create_player(req: CreatePlayerRequest, conn: Conn) -> PlayerRecord:
        label = req.label.strip()
        if not label:
            raise HTTPException(400, "label must be non-empty")
        if isinstance(req.agent_spec, HumanAgentSpec):
            raise HTTPException(400, "players must be buildable; human is not")
        record = PlayerRecord(
            id=f"p_{uuid.uuid4().hex[:12]}",
            label=label,
            agent_spec=req.agent_spec,
            is_default=False,
        )
        try:
            with conn:
                store.insert_player(conn, record)
        except sqlite3.IntegrityError as e:
            # Unique label collision is the only constraint that fails in practice.
            raise HTTPException(409, f"a player named {label!r} already exists") from e
        return record

    @app.delete("/api/players/{player_id}")
    def delete_player(player_id: str, conn: Conn) -> dict:
        existing = store.get_player(conn, player_id)
        if existing is None:
            raise HTTPException(404, "player not found")
        if existing.is_default:
            raise HTTPException(400, "default players cannot be deleted")
        with conn:
            store.delete_player(conn, player_id)
        return {"deleted": player_id}

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    def analyze(req: AnalyzeRequest, conn: Conn) -> AnalyzeResponse:
        if not req.player_ids:
            raise HTTPException(400, "player_ids must be non-empty")
        try:
            state = final_state(req.spec, req.actions)
        except ValueError as e:
            raise HTTPException(400, f"invalid action sequence: {e}") from e
        if state.is_terminal:
            raise HTTPException(400, "position is terminal; no move to analyze")

        oracle_value: int | None = None
        oracle_best_cell: int | None = None
        oracle_per_cell: dict[int, int] | None = None
        if _oracle_state_budget(req.spec) <= ORACLE_MAX_LEAVES:
            result = solve(state)
            oracle_value = result.value
            oracle_best_cell = result.best_cell
            oracle_per_cell = dict(result.per_cell_values)

        verdicts: list[PlayerVerdict] = []
        for pid in req.player_ids:
            player = store.get_player(conn, pid)
            if player is None:
                verdicts.append(
                    PlayerVerdict(player_id=pid, label="(missing)", error="player not found")
                )
                continue
            try:
                agent = build(player.agent_spec)
            except ValueError as e:
                verdicts.append(
                    PlayerVerdict(player_id=pid, label=player.label, error=str(e))
                )
                continue
            try:
                cell = int(agent.act(state))
            except Exception as e:  # noqa: BLE001 — surface any agent-level failure
                verdicts.append(
                    PlayerVerdict(player_id=pid, label=player.label, error=str(e))
                )
                continue
            agrees = (
                cell in oracle_per_cell
                and oracle_per_cell[cell] == oracle_value
                if oracle_per_cell is not None and oracle_value is not None
                else None
            )
            verdicts.append(
                PlayerVerdict(
                    player_id=pid,
                    label=player.label,
                    cell=cell,
                    agrees_with_oracle=agrees,
                )
            )

        return AnalyzeResponse(
            verdicts=verdicts,
            oracle_value=oracle_value,
            oracle_best_cell=oracle_best_cell,
            oracle_per_cell_values=oracle_per_cell,
        )

    @app.post("/api/oracle", response_model=OracleResponse)
    def oracle(req: OracleRequest) -> OracleResponse:
        if _oracle_state_budget(req.spec) > ORACLE_MAX_LEAVES:
            raise HTTPException(
                413, f"state space too large for the exact solver (>{ORACLE_MAX_LEAVES} leaves)"
            )
        try:
            state = final_state(req.spec, req.actions)
        except ValueError as e:
            raise HTTPException(400, f"invalid action sequence: {e}") from e
        result = solve(state)
        return OracleResponse(
            value=result.value,
            best_cell=result.best_cell,
            per_cell_values=result.per_cell_values,
            is_terminal=state.is_terminal,
        )

    @app.get("/api/evals", response_model=list[EvalRecord])
    def list_evals(
        conn: Conn,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[EvalRecord]:
        return store.list_evals(conn, limit=limit, offset=offset)

    @app.get("/api/evals/{eval_id}", response_model=EvalRecord)
    def get_eval(eval_id: str, conn: Conn) -> EvalRecord:
        rec = store.get_eval(conn, eval_id)
        if rec is None:
            raise HTTPException(404, "eval not found")
        return rec

    @app.post("/api/evals", response_model=EvalRecord)
    def post_eval(config: EvalConfig, conn: Conn) -> EvalRecord:
        # A tournament is bot-only; reject human agents up front for a clear error.
        for spec in (config.agent_a, config.agent_b):
            if isinstance(spec, HumanAgentSpec):
                raise HTTPException(400, "tournaments cannot include human agents")
        return run_tournament(conn, config)

    @app.get("/api/comparisons", response_model=list[ComparisonRecord])
    def list_comparisons(
        conn: Conn,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[ComparisonRecord]:
        return store.list_comparisons(conn, limit=limit, offset=offset)

    @app.get("/api/comparisons/{comparison_id}", response_model=ComparisonRecord)
    def get_comparison(comparison_id: str, conn: Conn) -> ComparisonRecord:
        rec = store.get_comparison(conn, comparison_id)
        if rec is None:
            raise HTTPException(404, "comparison not found")
        return rec

    @app.post("/api/comparisons", response_model=ComparisonRecord)
    def post_comparison(config: ComparisonConfig, conn: Conn) -> ComparisonRecord:
        try:
            return run_comparison(conn, config)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app


app = create_app()
