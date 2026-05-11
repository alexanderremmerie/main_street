# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Deep RL on a 1D placement game (CS285 project). Two players (X, O) place marks on a length-N 1D board over a fixed turn schedule; longest contiguous run wins, ties broken by rightmost end index. Python backend (FastAPI + SQLite) plus a React/Vite dashboard.

## Commands

Python (managed with `uv`; Python >=3.11):

- Install deps: `uv sync`
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_core.py::test_name`
- Lint: `uv run ruff check` (config in `pyproject.toml`, line length 100)
- Run the API server: `uv run uvicorn main_street.server:app --reload`

Frontend (in `web/`, uses `pnpm`):

- Dev server: `pnpm dev` (Vite, on :5173 — CORS is whitelisted for this origin)
- Build: `pnpm build` (output goes to `web/dist`; the FastAPI app auto-mounts it at `/` when present)
- Lint: `pnpm lint`

## Architecture

### Game core (`src/main_street/core.py`)

Single hot-path module. `GameSpec` is a frozen pydantic model (crosses the wire). `GameState` is a frozen dataclass wrapping a read-only numpy `uint8` board. Player encoding: `0=EMPTY, 1=X, 2=O`. Current player is **derived** from `turn_idx` (X on even, O on odd), never stored. `step` returns a new state with a fresh read-only board; never mutate in place. `outcome` is only defined at terminal states.

### Agents (`src/main_street/agents.py`)

`AgentSpec` is a discriminated union of pydantic specs (`random`, `greedy`, `rightmost`, `alphabeta`, `human`) keyed by `kind`. `build(spec)` returns a live `Agent` (Protocol with `act(state) -> int`). `HumanAgentSpec` exists as a serializable identity but **cannot be built** — humans act in the client; `build()` raises for human specs, and server endpoints must guard against this before calling `build()`.

To add a baseline: add a `Spec` class to the union, add it to `KINDS` / `SPEC_TYPES`, implement the runtime class, add a `build()` branch.

### Records, store, runner

- `records.py` — wire/DB pydantic models (`GameRecord`, `EvalConfig`, `EvalRecord`, `EvalSummary`). Frozen.
- `store.py` — SQLite DAO. Schema is in this file. Games are one row each; `actions` is a raw `uint8` BLOB; `spec`/agent specs are JSON; filtering/pagination happens in SQL. Default DB path: `data/main_street.db`.
- `runner.py` — `play` runs a full bot-vs-bot game; `record_from_actions` validates and replays an externally-played sequence (used for games involving humans, since the server never decides human moves); `run_tournament` plays many games and persists incrementally, updating the eval row's status (`running` → `done`/`failed`).

### Server (`src/main_street/server.py`)

All FastAPI routes live here, deliberately. `create_app(db_path)` builds the app; the active DB path is stored in module-level `_state` so tests can swap it. Key routes:

- `GET /api/agents` — list kinds with JSON schemas for client form rendering.
- `GET /api/games`, `GET /api/games/{id}`, `POST /api/games` (save an externally-played game).
- `POST /api/move` — compute one bot move from `(spec, actions, agent)`; rejects human agents.
- `GET /api/evals`, `POST /api/evals` (run a tournament synchronously), `GET /api/evals/{id}`.

If `web/dist` exists, it is mounted at `/` so a single uvicorn process serves both API and built frontend.

### Frontend (`web/`)

React 19 + TypeScript + Vite + Tailwind v4. Talks to the FastAPI API. The Python types in `records.py` / `agents.py` are the source of truth for shapes that cross the wire — keep `web/src/types.ts` in sync.
