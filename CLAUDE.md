# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Deep RL on a 1D placement game (CS285 project). Two players (X, O) place marks on a length-N 1D board over a fixed turn schedule; longest contiguous run wins, ties broken by rightmost end index. Python backend (FastAPI + SQLite) plus a React/Vite dashboard. AlphaZero-style training pipeline (encoder + PUCT MCTS + supervised/self-play replay) lives alongside the classical agents and the exact solver.

## Commands

Python (managed with `uv`; Python >=3.11):

- Install deps: `uv sync`
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_core.py::test_name`
- Lint: `uv run ruff check` (config in `pyproject.toml`, line length 100)
- Run the API server: `uv run uvicorn main_street.server:app --reload`
- Train: `uv run python -m main_street.nn --config experiments/<name>.json`
- Build/score eval sets: `uv run python -m main_street.eval build|score|list`
- Solver benchmark: `uv run python -m bench.solver_bench`

Frontend (in `web/`, uses `pnpm`):

- Dev server: `pnpm dev` (Vite, on :5173 — CORS is whitelisted for this origin)
- Build: `pnpm build` (output goes to `web/dist`; the FastAPI app auto-mounts it at `/` when present)
- Lint: `pnpm lint`

## Architecture

### Game core (`src/main_street/core.py`)

Single hot-path module. `GameSpec` is a frozen pydantic model (crosses the wire). `GameState` is a frozen dataclass wrapping a read-only numpy `uint8` board. Player encoding: `0=EMPTY, 1=X, 2=O`. Current player is **derived** from `turn_idx` (X on even, O on odd), never stored. `step` returns a new state with a fresh read-only board; never mutate in place. `outcome` is only defined at terminal states.

### Solver (`src/main_street/solve.py`)

Exact negamax + alpha-beta with a transposition table. Internally the hot loop works on two Python `int` bitboards (`xs`, `os_`) — one bit per cell — so the TT key avoids the `board.tobytes()` allocation a `GameState`-keyed cache would pay. Public surface: `Solver` (reuses TT across queries on the same spec, clearing on change; exposes `tt_size`), `solve(state) -> SolveResult`, `SolvedTable` + `build_table` + `save_table` / `load_table` (persisted oracle tables under `data/solved/`), `search_with_depth` (depth-limited variant used by the `alphabeta` agent with a finite depth), and `reachable_states(spec)` (shared traversal used by both `build_table` and the eval position-set builder). `SolveResult.value` is +1 / -1 from X's perspective; ties are impossible at terminal in a valid game.

### Agents (`src/main_street/agents.py`)

`AgentSpec` is a discriminated union of pydantic specs keyed by `kind`. Kinds: `human`, `random`, `greedy`, `rightmost`, `alphabeta`, `extension`, `blocker`, `center`, `forkaware`, `potentialaware`, `mcts`, `alphazero`. `build(spec)` returns a live `Agent` (Protocol with `act(state) -> int`).

- `HumanAgentSpec` exists as a serializable identity but **cannot be built** — humans act in the client; `build()` raises for human specs, and server endpoints must guard against this before calling `build()`.
- `AlphaZeroAgentSpec` carries a checkpoint path + search knobs. `build()` lazy-imports `nn.agent.build_alphazero` so callers that only use classical baselines don't pay the torch import cost.

To add a classical baseline: add a `Spec` class to the union, add it to `KINDS` / `SPEC_TYPES` / `LABELS`, implement the runtime class, add a `build()` branch.

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

### Eval harness (`src/main_street/eval/`)

`PositionSet` is a labeled bag of `(state, oracle_value, optimal_mask)` triples, stored as `data/eval/<name>/{positions.npz, manifest.json}` (gitignored). `score_agent(agent, ps)` computes top-1 oracle agreement plus per-spec and per-label breakdowns; works for *any* registered agent including AlphaZero (via `AlphaZeroAgentSpec`).

`sets.py` defines four spec lists with import-time disjointness checks:
- `DIAGNOSTICS` — named hard configs from the project notes (T(t) boundaries, schedule-order pairs, `(1,k,1)` sandwiches, local-sensitivity perturbations). `initial_only` mode; each position carries a label that survives into per-position metrics.
- `HOLDOUT` — out-of-distribution generalization probes.
- `TRAIN_SMALL` — broad training pool, disjoint from holdout + diagnostics.
- `STARTER` — subset of `TRAIN_SMALL` used as the in-distribution eval slice.

CLI: `python -m main_street.eval build <preset>` / `score <set> <kind|--spec path|--ckpt path>` / `list`. `--ckpt <path>` builds an `AlphaZeroAgentSpec` for you so you can score a trained checkpoint without writing a spec file.

### Neural-net pipeline (`src/main_street/nn/`)

AlphaZero-style training. Five protocols, each consumed via config:

- **Encoders** (`encode.py`) — `Encoder` ABC + `ENCODERS` registry; the default produces `{board, ctx, legal_mask, valid_mask}`. Each encoder exposes integer attributes (`board_channels`, `ctx_dim`, ...) so paired models can size their input layers at construction. Required output keys are `legal_mask` (cells legal now) and `valid_mask` (cells that exist, i.e. `i < n`); other keys are encoder-specific. The default encoder is fully vectorized in numpy.
- **Models** (`models.py`) — `Model` ABC + `MODELS` registry. `forward(inputs)` reads tensors by key; mismatched encoder/model pairs raise `KeyError` on first forward. `SimpleConv` is the baseline (residual 1D conv stack, schedule context as additive per-channel bias, masked global mean pool for value).
- **Search** (`mcts.py`) — pure-function `puct_search(state, model, encoder, ...)`. Carries value in X's frame; selection flips sign per-node based on `state.current_player`. Within a turn the current player does not flip between states (`step()` keeps the same player active until `placements_left == 0`) — the per-node sign-flip handles this correctly.
- **Sample sources** (`buffer.py`, `sources.py`, `selfplay.py`) — `SampleSource` ABC + a `ReplayBuffer` that draws weighted mini-batches from N sources. Two implementations: `SupervisedFromSet` (loads a `PositionSet` once, samples uniformly) and `SelfPlay` (generates games each `populate`, FIFO over a `CyclicBuffer`). `SelfPlay.opponent` is `"self"` (textbook AZ, both sides current model) or any `AgentSpec` (the learner plays one side and only its moves become samples; sides alternate per game — covers behavior cloning, frozen-checkpoint matchups, etc.). Each source config has a `.build()` method; no separate factory.
- **Trainer** (`train.py`) — single process. Each iter: `buffer.populate(model, encoder)` → `steps_per_iter` gradient steps on mixed batches → (every `eval_every`) raw + PUCT oracle-agreement eval, checkpoint, wandb log. The eval restricts PUCT to `diagnostics` (search meaningfully diverges from raw policy only on hard positions); raw on starter + holdout + diagnostics. Positions whose spec exceeds the encoder's `(max_n, max_turns)` are skipped and counted.

`TrainConfig` (pydantic) is the full configurable surface; one JSON file is one experiment. `data.sources` is a list of typed source configs — adding sources is config-only. Checkpoints at `data/runs/<run_id>/checkpoints/iter_NNNN.pt` (or `final.pt`) carry both the encoder config and the model name+params, so `load_checkpoint` reconstructs the `(encoder, model)` pair. `AlphaZeroAgentSpec(checkpoint_path=...)` plugs trained nets into the classical agent registry — `/api/move`, tournaments, and the eval harness all work unchanged.

Wandb: mode is `"online"`, `"offline"`, or `"disabled"`. Online runs sync to whatever account `wandb login` configured. Offline runs write to `data/runs/<run_id>/wandb/` and can be pushed later via `wandb sync`.

### Experiment configs (`experiments/`)

JSON `TrainConfig` files, one per experiment. Copy + edit to launch a variant. `smoke.json` runs in seconds (used by `tests/test_nn_train.py`); `quick.json` is a few-minute sanity run; `default.json` is a longer / fuller-coverage starting point.

### Solver bench (`bench/`)

`solver_bench.py` runs the exact solver on a fixed grid of `(N, schedule)` and writes `solver_bench.csv`. Used to validate that solver changes don't regress speed.

### Frontend (`web/`)

React 19 + TypeScript + Vite + Tailwind v4. Talks to the FastAPI API. The Python types in `records.py` / `agents.py` are the source of truth for shapes that cross the wire — keep `web/src/types.ts` in sync. `web/src/components/AgentField.tsx` renders per-`kind` form fields; adding a new agent kind requires updating both the type union (`types.ts`), the formatter (`format.ts`), and the field renderer (`AgentField.tsx`).

## Data layout

Everything under `data/` is gitignored (runtime artifacts only):
- `data/main_street.db` — SQLite store of games + evals.
- `data/solved/` — persisted oracle tables (`SolvedTable` JSON).
- `data/eval/<name>/` — built `PositionSet`s.
- `data/runs/<run_id>/` — training runs: `config.json`, `checkpoints/`, `wandb/`.
- `data/runs/_index.jsonl` — one line per completed run.
