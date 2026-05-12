"""SQLite store. Schema, connection, and DAO functions for games and evals.

Games are rows (one per game). Actions live in a BLOB of raw uint8 bytes; specs
and agent specs are JSON. Sorting and filtering happen in SQL.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from .agents import SQRT_2
from .records import (
    ComparisonConfig,
    ComparisonRecord,
    ComparisonSummary,
    EvalConfig,
    EvalRecord,
    EvalSummary,
    GameRecord,
    PlayerRecord,
    Status,
)

DEFAULT_DB_PATH: Final[Path] = Path("data") / "main_street.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id          TEXT PRIMARY KEY,
    spec_n      INTEGER NOT NULL,
    spec        TEXT    NOT NULL,
    x_agent     TEXT    NOT NULL,
    o_agent     TEXT    NOT NULL,
    x_kind      TEXT    NOT NULL,
    o_kind      TEXT    NOT NULL,
    actions     BLOB    NOT NULL,
    outcome     INTEGER NOT NULL,
    created_at  TEXT    NOT NULL,
    eval_id     TEXT,
    seed        INTEGER,
    FOREIGN KEY (eval_id) REFERENCES evals(id)
);

CREATE INDEX IF NOT EXISTS idx_games_eval    ON games(eval_id);
CREATE INDEX IF NOT EXISTS idx_games_created ON games(created_at);
CREATE INDEX IF NOT EXISTS idx_games_kinds   ON games(x_kind, o_kind);

CREATE TABLE IF NOT EXISTS evals (
    id          TEXT PRIMARY KEY,
    config      TEXT NOT NULL,
    status      TEXT NOT NULL,
    summary     TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evals_created ON evals(created_at);

-- A `player` is a named, persistent identity for an agent. Classical baselines
-- live here as `is_default=1` rows so the rest of the app can refer to them by
-- stable ID; user-created players (different seeds, deeper search, eventually
-- trained checkpoints) are `is_default=0`.
CREATE TABLE IF NOT EXISTS players (
    id          TEXT PRIMARY KEY,
    label       TEXT    NOT NULL UNIQUE,
    agent_spec  TEXT    NOT NULL,
    is_default  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_players_default ON players(is_default);

-- An N-player × M-spec comparison. Each comparison spawns one pairwise
-- tournament per unordered (player_i, player_j) pair; those tournaments are
-- stored as ordinary rows in `evals` and referenced from the comparison's
-- summary so the comparison itself is just metadata + aggregation.
CREATE TABLE IF NOT EXISTS comparisons (
    id          TEXT PRIMARY KEY,
    config      TEXT NOT NULL,
    status      TEXT NOT NULL,
    summary     TEXT,
    progress_done  INTEGER NOT NULL,
    progress_total INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_comparisons_created ON comparisons(created_at);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection. The schema and default players are assumed to already
    exist; call `bootstrap` once at process startup before serving requests."""
    p = Path(path) if path is not None else DEFAULT_DB_PATH
    # check_same_thread=False because FastAPI's sync handlers run in a threadpool;
    # the dependency and the route can land on different worker threads within
    # one request. Each request still gets its own connection (no sharing).
    conn = sqlite3.connect(p, isolation_level="DEFERRED", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def bootstrap(path: Path | str | None = None) -> None:
    """Create the database file (if missing), install the schema, and seed the
    default players. Idempotent; call once at startup."""
    p = Path(path) if path is not None else DEFAULT_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with connect(p) as conn:
        conn.executescript(_SCHEMA)
        _seed_default_players(conn)


def _default_player_definitions() -> list[tuple[str, str, dict]]:
    """Return (id, label, agent_spec_dict) tuples for seeded default players.

    The defaults form a strength ladder. Extension / Blocker / Center are
    deliberately omitted: they're strawmen for characterization studies, not
    serious players, and remain available as agent kinds for ad-hoc use.
    """
    return [
        ("p_random_0", "Random", {"kind": "random", "seed": 0}),
        ("p_rightmost", "Rightmost", {"kind": "rightmost"}),
        ("p_greedy_0", "Greedy", {"kind": "greedy", "seed": 0}),
        ("p_forkaware", "ForkAware", {"kind": "forkaware", "seed": 0}),
        ("p_potentialaware", "PotentialAware", {"kind": "potentialaware", "seed": 0}),
        (
            "p_mcts_200",
            "MCTS (200, random)",
            {
                "kind": "mcts",
                "n_simulations": 200,
                "exploration_c": SQRT_2,
                "seed": 0,
                "rollout": "random",
            },
        ),
        (
            "p_mcts_h_200",
            "MCTS (200, forkaware)",
            {
                "kind": "mcts",
                "n_simulations": 200,
                "exploration_c": SQRT_2,
                "seed": 0,
                "rollout": "forkaware",
            },
        ),
        ("p_alphabeta_4", "AlphaBeta (depth=4)", {"kind": "alphabeta", "depth": 4}),
        ("p_alphabeta_exact", "AlphaBeta (exact)", {"kind": "alphabeta", "depth": None}),
    ]


def _seed_default_players(conn: sqlite3.Connection) -> None:
    ts = datetime.now(UTC).isoformat()
    for pid, label, spec in _default_player_definitions():
        conn.execute(
            "INSERT OR IGNORE INTO players (id, label, agent_spec, is_default, created_at) "
            "VALUES (?, ?, ?, 1, ?)",
            (pid, label, json.dumps(spec), ts),
        )


def insert_game(conn: sqlite3.Connection, r: GameRecord) -> None:
    conn.execute(
        """
        INSERT INTO games
          (id, spec_n, spec, x_agent, o_agent, x_kind, o_kind, actions,
           outcome, created_at, eval_id, seed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            r.id,
            r.spec.n,
            r.spec.model_dump_json(),
            r.x_agent.model_dump_json(),
            r.o_agent.model_dump_json(),
            r.x_agent.kind,
            r.o_agent.kind,
            bytes(r.actions),
            r.outcome,
            r.created_at.isoformat(),
            r.eval_id,
            r.seed,
        ),
    )


def _row_to_game(row: sqlite3.Row) -> GameRecord:
    return GameRecord.model_validate(
        {
            "id": row["id"],
            "spec": json.loads(row["spec"]),
            "x_agent": json.loads(row["x_agent"]),
            "o_agent": json.loads(row["o_agent"]),
            "actions": tuple(row["actions"]),
            "outcome": row["outcome"],
            "created_at": row["created_at"],
            "eval_id": row["eval_id"],
            "seed": row["seed"],
        }
    )


def get_game(conn: sqlite3.Connection, game_id: str) -> GameRecord | None:
    row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    return _row_to_game(row) if row else None


def _games_where(
    eval_id: str | None, x_kind: str | None, o_kind: str | None
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    args: list[object] = []
    if eval_id is not None:
        clauses.append("eval_id = ?")
        args.append(eval_id)
    if x_kind is not None:
        clauses.append("x_kind = ?")
        args.append(x_kind)
    if o_kind is not None:
        clauses.append("o_kind = ?")
        args.append(o_kind)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, args


def list_games(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    offset: int = 0,
    eval_id: str | None = None,
    x_kind: str | None = None,
    o_kind: str | None = None,
) -> tuple[list[GameRecord], int]:
    """Page of games matching the filters, plus the total count across all
    matching rows. One query (with a window function) so the total is always
    consistent with the page."""
    where, args = _games_where(eval_id, x_kind, o_kind)
    sql = f"""
        SELECT *, COUNT(*) OVER () AS _total
        FROM games
        {where}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(sql, [*args, limit, offset]).fetchall()
    if not rows:
        return [], count_games(conn, eval_id=eval_id, x_kind=x_kind, o_kind=o_kind)
    return [_row_to_game(r) for r in rows], int(rows[0]["_total"])


def count_games(
    conn: sqlite3.Connection,
    *,
    eval_id: str | None = None,
    x_kind: str | None = None,
    o_kind: str | None = None,
) -> int:
    """Total matching games. Used when only the count is needed; for paginated
    queries, use `list_games`, which returns the total alongside the rows."""
    where, args = _games_where(eval_id, x_kind, o_kind)
    return int(conn.execute(f"SELECT COUNT(*) FROM games {where}", args).fetchone()[0])


def insert_eval(conn: sqlite3.Connection, r: EvalRecord) -> None:
    conn.execute(
        "INSERT INTO evals VALUES (?,?,?,?,?)",
        (
            r.id,
            r.config.model_dump_json(),
            r.status,
            r.summary.model_dump_json() if r.summary else None,
            r.created_at.isoformat(),
        ),
    )


def update_eval(
    conn: sqlite3.Connection,
    eval_id: str,
    status: Status,
    summary: EvalSummary | None,
) -> None:
    conn.execute(
        "UPDATE evals SET status = ?, summary = ? WHERE id = ?",
        (status, summary.model_dump_json() if summary else None, eval_id),
    )


def _row_to_eval(row: sqlite3.Row) -> EvalRecord:
    return EvalRecord(
        id=row["id"],
        config=EvalConfig.model_validate_json(row["config"]),
        status=row["status"],
        summary=EvalSummary.model_validate_json(row["summary"]) if row["summary"] else None,
        created_at=row["created_at"],
    )


def get_eval(conn: sqlite3.Connection, eval_id: str) -> EvalRecord | None:
    row = conn.execute("SELECT * FROM evals WHERE id = ?", (eval_id,)).fetchone()
    return _row_to_eval(row) if row else None


def list_evals(conn: sqlite3.Connection, *, limit: int = 50, offset: int = 0) -> list[EvalRecord]:
    rows = conn.execute(
        "SELECT * FROM evals ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [_row_to_eval(r) for r in rows]


# ---------- Players ---------------------------------------------------------


def _row_to_player(row: sqlite3.Row) -> PlayerRecord:
    return PlayerRecord.model_validate(
        {
            "id": row["id"],
            "label": row["label"],
            "agent_spec": json.loads(row["agent_spec"]),
            "is_default": bool(row["is_default"]),
            "created_at": row["created_at"],
        }
    )


def insert_player(conn: sqlite3.Connection, r: PlayerRecord) -> None:
    conn.execute(
        "INSERT INTO players (id, label, agent_spec, is_default, created_at) VALUES (?,?,?,?,?)",
        (
            r.id,
            r.label,
            r.agent_spec.model_dump_json(),
            1 if r.is_default else 0,
            r.created_at.isoformat(),
        ),
    )


def get_player(conn: sqlite3.Connection, player_id: str) -> PlayerRecord | None:
    row = conn.execute("SELECT * FROM players WHERE id = ?", (player_id,)).fetchone()
    return _row_to_player(row) if row else None


def list_players(conn: sqlite3.Connection) -> list[PlayerRecord]:
    """All players, defaults first then user-created, each group by `created_at`."""
    rows = conn.execute(
        "SELECT * FROM players ORDER BY is_default DESC, created_at ASC"
    ).fetchall()
    return [_row_to_player(r) for r in rows]


def delete_player(conn: sqlite3.Connection, player_id: str) -> bool:
    """Delete a non-default player. Returns True iff a row was removed.
    Default players are protected — they're the stable identities the rest of
    the app references."""
    cur = conn.execute(
        "DELETE FROM players WHERE id = ? AND is_default = 0", (player_id,)
    )
    return cur.rowcount > 0


# ---------- Comparisons -----------------------------------------------------


def _row_to_comparison(row: sqlite3.Row) -> ComparisonRecord:
    return ComparisonRecord(
        id=row["id"],
        config=ComparisonConfig.model_validate_json(row["config"]),
        status=row["status"],
        summary=(
            ComparisonSummary.model_validate_json(row["summary"]) if row["summary"] else None
        ),
        progress_done=row["progress_done"],
        progress_total=row["progress_total"],
        created_at=row["created_at"],
    )


def insert_comparison(conn: sqlite3.Connection, r: ComparisonRecord) -> None:
    conn.execute(
        """
        INSERT INTO comparisons
          (id, config, status, summary, progress_done, progress_total, created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            r.id,
            r.config.model_dump_json(),
            r.status,
            r.summary.model_dump_json() if r.summary else None,
            r.progress_done,
            r.progress_total,
            r.created_at.isoformat(),
        ),
    )


def update_comparison(
    conn: sqlite3.Connection,
    comparison_id: str,
    status: Status,
    summary: ComparisonSummary | None,
) -> None:
    conn.execute(
        "UPDATE comparisons SET status = ?, summary = ? WHERE id = ?",
        (status, summary.model_dump_json() if summary else None, comparison_id),
    )


def update_comparison_progress(
    conn: sqlite3.Connection,
    comparison_id: str,
    progress_done: int,
    progress_total: int,
) -> None:
    conn.execute(
        "UPDATE comparisons SET progress_done = ?, progress_total = ? WHERE id = ?",
        (progress_done, progress_total, comparison_id),
    )


def request_comparison_cancel(conn: sqlite3.Connection, comparison_id: str) -> bool:
    cur = conn.execute(
        """
        UPDATE comparisons
        SET status = 'cancelling'
        WHERE id = ? AND status = 'running'
        """,
        (comparison_id,),
    )
    return cur.rowcount > 0


def get_comparison(conn: sqlite3.Connection, comparison_id: str) -> ComparisonRecord | None:
    row = conn.execute(
        "SELECT * FROM comparisons WHERE id = ?", (comparison_id,)
    ).fetchone()
    return _row_to_comparison(row) if row else None


def list_comparisons(
    conn: sqlite3.Connection, *, limit: int = 50, offset: int = 0
) -> list[ComparisonRecord]:
    rows = conn.execute(
        "SELECT * FROM comparisons ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [_row_to_comparison(r) for r in rows]


# ---------- Spec aggregation ------------------------------------------------


def list_spec_summaries(conn: sqlite3.Connection) -> list[dict]:
    """Return one row per distinct GameSpec that's appeared in games, with
    win counts. Specs are aggregated by their JSON serialization so that the
    same `(n, schedule)` always maps to one row regardless of order of insert."""
    rows = conn.execute(
        """
        SELECT spec,
               COUNT(*)                                       AS n_games,
               SUM(CASE WHEN outcome =  1 THEN 1 ELSE 0 END)  AS x_wins,
               SUM(CASE WHEN outcome = -1 THEN 1 ELSE 0 END)  AS o_wins,
               SUM(CASE WHEN outcome =  0 THEN 1 ELSE 0 END)  AS ties,
               MAX(created_at)                                AS last_game_at
        FROM   games
        GROUP  BY spec
        ORDER  BY n_games DESC, last_game_at DESC
        """
    ).fetchall()
    return [
        {
            "spec": json.loads(r["spec"]),
            "n_games": int(r["n_games"]),
            "x_wins": int(r["x_wins"]),
            "o_wins": int(r["o_wins"]),
            "ties": int(r["ties"]),
            "last_game_at": r["last_game_at"],
        }
        for r in rows
    ]
