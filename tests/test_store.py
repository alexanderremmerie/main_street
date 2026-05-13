import json
from pathlib import Path

from main_street import store
from main_street.agents import GreedyAgentSpec, LLMAgentSpec, RandomAgentSpec, RightmostAgentSpec
from main_street.core import GameSpec
from main_street.records import EvalConfig
from main_street.runner import play, run_tournament
from tests._llm_test_utils import llm_server, openai_chat_response


def test_game_roundtrip(tmp_path: Path):
    db = tmp_path / "t.db"
    store.bootstrap(db)
    conn = store.connect(db)
    g = play(
        GameSpec(n=6, schedule=(1, 2, 1)),
        RandomAgentSpec(seed=1),
        GreedyAgentSpec(seed=1),
    )
    with conn:
        store.insert_game(conn, g)
    loaded = store.get_game(conn, g.id)
    assert loaded == g


def test_count_games_respects_kind_filters(tmp_path: Path):
    db = tmp_path / "t.db"
    store.bootstrap(db)
    conn = store.connect(db)
    spec = GameSpec(n=4, schedule=(1, 1, 1, 1))
    with conn:
        store.insert_game(conn, play(spec, RandomAgentSpec(seed=0), GreedyAgentSpec(seed=0)))
        store.insert_game(conn, play(spec, RandomAgentSpec(seed=1), RandomAgentSpec(seed=2)))
        store.insert_game(conn, play(spec, GreedyAgentSpec(seed=3), GreedyAgentSpec(seed=4)))
    assert store.count_games(conn) == 3
    assert store.count_games(conn, x_kind="random") == 2
    assert store.count_games(conn, x_kind="greedy", o_kind="greedy") == 1
    assert store.count_games(conn, x_kind="random", o_kind="greedy") == 1


def test_tournament_persists_summary(tmp_path: Path):
    db = tmp_path / "t.db"
    store.bootstrap(db)
    conn = store.connect(db)
    cfg = EvalConfig(
        agent_a=RandomAgentSpec(seed=0),
        agent_b=GreedyAgentSpec(seed=0),
        specs=(GameSpec(n=6, schedule=(1, 1, 1, 1)),),
        n_games_per_spec=4,
    )
    rec = run_tournament(conn, cfg)
    assert rec.status == "done"
    assert rec.summary is not None
    assert rec.summary.n_games == 4
    assert store.count_games(conn, eval_id=rec.id) == 4
    loaded = store.get_eval(conn, rec.id)
    assert loaded is not None
    assert loaded.summary == rec.summary


def test_play_with_llm_emits_trace_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")
    monkeypatch.setenv("MAIN_STREET_LLM_TRACE_ROOT", str(tmp_path / "llm_runs"))

    def responder(
        _path: str, payload: dict[str, object]
    ) -> tuple[int, dict[str, object], float | None]:
        messages = payload["messages"]
        assert isinstance(messages, list)
        prompt = messages[-1]["content"]
        assert isinstance(prompt, str)
        legal_text = prompt.split("legal_cells=[", 1)[1].rstrip("]")
        legal = [int(part) for part in legal_text.split(",") if part]
        return 200, openai_chat_response(json.dumps({"cell": legal[-1]})), None

    with llm_server(responder) as (base_url, _requests):
        game = play(
            GameSpec(n=5, schedule=(1, 1, 1)),
            LLMAgentSpec(
                base_url=base_url,
                api_key_env="TEST_LLM_KEY",
                model="test-model",
                temperature=0.0,
                max_tokens=16,
                timeout_s=1.0,
            ),
            RightmostAgentSpec(),
        )
    trace_path = tmp_path / "llm_runs" / "play" / f"{game.id}.jsonl"
    assert trace_path.exists()
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [row["move_index"] for row in rows] == [0, 2]
    assert all(row["chosen_cell"] in row["state"]["legal_cells"] for row in rows)
    assert all(row["fallback_used"] is False for row in rows)


def test_tournament_runs_with_llm_player(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "dummy")

    def responder(
        _path: str, payload: dict[str, object]
    ) -> tuple[int, dict[str, object], float | None]:
        messages = payload["messages"]
        assert isinstance(messages, list)
        prompt = messages[-1]["content"]
        assert isinstance(prompt, str)
        legal_text = prompt.split("legal_cells=[", 1)[1].rstrip("]")
        legal = [int(part) for part in legal_text.split(",") if part]
        return 200, openai_chat_response(json.dumps({"cell": legal[-1]})), None

    with llm_server(responder) as (base_url, _requests):
        db = tmp_path / "t.db"
        store.bootstrap(db)
        conn = store.connect(db)
        cfg = EvalConfig(
            agent_a=LLMAgentSpec(
                base_url=base_url,
                api_key_env="TEST_LLM_KEY",
                model="test-model",
                temperature=0.0,
                max_tokens=16,
                timeout_s=1.0,
            ),
            agent_b=GreedyAgentSpec(seed=0),
            specs=(GameSpec(n=5, schedule=(1, 1, 1)),),
            n_games_per_spec=2,
        )
        rec = run_tournament(conn, cfg)
    assert rec.status == "done"
    assert rec.summary is not None
    assert rec.summary.n_games == 2
