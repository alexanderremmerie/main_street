from pathlib import Path

from main_street import store
from main_street.agents import GreedyAgentSpec, RandomAgentSpec
from main_street.core import GameSpec
from main_street.records import EvalConfig
from main_street.runner import play, run_tournament


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
