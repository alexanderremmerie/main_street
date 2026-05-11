import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, errorMessage } from "../api";
import { Mark } from "../components/Mark";
import { Notice } from "../components/Notice";
import { DescriptionRow as Row, LoadingState, PageHeader, SectionHeader } from "../components/Page";
import { StatRow, StatTile } from "../components/Stat";
import { TimeAgo } from "../components/TimeAgo";
import { formatAgent, formatSpec } from "../format";
import type { EvalRecord, GameRecord } from "../types";

export function EvalDetailPage() {
  const { id } = useParams<{ id: string }>();
  const nav = useNavigate();
  const [rec, setRec] = useState<EvalRecord | null>(null);
  const [games, setGames] = useState<GameRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setRec(null);
    setGames([]);
    setError(null);
    Promise.all([api.getEval(id), api.listGames({ eval_id: id, limit: 500 })])
      .then(([r, gs]) => {
        if (cancelled) return;
        setRec(r);
        setGames(gs.games);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => { cancelled = true; };
  }, [id]);

  if (error) return <Notice kind="error">{error}</Notice>;
  if (!rec) return <LoadingState>Loading eval…</LoadingState>;

  const c = rec.config;
  const s = rec.summary;

  return (
    <div>
      <PageHeader
        title="Eval"
        description={<span className="font-mono text-xs">{rec.id}</span>}
        actions={
          <Link to="/evals" className="text-xs text-neutral-500 hover:text-neutral-900">
            ← all evals
          </Link>
        }
      />

      <div className="border border-neutral-200 bg-white mb-6">
        <Row label="Agents">
          <span className="font-mono text-sm">{formatAgent(c.agent_a)}</span>{" "}
          <span className="text-neutral-400">(A)</span> vs{" "}
          <span className="font-mono text-sm">{formatAgent(c.agent_b)}</span>{" "}
          <span className="text-neutral-400">(B)</span>
        </Row>
        <Row label="Specs">
          <div className="font-mono text-sm flex flex-wrap gap-x-4 gap-y-1">
            {c.specs.map((sp, i) => (
              <span key={i}>
                N={sp.n} [{sp.schedule.join(",")}]
              </span>
            ))}
          </div>
        </Row>
        <Row label="Configuration">
          <span className="text-sm text-neutral-700">
            {c.n_games_per_spec} games/spec, {c.swap_sides ? "sides swapped" : "fixed sides"},
            seed {c.seed}
          </span>
        </Row>
        <Row label="Status">
          <span className="text-sm">{rec.status}</span>
          <span className="text-xs text-neutral-500 ml-3">
            <TimeAgo iso={rec.created_at} />
          </span>
        </Row>
      </div>

      {s && (
        <div className="mb-8">
          <StatRow>
            <StatTile label="games" value={s.n_games} />
            <StatTile label="A wins" value={s.a_wins} />
            <StatTile label="B wins" value={s.b_wins} />
            <StatTile
              label="A winrate"
              value={s.a_winrate.toFixed(3)}
              emphasis={s.a_winrate > 0.5 ? "accent" : "default"}
              hint={
                s.a_winrate === 0.5
                  ? "balanced"
                  : s.a_winrate > 0.5
                    ? "A favored"
                    : "B favored"
              }
            />
          </StatRow>
        </div>
      )}

      <div>
        <SectionHeader title="Games" />
        <div className="border border-neutral-200 bg-white">
          <table className="w-full text-sm">
            <thead className="text-[11px] uppercase text-neutral-500 bg-neutral-50/50 border-b border-neutral-200">
              <tr>
                <th className="text-left px-3 py-2 font-normal">id</th>
                <th className="text-left px-3 py-2 font-normal">spec</th>
                <th className="text-left px-3 py-2 font-normal">X</th>
                <th className="text-left px-3 py-2 font-normal">O</th>
                <th className="text-left px-3 py-2 font-normal">winner</th>
              </tr>
            </thead>
            <tbody>
              {games.map((g) => (
                <tr
                  key={g.id}
                  onClick={() => nav(`/games/${g.id}`)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") nav(`/games/${g.id}`);
                  }}
                  role="link"
                  tabIndex={0}
                  className="border-b border-neutral-100 last:border-0 cursor-pointer hover:bg-neutral-50 focus:bg-neutral-50 focus:outline-none"
                >
                  <td className="px-3 py-2">
                    <Link to={`/games/${g.id}`} className="font-mono text-neutral-900 underline underline-offset-2 hover:text-black hover:decoration-2">
                      {g.id.slice(0, 8)}
                    </Link>
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-neutral-600">
                    {formatSpec(g.spec)}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">{formatAgent(g.x_agent)}</td>
                  <td className="px-3 py-2 font-mono text-xs">{formatAgent(g.o_agent)}</td>
                  <td className="px-3 py-2">
                    {g.outcome === 1 ? <Mark player="X" /> : g.outcome === -1 ? <Mark player="O" /> : (
                      <span className="text-neutral-400">tie</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

