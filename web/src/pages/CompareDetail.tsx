/**
 * Comparison detail: the winrate matrix.
 *
 * Rows and columns are players in the order chosen by the user. Cell (row,
 * col) is the winrate of `row` against `col` aggregated across all specs and
 * (when enabled) side swaps. The diagonal is blank — a player vs itself
 * isn't a meaningful match. Clicking a cell opens the underlying eval where
 * the games live.
 */

import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, errorMessage } from "../api";
import { comparisonProgress, formatSpec } from "../format";
import { SecondaryButton } from "../components/Form";
import { Notice } from "../components/Notice";
import {
  DescriptionRow as Row,
  LoadingState,
  PageHeader,
  SectionHeader,
} from "../components/Page";
import { StatRow, StatTile } from "../components/Stat";
import { TimeAgo } from "../components/TimeAgo";
import type {
  ComparisonRecord,
  PairResult,
  PlayerRecord,
} from "../types";

export function CompareDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [rec, setRec] = useState<ComparisonRecord | null>(null);
  const [players, setPlayers] = useState<PlayerRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cancelBusy, setCancelBusy] = useState(false);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    Promise.all([api.getComparison(id), api.listPlayers()])
      .then(([r, ps]) => {
        if (cancelled) return;
        setRec(r);
        setPlayers(ps);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  // Poll while the comparison is unfinished. Keying the effect on
  // `rec?.status` (not the whole record) keeps the interval alive across the
  // many progress-tick rerenders that happen while running; we only tear it
  // down when the status itself transitions to terminal.
  const status = rec?.status;
  useEffect(() => {
    if (!id || !status || !["running", "cancelling"].includes(status)) {
      return;
    }
    const timer = window.setInterval(() => {
      api
        .getComparison(id)
        .then((r) =>
          setRec((prev) =>
            prev &&
            prev.status === r.status &&
            prev.progress_done === r.progress_done &&
            prev.progress_total === r.progress_total &&
            prev.summary === r.summary
              ? prev
              : r,
          ),
        )
        .catch((e) => setError(errorMessage(e)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [id, status]);

  if (error) return <Notice kind="error">{error}</Notice>;
  if (!rec || !players) return <LoadingState>Loading comparison…</LoadingState>;

  const playerById = new Map(players.map((p) => [p.id, p]));
  const pairs = rec.summary?.pairs ?? [];
  const canCancel = rec.status === "running";
  const { done: progressDone, total: progressTotal, pct: progressPct } =
    comparisonProgress(rec);

  const cancel = async () => {
    if (!id || !canCancel) return;
    setCancelBusy(true);
    setError(null);
    try {
      const updated = await api.cancelComparison(id);
      setRec(updated);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setCancelBusy(false);
    }
  };

  // Compute the leader once so we can highlight the top player in the stat tiles.
  const leader =
    rec.summary && pairs.length > 0 ? findLeader(rec.config.player_ids, pairs) : null;
  const leaderLabel = leader ? playerById.get(leader.id)?.label ?? leader.id : "—";

  return (
    <div>
      <PageHeader
        title="Comparison"
        description={<span className="font-mono text-xs">{rec.id}</span>}
        actions={
          <div className="flex items-center gap-2">
            {canCancel && (
              <SecondaryButton size="sm" onClick={cancel} disabled={cancelBusy}>
                {cancelBusy ? "cancelling..." : "cancel"}
              </SecondaryButton>
            )}
            <Link to="/compare" className="text-xs text-neutral-500 hover:text-neutral-900">
              ← new comparison
            </Link>
          </div>
        }
      />

      {rec.summary && (
        <div className="mb-6">
          <StatRow>
            <StatTile
              label="players"
              value={rec.config.player_ids.length}
              hint={`${pairs.length} pairs`}
            />
            <StatTile label="specs" value={rec.config.specs.length} />
            <StatTile label="games" value={rec.summary.n_total_games} />
            <StatTile
              label="leader"
              value={
                <span className="text-base font-semibold leading-snug">{leaderLabel}</span>
              }
              hint={leader ? `${leader.winrate.toFixed(3)} winrate` : undefined}
              emphasis="accent"
            />
          </StatRow>
        </div>
      )}

      <div className="border border-neutral-200 bg-white mb-8">
        <Row label="Status">
          <span className="text-sm">{rec.status}</span>
          <span className="text-xs text-neutral-500 ml-3">
            <TimeAgo iso={rec.created_at} />
          </span>
        </Row>
        <Row label="Progress">
          <div className="min-w-[16rem] max-w-lg flex-1">
            <div className="h-2 border border-neutral-200 bg-neutral-50">
              <div
                className="h-full bg-neutral-900 transition-[width] duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <div className="mt-1 text-xs text-neutral-500 tabular-nums">
              {progressDone} / {progressTotal} games
              {progressTotal > 0 && ` · ${progressPct}%`}
            </div>
          </div>
        </Row>
        <Row label="Players">
          <div className="flex flex-wrap gap-1.5 text-xs">
            {rec.config.player_ids.map((pid) => (
              <span
                key={pid}
                className="px-2 py-1 border border-neutral-200 text-neutral-700"
              >
                {playerById.get(pid)?.label ?? pid}
              </span>
            ))}
          </div>
        </Row>
        <Row label="Specs">
          <div className="font-mono text-sm flex flex-wrap gap-x-4 gap-y-1">
            {rec.config.specs.slice(0, 20).map((sp, i) => (
              <span key={i}>{formatSpec(sp)}</span>
            ))}
            {rec.config.specs.length > 20 && (
              <span className="text-neutral-400">
                +{rec.config.specs.length - 20} more
              </span>
            )}
          </div>
        </Row>
        {rec.config.spec_sampler && (
          <Row label="Spec source">
            <span className="text-sm text-neutral-700">
              sampled {rec.config.n_sampled_specs} specs, N={rec.config.spec_sampler.n_min}
              -{rec.config.spec_sampler.n_max}, turns={rec.config.spec_sampler.turns_min}
              -{rec.config.spec_sampler.turns_max}, fill={rec.config.spec_sampler.fill_min}
              -{rec.config.spec_sampler.fill_max}
            </span>
          </Row>
        )}
        <Row label="Configuration">
          <span className="text-sm text-neutral-700">
            {rec.config.n_games_per_spec} games per (spec, pair),{" "}
            {rec.config.swap_sides ? "sides swapped" : "fixed sides"}, seed {rec.config.seed}
          </span>
        </Row>
      </div>

      {rec.status === "done" && rec.summary ? (
        <Matrix
          playerIds={rec.config.player_ids}
          playerById={playerById}
          pairs={pairs}
        />
      ) : rec.status === "running" ? (
        <Notice kind="info">Comparison is running. This page updates automatically.</Notice>
      ) : rec.status === "cancelling" ? (
        <Notice kind="info">Cancellation requested. The worker will stop after the current game.</Notice>
      ) : rec.status === "cancelled" ? (
        <Notice kind="info">Comparison was cancelled.</Notice>
      ) : (
        <Notice kind="error">Comparison failed; no matrix available.</Notice>
      )}
    </div>
  );
}

function findLeader(
  playerIds: readonly string[],
  pairs: readonly PairResult[],
): { id: string; winrate: number } | null {
  const totals = new Map<string, { sum: number; n: number }>();
  for (const p of pairs) {
    const a = totals.get(p.a_player_id) ?? { sum: 0, n: 0 };
    const b = totals.get(p.b_player_id) ?? { sum: 0, n: 0 };
    a.sum += p.a_winrate * p.n_games;
    a.n += p.n_games;
    b.sum += (1 - p.a_winrate) * p.n_games;
    b.n += p.n_games;
    totals.set(p.a_player_id, a);
    totals.set(p.b_player_id, b);
  }
  let best: { id: string; winrate: number } | null = null;
  for (const id of playerIds) {
    const t = totals.get(id);
    if (!t || t.n === 0) continue;
    const wr = t.sum / t.n;
    if (best === null || wr > best.winrate) best = { id, winrate: wr };
  }
  return best;
}

function Matrix({
  playerIds,
  playerById,
  pairs,
}: {
  playerIds: string[];
  playerById: Map<string, PlayerRecord>;
  pairs: PairResult[];
}) {
  // Build a lookup: (row, col) -> {winrate of row vs col, eval id}. Each pair
  // is stored once with a deterministic ordering; we fill in both (a,b) and
  // (b,a) from the same record. winrate(b vs a) = 1 - winrate(a vs b) for
  // games without ties; with ties, our `a_winrate` already counts ties as 0.5
  // so the complement is symmetric in the same way.
  type Cell = { winrate: number; eval_id: string; n_games: number };
  const cells: Map<string, Cell> = new Map();
  const key = (r: string, c: string) => `${r}|${c}`;
  for (const p of pairs) {
    cells.set(key(p.a_player_id, p.b_player_id), {
      winrate: p.a_winrate,
      eval_id: p.eval_id,
      n_games: p.n_games,
    });
    cells.set(key(p.b_player_id, p.a_player_id), {
      winrate: 1 - p.a_winrate,
      eval_id: p.eval_id,
      n_games: p.n_games,
    });
  }

  // Overall winrate per player across the whole comparison (mean of row).
  const totals: { id: string; winrate: number; n: number }[] = playerIds.map((rid) => {
    let sumRate = 0;
    let games = 0;
    for (const cid of playerIds) {
      if (cid === rid) continue;
      const c = cells.get(key(rid, cid));
      if (!c) continue;
      sumRate += c.winrate * c.n_games;
      games += c.n_games;
    }
    return {
      id: rid,
      winrate: games > 0 ? sumRate / games : 0,
      n: games,
    };
  });
  const ranked = [...totals].sort((a, b) => b.winrate - a.winrate);

  return (
    <div className="space-y-8">
      <div>
        <SectionHeader
          title="Matrix"
          description="Cell (row, col) is row's winrate against col across every spec and side. Ties count as half a win. Click a cell to open the underlying eval."
        />
        <div className="overflow-x-auto border border-neutral-200 bg-white">
          <table className="text-sm border-collapse">
            <thead className="text-[10px] uppercase text-neutral-500 bg-neutral-50/50 border-b border-neutral-200">
              <tr>
                <th className="px-3 py-2 font-normal text-left">
                  <span className="text-neutral-400">row</span> \{" "}
                  <span className="text-neutral-400">col</span>
                </th>
                {playerIds.map((cid) => (
                  <th key={cid} className="px-3 py-2 font-normal text-center min-w-[8rem]">
                    {playerById.get(cid)?.label ?? cid.slice(0, 6)}
                  </th>
                ))}
                <th className="px-3 py-2 font-normal text-right border-l border-neutral-200">
                  overall
                </th>
              </tr>
            </thead>
            <tbody>
              {playerIds.map((rid) => {
                const total = totals.find((t) => t.id === rid);
                const isLeader = total && ranked[0]?.id === rid;
                return (
                  <tr key={rid} className="border-b border-neutral-100 last:border-0">
                    <td
                      className={`px-3 py-2 whitespace-nowrap ${
                        isLeader ? "text-neutral-900 font-semibold" : "text-neutral-900 font-medium"
                      }`}
                    >
                      {playerById.get(rid)?.label ?? rid.slice(0, 6)}
                    </td>
                    {playerIds.map((cid) => {
                      if (rid === cid) {
                        return (
                          <td
                            key={cid}
                            className="text-center text-neutral-300 bg-neutral-50"
                          >
                            —
                          </td>
                        );
                      }
                      const cell = cells.get(key(rid, cid));
                      return (
                        <td key={cid} className="p-0 text-center">
                          {cell ? <MatrixCell cell={cell} /> : (
                            <span className="text-neutral-300">·</span>
                          )}
                        </td>
                      );
                    })}
                    <td
                      className={`px-3 py-2 text-right border-l border-neutral-200 font-mono tabular-nums ${
                        isLeader ? "text-neutral-900 font-semibold" : "text-neutral-900"
                      }`}
                    >
                      {total ? total.winrate.toFixed(3) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <SectionHeader title="Ranking" />
        <div className="border border-neutral-200 bg-white">
          <table className="w-full text-sm">
            <thead className="text-[10px] uppercase text-neutral-500 bg-neutral-50/50 border-b border-neutral-200">
              <tr>
                <th className="text-left px-3 py-2 font-normal w-12">#</th>
                <th className="text-left px-3 py-2 font-normal">player</th>
                <th className="text-right px-3 py-2 font-normal">winrate</th>
                <th className="text-right px-3 py-2 font-normal">games</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((t, i) => (
                <tr key={t.id} className="border-b border-neutral-100 last:border-0">
                  <td className="px-3 py-2 text-neutral-400 tabular-nums">{i + 1}</td>
                  <td className="px-3 py-2">
                    <span className={i === 0 ? "font-semibold text-neutral-900" : ""}>
                      {playerById.get(t.id)?.label ?? t.id}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">
                    {t.winrate.toFixed(3)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-neutral-500">{t.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function MatrixCell({ cell }: { cell: { winrate: number; eval_id: string; n_games: number } }) {
  // Color cells on a red→white→green gradient so the matrix is scannable at
  // a glance. Inline background keeps the gradient continuous (vs Tailwind
  // buckets). The inset ring on hover/focus is the *clickability cue* —
  // before this the cell had no affordance beyond an easily-missed
  // underline-on-hover on the number itself.
  const w = cell.winrate;
  const dist = Math.abs(w - 0.5) * 2; // 0 at draw, 1 at sweep
  const hue = w >= 0.5 ? 145 : 5; // green / red
  const sat = 60 * dist;
  const light = 96 - 20 * dist;
  const bg = `hsl(${hue}, ${sat}%, ${light}%)`;
  return (
    <Link
      to={`/evals/${cell.eval_id}`}
      className="block h-full w-full px-3 py-2 font-mono tabular-nums text-neutral-900 hover:shadow-[inset_0_0_0_2px_#0a0a0a] focus:outline-none focus:shadow-[inset_0_0_0_2px_#0a0a0a]"
      style={{ backgroundColor: bg }}
      title={`${cell.winrate.toFixed(3)} over ${cell.n_games} games · open eval`}
    >
      {cell.winrate.toFixed(3)}
    </Link>
  );
}
