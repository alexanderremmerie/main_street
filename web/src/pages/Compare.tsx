/**
 * Compare: N-player × M-spec round-robin.
 *
 * Pick at least two players and at least one spec, set games-per-spec, and the
 * server runs every unordered pair as a tournament across all specs. The
 * result is a winrate matrix you can drill into for per-pair details.
 *
 * This is the page you live in once you have more than two things to compare.
 * Each cell is just a saved eval, so its games persist and stay browsable
 * from the Evals/Games views.
 */

import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, errorMessage } from "../api";
import {
  Checkbox,
  FormBox,
  FormFooter,
  FormRow,
  NumberInput,
  PrimaryButton,
  SecondaryButton,
  TextInput,
} from "../components/Form";
import { Notice } from "../components/Notice";
import { EmptyState, PageHeader, SectionHeader } from "../components/Page";
import { PlayerPicker } from "../components/PlayerPicker";
import { TimeAgo } from "../components/TimeAgo";
import { parseSchedule } from "../schedule";
import type {
  ComparisonRecord,
  GameSpec,
  PlayerRecord,
} from "../types";

type SpecRow = { id: string; n: number; scheduleText: string };

export function ComparePage() {
  const nav = useNavigate();
  const [players, setPlayers] = useState<PlayerRecord[] | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [rows, setRows] = useState<SpecRow[]>(() => [
    { id: crypto.randomUUID(), n: 14, scheduleText: "1,2,3,3,2,1" },
  ]);
  const [nGames, setNGames] = useState(10);
  const [swap, setSwap] = useState(true);
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [recent, setRecent] = useState<ComparisonRecord[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listPlayers()
      .then((ps) => {
        if (cancelled) return;
        setPlayers(ps);
        setSelectedIds(
          new Set(ps.filter((p) => p.is_default).slice(0, 3).map((p) => p.id)),
        );
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    api
      .listComparisons()
      .then((cs) => {
        if (!cancelled) setRecent(cs.slice(0, 6));
      })
      .catch((e) => {
        // Surface this — silently empty "Recent comparisons" hid real backend
        // breakage during development. The user can still run a comparison
        // even if the list is broken, so the error is informational.
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const parsedSpecs = useMemo<(GameSpec | null)[]>(
    () =>
      rows.map((r) => {
        const sched = parseSchedule(r.scheduleText);
        if (!sched || r.n <= 0) return null;
        if (sched.reduce((a, b) => a + b, 0) > r.n) return null;
        return { n: r.n, schedule: sched };
      }),
    [rows],
  );

  const allValid =
    selectedIds.size >= 2 &&
    parsedSpecs.length > 0 &&
    parsedSpecs.every((s) => s !== null) &&
    nGames > 0;

  const nPairs = (selectedIds.size * (selectedIds.size - 1)) / 2;
  const totalGames = nPairs * parsedSpecs.length * nGames;

  const submit = async () => {
    if (!allValid) return;
    setBusy(true);
    setError(null);
    try {
      const rec = await api.runComparison({
        player_ids: [...selectedIds],
        specs: parsedSpecs as GameSpec[],
        n_games_per_spec: nGames,
        swap_sides: swap,
        seed,
      });
      nav(`/compare/${rec.id}`);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Compare"
        description="Pick two or more players and one or more specs. Every unordered pair plays a tournament across every spec; the result is a winrate matrix you can drill into."
      />

      <FormBox>
        <FormRow label="Players">
          <PlayerPicker
            players={players}
            selectedIds={selectedIds}
            onToggle={(id) =>
              setSelectedIds((s) => {
                const next = new Set(s);
                if (next.has(id)) next.delete(id);
                else next.add(id);
                return next;
              })
            }
          />
        </FormRow>
        <FormRow label="Specs">
          <SpecRows
            rows={rows}
            update={(id, patch) =>
              setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)))
            }
            add={() =>
              setRows((rs) => [
                ...rs,
                { id: crypto.randomUUID(), n: rs.at(-1)?.n ?? 7, scheduleText: rs.at(-1)?.scheduleText ?? "2,2,1" },
              ])
            }
            remove={(id) => setRows((rs) => (rs.length > 1 ? rs.filter((r) => r.id !== id) : rs))}
          />
        </FormRow>
        <FormRow label="Run">
          <NumberInput value={nGames} onChange={(v) => setNGames(v ?? 0)} />
          <span className="text-xs text-neutral-500">games per (spec, pair)</span>
          <Checkbox checked={swap} onChange={setSwap} label="swap sides" />
          <span className="text-xs text-neutral-500 ml-2">seed</span>
          <NumberInput value={seed} onChange={(v) => setSeed(v ?? 0)} />
        </FormRow>
        <FormFooter>
          <span
            className={`text-xs tabular-nums ${
              totalGames > 500 ? "text-amber-700" : "text-neutral-500"
            }`}
          >
            {selectedIds.size} players · {parsedSpecs.length} specs · {nPairs} pairs ·
            {" "}{totalGames} games total
            {totalGames > 500 && " · this may take a while"}
          </span>
          <PrimaryButton onClick={submit} disabled={!allValid || busy}>
            {busy ? "Running..." : "Run comparison"}
          </PrimaryButton>
        </FormFooter>
      </FormBox>

      {error && <div className="mt-4"><Notice kind="error">{error}</Notice></div>}
      {busy && (
        <p className="text-xs text-neutral-500 mt-3">
          Runs synchronously. Large comparisons (many pairs × specs × games)
          can take a while; the request stays open until complete.
        </p>
      )}

      <div className="mt-10">
        <RecentComparisons recent={recent} players={players} />
      </div>
    </div>
  );
}

function SpecRows({
  rows,
  update,
  remove,
  add,
}: {
  rows: SpecRow[];
  update: (id: string, patch: Partial<SpecRow>) => void;
  remove: (id: string) => void;
  add: () => void;
}) {
  return (
    <div className="space-y-1.5 w-full">
      {rows.map((r) => {
        const sched = parseSchedule(r.scheduleText);
        const sum = sched?.reduce((a, b) => a + b, 0) ?? 0;
        const valid = sched !== null && r.n > 0 && sum <= r.n;
        return (
          <div key={r.id} className="flex items-center gap-2">
            <span className="text-xs text-neutral-500 w-4">N</span>
            <NumberInput value={r.n} onChange={(v) => update(r.id, { n: v ?? 0 })} />
            <span className="text-xs text-neutral-500">schedule</span>
            <TextInput
              value={r.scheduleText}
              onChange={(s) => update(r.id, { scheduleText: s })}
              mono
              width="w-40"
            />
            <span
              className={`text-xs font-mono tabular-nums ${
                valid ? "text-neutral-400" : "text-red-600"
              }`}
            >
              {sched ? `sum ${sum}/${r.n}` : "invalid"}
            </span>
            <div className="ml-auto">
              <SecondaryButton
                size="sm"
                onClick={() => remove(r.id)}
                disabled={rows.length === 1}
                title="remove this spec"
              >
                remove
              </SecondaryButton>
            </div>
          </div>
        );
      })}
      <SecondaryButton size="sm" onClick={add}>+ add spec</SecondaryButton>
    </div>
  );
}

function RecentComparisons({
  recent,
  players,
}: {
  recent: ComparisonRecord[] | null;
  players: PlayerRecord[] | null;
}) {
  const nav = useNavigate();
  if (!recent) return null;
  const byId = new Map((players ?? []).map((p) => [p.id, p.label]));
  if (recent.length === 0) {
    return (
      <div>
        <SectionHeader title="Recent comparisons" />
        <EmptyState>No comparisons yet. Run one above.</EmptyState>
      </div>
    );
  }
  return (
    <div>
      <SectionHeader title="Recent comparisons" />
      <div className="border border-neutral-200 bg-white">
        <table className="w-full text-sm">
          <thead className="text-[11px] uppercase text-neutral-500 bg-neutral-50/50 border-b border-neutral-200">
            <tr>
              <th className="text-left px-3 py-2 font-normal">id</th>
              <th className="text-left px-3 py-2 font-normal">players</th>
              <th className="text-left px-3 py-2 font-normal">specs</th>
              <th className="text-right px-3 py-2 font-normal">games</th>
              <th className="text-left px-3 py-2 font-normal">status</th>
              <th className="text-left px-3 py-2 font-normal">when</th>
            </tr>
          </thead>
          <tbody>
            {recent.map((c) => (
              <tr
                key={c.id}
                onClick={() => nav(`/compare/${c.id}`)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") nav(`/compare/${c.id}`);
                }}
                role="link"
                tabIndex={0}
                className="border-b border-neutral-100 last:border-0 cursor-pointer hover:bg-neutral-50 focus:bg-neutral-50 focus:outline-none"
              >
                <td className="px-3 py-2">
                  <Link to={`/compare/${c.id}`} className="font-mono text-xs text-neutral-900 underline underline-offset-2 hover:text-black hover:decoration-2">
                    {c.id.slice(0, 8)}
                  </Link>
                </td>
                <td className="px-3 py-2 text-xs">
                  {c.config.player_ids.map((pid) => byId.get(pid) ?? pid.slice(0, 6)).join(", ")}
                </td>
                <td className="px-3 py-2 font-mono text-xs text-neutral-600">
                  {c.config.specs
                    .map((s) => `N=${s.n}[${s.schedule.join(",")}]`)
                    .join(" · ")}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {c.summary?.n_total_games ?? "—"}
                </td>
                <td className="px-3 py-2 text-xs">{c.status}</td>
                <td className="px-3 py-2 text-xs text-neutral-500">
                  <TimeAgo iso={c.created_at} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
