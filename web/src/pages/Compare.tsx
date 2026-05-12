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
import { comparisonProgress, formatSpec } from "../format";
import {
  Checkbox,
  FormBox,
  FormFooter,
  FormRow,
  NumberInput,
  PrimaryButton,
  SecondaryButton,
  Select,
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
  SpecSamplerConfig,
} from "../types";

type SpecRow = { id: string; n: number; scheduleText: string };
type SpecMode = "fixed" | "sampled";

const DEFAULT_SAMPLER: SpecSamplerConfig = {
  kind: "mixture",
  n_min: 12,
  n_max: 24,
  turns_min: 3,
  turns_max: 8,
  fill_min: 0.6,
  fill_max: 0.95,
  max_marks_per_turn: 6,
  random_weight: 0.4,
  arc_weight: 0.25,
  few_big_weight: 0.2,
  many_small_weight: 0.15,
};

export function ComparePage() {
  const nav = useNavigate();
  const [players, setPlayers] = useState<PlayerRecord[] | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [rows, setRows] = useState<SpecRow[]>(() => [
    { id: crypto.randomUUID(), n: 14, scheduleText: "1,2,3,3,2,1" },
  ]);
  const [specMode, setSpecMode] = useState<SpecMode>("fixed");
  const [sampler, setSampler] = useState<SpecSamplerConfig>(DEFAULT_SAMPLER);
  const [nSampledSpecs, setNSampledSpecs] = useState(200);
  const [preview, setPreview] = useState<GameSpec[] | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
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

  const fixedSpecsValid =
    parsedSpecs.length > 0 && parsedSpecs.every((s) => s !== null);
  const sampledSpecsValid =
    sampler.n_min > 0 &&
    sampler.n_max >= sampler.n_min &&
    sampler.turns_min > 0 &&
    sampler.turns_max >= sampler.turns_min &&
    sampler.fill_min >= 0 &&
    sampler.fill_max <= 1 &&
    sampler.fill_max >= sampler.fill_min &&
    sampler.max_marks_per_turn > 0 &&
    nSampledSpecs > 0;

  const allValid =
    selectedIds.size >= 2 &&
    (specMode === "fixed" ? fixedSpecsValid : sampledSpecsValid) &&
    nGames > 0;

  const nPairs = (selectedIds.size * (selectedIds.size - 1)) / 2;
  const nSpecs = specMode === "fixed" ? parsedSpecs.length : nSampledSpecs;
  const totalGames = nPairs * nSpecs * nGames;

  const loadPreview = async () => {
    if (!sampledSpecsValid) return;
    setPreviewBusy(true);
    setError(null);
    try {
      const specs = await api.sampleSpecs({
        sampler,
        count: Math.min(nSampledSpecs, 20),
        seed,
      });
      setPreview(specs);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setPreviewBusy(false);
    }
  };

  const submit = async () => {
    if (!allValid) return;
    setBusy(true);
    setError(null);
    try {
      const rec = await api.runComparison({
        player_ids: [...selectedIds],
        specs: specMode === "fixed" ? (parsedSpecs as GameSpec[]) : [],
        spec_sampler: specMode === "sampled" ? sampler : null,
        n_sampled_specs: specMode === "sampled" ? nSampledSpecs : null,
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
        <FormRow label="Spec source">
          <Select
            value={specMode}
            options={[
              { value: "fixed", label: "Fixed specs" },
              { value: "sampled", label: "Sampled specs" },
            ]}
            onChange={setSpecMode}
            width="w-40"
          />
          <span className="text-xs text-neutral-500">
            {specMode === "fixed"
              ? "manually list exact specs"
              : "sample a reproducible arena from a distribution"}
          </span>
        </FormRow>
        {specMode === "fixed" ? (
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
        ) : (
          <SampledSpecRows
            sampler={sampler}
            setSampler={setSampler}
            count={nSampledSpecs}
            setCount={setNSampledSpecs}
            seed={seed}
            preview={preview}
            previewBusy={previewBusy}
            onPreview={loadPreview}
          />
        )}
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
            {selectedIds.size} players · {nSpecs} specs · {nPairs} pairs ·
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
          Submitting comparison. The run continues in the background; you can
          watch or cancel it from the detail page.
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

function SampledSpecRows({
  sampler,
  setSampler,
  count,
  setCount,
  seed,
  preview,
  previewBusy,
  onPreview,
}: {
  sampler: SpecSamplerConfig;
  setSampler: (s: SpecSamplerConfig) => void;
  count: number;
  setCount: (n: number) => void;
  seed: number;
  preview: GameSpec[] | null;
  previewBusy: boolean;
  onPreview: () => void;
}) {
  const patch = (p: Partial<SpecSamplerConfig>) => setSampler({ ...sampler, ...p });
  return (
    <>
      <FormRow label="Sample count">
        <NumberInput value={count} onChange={(v) => setCount(v ?? 0)} />
        <span className="text-xs text-neutral-500">unique specs sampled with seed {seed}</span>
      </FormRow>
      <FormRow label="N range">
        <NumberInput value={sampler.n_min} onChange={(v) => patch({ n_min: v ?? 0 })} />
        <span className="text-xs text-neutral-400">to</span>
        <NumberInput value={sampler.n_max} onChange={(v) => patch({ n_max: v ?? 0 })} />
      </FormRow>
      <FormRow label="Turns">
        <NumberInput value={sampler.turns_min} onChange={(v) => patch({ turns_min: v ?? 0 })} />
        <span className="text-xs text-neutral-400">to</span>
        <NumberInput value={sampler.turns_max} onChange={(v) => patch({ turns_max: v ?? 0 })} />
      </FormRow>
      <FormRow label="Fill">
        <NumberInput
          value={sampler.fill_min}
          onChange={(v) => patch({ fill_min: v ?? 0 })}
        />
        <span className="text-xs text-neutral-400">to</span>
        <NumberInput
          value={sampler.fill_max}
          onChange={(v) => patch({ fill_max: v ?? 0 })}
        />
        <span className="text-xs text-neutral-500">fraction of board filled</span>
      </FormRow>
      <FormRow label="Max/turn">
        <NumberInput
          value={sampler.max_marks_per_turn}
          onChange={(v) => patch({ max_marks_per_turn: v ?? 0 })}
        />
        <SecondaryButton onClick={onPreview} disabled={previewBusy}>
          {previewBusy ? "previewing..." : "preview"}
        </SecondaryButton>
      </FormRow>
      {preview && (
        <FormRow label="Preview">
          <div className="flex flex-wrap gap-1.5 font-mono text-[11px] text-neutral-600">
            {preview.map((s, i) => (
              <span key={i} className="border border-neutral-200 px-1.5 py-0.5">
                {formatSpec(s)}
              </span>
            ))}
            {count > preview.length && (
              <span className="text-neutral-400 px-1.5 py-0.5">
                +{count - preview.length} more
              </span>
            )}
          </div>
        </FormRow>
      )}
    </>
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
                  {formatSpecList(c.config.specs)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {formatProgress(c)}
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

function formatSpecList(specs: GameSpec[]): string {
  const shown: string[] = specs.slice(0, 4).map(formatSpec);
  if (specs.length > shown.length) shown.push(`+${specs.length - shown.length} more`);
  return shown.join(" · ");
}

function formatProgress(c: ComparisonRecord): string {
  if (c.summary) return String(c.summary.n_total_games);
  const { done, total, pct } = comparisonProgress(c);
  if (total <= 0) return "—";
  return `${done}/${total} (${pct}%)`;
}
