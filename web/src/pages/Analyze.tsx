/**
 * Analyze a single position with multiple players at once.
 *
 * Workflow:
 * 1. Pick a spec (N + schedule). Optionally seed from a saved game id.
 * 2. Set up a position by clicking empty cells; clicks alternate following
 *    the spec's schedule so any sequence you build is legal.
 * 3. Choose which players to ask. Each player's chosen cell is shown next
 *    to its agreement with the exact solver (when the spec is small enough).
 *
 * This is the project's microscope: it's the same view you use to debug the
 * engine today and to inspect a trained model's decisions later.
 */

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, errorMessage } from "../api";
import { Board } from "../components/Board";
import {
  FormBox,
  FormRow,
  NumberInput,
  PrimaryButton,
  SecondaryButton,
  TextInput,
} from "../components/Form";
import { Mark } from "../components/Mark";
import { Notice } from "../components/Notice";
import { oracleAnnotations } from "../components/Oracle";
import { PageHeader, SectionHeader } from "../components/Page";
import { PlayerPicker } from "../components/PlayerPicker";
import { Schedule } from "../components/Schedule";
import { formatAgent } from "../format";
import { statesAlong } from "../game";
import { parseSchedule } from "../schedule";
import type {
  AnalyzeResponse,
  GameSpec,
  PlayerRecord,
  PlayerVerdict,
} from "../types";

export function AnalyzePage() {
  const [searchParams] = useSearchParams();
  const seededGameId = searchParams.get("game");
  const seededStep = searchParams.get("step");
  const seededN = searchParams.get("n");
  const seededSchedule = searchParams.get("schedule");

  // Default to (N=8, [1,2,2,2,1]) — solver-tractable (40k leaves, fits the
  // oracle), equal 4-4 marks across 5 turns, fills the entire board. The
  // most strategically rich config inside the solver's reach.
  const [n, setN] = useState(() => (seededN ? Number(seededN) : 8));
  const [scheduleText, setScheduleText] = useState(() => seededSchedule ?? "1,2,2,2,1");
  const [actions, setActions] = useState<number[]>([]);
  const [players, setPlayers] = useState<PlayerRecord[] | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadGameId, setLoadGameId] = useState(seededGameId ?? "");

  const schedule = useMemo(() => parseSchedule(scheduleText), [scheduleText]);
  const sum = schedule?.reduce((a, b) => a + b, 0) ?? 0;
  const spec: GameSpec | null =
    schedule !== null && n > 0 && sum <= n ? { n, schedule } : null;

  // After an N/schedule change, `actions` is briefly stale (a cell from the
  // old spec may be out of range). Catch the one-tick mismatch; the
  // clear-actions effect below resolves it on the next render.
  const states = useMemo(() => {
    if (!spec) return null;
    try {
      return statesAlong(spec, actions);
    } catch {
      return null;
    }
  }, [spec, actions]);
  const state = states?.[states.length - 1] ?? null;

  useEffect(() => {
    let cancelled = false;
    api
      .listPlayers()
      .then((ps) => {
        if (cancelled) return;
        setPlayers(ps);
        const defaults = ps
          .filter((p) =>
            ["p_alphabeta_exact", "p_greedy_0", "p_rightmost"].includes(p.id),
          )
          .map((p) => p.id);
        setSelectedIds(new Set(defaults));
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Re-sync from URL params on SPA nav (useState initializers only fire once).
  useEffect(() => {
    if (seededN !== null) {
      const parsed = Number(seededN);
      if (Number.isFinite(parsed) && parsed > 0) setN(parsed);
    }
    if (seededSchedule !== null) setScheduleText(seededSchedule);
    if (seededN !== null || seededSchedule !== null) setActions([]);
  }, [seededN, seededSchedule]);

  useEffect(() => {
    if (!seededGameId) return;
    let cancelled = false;
    api
      .getGame(seededGameId)
      .then((g) => {
        if (cancelled) return;
        setN(g.spec.n);
        setScheduleText(g.spec.schedule.join(","));
        const cutoff =
          seededStep !== null
            ? Math.max(0, Math.min(g.actions.length, Number(seededStep)))
            : g.actions.length;
        setActions(g.actions.slice(0, cutoff));
        setAnalysis(null);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, [seededGameId, seededStep]);

  // Drop stale actions when the spec changes — old indices may be out of
  // range for the new board.
  useEffect(() => {
    setActions([]);
    setAnalysis(null);
  }, [n, scheduleText]);

  useEffect(() => {
    setAnalysis(null);
  }, [actions]);

  const onCellClick = (cell: number) => {
    if (!state || state.isTerminal) return;
    if (state.board[cell] !== 0) return;
    setActions((prev) => [...prev, cell]);
  };

  const undo = () => setActions((prev) => prev.slice(0, -1));
  const clear = () => {
    setActions([]);
    setAnalysis(null);
  };

  const loadGame = async () => {
    if (!loadGameId.trim()) return;
    try {
      const g = await api.getGame(loadGameId.trim());
      setN(g.spec.n);
      setScheduleText(g.spec.schedule.join(","));
      setActions(g.actions);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const toggle = (id: string) =>
    setSelectedIds((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const analyze = async () => {
    if (!spec || !state || state.isTerminal) return;
    if (selectedIds.size === 0) return;
    setAnalyzing(true);
    setError(null);
    try {
      const r = await api.analyze({
        spec,
        actions,
        player_ids: [...selectedIds],
      });
      setAnalysis(r);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setAnalyzing(false);
    }
  };

  // Oracle fields are always set together or all null on the server. The
  // explicit guard keeps the discriminated union narrowing intact.
  const annotations =
    analysis &&
    analysis.oracle_per_cell_values !== null &&
    analysis.oracle_value !== null &&
    analysis.oracle_best_cell !== null &&
    state &&
    !state.isTerminal
      ? oracleAnnotations({
          value: analysis.oracle_value,
          best_cell: analysis.oracle_best_cell,
          per_cell_values: analysis.oracle_per_cell_values,
          is_terminal: false,
        })
      : undefined;

  return (
    <div className="space-y-8">
      <PageHeader
        title="Analyze"
        description="Set up a position and ask every selected player what it would do. The exact solver is included automatically when the spec is small enough."
      />

      <FormBox>
        <FormRow label="N" hint={`${n} cells`}>
          <NumberInput value={n} onChange={(v) => setN(v ?? 0)} />
        </FormRow>
        <FormRow
          label="Schedule"
          hint={schedule ? `sum ${sum} of ${n}` : "comma-separated positive ints"}
        >
          <TextInput
            value={scheduleText}
            onChange={setScheduleText}
            mono
            placeholder="2,2,1"
          />
        </FormRow>
        <FormRow label="Seed from game">
          <TextInput
            value={loadGameId}
            onChange={setLoadGameId}
            mono
            placeholder="game id"
            width="w-72"
          />
          <SecondaryButton onClick={loadGame} disabled={!loadGameId.trim()}>
            load
          </SecondaryButton>
          <span className="text-xs text-neutral-400">
            replaces N, schedule, and current actions
          </span>
        </FormRow>
      </FormBox>

      {!spec && (
        <Notice kind="error">Invalid spec. Fix N or the schedule.</Notice>
      )}

      {spec && state && (
        <div className="space-y-4">
          <Schedule
            schedule={spec.schedule}
            turnIdx={state.turnIdx}
            placementsLeft={state.placementsLeft}
          />

          <Board
            state={state}
            highlightWinner={state.isTerminal}
            lastAction={actions.length > 0 ? actions[actions.length - 1] : null}
            onCellClick={state.isTerminal ? undefined : onCellClick}
            hoverPlayer={
              !state.isTerminal
                ? state.currentPlayer === 1
                  ? "X"
                  : "O"
                : null
            }
            annotations={annotations}
          />

          <div className="flex items-center gap-3 flex-wrap text-sm">
            {state.isTerminal ? (
              <span className="text-neutral-500">Position is terminal.</span>
            ) : (
              <span className="flex items-center gap-2 text-neutral-700">
                <Mark player={state.currentPlayer === 1 ? "X" : "O"} />
                to move · click any empty cell
              </span>
            )}
            <div className="ml-auto flex items-center gap-2">
              <SecondaryButton onClick={undo} disabled={actions.length === 0}>
                undo
              </SecondaryButton>
              <SecondaryButton onClick={clear} disabled={actions.length === 0}>
                clear
              </SecondaryButton>
            </div>
          </div>
        </div>
      )}

      <div>
        <SectionHeader
          title="Players"
          actions={
            <span className="text-xs text-neutral-500 tabular-nums">
              {selectedIds.size} selected
            </span>
          }
        />
        <div className="space-y-3">
          <PlayerPicker players={players} selectedIds={selectedIds} onToggle={toggle} />
          <div className="flex items-center gap-2 text-xs text-neutral-500">
            <SecondaryButton
              size="sm"
              onClick={() => setSelectedIds(new Set(players?.map((p) => p.id) ?? []))}
            >
              all
            </SecondaryButton>
            <SecondaryButton size="sm" onClick={() => setSelectedIds(new Set())}>
              none
            </SecondaryButton>
            <div className="ml-auto">
              <PrimaryButton
                onClick={analyze}
                disabled={
                  !spec || !state || state.isTerminal || selectedIds.size === 0 || analyzing
                }
              >
                {analyzing ? "Analyzing..." : "Analyze position"}
              </PrimaryButton>
            </div>
          </div>
        </div>
      </div>

      {error && <Notice kind="error">{error}</Notice>}

      {analysis && <VerdictsTable analysis={analysis} players={players ?? []} />}
    </div>
  );
}

function VerdictsTable({
  analysis,
  players,
}: {
  analysis: AnalyzeResponse;
  players: PlayerRecord[];
}) {
  const byId = new Map(players.map((p) => [p.id, p]));
  const winner =
    analysis.oracle_value === 1 ? "X" : analysis.oracle_value === -1 ? "O" : null;

  return (
    <div className="space-y-3">
      <SectionHeader title="Verdicts" />
      {winner ? (
        <div className="border border-neutral-200 bg-white px-3 py-2 text-sm flex items-center gap-2">
          <span className="text-neutral-500">Oracle:</span>
          <Mark player={winner} />
          <span className="text-neutral-900">wins with best play</span>
          {analysis.oracle_best_cell !== null && analysis.oracle_best_cell >= 0 && (
            <>
              <span className="text-neutral-300">·</span>
              <span className="text-neutral-500">best cell</span>
              <span className="font-mono tabular-nums text-neutral-900">
                {analysis.oracle_best_cell}
              </span>
            </>
          )}
        </div>
      ) : (
        <div className="border border-neutral-200 bg-white px-3 py-2 text-sm text-neutral-500">
          Oracle unavailable for this spec (state space too large). Agreement
          columns are not populated.
        </div>
      )}

      <div className="border border-neutral-200 bg-white">
        <table className="w-full text-sm">
          <thead className="text-[11px] uppercase text-neutral-500 bg-neutral-50/50 border-b border-neutral-200">
            <tr>
              <th className="text-left px-3 py-2 font-normal">player</th>
              <th className="text-left px-3 py-2 font-normal">picks cell</th>
              <th className="text-left px-3 py-2 font-normal">vs oracle</th>
              <th className="text-left px-3 py-2 font-normal">agent</th>
            </tr>
          </thead>
          <tbody>
            {analysis.verdicts.map((v) => (
              <VerdictRow key={v.player_id} v={v} agentLabel={byId.get(v.player_id)?.agent_spec} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function VerdictRow({
  v,
  agentLabel,
}: {
  v: PlayerVerdict;
  agentLabel: PlayerRecord["agent_spec"] | undefined;
}) {
  return (
    <tr className="border-b border-neutral-100 last:border-0">
      <td className="px-3 py-2 text-neutral-900">{v.label}</td>
      <td className="px-3 py-2 font-mono tabular-nums">
        {v.cell !== null ? v.cell : <span className="text-neutral-400">—</span>}
      </td>
      <td className="px-3 py-2 text-xs">
        {v.agrees_with_oracle === null ? (
          <span className="text-neutral-400">n/a</span>
        ) : v.agrees_with_oracle ? (
          <span className="text-emerald-700">optimal</span>
        ) : (
          <span className="text-rose-700">suboptimal</span>
        )}
        {v.error && <span className="ml-2 text-rose-700 font-mono">{v.error}</span>}
      </td>
      <td className="px-3 py-2 font-mono text-xs text-neutral-500">
        {agentLabel ? formatAgent(agentLabel) : ""}
      </td>
    </tr>
  );
}
