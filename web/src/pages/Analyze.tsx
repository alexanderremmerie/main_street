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
  Select,
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
  GameRecord,
  GameSpec,
  InspectModelResponse,
  PlayerRecord,
  PlayerVerdict,
} from "../types";

function gameCutoff(actionCount: number, stepText: string | null) {
  if (stepText !== null) {
    const parsed = Number(stepText);
    if (Number.isFinite(parsed)) {
      return Math.max(0, Math.min(actionCount, parsed));
    }
  }
  return actionCount;
}

function seededActions(g: GameRecord, stepText: string | null) {
  return g.actions.slice(0, gameCutoff(g.actions.length, stepText));
}

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
  const [inspection, setInspection] = useState<InspectModelResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadGameId, setLoadGameId] = useState(seededGameId ?? "");
  const [inspectPlayerId, setInspectPlayerId] = useState("");
  const [inspectSims, setInspectSims] = useState(200);

  const schedule = useMemo(() => parseSchedule(scheduleText), [scheduleText]);
  const sum = schedule?.reduce((a, b) => a + b, 0) ?? 0;
  const spec: GameSpec | null = useMemo(
    () => (schedule !== null && n > 0 && sum <= n ? { n, schedule } : null),
    [n, schedule, sum],
  );

  // After an N/schedule change, `actions` can be briefly stale while React
  // batches updates. Catch the one-tick mismatch; input handlers clear actions.
  const states = useMemo(() => {
    if (!spec) return null;
    try {
      return statesAlong(spec, actions);
    } catch {
      return null;
    }
  }, [spec, actions]);
  const state = states?.[states.length - 1] ?? null;

  const updateN = (value: number | null) => {
    setN(value ?? 0);
    setActions([]);
    setAnalysis(null);
    setInspection(null);
  };

  const updateScheduleText = (value: string) => {
    setScheduleText(value);
    setActions([]);
    setAnalysis(null);
    setInspection(null);
  };

  const applySeededGame = (g: GameRecord, stepText: string | null) => {
    setN(g.spec.n);
    setScheduleText(g.spec.schedule.join(","));
    setActions(seededActions(g, stepText));
    setAnalysis(null);
    setInspection(null);
  };

  useEffect(() => {
    let cancelled = false;
    api
      .listPlayers()
      .then((ps) => {
        if (cancelled) return;
        setPlayers(ps);
        const firstModel = ps.find((p) => p.agent_spec.kind === "alphazero");
        if (firstModel) setInspectPlayerId(firstModel.id);
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
      // eslint-disable-next-line react-hooks/set-state-in-effect -- route params are external navigation state.
      if (Number.isFinite(parsed) && parsed > 0) setN(parsed);
    }
    if (seededSchedule !== null) setScheduleText(seededSchedule);
    if (seededN !== null || seededSchedule !== null) setActions([]);
    if (seededN !== null || seededSchedule !== null) setAnalysis(null);
    if (seededN !== null || seededSchedule !== null) setInspection(null);
  }, [seededN, seededSchedule]);

  useEffect(() => {
    if (!seededGameId) return;
    let cancelled = false;
    api
      .getGame(seededGameId)
      .then((g) => {
        if (cancelled) return;
        applySeededGame(g, seededStep);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, [seededGameId, seededStep]);

  const onCellClick = (cell: number) => {
    if (!state || state.isTerminal) return;
    if (state.board[cell] !== 0) return;
    setAnalysis(null);
    setInspection(null);
    setActions((prev) => [...prev, cell]);
  };

  const undo = () => {
    setAnalysis(null);
    setInspection(null);
    setActions((prev) => prev.slice(0, -1));
  };
  const clear = () => {
    setActions([]);
    setAnalysis(null);
    setInspection(null);
  };

  const loadGame = async () => {
    if (!loadGameId.trim()) return;
    try {
      const g = await api.getGame(loadGameId.trim());
      applySeededGame(g, null);
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

  const inspect = async () => {
    if (!spec || !state || state.isTerminal || !inspectPlayerId) return;
    setInspecting(true);
    setError(null);
    try {
      const r = await api.inspectModel({
        spec,
        actions,
        player_id: inspectPlayerId,
        n_simulations: inspectSims,
      });
      setInspection(r);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setInspecting(false);
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
          <NumberInput value={n} onChange={updateN} />
        </FormRow>
        <FormRow
          label="Schedule"
          hint={schedule ? `sum ${sum} of ${n}` : "comma-separated positive ints"}
        >
          <TextInput
            value={scheduleText}
            onChange={updateScheduleText}
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

      <ModelInspectPanel
        players={players ?? []}
        selectedId={inspectPlayerId}
        setSelectedId={setInspectPlayerId}
        sims={inspectSims}
        setSims={setInspectSims}
        disabled={!spec || !state || state.isTerminal || inspecting}
        inspecting={inspecting}
        onInspect={inspect}
        result={inspection}
      />
    </div>
  );
}

function ModelInspectPanel({
  players,
  selectedId,
  setSelectedId,
  sims,
  setSims,
  disabled,
  inspecting,
  onInspect,
  result,
}: {
  players: PlayerRecord[];
  selectedId: string;
  setSelectedId: (id: string) => void;
  sims: number;
  setSims: (n: number) => void;
  disabled: boolean;
  inspecting: boolean;
  onInspect: () => void;
  result: InspectModelResponse | null;
}) {
  const modelPlayers = players.filter((p) => p.agent_spec.kind === "alphazero");
  return (
    <div className="space-y-3">
      <SectionHeader title="Model inspect" />
      <FormBox>
        <FormRow label="Model">
          <Select
            value={selectedId}
            onChange={setSelectedId}
            options={modelPlayers.map((p) => ({ value: p.id, label: p.label }))}
            width="w-80"
          />
          <span className="text-xs text-neutral-500">
            raw policy/value plus PUCT visits for the current position
          </span>
        </FormRow>
        <FormRow label="Search">
          <NumberInput value={sims} onChange={(v) => setSims(v ?? 0)} />
          <span className="text-xs text-neutral-500">PUCT simulations</span>
          <div className="ml-auto">
            <PrimaryButton
              onClick={onInspect}
              disabled={disabled || !selectedId || modelPlayers.length === 0}
            >
              {inspecting ? "Inspecting..." : "Inspect model"}
            </PrimaryButton>
          </div>
        </FormRow>
        {modelPlayers.length === 0 && (
          <div className="text-xs text-neutral-500">
            Create or keep an AlphaZero player first; classical agents do not
            expose policy/value tensors.
          </div>
        )}
      </FormBox>

      {result && <InspectionResult result={result} />}
    </div>
  );
}

function InspectionResult({ result }: { result: InspectModelResponse }) {
  const maxRaw = Math.max(...result.moves.map((m) => m.raw_policy), 1e-9);
  const maxVisits = Math.max(...result.moves.map((m) => m.puct_visits), 1);
  return (
    <div className="border border-neutral-200 bg-white">
      <div className="px-3 py-2 border-b border-neutral-200 flex items-center gap-3 text-sm">
        <span className="font-medium text-neutral-900">{result.label}</span>
        <span className="text-neutral-300">·</span>
        <span className="text-neutral-500 flex items-center gap-1">
          value for <Mark player={result.current_player === 1 ? "X" : "O"} />
          <span className="font-mono tabular-nums text-neutral-900">
            {result.raw_value.toFixed(3)}
          </span>
        </span>
        <span className="text-neutral-300">·</span>
        <span className="text-neutral-500">
          PUCT picks{" "}
          <span className="font-mono tabular-nums text-neutral-900">
            {result.puct_action}
          </span>
        </span>
      </div>
      <table className="w-full text-sm">
        <thead className="text-[11px] uppercase text-neutral-500 bg-neutral-50/50 border-b border-neutral-200">
          <tr>
            <th className="text-left px-3 py-2 font-normal">cell</th>
            <th className="text-left px-3 py-2 font-normal">raw policy</th>
            <th className="text-left px-3 py-2 font-normal">PUCT visits</th>
            <th className="text-right px-3 py-2 font-normal">move value</th>
          </tr>
        </thead>
        <tbody>
          {result.moves.map((m) => (
            <tr
              key={m.cell}
              className={`border-b border-neutral-100 last:border-0 ${
                m.cell === result.puct_action ? "bg-emerald-50/70" : ""
              }`}
            >
              <td className="px-3 py-2 font-mono tabular-nums">{m.cell}</td>
              <td className="px-3 py-2">
                <MetricBar
                  value={m.raw_policy}
                  max={maxRaw}
                  label={`${(m.raw_policy * 100).toFixed(1)}%`}
                />
              </td>
              <td className="px-3 py-2">
                <MetricBar
                  value={m.puct_visits}
                  max={maxVisits}
                  label={`${m.puct_visits} (${(m.puct_visit_prob * 100).toFixed(1)}%)`}
                />
              </td>
              <td className="px-3 py-2 text-right font-mono tabular-nums">
                {m.puct_value === null ? "—" : m.puct_value.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricBar({
  value,
  max,
  label,
}: {
  value: number;
  max: number;
  label: string;
}) {
  const pct = max > 0 ? Math.max(1, Math.round((value / max) * 100)) : 0;
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-24 bg-neutral-100 border border-neutral-200">
        <div className="h-full bg-neutral-900" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-xs tabular-nums text-neutral-700">
        {label}
      </span>
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
