import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, errorMessage } from "../api";
import { AgentField } from "../components/AgentField";
import { Board } from "../components/Board";
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
import { Mark } from "../components/Mark";
import { Notice } from "../components/Notice";
import { OraclePanel, oracleAnnotations } from "../components/Oracle";
import { Outcome } from "../components/Outcome";
import { Schedule } from "../components/Schedule";
import { formatAgent, formatSpec } from "../format";
import { statesAlong, winner } from "../game";
import { useOracle } from "../oracle";
import { parseSchedule } from "../schedule";
import type { AgentSpec, GameSpec } from "../types";

type Phase = "config" | "playing" | "terminal";

export function SandboxPanel() {
  const nav = useNavigate();

  // Palindromic tempo arc, 12 marks on a 14-board: open slow, peak with
  // back-to-back 3-turns mid-game, close slow. Beyond solver reach but
  // playable in seconds.
  const [n, setN] = useState(14);
  const [scheduleText, setScheduleText] = useState("1,2,3,3,2,1");
  const [xAgent, setXAgent] = useState<AgentSpec>({ kind: "human" });
  const [oAgent, setOAgent] = useState<AgentSpec>({ kind: "alphabeta", depth: 4 });

  const [phase, setPhase] = useState<Phase>("config");
  const [actions, setActions] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [savedAs, setSavedAs] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [showOracle, setShowOracle] = useState(false);

  const schedule = useMemo(() => parseSchedule(scheduleText), [scheduleText]);
  const sum = schedule?.reduce((a, b) => a + b, 0) ?? 0;
  const configValid = schedule !== null && n > 0 && sum <= n;

  const spec: GameSpec | null = configValid ? { n, schedule: schedule! } : null;
  const states = useMemo(
    () => (spec && phase !== "config" ? statesAlong(spec, actions) : null),
    [spec, actions, phase],
  );
  const state = states?.[states.length - 1] ?? null;
  const currentAgent =
    state && !state.isTerminal ? (state.currentPlayer === 1 ? xAgent : oAgent) : null;
  const lastAction = actions.length > 0 ? actions[actions.length - 1] : null;

  const oracle = useOracle(
    spec && phase !== "config" ? { spec, actions } : null,
    showOracle && phase !== "config",
  );
  const annotations =
    showOracle && oracle.kind === "ok" && !oracle.data.is_terminal
      ? oracleAnnotations(oracle.data)
      : undefined;

  useEffect(() => {
    if (phase === "playing" && state?.isTerminal) setPhase("terminal");
  }, [phase, state?.isTerminal]);

  // When the current player is a bot, ask the server for its move and append.
  useEffect(() => {
    if (phase !== "playing" || !spec || !state || !currentAgent) return;
    if (state.isTerminal || currentAgent.kind === "human") return;

    let cancelled = false;
    api
      .computeMove({ spec, actions, agent: currentAgent })
      .then(({ cell }) => {
        if (cancelled) return;
        setActions((prev) => [...prev, cell]);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
    // Keyed on actions.length (not identity) so we re-fire only after a move
    // completes; agent/spec changes are gated by `phase === "playing"`.
  }, [phase, actions.length, currentAgent?.kind, spec?.n]);

  const onCellClick = (cell: number) => {
    if (phase !== "playing" || !state || state.isTerminal) return;
    if (currentAgent?.kind !== "human") return;
    if (state.board[cell] !== 0) return;
    setActions((prev) => [...prev, cell]);
  };

  const start = () => {
    if (!configValid) return;
    setActions([]);
    setError(null);
    setSavedAs(null);
    setPhase("playing");
  };

  const reset = () => {
    setActions([]);
    setError(null);
    setSavedAs(null);
    setPhase("config");
  };

  const save = async () => {
    if (!spec || phase !== "terminal") return;
    setSaving(true);
    setError(null);
    try {
      const rec = await api.saveGame({
        spec,
        x_agent: xAgent,
        o_agent: oAgent,
        actions,
      });
      setSavedAs(rec.id);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setSaving(false);
    }
  };

  if (phase === "config") {
    return (
      <div className="space-y-4">
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
              placeholder="1,2,3,1"
            />
          </FormRow>
          <FormRow label="X first">
            <AgentField value={xAgent} onChange={setXAgent} />
          </FormRow>
          <FormRow label="O second">
            <AgentField value={oAgent} onChange={setOAgent} />
          </FormRow>
          <FormFooter>
            <span className="text-xs text-neutral-500">
              {formatAgent(xAgent)} (X) vs {formatAgent(oAgent)} (O)
            </span>
            <PrimaryButton onClick={start} disabled={!configValid}>
              Start
            </PrimaryButton>
          </FormFooter>
        </FormBox>
      </div>
    );
  }

  // phase === "playing" || "terminal"
  if (!spec || !state) return null;
  const turnLabel = describeTurn({
    phase,
    state,
    currentAgent,
    xAgent,
    oAgent,
  });

  return (
    <div className="space-y-6">
      <SummaryBar
        spec={spec}
        xAgent={xAgent}
        oAgent={oAgent}
        onReset={reset}
      />

      <Schedule
        schedule={spec.schedule}
        turnIdx={state.turnIdx}
        placementsLeft={state.placementsLeft}
      />

      <Board
        state={state}
        highlightWinner={state.isTerminal}
        lastAction={lastAction}
        onCellClick={
          phase === "playing" && currentAgent?.kind === "human" ? onCellClick : undefined
        }
        hoverPlayer={
          phase === "playing" && currentAgent?.kind === "human"
            ? state.currentPlayer === 1
              ? "X"
              : "O"
            : null
        }
        annotations={annotations}
      />

      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="text-sm text-neutral-700 min-h-5">{turnLabel}</div>
        <Checkbox checked={showOracle} onChange={setShowOracle} label="show oracle" />
      </div>

      {showOracle && <OraclePanel result={oracle} />}

      {phase === "terminal" && (
        <Outcome state={state} outcome={winner(state.board)} />
      )}

      {error && <Notice kind="error">{error}</Notice>}

      {phase === "terminal" && (
        <div className="flex items-center gap-3">
          {savedAs ? (
            <span className="text-sm text-neutral-700">
              Saved as{" "}
              <Link
                to={`/games/${savedAs}`}
                className="font-mono text-neutral-900 underline underline-offset-2 hover:text-black hover:decoration-2"
              >
                {savedAs.slice(0, 8)}
              </Link>{" "}
              ·{" "}
              <button
                onClick={() => nav(`/games/${savedAs}`)}
                className="text-neutral-900 underline underline-offset-2 hover:text-black hover:decoration-2"
              >
                view replay
              </button>
            </span>
          ) : (
            <PrimaryButton onClick={save} disabled={saving}>
              {saving ? "Saving..." : "Save game"}
            </PrimaryButton>
          )}
        </div>
      )}
    </div>
  );
}

function SummaryBar({
  spec,
  xAgent,
  oAgent,
  onReset,
}: {
  spec: GameSpec;
  xAgent: AgentSpec;
  oAgent: AgentSpec;
  onReset: () => void;
}) {
  return (
    <div className="flex items-center gap-3 flex-wrap text-sm border border-neutral-200 bg-white px-3 py-2">
      <span className="flex items-center gap-2">
        <Mark player="X" />
        <span className="font-mono text-xs">{formatAgent(xAgent)}</span>
      </span>
      <span className="text-neutral-300">vs</span>
      <span className="flex items-center gap-2">
        <Mark player="O" />
        <span className="font-mono text-xs">{formatAgent(oAgent)}</span>
      </span>
      <span className="text-neutral-300">·</span>
      <span className="font-mono text-xs text-neutral-500">{formatSpec(spec)}</span>
      <div className="ml-auto">
        <SecondaryButton onClick={onReset} size="sm">reset</SecondaryButton>
      </div>
    </div>
  );
}

function describeTurn({
  phase,
  state,
  currentAgent,
  xAgent,
  oAgent,
}: {
  phase: Phase;
  state: ReturnType<typeof statesAlong>[number];
  currentAgent: AgentSpec | null;
  xAgent: AgentSpec;
  oAgent: AgentSpec;
}): React.ReactNode {
  if (state.isTerminal) return null;
  if (phase !== "playing" || !currentAgent) return null;

  const isHuman = currentAgent.kind === "human";
  const player: "X" | "O" = state.currentPlayer === 1 ? "X" : "O";
  const samePlayer = player === "X" ? xAgent : oAgent;

  if (isHuman) {
    return (
      <span className="flex items-center gap-2">
        <Mark player={player} /> your move ·{" "}
        <span className="text-neutral-500">click an empty cell</span>
      </span>
    );
  }
  return (
    <span className="flex items-center gap-2 text-neutral-500">
      <Mark player={player} />
      <span>{formatAgent(samePlayer)} thinking…</span>
    </span>
  );
}

