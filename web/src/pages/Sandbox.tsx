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
  Select,
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
import type { AgentSpec, GameSpec, PlayerRecord } from "../types";

type Phase = "config" | "playing" | "terminal";
type Seat =
  | { kind: "human" }
  | { kind: "player"; playerId: string }
  | { kind: "custom"; agent: AgentSpec };

const HUMAN_AGENT: AgentSpec = { kind: "human" };

function seatAgent(seat: Seat, players: PlayerRecord[] | null): AgentSpec {
  if (seat.kind === "human") return HUMAN_AGENT;
  if (seat.kind === "custom") return seat.agent;
  return players?.find((p) => p.id === seat.playerId)?.agent_spec ?? HUMAN_AGENT;
}

function seatLabel(seat: Seat, players: PlayerRecord[] | null): string {
  if (seat.kind === "human") return "Human";
  if (seat.kind === "custom") return formatAgent(seat.agent);
  return players?.find((p) => p.id === seat.playerId)?.label ?? "(missing player)";
}

export function SandboxPanel() {
  const nav = useNavigate();

  // Palindromic tempo arc, 12 marks on a 14-board: open slow, peak with
  // back-to-back 3-turns mid-game, close slow. Beyond solver reach but
  // playable in seconds.
  const [n, setN] = useState(14);
  const [scheduleText, setScheduleText] = useState("1,2,3,3,2,1");
  const [players, setPlayers] = useState<PlayerRecord[] | null>(null);
  const [xSeat, setXSeat] = useState<Seat>({ kind: "human" });
  const [oSeat, setOSeat] = useState<Seat>({ kind: "player", playerId: "p_alphabeta_4" });

  const [phase, setPhase] = useState<Phase>("config");
  const [actions, setActions] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [savedAs, setSavedAs] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [showOracle, setShowOracle] = useState(false);

  const schedule = useMemo(() => parseSchedule(scheduleText), [scheduleText]);
  const sum = schedule?.reduce((a, b) => a + b, 0) ?? 0;
  const configValid = schedule !== null && n > 0 && sum <= n;

  const spec: GameSpec | null = useMemo(
    () => (configValid ? { n, schedule: schedule! } : null),
    [configValid, n, schedule],
  );
  const xAgent = seatAgent(xSeat, players);
  const oAgent = seatAgent(oSeat, players);
  const xLabel = seatLabel(xSeat, players);
  const oLabel = seatLabel(oSeat, players);
  const states = useMemo(
    () => (spec && phase !== "config" ? statesAlong(spec, actions) : null),
    [spec, actions, phase],
  );
  const state = states?.[states.length - 1] ?? null;
  const displayPhase: Phase =
    phase === "playing" && state?.isTerminal ? "terminal" : phase;
  const currentAgent =
    state && !state.isTerminal ? (state.currentPlayer === 1 ? xAgent : oAgent) : null;
  const lastAction = actions.length > 0 ? actions[actions.length - 1] : null;
  const botMove = useMemo(
    () =>
      phase === "playing" &&
      spec &&
      state &&
      !state.isTerminal &&
      currentAgent &&
      currentAgent.kind !== "human"
        ? { spec, actions, agent: currentAgent }
        : null,
    [phase, spec, state, currentAgent, actions],
  );

  const oracle = useOracle(
    spec && phase !== "config" ? { spec, actions } : null,
    showOracle && phase !== "config",
  );
  const annotations =
    showOracle && oracle.kind === "ok" && !oracle.data.is_terminal
      ? oracleAnnotations(oracle.data)
      : undefined;

  useEffect(() => {
    let cancelled = false;
    api
      .listPlayers()
      .then((ps) => {
        if (!cancelled) setPlayers(ps);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // When the current player is a bot, ask the server for its move and append.
  useEffect(() => {
    if (!botMove) return;

    let cancelled = false;
    api
      .computeMove(botMove)
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
  }, [botMove]);

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

  const swapSeats = () => {
    setXSeat(oSeat);
    setOSeat(xSeat);
    setActions([]);
    setError(null);
    setSavedAs(null);
    setPhase("config");
  };

  const rematchSwapped = () => {
    setXSeat(oSeat);
    setOSeat(xSeat);
    setActions([]);
    setError(null);
    setSavedAs(null);
    setPhase("playing");
  };

  const save = async () => {
    if (!spec || !state?.isTerminal) return;
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
            <SeatField
              value={xSeat}
              onChange={setXSeat}
              players={players}
              allowHuman
            />
          </FormRow>
          <FormRow label="O second">
            <SeatField
              value={oSeat}
              onChange={setOSeat}
              players={players}
              allowHuman
            />
          </FormRow>
          <FormFooter>
            <span className="text-xs text-neutral-500">
              {xLabel} (X) vs {oLabel} (O)
            </span>
            <div className="flex items-center gap-2">
              <SecondaryButton onClick={swapSeats}>swap X/O</SecondaryButton>
              <PrimaryButton onClick={start} disabled={!configValid}>
                Start
              </PrimaryButton>
            </div>
          </FormFooter>
        </FormBox>
      </div>
    );
  }

  // displayPhase === "playing" || "terminal"
  if (!spec || !state) return null;
  const turnLabel = describeTurn({
    phase: displayPhase,
    state,
    currentAgent,
    xAgent,
    oAgent,
  });

  return (
    <div className="space-y-6">
      <SummaryBar
        spec={spec}
        xLabel={xLabel}
        oLabel={oLabel}
        xAgent={xAgent}
        oAgent={oAgent}
        onReset={reset}
        onSwap={swapSeats}
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

      {displayPhase === "terminal" && (
        <Outcome state={state} outcome={winner(state.board)} />
      )}

      {error && <Notice kind="error">{error}</Notice>}

      {displayPhase === "terminal" && (
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
          <SecondaryButton onClick={rematchSwapped}>rematch swapped</SecondaryButton>
        </div>
      )}
    </div>
  );
}

function SeatField({
  value,
  onChange,
  players,
  allowHuman,
}: {
  value: Seat;
  onChange: (seat: Seat) => void;
  players: PlayerRecord[] | null;
  allowHuman: boolean;
}) {
  const selectValue =
    value.kind === "human"
      ? "__human"
      : value.kind === "custom"
        ? "__custom"
        : value.playerId;
  const options = [
    ...(allowHuman ? [{ value: "__human", label: "Human" }] : []),
    ...(players ?? []).map((p) => ({ value: p.id, label: p.label })),
    { value: "__custom", label: "Custom agent..." },
  ];
  return (
    <>
      <Select
        value={selectValue}
        options={options}
        width="w-56"
        onChange={(next) => {
          if (next === "__human") onChange({ kind: "human" });
          else if (next === "__custom") {
            onChange({ kind: "custom", agent: { kind: "greedy", seed: 0 } });
          }
          else onChange({ kind: "player", playerId: next });
        }}
      />
      {value.kind === "custom" && (
        <AgentField
          value={value.agent}
          onChange={(agent) => onChange({ kind: "custom", agent })}
          allowHuman={allowHuman}
        />
      )}
      {players === null && (
        <span className="text-xs text-neutral-400">loading players</span>
      )}
    </>
  );
}

function SummaryBar({
  spec,
  xLabel,
  oLabel,
  xAgent,
  oAgent,
  onReset,
  onSwap,
}: {
  spec: GameSpec;
  xLabel: string;
  oLabel: string;
  xAgent: AgentSpec;
  oAgent: AgentSpec;
  onReset: () => void;
  onSwap: () => void;
}) {
  return (
    <div className="flex items-center gap-3 flex-wrap text-sm border border-neutral-200 bg-white px-3 py-2">
      <span className="flex items-center gap-2">
        <Mark player="X" />
        <span className="text-xs text-neutral-900">{xLabel}</span>
        <span className="font-mono text-xs text-neutral-500">{formatAgent(xAgent)}</span>
      </span>
      <span className="text-neutral-300">vs</span>
      <span className="flex items-center gap-2">
        <Mark player="O" />
        <span className="text-xs text-neutral-900">{oLabel}</span>
        <span className="font-mono text-xs text-neutral-500">{formatAgent(oAgent)}</span>
      </span>
      <span className="text-neutral-300">·</span>
      <span className="font-mono text-xs text-neutral-500">{formatSpec(spec)}</span>
      <div className="ml-auto flex items-center gap-2">
        <SecondaryButton onClick={onSwap} size="sm">swap X/O</SecondaryButton>
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
