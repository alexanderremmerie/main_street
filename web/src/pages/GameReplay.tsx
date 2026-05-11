import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, errorMessage } from "../api";
import { Board } from "../components/Board";
import { Checkbox, SecondaryButton } from "../components/Form";
import { Mark } from "../components/Mark";
import { Notice } from "../components/Notice";
import { OraclePanel, oracleAnnotations } from "../components/Oracle";
import { Outcome } from "../components/Outcome";
import { LoadingState, PageHeader } from "../components/Page";
import { Schedule } from "../components/Schedule";
import { formatAgent, formatSpec } from "../format";
import { statesAlong } from "../game";
import { useOracle } from "../oracle";
import type { GameRecord } from "../types";

export function GameReplayPage() {
  const { id } = useParams<{ id: string }>();
  const [game, setGame] = useState<GameRecord | null>(null);
  const [step, setStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [showOracle, setShowOracle] = useState(false);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setGame(null);
    setError(null);
    api
      .getGame(id)
      .then((g) => {
        if (cancelled) return;
        setGame(g);
        setStep(g.actions.length);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => { cancelled = true; };
  }, [id]);

  useEffect(() => {
    if (!game) return;
    const total = game.actions.length;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName ?? "";
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        setStep((s) => Math.min(s + 1, total));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setStep((s) => Math.max(s - 1, 0));
      } else if (e.key === "Home") {
        e.preventDefault();
        setStep(0);
      } else if (e.key === "End") {
        e.preventDefault();
        setStep(total);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [game]);

  const oracle = useOracle(
    game ? { spec: game.spec, actions: game.actions.slice(0, step) } : null,
    showOracle && game !== null,
  );
  const states = useMemo(
    () => (game ? statesAlong(game.spec, game.actions) : null),
    [game],
  );

  if (error) return <Notice kind="error">{error}</Notice>;
  if (!game || !states) return <LoadingState>Loading game…</LoadingState>;

  const state = states[step];
  const lastAction = step > 0 ? game.actions[step - 1] : null;
  const total = game.actions.length;

  const annotations =
    showOracle && oracle.kind === "ok" && !oracle.data.is_terminal
      ? oracleAnnotations(oracle.data)
      : undefined;

  return (
    <div className="space-y-8">
      <PageHeader
        title="Game"
        description={<span className="font-mono text-xs">{game.id}</span>}
        actions={
          <Link to="/games" className="text-xs text-neutral-500 hover:text-neutral-900">
            ← all games
          </Link>
        }
      />

      <div className="flex items-center gap-3 flex-wrap text-sm">
        <span className="flex items-center gap-2">
          <Mark player="X" />
          <span className="font-mono text-xs">{formatAgent(game.x_agent)}</span>
        </span>
        <span className="text-neutral-300">vs</span>
        <span className="flex items-center gap-2">
          <Mark player="O" />
          <span className="font-mono text-xs">{formatAgent(game.o_agent)}</span>
        </span>
        <span className="ml-auto font-mono text-xs text-neutral-500">
          {formatSpec(game.spec)}
        </span>
      </div>

      <Schedule
        schedule={game.spec.schedule}
        turnIdx={state.turnIdx}
        placementsLeft={state.placementsLeft}
      />

      <Board
        state={state}
        highlightWinner={state.isTerminal}
        lastAction={lastAction}
        annotations={annotations}
      />

      <div className="flex items-center justify-between gap-3 flex-wrap">
        <Checkbox checked={showOracle} onChange={setShowOracle} label="show oracle" />
        {showOracle && (
          <div className="flex-1 min-w-0 sm:min-w-[20rem]">
            <OraclePanel result={oracle} />
          </div>
        )}
      </div>

      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <SecondaryButton
            onClick={() => setStep(0)}
            disabled={step === 0}
            title="first move (Home)"
          >
            « start
          </SecondaryButton>
          <SecondaryButton
            onClick={() => setStep((s) => Math.max(s - 1, 0))}
            disabled={step === 0}
            title="previous (←)"
          >
            ‹ prev
          </SecondaryButton>
          <input
            type="range"
            className="flex-1 min-w-0 accent-neutral-900"
            min={0}
            max={total}
            value={step}
            onChange={(e) => setStep(Number(e.target.value))}
          />
          <SecondaryButton
            onClick={() => setStep((s) => Math.min(s + 1, total))}
            disabled={step === total}
            title="next (→)"
          >
            next ›
          </SecondaryButton>
          <SecondaryButton
            onClick={() => setStep(total)}
            disabled={step === total}
            title="last move (End)"
          >
            end »
          </SecondaryButton>
          <span className="font-mono text-xs text-neutral-500 whitespace-nowrap tabular-nums ml-1">
            step {step}/{total}
            {lastAction !== null ? ` · cell ${lastAction}` : ""}
          </span>
        </div>
        <div className="flex items-center justify-between gap-3">
          <p className="text-xs text-neutral-400">←/→ Home/End to navigate</p>
          {!state.isTerminal && game && (
            <Link
              to={`/analyze?game=${game.id}&step=${step}`}
              className="h-7 px-2.5 text-xs font-medium text-neutral-900 bg-neutral-100 border border-neutral-400 inline-flex items-center hover:bg-neutral-200 hover:border-neutral-700 transition-colors"
            >
              analyze this position →
            </Link>
          )}
        </div>
      </div>

      {state.isTerminal && <Outcome state={state} outcome={game.outcome} />}
    </div>
  );
}

