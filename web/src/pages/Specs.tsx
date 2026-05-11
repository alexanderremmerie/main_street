/**
 * Index of game specs (N, schedule) that have appeared in saved games.
 *
 * Specs aren't a stored entity — they're aggregated from the games table, so
 * this page is always live with whatever the database actually contains.
 *
 * The card grid optimizes for scannability: at a glance you should be able
 * to spot lopsided configs (X dominates / O dominates) by the colored bar
 * and pick one to dig into.
 */

import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, errorMessage } from "../api";
import { Notice } from "../components/Notice";
import { EmptyState, LoadingState, PageHeader } from "../components/Page";
import { ProportionBar } from "../components/Stat";
import type { SpecSummary } from "../types";

export function SpecsPage() {
  const nav = useNavigate();
  const [specs, setSpecs] = useState<SpecSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listSpecs()
      .then((s) => {
        if (!cancelled) setSpecs(s);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const analyzeUrl = (s: SpecSummary) => {
    const params = new URLSearchParams();
    params.set("n", String(s.spec.n));
    params.set("schedule", s.spec.schedule.join(","));
    return `/analyze?${params}`;
  };

  return (
    <div>
      <PageHeader
        title="Specs"
        description="Every game configuration that's appeared in a saved match. Click a card to analyze the initial position with the current set of players."
      />

      {error && <Notice kind="error">{error}</Notice>}

      {!specs && !error && <LoadingState>Loading specs…</LoadingState>}

      {specs && specs.length === 0 && !error && (
        <EmptyState>
          No games yet.{" "}
          <Link to="/" className="text-neutral-900 underline underline-offset-2 hover:text-black hover:decoration-2">
            Play one
          </Link>{" "}
          to populate this list.
        </EmptyState>
      )}

      {specs && specs.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 lg:gap-5">
          {specs.map((s, i) => (
            <SpecCard
              key={i}
              spec={s}
              onClick={() => nav(analyzeUrl(s))}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function SpecCard({ spec: s, onClick }: { spec: SpecSummary; onClick: () => void }) {
  const decided = s.x_wins + s.o_wins + s.ties;
  const winrate = decided > 0 ? (s.x_wins + 0.5 * s.ties) / decided : 0.5;
  // A spec is "lopsided" if one side wins ≥75% of decided games on a non-
  // trivial sample. Used purely as a UI cue.
  const lopsided = decided >= 4 && (winrate >= 0.75 || winrate <= 0.25);
  return (
    <button
      onClick={onClick}
      className="text-left bg-white border border-neutral-200 hover:border-neutral-900 transition-colors p-5 space-y-4 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900 focus-visible:ring-offset-2"
    >
      <div className="flex items-baseline justify-between gap-3">
        <div className="font-mono text-base text-neutral-900">
          N={s.spec.n} <span className="text-neutral-500">[{s.spec.schedule.join(",")}]</span>
        </div>
        <div className="text-xs text-neutral-500 tabular-nums shrink-0">
          {s.n_games} {s.n_games === 1 ? "game" : "games"}
        </div>
      </div>

      <ProportionBar
        value={winrate}
        leftLabel={`X ${s.x_wins}`}
        rightLabel={`${s.o_wins} O`}
      />

      <div className="flex items-center justify-between text-xs">
        <span className="text-neutral-500">
          X winrate{" "}
          <span className={`font-mono tabular-nums ${lopsided ? "text-neutral-900 font-medium" : "text-neutral-700"}`}>
            {winrate.toFixed(3)}
          </span>
          {s.ties > 0 && (
            <span className="text-neutral-400"> · {s.ties} {s.ties === 1 ? "tie" : "ties"}</span>
          )}
        </span>
        <span className="text-neutral-400">analyze →</span>
      </div>
    </button>
  );
}
