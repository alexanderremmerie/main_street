import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, errorMessage } from "../api";
import { Notice } from "../components/Notice";
import { SecondaryButton, Select } from "../components/Form";
import { EmptyState, LoadingState, PageHeader } from "../components/Page";
import { GameTable } from "../components/Tables";
import type { AgentKind, GameRecord } from "../types";

const PAGE_SIZE = 50;

type KindFilter = "any" | AgentKind;

const KIND_OPTIONS: { value: KindFilter; label: string }[] = [
  { value: "any", label: "any" },
  { value: "human", label: "human" },
  { value: "random", label: "random" },
  { value: "greedy", label: "greedy" },
  { value: "rightmost", label: "rightmost" },
  { value: "extension", label: "extension" },
  { value: "blocker", label: "blocker" },
  { value: "center", label: "center" },
  { value: "forkaware", label: "forkaware" },
  { value: "potentialaware", label: "potentialaware" },
  { value: "mcts", label: "mcts" },
  { value: "alphabeta", label: "alphabeta" },
];

export function GamesPage() {
  const [games, setGames] = useState<GameRecord[] | null>(null);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const [xKind, setXKind] = useState<KindFilter>("any");
  const [oKind, setOKind] = useState<KindFilter>("any");
  const [offset, setOffset] = useState(0);

  // Reset offset when filters change so we never land on an empty page.
  useEffect(() => {
    setOffset(0);
  }, [xKind, oKind]);

  useEffect(() => {
    let cancelled = false;
    setGames(null);
    setError(null);
    api
      .listGames({
        limit: PAGE_SIZE,
        offset,
        x_kind: xKind === "any" ? undefined : xKind,
        o_kind: oKind === "any" ? undefined : oKind,
      })
      .then((r) => {
        if (cancelled) return;
        setGames(r.games);
        setTotal(r.total);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, [xKind, oKind, offset]);

  const pageStart = total === 0 ? 0 : offset + 1;
  const pageEnd = Math.min(offset + (games?.length ?? 0), total);
  const canPrev = offset > 0;
  const canNext = offset + PAGE_SIZE < total;

  return (
    <div>
      <PageHeader
        title="Games"
        description="Every saved match. Filter by who played which side; click a row to scrub through the moves and inspect any position."
        actions={
          games && total > 0 ? (
            <span className="text-xs text-neutral-500 tabular-nums">
              {pageStart}–{pageEnd} of {total}
            </span>
          ) : null
        }
      />

      <div className="flex items-center gap-3 flex-wrap mb-4 text-sm">
        <span className="text-neutral-500">filter</span>
        <span className="flex items-center gap-1.5">
          <span className="text-xs text-neutral-500">X</span>
          <Select value={xKind} options={KIND_OPTIONS} onChange={setXKind} width="w-32" />
        </span>
        <span className="flex items-center gap-1.5">
          <span className="text-xs text-neutral-500">O</span>
          <Select value={oKind} options={KIND_OPTIONS} onChange={setOKind} width="w-32" />
        </span>
        {(xKind !== "any" || oKind !== "any") && (
          <SecondaryButton
            onClick={() => {
              setXKind("any");
              setOKind("any");
            }}
          >
            clear filters
          </SecondaryButton>
        )}
      </div>

      {error && <Notice kind="error">{error}</Notice>}

      {!games && !error && <LoadingState>Loading games…</LoadingState>}

      {games && games.length === 0 && !error ? (
        <EmptyState>
          {total === 0 ? (
            <>
              No games yet.{" "}
              <Link to="/" className="text-neutral-900 underline underline-offset-2 hover:text-black hover:decoration-2">
                Play one
              </Link>
              .
            </>
          ) : (
            "No games match those filters."
          )}
        </EmptyState>
      ) : games ? (
        <>
          <GameTable games={games} />
          <div className="flex items-center justify-end gap-2 mt-4">
            <SecondaryButton
              onClick={() => setOffset(Math.max(offset - PAGE_SIZE, 0))}
              disabled={!canPrev}
            >
              ‹ prev
            </SecondaryButton>
            <SecondaryButton
              onClick={() => setOffset(offset + PAGE_SIZE)}
              disabled={!canNext}
            >
              next ›
            </SecondaryButton>
          </div>
        </>
      ) : null}
    </div>
  );
}
