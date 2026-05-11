import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, errorMessage } from "../api";
import { EmptyState, PageHeader, SectionHeader } from "../components/Page";
import { Notice } from "../components/Notice";
import { GameTable } from "../components/Tables";
import type { GameRecord } from "../types";
import { SandboxPanel } from "./Sandbox";

/**
 * Home is the scratch space: spin up one game, watch it, save it if it's
 * interesting. Multi-player and multi-spec comparisons live on /compare;
 * position analysis lives on /analyze. This page deliberately stays narrow.
 */
export function HomePage() {
  return (
    <div>
      <PageHeader
        title="Sandbox"
        description="Set up a single game, watch any combination of players, save it if it's interesting. For multi-player matrices use Compare; for position analysis use Analyze."
      />
      <SandboxPanel />
      <div className="mt-12">
        <RecentMatches />
      </div>
    </div>
  );
}

function RecentMatches() {
  const [games, setGames] = useState<GameRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listGames({ limit: 8 })
      .then((r) => { if (!cancelled) setGames(r.games); })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => { cancelled = true; };
  }, []);

  return (
    <div>
      <SectionHeader
        title="Recent matches"
        actions={
          <Link to="/games" className="text-xs text-neutral-500 hover:text-neutral-900">
            all games →
          </Link>
        }
      />
      {error ? (
        <Notice kind="error">{error}</Notice>
      ) : games && games.length === 0 ? (
        <EmptyState>No matches yet. Play a game above and click Save.</EmptyState>
      ) : games ? (
        <GameTable games={games} />
      ) : null}
    </div>
  );
}
