import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, errorMessage } from "../api";
import { Notice } from "../components/Notice";
import { EmptyState, LoadingState, PageHeader } from "../components/Page";
import { EvalTable } from "../components/Tables";
import type { EvalRecord } from "../types";

export function EvalsPage() {
  const [evals, setEvals] = useState<EvalRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listEvals()
      .then((r) => { if (!cancelled) setEvals(r); })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => { cancelled = true; };
  }, []);

  return (
    <div>
      <PageHeader
        title="Evals"
        description="Pairwise tournaments. Each row is one head-to-head played across one or more specs. Comparisons create these in batches."
      />
      {error && <Notice kind="error">{error}</Notice>}
      {!evals && !error && <LoadingState>Loading evals…</LoadingState>}
      {evals && evals.length === 0 && !error ? (
        <EmptyState>
          No evals yet.{" "}
          <Link to="/compare" className="text-neutral-900 underline underline-offset-2 hover:text-black hover:decoration-2">
            Run a comparison
          </Link>
          .
        </EmptyState>
      ) : evals ? (
        <EvalTable evals={evals} />
      ) : null}
    </div>
  );
}
