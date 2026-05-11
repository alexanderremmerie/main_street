import { useEffect, useState } from "react";
import { api, ApiError, errorMessage } from "./api";
import type { GameSpec, OracleResponse } from "./types";

export type OracleQuery = { spec: GameSpec; actions: number[] };

/** State of an oracle query. `tooLarge` is set when the server returns 413,
 *  which we treat as "this config is not solver-eligible" rather than an
 *  error, since it's an expected condition on big boards. */
export type OracleResult =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; data: OracleResponse }
  | { kind: "too-large" }
  | { kind: "error"; message: string };

/** Run the exact solver against the current `(spec, actions)`. Returns
 *  whatever state the latest request is in. Requests for stale inputs are
 *  ignored. */
export function useOracle(query: OracleQuery | null, enabled: boolean): OracleResult {
  const [result, setResult] = useState<OracleResult>({ kind: "idle" });

  // Encode the query into a primitive key so the effect doesn't re-fire on
  // identity changes. `actions` is the only mutating list.
  const key = query ? JSON.stringify(query) : null;

  useEffect(() => {
    if (!enabled || !query || key === null) {
      setResult({ kind: "idle" });
      return;
    }
    let cancelled = false;
    setResult({ kind: "loading" });
    api
      .oracle(query)
      .then((data) => {
        if (!cancelled) setResult({ kind: "ok", data });
      })
      .catch((e) => {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 413) {
          setResult({ kind: "too-large" });
        } else {
          setResult({
            kind: "error",
            message: errorMessage(e),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [key, enabled]); // eslint-disable-line react-hooks/exhaustive-deps
  return result;
}
