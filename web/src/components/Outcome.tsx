import { longestRun, type State } from "../game";
import { Mark } from "./Mark";

export function Outcome({ state, outcome }: { state: State; outcome: -1 | 0 | 1 }) {
  const x = longestRun(state.board, 1);
  const o = longestRun(state.board, 2);

  let headline: React.ReactNode;
  if (outcome === 0) {
    headline = <span className="text-neutral-700">Tie. The board is empty.</span>;
  } else {
    const reason = x.len !== o.len ? "by run length" : "by rightmost end (tied lengths)";
    headline = (
      <span className="flex items-center gap-2">
        <Mark player={outcome === 1 ? "X" : "O"} size="md" />
        <span className="text-neutral-900">wins, {reason}</span>
      </span>
    );
  }

  return (
    <div className="border border-neutral-200 bg-white px-4 py-3 space-y-2">
      <div className="text-sm">{headline}</div>
      <div className="font-mono text-xs text-neutral-600 flex gap-6 flex-wrap">
        <RunStat player="X" len={x.len} end={x.end} />
        <RunStat player="O" len={o.len} end={o.end} />
      </div>
    </div>
  );
}

function RunStat({ player, len, end }: { player: "X" | "O"; len: number; end: number }) {
  return (
    <span className="flex items-center gap-2">
      <Mark player={player} />
      <span>longest run {len}</span>
      {len > 0 && <span className="text-neutral-400">ending at cell {end}</span>}
    </span>
  );
}
