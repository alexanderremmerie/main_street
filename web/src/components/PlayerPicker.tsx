import { formatAgent } from "../format";
import type { PlayerRecord } from "../types";

/**
 * Chip-style multi-select for players. Used by Analyze (which players to ask
 * about a position) and Compare (which players are in the matrix). Keeps
 * styling identical across both so the two pages feel like the same workflow.
 */
export function PlayerPicker({
  players,
  selectedIds,
  onToggle,
}: {
  players: PlayerRecord[] | null;
  selectedIds: Set<string>;
  onToggle: (id: string) => void;
}) {
  if (players === null) {
    return <span className="text-sm text-neutral-500">Loading players…</span>;
  }
  if (players.length === 0) {
    return <span className="text-sm text-neutral-500">No players available.</span>;
  }
  return (
    <div className="flex flex-wrap gap-2 w-full">
      {players.map((p) => {
        const on = selectedIds.has(p.id);
        return (
          <button
            key={p.id}
            onClick={() => onToggle(p.id)}
            title={formatAgent(p.agent_spec)}
            className={`text-xs font-medium h-8 px-3 border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-900 focus-visible:ring-offset-2 ${
              on
                ? "bg-neutral-900 text-white border-neutral-900 hover:bg-neutral-800"
                : "bg-white text-neutral-700 border-neutral-300 hover:bg-neutral-50 hover:border-neutral-500 hover:text-neutral-900"
            }`}
          >
            {p.label}
          </button>
        );
      })}
    </div>
  );
}
