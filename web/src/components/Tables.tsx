import { Link, useNavigate } from "react-router-dom";
import { formatAgent, formatSpec } from "../format";
import type { EvalRecord, GameRecord } from "../types";
import { Mark } from "./Mark";
import { TimeAgo } from "./TimeAgo";

/**
 * Tables where every row links somewhere use `LinkRow`, which makes the whole
 * row interactive: pointer cursor, hover background, keyboard navigable. The
 * row-level affordance was previously a lie — only the id cell was a Link,
 * but the entire row lit up on hover.
 *
 * We keep a real `<Link>` on the id text so users can still cmd/middle-click
 * to open in a new tab. Clicks on the rest of the row navigate via the
 * row's onClick.
 */
function LinkRow({
  to,
  highlighted,
  children,
}: {
  to: string;
  highlighted?: boolean;
  children: React.ReactNode;
}) {
  const nav = useNavigate();
  return (
    <tr
      onClick={() => nav(to)}
      onKeyDown={(e) => {
        if (e.key === "Enter") nav(to);
      }}
      role="link"
      tabIndex={0}
      className={`border-b border-neutral-100 last:border-0 cursor-pointer hover:bg-neutral-100 focus:bg-neutral-100 focus:outline-none transition-colors ${
        highlighted ? "bg-amber-50" : ""
      }`}
    >
      {children}
    </tr>
  );
}

export function GameTable({ games, highlightId }: { games: GameRecord[]; highlightId?: string | null }) {
  return (
    <div className="border border-neutral-200 bg-white">
      <table className="w-full text-sm">
        <thead className="text-[11px] uppercase text-neutral-500 bg-neutral-50/50 border-b border-neutral-200">
          <tr>
            <Th>id</Th>
            <Th>spec</Th>
            <Th>X</Th>
            <Th>O</Th>
            <Th>winner</Th>
            <Th align="right">when</Th>
          </tr>
        </thead>
        <tbody>
          {games.map((g) => (
            <LinkRow key={g.id} to={`/games/${g.id}`} highlighted={g.id === highlightId}>
              <Td>
                <Link to={`/games/${g.id}`} className="font-mono text-neutral-900 underline underline-offset-2 hover:text-black hover:decoration-2">
                  {g.id.slice(0, 8)}
                </Link>
              </Td>
              <Td className="font-mono text-neutral-600 text-xs">
                {formatSpec(g.spec)}
              </Td>
              <Td className="font-mono text-xs">{formatAgent(g.x_agent)}</Td>
              <Td className="font-mono text-xs">{formatAgent(g.o_agent)}</Td>
              <Td>
                {g.outcome === 1 ? (
                  <Mark player="X" />
                ) : g.outcome === -1 ? (
                  <Mark player="O" />
                ) : (
                  <span className="text-neutral-400">tie</span>
                )}
              </Td>
              <Td align="right" className="text-neutral-500 text-xs">
                <TimeAgo iso={g.created_at} />
              </Td>
            </LinkRow>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function EvalTable({ evals, highlightId }: { evals: EvalRecord[]; highlightId?: string | null }) {
  return (
    <div className="border border-neutral-200 bg-white">
      <table className="w-full text-sm">
        <thead className="text-[11px] uppercase text-neutral-500 bg-neutral-50/50 border-b border-neutral-200">
          <tr>
            <Th>id</Th>
            <Th>A</Th>
            <Th>B</Th>
            <Th align="right">games</Th>
            <Th align="right">A winrate</Th>
            <Th>status</Th>
            <Th align="right">when</Th>
          </tr>
        </thead>
        <tbody>
          {evals.map((e) => (
            <LinkRow key={e.id} to={`/evals/${e.id}`} highlighted={e.id === highlightId}>
              <Td>
                <Link to={`/evals/${e.id}`} className="font-mono text-neutral-900 underline underline-offset-2 hover:text-black hover:decoration-2">
                  {e.id.slice(0, 8)}
                </Link>
              </Td>
              <Td className="font-mono text-xs">{formatAgent(e.config.agent_a)}</Td>
              <Td className="font-mono text-xs">{formatAgent(e.config.agent_b)}</Td>
              <Td align="right" className="tabular-nums">{e.summary?.n_games ?? "-"}</Td>
              <Td align="right" className="tabular-nums font-mono">
                {e.summary ? e.summary.a_winrate.toFixed(3) : "-"}
              </Td>
              <Td className="text-neutral-600">{e.status}</Td>
              <Td align="right" className="text-neutral-500 text-xs">
                <TimeAgo iso={e.created_at} />
              </Td>
            </LinkRow>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Th({ children, align = "left" }: { children: React.ReactNode; align?: "left" | "right" }) {
  return <th className={`px-3 py-2 font-normal text-${align}`}>{children}</th>;
}

export function Td({
  children,
  className = "",
  align = "left",
}: {
  children: React.ReactNode;
  className?: string;
  align?: "left" | "right";
}) {
  return <td className={`px-3 py-2 text-${align} ${className}`}>{children}</td>;
}
