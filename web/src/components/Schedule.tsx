type Props = {
  schedule: number[];
  turnIdx: number;
  placementsLeft: number;
};

const X_COLOR = "text-[#1d4ed8]";
const O_COLOR = "text-[#b45309]";

export function Schedule({ schedule, turnIdx, placementsLeft }: Props) {
  return (
    <div className="font-mono text-xs flex flex-wrap items-center gap-x-4 gap-y-1">
      <span className="text-neutral-400 select-none">schedule</span>
      {schedule.map((k, i) => {
        const isX = i % 2 === 0;
        const done = i < turnIdx;
        const active = i === turnIdx;
        const playerCls = done ? "text-neutral-300" : isX ? X_COLOR : O_COLOR;
        const numCls = done
          ? "text-neutral-300"
          : active
            ? "text-neutral-900 font-semibold"
            : "text-neutral-700";
        return (
          <span key={i} className="tabular-nums">
            <span className={playerCls}>{isX ? "X" : "O"}</span>{" "}
            <span className={numCls}>{active ? `${placementsLeft}/${k}` : k}</span>
          </span>
        );
      })}
    </div>
  );
}
