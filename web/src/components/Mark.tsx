type Props = { player: "X" | "O"; size?: "sm" | "md" };

export function Mark({ player, size = "sm" }: Props) {
  const dim = size === "sm" ? "w-4 h-4 text-[10px]" : "w-5 h-5 text-xs";
  const color = player === "X" ? "bg-[#1d4ed8] text-white" : "bg-[#b45309] text-white";
  return (
    <span className={`inline-flex items-center justify-center font-mono font-semibold shrink-0 ${dim} ${color}`}>
      {player}
    </span>
  );
}
