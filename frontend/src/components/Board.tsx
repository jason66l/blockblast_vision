import type { Matrix, Step } from "../types";
import { SLOT_COLORS } from "../matrix";

interface Props {
  board: Matrix;
  steps: Step[];
  stepIndex: number; // which step's placement to preview
  onToggle: (r: number, c: number) => void;
}

export default function Board({ board, steps, stepIndex, onToggle }: Props) {
  const step = steps[stepIndex];
  const before = stepIndex === 0 ? board : steps[stepIndex - 1].board_after;
  const placed = new Set(step?.cells.map(([r, c]) => r * 8 + c));
  const clearing = (r: number, c: number) => step && (step.cleared_rows.includes(r) || step.cleared_cols.includes(c));
  const editable = stepIndex === 0;

  return (
    <div className="board" role="grid" aria-label="Game board">
      {before.map((row, r) =>
        row.map((v, c) => {
          const isPlaced = placed.has(r * 8 + c);
          const cls = ["cell", v ? "filled" : "", isPlaced ? "placed" : "", clearing(r, c) ? "clearing" : ""].join(" ");
          return (
            <button
              key={`${r}-${c}`}
              className={cls}
              style={isPlaced ? { background: SLOT_COLORS[step.slot] } : undefined}
              onClick={() => editable && onToggle(r, c)}
              disabled={!editable}
              aria-label={`Row ${r + 1}, column ${c + 1}: ${v ? "filled" : "empty"}`}
            />
          );
        }),
      )}
    </div>
  );
}
