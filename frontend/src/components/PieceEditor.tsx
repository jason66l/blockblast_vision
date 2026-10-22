import type { Matrix } from "../types";
import { PRESETS, SLOT_COLORS, rotate, toPieceGrid, toggle, trimMatrix } from "../matrix";

interface Props {
  slot: number;
  piece: Matrix | null;
  active: boolean; // highlighted when the previewed step uses this piece
  order: number | null; // 1-based position in the plan
  unplaceable: boolean;
  onChange: (p: Matrix | null) => void;
}

export default function PieceEditor({ slot, piece, active, order, unplaceable, onChange }: Props) {
  const grid = toPieceGrid(piece);
  const color = SLOT_COLORS[slot];
  const cycle = () => {
    const i = PRESETS.findIndex((p) => JSON.stringify(p) === JSON.stringify(piece));
    onChange(PRESETS[(i + 1) % PRESETS.length]);
  };

  return (
    <div className={`piece ${active ? "active" : ""}`} style={{ "--slot": color } as React.CSSProperties}>
      <div className="piece-head">
        <span>Slot {slot + 1}</span>
        {order !== null && <span className="badge">Move {order}</span>}
        {unplaceable && <span className="badge warn">No room</span>}
      </div>
      <div className="piece-grid">
        {grid.map((row, r) =>
          row.map((v, c) => (
            <button
              key={`${r}-${c}`}
              className={`pcell ${v ? "on" : ""}`}
              onClick={() => onChange(trimMatrix(toggle(grid, r, c)))}
              aria-label={`Slot ${slot + 1} cell ${r + 1},${c + 1}`}
            />
          )),
        )}
      </div>
      <div className="piece-actions">
        <button onClick={cycle} title="Cycle common shapes">Shape</button>
        <button onClick={() => piece && onChange(rotate(piece))} disabled={!piece} title="Rotate 90°">Rotate</button>
        <button onClick={() => onChange(null)} disabled={!piece}>Clear</button>
      </div>
    </div>
  );
}
