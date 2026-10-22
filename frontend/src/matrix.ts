import type { Matrix } from "./types";

export const PIECE_GRID = 5;
export const SLOT_COLORS = ["var(--slot-0)", "var(--slot-1)", "var(--slot-2)"];

export const emptyBoard = (): Matrix => Array.from({ length: 8 }, () => Array(8).fill(0));

/** Drop empty outer rows/columns; returns null for an all-empty piece. */
export function trimMatrix(m: Matrix): Matrix | null {
  const rows = m.map((r, i) => (r.some(Boolean) ? i : -1)).filter((i) => i >= 0);
  const cols = m[0].map((_, c) => (m.some((r) => r[c]) ? c : -1)).filter((c) => c >= 0);
  if (!rows.length || !cols.length) return null;
  return m.slice(rows[0], rows[rows.length - 1] + 1).map((r) => r.slice(cols[0], cols[cols.length - 1] + 1));
}

/** Pad a trimmed piece into the fixed 5x5 editing grid (top-left aligned). */
export function toPieceGrid(p: Matrix | null): Matrix {
  const g = Array.from({ length: PIECE_GRID }, () => Array(PIECE_GRID).fill(0));
  p?.forEach((row, r) => row.forEach((v, c) => (g[r][c] = v)));
  return g;
}

export const toggle = (m: Matrix, r: number, c: number): Matrix =>
  m.map((row, i) => (i === r ? row.map((v, j) => (j === c ? 1 - v : v)) : row));

export const PRESETS: Matrix[] = [
  [[1]],
  [[1, 1]],
  [[1, 1, 1]],
  [[1, 1, 1, 1]],
  [[1, 1, 1, 1, 1]],
  [[1, 1], [1, 1]],
  [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
  [[1, 0], [1, 1]],
  [[1, 0, 0], [1, 0, 0], [1, 1, 1]],
  [[1, 1, 1], [0, 1, 0]],
  [[1, 1, 0], [0, 1, 1]],
];

export const rotate = (m: Matrix): Matrix => m[0].map((_, c) => m.map((row) => row[c]).reverse());
