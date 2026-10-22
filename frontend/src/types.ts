export type Matrix = number[][];

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Regions {
  board: Box;
  tray: Box;
  confident: boolean;
}

export interface Step {
  slot: number;
  top: number;
  left: number;
  cells: [number, number][];
  cleared_rows: number[];
  cleared_cols: number[];
  board_after: Matrix;
}

export interface Plan {
  steps: Step[];
  score: number;
  placed_all: boolean;
  unplaced_slots: number[];
  nodes_searched: number;
}

export interface AnalyzeResponse {
  width: number;
  height: number;
  regions: Regions;
  board: Matrix;
  pieces: (Matrix | null)[];
  plan: Plan;
}
