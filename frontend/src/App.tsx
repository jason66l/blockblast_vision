import { useCallback, useEffect, useRef, useState } from "react";
import { analyze, solve } from "./api";
import Board from "./components/Board";
import PieceEditor from "./components/PieceEditor";
import RegionSelector from "./components/RegionSelector";
import { emptyBoard, toggle } from "./matrix";
import type { Box, Matrix, Plan, Regions } from "./types";
import "./App.css";

interface Shot {
  file: File;
  url: string;
  width: number;
  height: number;
}

const SOLVE_DEBOUNCE_MS = 250;

// Mid-game position shown before anything is uploaded.
const DEMO_BOARD: Matrix = [
  "........",
  "........",
  "##......",
  "###...##",
  "####.###",
  "##.#####",
  ".#######",
  ".##.####",
].map((row) => [...row].map((ch) => (ch === "#" ? 1 : 0)));

export default function App() {
  const [shot, setShot] = useState<Shot | null>(null);
  const [regions, setRegions] = useState<Regions | null>(null);
  const [board, setBoard] = useState<Matrix>(DEMO_BOARD);
  const [pieces, setPieces] = useState<(Matrix | null)[]>([[[1, 1, 1]], [[1, 1], [1, 1]], [[1, 0], [1, 1]]]);
  const [msc, setMsc] = useState(0);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [busy, setBusy] = useState<"reading" | "solving" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [solveMs, setSolveMs] = useState<number | null>(null);
  // Bumped on every user edit. Plans from /api/analyze arrive already solved, so
  // only edits trigger a re-solve.
  const [editVersion, setEditVersion] = useState(1);
  const fileInput = useRef<HTMLInputElement>(null);

  const edit = <T,>(setter: (v: T) => void) => (v: T) => {
    setter(v);
    setEditVersion((n) => n + 1);
  };

  // ---------- screenshot flow ----------
  const read = useCallback(async (file: File, boxes?: { board: Box; tray: Box }) => {
    setBusy("reading");
    setError(null);
    const t0 = performance.now();
    try {
      const res = await analyze(file, msc, boxes);
      setShot((s) => (s && s.file === file ? { ...s, width: res.width, height: res.height } : s));
      setRegions(res.regions);
      setBoard(res.board);
      setPieces(res.pieces);
      setPlan(res.plan);
      setStepIndex(0);
      setSolveMs(performance.now() - t0);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, [msc]);

  const loadFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("That file isn't an image.");
      return;
    }
    setShot((prev) => {
      if (prev) URL.revokeObjectURL(prev.url);
      return { file, url: URL.createObjectURL(file), width: 0, height: 0 };
    });
    setRegions(null);
    read(file);
  }, [read]);

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
      const file = item?.getAsFile();
      if (file) loadFile(file);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [loadFile]);

  // ---------- re-solve after manual edits ----------
  useEffect(() => {
    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      setBusy("solving");
      setError(null);
      const t0 = performance.now();
      try {
        setPlan(await solve(board, pieces, msc, ctrl.signal));
        setStepIndex(0);
        setSolveMs(performance.now() - t0);
      } catch (e) {
        if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!ctrl.signal.aborted) setBusy(null);
      }
    }, SOLVE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editVersion]);

  // ---------- plan navigation ----------
  const steps = plan?.steps ?? [];
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") setStepIndex((i) => Math.min(i + 1, Math.max(0, steps.length - 1)));
      if (e.key === "ArrowLeft") setStepIndex((i) => Math.max(i - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [steps.length]);

  /** Commit the whole plan: the final board becomes the new board and the tray empties. */
  const applyPlan = () => {
    if (!steps.length) return;
    let streak = msc;
    for (const s of steps) streak = s.cleared_rows.length + s.cleared_cols.length ? 0 : Math.min(2, streak + 1);
    setBoard(steps[steps.length - 1].board_after);
    setPieces([null, null, null]);
    setMsc(streak);
    setPlan(null);
    setStepIndex(0);
  };

  const order = (slot: number) => {
    const i = steps.findIndex((s) => s.slot === slot);
    return i < 0 ? null : i + 1;
  };
  const lines = (s: { cleared_rows: number[]; cleared_cols: number[] }) => s.cleared_rows.length + s.cleared_cols.length;
  const totalCleared = steps.reduce((n, s) => n + lines(s), 0);
  const current = steps[stepIndex];

  return (
    <div className="app">
      <header>
        <h1>Block Blast Solver</h1>
        <p>Upload or paste a screenshot, or set up the board by hand. Click any cell to fix what the reader got wrong.</p>
      </header>

      <main>
        <section className="panel shot-panel">
          <div className="panel-head">
            <h2>Screenshot</h2>
            {shot && (
              <button className="ghost" onClick={() => fileInput.current?.click()}>
                Replace
              </button>
            )}
          </div>
          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => e.target.files?.[0] && loadFile(e.target.files[0])}
          />
          {shot && regions && shot.width > 0 ? (
            <>
              <RegionSelector
                src={shot.url}
                naturalWidth={shot.width}
                naturalHeight={shot.height}
                boxes={{ board: regions.board, tray: regions.tray }}
                onCommit={(b) => read(shot.file, b)}
              />
              <p className="hint">
                {regions.confident ? "Board found automatically." : "Couldn't find the board with confidence."} Drag a
                box to move it; the corner handle resizes it.
              </p>
            </>
          ) : (
            <div
              className={`drop ${busy === "reading" ? "loading" : ""}`}
              role="button"
              tabIndex={0}
              onClick={() => fileInput.current?.click()}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileInput.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files?.[0];
                if (f) loadFile(f);
              }}
            >
              {busy === "reading" ? (
                "Reading screenshot…"
              ) : (
                <>
                  <strong>Drop a screenshot here</strong>
                  <span>or click to choose one, or paste with ⌘V / Ctrl+V</span>
                </>
              )}
            </div>
          )}
        </section>

        <section className="panel solve-panel">
          <div className="panel-head">
            <h2>Board</h2>
            <div className="row">
              <label className="streak">
                Moves since last clear
                <select value={msc} onChange={(e) => edit(setMsc)(Number(e.target.value))}>
                  <option value={0}>0</option>
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                </select>
              </label>
              <button className="ghost" onClick={() => edit(setBoard)(emptyBoard())}>
                Clear board
              </button>
            </div>
          </div>

          <div className="board-wrap">
            <Board board={board} steps={steps} stepIndex={stepIndex} onToggle={(r, c) => edit(setBoard)(toggle(board, r, c))} />
            {busy === "solving" && <div className="spinner" aria-label="Solving" />}
          </div>

          <div className="stepper" aria-live="polite">
            {steps.length > 0 ? (
              <>
                <div className="steps">
                  {steps.map((s, i) => (
                    <button
                      key={i}
                      className={i === stepIndex ? "on" : ""}
                      style={{ "--slot": `var(--slot-${s.slot})` } as React.CSSProperties}
                      onClick={() => setStepIndex(i)}
                    >
                      Move {i + 1}
                    </button>
                  ))}
                </div>
                {current && (
                  <p className="step-desc">
                    Place <b>slot {current.slot + 1}</b> at row {current.top + 1}, column {current.left + 1}
                    {lines(current) > 0 && <> · clears {lines(current)} line{lines(current) > 1 ? "s" : ""}</>}
                  </p>
                )}
                {stepIndex > 0 && <p className="hint">Go back to move 1 to edit the board.</p>}
              </>
            ) : (
              <p className="step-desc">{pieces.some(Boolean) ? "No piece fits anywhere." : "Add pieces to get a plan."}</p>
            )}
          </div>

          <h2>Pieces</h2>
          <div className="pieces">
            {pieces.map((p, i) => (
              <PieceEditor
                key={i}
                slot={i}
                piece={p}
                active={current?.slot === i}
                order={order(i)}
                unplaceable={plan?.unplaced_slots.includes(i) ?? false}
                onChange={(np) => edit(setPieces)(pieces.map((q, j) => (j === i ? np : q)))}
              />
            ))}
          </div>

          {plan && (
            <div className="summary">
              <span>{plan.placed_all ? "All pieces fit" : "Not every piece fits; this plan places as many as possible"}</span>
              <span>
                {totalCleared} line{totalCleared === 1 ? "" : "s"} cleared
              </span>
              <span>
                {plan.nodes_searched.toLocaleString()} positions searched
                {solveMs !== null && ` · ${Math.round(solveMs)} ms`}
              </span>
              <button onClick={applyPlan} disabled={!steps.length}>
                Apply moves
              </button>
            </div>
          )}

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </section>
      </main>
    </div>
  );
}
