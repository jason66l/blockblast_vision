# Block Blast Solver

**Live demo:** [blockblast-solver.onrender.com](https://blockblast-solver.onrender.com/) (hosted on a free tier, so the first load may take ~30s to wake up)

Upload (or paste) a Block Blast screenshot and get the best way to place all three pieces, shown move by move. Anything the image reader gets wrong can be fixed by clicking cells, and the plan re-solves instantly. You can also skip the screenshot and set up a board by hand.

**Stack:** React + TypeScript (Vite) front end · FastAPI back end · OpenCV + NumPy for screenshot reading · bitboard search in pure Python.

## How it works

**Reading the screenshot** (`backend/vision.py`)
1. **Find the board.** The row and column brightness profiles are high-passed so thin grid lines and the board border stand out. The detector then searches for one step size where 9 evenly spaced lines line up on both axes. It scores with the median line strength, so decorative edges and the score text can't anchor a false grid, and requires both outer edges so a window shifted by one cell loses. The pieces tray is assumed to sit just below the board. Both boxes can be dragged in the UI.
2. **Read the board.** Each of the 64 cells is classified from the brightness of its center region (HSV value channel).
3. **Read the pieces.** Each tray slot is segmented with Otsu thresholding on saturation, cleaned with morphology, reduced to its largest component, and snapped to a grid. The tray cell size is derived from the board cell size, so any screenshot resolution works.

**Solving** (`backend/solver.py`)
- The board is a 64-bit integer. Legal placements for a piece are found for all 64 anchors at once by shifting the empty-cell mask once per piece cell and ANDing the results, and line clears are 16 mask compares.
- The search covers **every order and every placement of all three pieces**, with memoization on (board, remaining pieces, combo streak). Identical pieces are deduplicated.
- Final positions are scored on lines cleared (with a multi-line bonus), empty cells, sealed-off holes (bitwise flood fill from the border), fragmentation into separate empty regions, and how many ways hard pieces (3×3, 1×5, 5×1, …) could still fit. A plan that leaves a piece unplaced is game over and ranks below any plan that places everything.
- "Moves since last clear" models the combo streak: going three moves without a clear is penalized.
- Typical positions solve in under 50 ms. The worst case (a near-empty board with three tiny pieces, about 180k positions) takes about 1.5 s.

## Running locally

```bash
# Back end
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --reload --port 8000

# Front end (separate terminal); proxies /api to :8000
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

Production: `npm run build` in `frontend/`, then the FastAPI app serves `frontend/dist` itself at http://localhost:8000. Or use Docker:

```bash
docker build -t blockblast-solver .
docker run -p 8000:8000 blockblast-solver
```

## API

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/analyze` | multipart: `image`, optional `board_box` / `tray_box` (JSON `{x,y,w,h}` in image pixels), `moves_since_clear` | detected regions, 8×8 board, 3 pieces, plan |
| `POST /api/solve` | JSON: `board` (8×8 of 0/1), `pieces` (up to 3 matrices or `null`), `moves_since_clear` | plan: steps with cells, cleared rows/cols, and the board after each move |

## Tests

```bash
cd backend && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pytest -q
```

Covers bitboard legality checks against brute force, line clears, holes and regions, move ordering, unplaceable pieces, the combo rule, board consistency across steps, and screenshot reading on frames from a real game recording.

## v1: real-time desktop solver

The original version (`v1/v1blockblast.py`) mirrored my iPhone screen onto my computer and read the board and pieces straight from the live screen feed at 20 fps with OpenCV. It suggested one move at a time, using a placement search adapted and improved from open-source research, with survival and scoring heuristics and a 2-ply lookahead.

https://github.com/user-attachments/assets/08b00fed-a4a2-41ce-b8e6-76b652ea1f75

v2 turns it into a web app, makes detection independent of screen position and resolution, and replaces the one-move search with a full search over all three pieces.
