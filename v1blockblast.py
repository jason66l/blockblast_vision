import numpy as np
import cv2
import mss
import time
import math
from collections import deque

# ============================================================
# WORKING SCRIPT (single file) ✅
# - Bank piece detection: HSV Saturation+Value + sanity checks
# - Shows NEXT_MOVE overlay (one move at a time)
# - Shows BANK_SLOT_0/1/2 windows with detected piece matrix + drawing
# - Better move scoring: clears + emptiness + holes + fragmentation + mobility + 2-ply lookahead
# - Combo rule: never go 3 moves without clearing
#   - If tsc==2 and no clearing move exists, falls back to best legal move and marks COMBO BREAK
# - Debounced stable reads to reduce drag/hover warping
# ============================================================

# ============================================================
# 0) INPUT: YOUR SCREEN CORNERS
# ============================================================
BOARD_TL = (1130.265625, 257.984375)
BOARD_TR = (1488.3984375, 257.94140625)
BOARD_BL = (1130.734375, 617.99609375)
BOARD_BR = (1490.06640625, 618.828125)

BANK_TL = (1142, 630)
BANK_TR = (1481, 630)
BANK_BL = (1138, 755)
BANK_BR = (1483, 755)

# ============================================================
# 1) SETTINGS
# ============================================================
# --- Board occupancy extraction ---
N = 8
V_THRESH = 165
CELL_INNER_PAD_FRAC = 0.30
SCAN_FPS = 20

# --- Bank / piece extraction ---
BLUR_K = 3
N_SLOTS = 3
SLOT_PAD = 20
BBOX_PAD = 2

CELL = 21
FILL_THR = 0.40

# --- Bank segmentation thresholds (HSV) ---
S_MIN = 50
V_MIN = 40
USE_OTSU_ON_S = True  # Otsu on saturation usually works best

# --- Sanity checks for detected pieces ---
MAX_DIM = 5
MAX_BLOCKS = 9  # set to 6 if your game has 6-block pieces
MIN_MASK_AREA = 350  # reject tiny blobs (phantom 1x1)

# --- Overlay render settings ---
CELL_PX = 52
GRID_THICK = 2
GRID_COLOR = (70, 70, 70)
BG_COLOR   = (15, 15, 15)
OCC_COLOR  = (255, 255, 255)
PLAN_COLOR = (0, 0, 255)
ALPHA_PLAN = 0.65

# --- Debounce (drag/hover protection) ---
BOARD_STABLE_FRAMES = 1
BANK_STABLE_FRAMES  = 1
STABLE_TIMEOUT_SEC  = 0.25

# --- Debug windows: detected bank piece matrices ---
SHOW_BANK_DETECTED_WINDOWS = True
BANK_WIN_X0 = 60
BANK_WIN_Y0 = 640
BANK_WIN_DX = 300

# --- Debug prints ---
DEBUG_PRINT = False

# ============================================================
# 2) ROI HELPERS
# ============================================================
def box_from_corners(TL, TR, BL, BR):
    left = int(round((TL[0] + BL[0]) / 2))
    right = int(round((TR[0] + BR[0]) / 2))
    top = int(round((TL[1] + TR[1]) / 2))
    bottom = int(round((BL[1] + BR[1]) / 2))
    return {"left": left, "top": top, "width": max(2, right-left), "height": max(2, bottom-top)}

BOARD_MON = box_from_corners(BOARD_TL, BOARD_TR, BOARD_BL, BOARD_BR)
BANK_MON  = box_from_corners(BANK_TL, BANK_TR, BANK_BL, BANK_BR)
BANK_MON["top"] -= 10
BANK_MON["height"] += 20

BANK_LEFT_EXPAND = 30  # try 80, 120, 160
BANK_MON["left"] = max(0, BANK_MON["left"] - BANK_LEFT_EXPAND)
BANK_MON["width"] += BANK_LEFT_EXPAND

print("BOARD_MON:", BOARD_MON)
print("BANK_MON :", BANK_MON)

# ============================================================
# 3) BOARD -> 8x8 MATRIX
# ============================================================
def even_grid_lines(w, h, n=8):
    xs = [int(round(i * w / n)) for i in range(n + 1)]
    ys = [int(round(i * h / n)) for i in range(n + 1)]
    xs[0], xs[-1] = 0, w
    ys[0], ys[-1] = 0, h
    return xs, ys

def board_to_matrix(board_bgr):
    hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)
    _, _, V = cv2.split(hsv)

    h, w = V.shape[:2]
    xs, ys = even_grid_lines(w, h, N)

    grid = np.zeros((N, N), dtype=np.uint8)

    for r in range(N):
        for c in range(N):
            x1, x2 = xs[c], xs[c + 1]
            y1, y2 = ys[r], ys[r + 1]
            cell = V[y1:y2, x1:x2]

            pad_x = int((x2 - x1) * CELL_INNER_PAD_FRAC)
            pad_y = int((y2 - y1) * CELL_INNER_PAD_FRAC)
            core = cell[pad_y:(y2 - y1) - pad_y, pad_x:(x2 - x1) - pad_x]
            if core.size == 0:
                core = cell

            grid[r, c] = 1 if np.mean(core > V_THRESH) > 0.05 else 0

    return grid

# ============================================================
# 4) BANK -> PIECE MATRICES (HSV saturation/value)
# ============================================================
def trim(mat: np.ndarray) -> np.ndarray:
    rows = np.where(mat.sum(axis=1) > 0)[0]
    cols = np.where(mat.sum(axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return mat
    return mat[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

def keep_largest_component(mask01: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num <= 1:
        return mask01
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8)

def matrix_from_mask_bbox_grid(mask01: np.ndarray, cell: int = 21, fill_thr: float = 0.40):
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    crop = mask01[y0:y1 + 1, x0:x1 + 1]
    h, w = crop.shape
    if h < 2 or w < 2:
        return None

    H = int(math.ceil(h / cell) * cell)
    W = int(math.ceil(w / cell) * cell)
    pad_bottom = H - h
    pad_right = W - w

    crop_p = cv2.copyMakeBorder(
        crop, 0, pad_bottom, 0, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )

    rows, cols = H // cell, W // cell
    fill = crop_p.reshape(rows, cell, cols, cell).mean(axis=(1, 3))
    mat = (fill >= float(fill_thr)).astype(np.uint8)
    mat_t = trim(mat)
    return mat_t

def sanitize_piece_mat(mat: np.ndarray):
    if mat is None:
        return None
    mat = (mat > 0).astype(np.uint8)
    mat = trim(mat)
    if mat is None or mat.size == 0:
        return None
    h, w = mat.shape
    s = int(mat.sum())
    if h > MAX_DIM or w > MAX_DIM:
        return None
    if s <= 0 or s > MAX_BLOCKS:
        return None
    return mat

def matrix_from_slot_bgr(slot_bgr: np.ndarray):
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    if BLUR_K and BLUR_K > 1:
        S = cv2.GaussianBlur(S, (BLUR_K, BLUR_K), 0)
        V = cv2.GaussianBlur(V, (BLUR_K, BLUR_K), 0)

    if USE_OTSU_ON_S:
        _, s_mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, s_mask = cv2.threshold(S, S_MIN, 255, cv2.THRESH_BINARY)

    _, v_mask = cv2.threshold(V, V_MIN, 255, cv2.THRESH_BINARY)

    mask_u8 = cv2.bitwise_and(s_mask, v_mask)
    mask01 = (mask_u8 > 0).astype(np.uint8)

    k = np.ones((3, 3), np.uint8)
    mask_open = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, k, iterations=1)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, k, iterations=1)

    mask_largest = keep_largest_component(mask_close)

    # area gate (phantom 1x1 prevention)
    if int(mask_largest.sum()) < MIN_MASK_AREA:
        return None

    ys, xs = np.where(mask_largest > 0)
    if xs.size == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0p = max(0, y0 - BBOX_PAD)
    x0p = max(0, x0 - BBOX_PAD)
    y1p = min(mask_largest.shape[0] - 1, y1 + BBOX_PAD)
    x1p = min(mask_largest.shape[1] - 1, x1 + BBOX_PAD)

    mask_crop = mask_largest[y0p:y1p + 1, x0p:x1p + 1]
    mat = matrix_from_mask_bbox_grid(mask_crop, cell=CELL, fill_thr=FILL_THR)
    return sanitize_piece_mat(mat)

# ============================================================
# 5) GAME MECHANICS
# ============================================================
def clear_lines(board01: np.ndarray):
    b = board01.copy()
    full_rows = np.where(b.sum(axis=1) == 8)[0]
    full_cols = np.where(b.sum(axis=0) == 8)[0]
    if full_rows.size:
        b[full_rows, :] = 0
    if full_cols.size:
        b[:, full_cols] = 0
    cleared = int(full_rows.size + full_cols.size)
    return b, cleared

def can_place(board01: np.ndarray, piece01: np.ndarray, top: int, left: int):
    ph, pw = piece01.shape
    if top < 0 or left < 0 or top + ph > 8 or left + pw > 8:
        return False
    region = board01[top:top + ph, left:left + pw]
    return np.all((region + piece01) <= 1)

def place_and_clear(board01: np.ndarray, piece01: np.ndarray, top: int, left: int):
    b = board01.copy()
    ph, pw = piece01.shape
    b[top:top + ph, left:left + pw] |= piece01
    b2, cleared = clear_lines(b)
    return b2, int(cleared)

# ============================================================
# 6) BETTER MOVE HEURISTIC (with 2-ply)
# ============================================================
def count_holes(board01: np.ndarray) -> int:
    empty = (board01 == 0)
    visited = np.zeros((8, 8), dtype=np.uint8)
    q = deque()

    for r in range(8):
        for c in (0, 7):
            if empty[r, c] and not visited[r, c]:
                visited[r, c] = 1
                q.append((r, c))
    for c in range(8):
        for r in (0, 7):
            if empty[r, c] and not visited[r, c]:
                visited[r, c] = 1
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8 and empty[rr, cc] and not visited[rr, cc]:
                visited[rr, cc] = 1
                q.append((rr, cc))

    return int(np.logical_and(empty, visited == 0).sum())

def count_empty_components(board01: np.ndarray) -> int:
    empty = (board01 == 0)
    visited = np.zeros((8, 8), dtype=np.uint8)
    comps = 0
    for r in range(8):
        for c in range(8):
            if empty[r, c] and not visited[r, c]:
                comps += 1
                q = deque([(r, c)])
                visited[r, c] = 1
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        r2, c2 = rr + dr, cc + dc
                        if 0 <= r2 < 8 and 0 <= c2 < 8 and empty[r2, c2] and not visited[r2, c2]:
                            visited[r2, c2] = 1
                            q.append((r2, c2))
    return comps

def count_total_placements(board01: np.ndarray, pieces: list) -> int:
    total = 0
    for p in pieces:
        if p is None:
            continue
        ph, pw = p.shape
        for top in range(0, 8 - ph + 1):
            for left in range(0, 8 - pw + 1):
                if can_place(board01, p, top, left):
                    total += 1
    return total

def best_followup_score(board01: np.ndarray, pieces: list, eval_board_fn) -> float:
    best = -1e18
    for i, p in enumerate(pieces):
        if p is None:
            continue
        ph, pw = p.shape
        for top in range(0, 8 - ph + 1):
            for left in range(0, 8 - pw + 1):
                if not can_place(board01, p, top, left):
                    continue
                b2, cleared = place_and_clear(board01, p, top, left)
                rem2 = [pieces[j] for j in range(len(pieces)) if j != i and pieces[j] is not None]
                s = eval_board_fn(b2, int(cleared), rem2)
                if s > best:
                    best = s
    if best == -1e18:
        return -1000.0
    return float(best)

def solve_best_one_move(board01: np.ndarray, slot_pieces: list, turns_since_clear: int):
    # weights
    W_CLEAR = 220.0
    W_EMPTY = 1.0
    W_HOLES = 14.0
    W_FRAG  = 10.0
    W_MOB   = 0.9
    W_2PLY  = 1.4
    USE_2PLY = True



    best_clear = None
    best_any = None

    pieces = []
    for p in slot_pieces:
        if p is None:
            pieces.append(None)
        else:
            p = (p > 0).astype(np.uint8)
            p = trim(p)
            pieces.append(p if (p is not None and p.size > 0 and int(p.sum()) > 0) else None)

    def eval_board(b: np.ndarray, cleared: int, remaining: list) -> float:
        empty = int((b == 0).sum())
        holes = count_holes(b)
        frag  = count_empty_components(b)
        mob   = count_total_placements(b, remaining)
        return (W_CLEAR * cleared + W_EMPTY * empty - W_HOLES * holes - W_FRAG * frag + W_MOB * mob)

    for slot_idx, p in enumerate(pieces):
        if p is None:
            continue
        ph, pw = p.shape
        for top in range(0, 8 - ph + 1):
            for left in range(0, 8 - pw + 1):
                if not can_place(board01, p, top, left):
                    continue

                b2, cleared = place_and_clear(board01, p, top, left)
                rem = [pieces[i] for i in range(len(pieces)) if i != slot_idx and pieces[i] is not None]

                base_score = eval_board(b2, int(cleared), rem)
                if USE_2PLY and rem:
                    follow = best_followup_score(b2, rem, eval_board)
                    score_val = base_score + W_2PLY * follow
                else:
                    score_val = base_score

                move = {
                    "slot": slot_idx,
                    "mat": p,
                    "top": top,
                    "left": left,
                    "cleared": int(cleared),
                    "filled_after": int(b2.sum()),
                    "score": float(score_val),
                }

                if best_any is None or score_val > best_any["score"]:
                    best_any = move
                if move["cleared"] > 0:
                    if best_clear is None or score_val > best_clear["score"]:
                        best_clear = move

    if turns_since_clear == 2:
        if best_clear is not None:
            best_clear["combo_break"] = False
            return best_clear
        if best_any is not None:
            best_any["combo_break"] = True
            return best_any
        return None

    if best_any is not None:
        best_any["combo_break"] = False
    return best_any

# ============================================================
# 7) RENDER: NEXT MOVE
# ============================================================
def render_one_move(board01: np.ndarray, move: dict | None, turns_since_clear: int):
    H = 8 * CELL_PX
    W = 8 * CELL_PX
    img = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)

    for r in range(8):
        for c in range(8):
            if board01[r, c] == 1:
                y0, y1 = r * CELL_PX, (r + 1) * CELL_PX
                x0, x1 = c * CELL_PX, (c + 1) * CELL_PX
                img[y0:y1, x0:x1] = OCC_COLOR

    if move is not None:
        overlay = img.copy()
        piece = move["mat"]
        top = move["top"]
        left = move["left"]
        ph, pw = piece.shape
        for rr in range(ph):
            for cc in range(pw):
                if piece[rr, cc] == 1:
                    br = top + rr
                    bc = left + cc
                    y0, y1 = br * CELL_PX, (br + 1) * CELL_PX
                    x0, x1 = bc * CELL_PX, (bc + 1) * CELL_PX
                    overlay[y0:y1, x0:x1] = PLAN_COLOR
                    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), 1)
        img = cv2.addWeighted(overlay, ALPHA_PLAN, img, 1 - ALPHA_PLAN, 0)

    for i in range(9):
        x = i * CELL_PX
        y = i * CELL_PX
        cv2.line(img, (x, 0), (x, H), GRID_COLOR, GRID_THICK)
        cv2.line(img, (0, y), (W, y), GRID_COLOR, GRID_THICK)

    if move is None:
        cv2.putText(img, f"tsc={turns_since_clear} | No legal placement",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        flag = "^" if move["cleared"] > 0 else "."
        combo_txt = " | COMBO BREAK" if move.get("combo_break", False) else ""
        cv2.putText(img,
                    f"Next{flag}: SLOT {move['slot']} at (r={move['top']}, c={move['left']}) | tsc={turns_since_clear} | clears={move['cleared']}{combo_txt}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)

    cv2.imshow("NEXT_MOVE", img)

# ============================================================
# 8) RENDER: BANK DETECTED PIECES + MATRIX TEXT
# ============================================================
def format_matrix_text(mat: np.ndarray) -> str:
    if mat is None:
        return "None"
    rows = ["".join(str(int(x)) for x in row) for row in mat.tolist()]
    return "[" + " / ".join(rows) + "]"

def render_piece_matrix_window(mat: np.ndarray, slot_idx: int, win_name: str):
    canvas_h, canvas_w = 220, 260
    img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    title = f"SLOT {slot_idx}"
    txt = format_matrix_text(mat)
    cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    if txt == "None":
        lines = ["None"]
    else:
        core = txt[1:-1]
        parts = core.split(" / ")
        lines = []
        cur = "["
        for i, p in enumerate(parts):
            add = p if i == 0 else (" / " + p)
            if len(cur) + len(add) > 24:
                lines.append(cur)
                cur = p
            else:
                cur += add
        lines.append(cur + "]")

    y = 46
    for ln in lines[:3]:
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18

    if mat is None or mat.size == 0 or int(mat.sum()) == 0:
        cv2.putText(img, "EMPTY", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(win_name, img)
        return

    mat01 = (mat > 0).astype(np.uint8)
    h, w = mat01.shape

    cell = min(40, max(16, int(min((canvas_w - 20) / max(1, w), (canvas_h - 90) / max(1, h)))))
    start_x = 10
    start_y = 90

    for r in range(h):
        for c in range(w):
            x0 = start_x + c * cell
            y0 = start_y + r * cell
            x1 = x0 + cell - 1
            y1 = y0 + cell - 1
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), 1)
            if mat01[r, c] == 1:
                img[y0:y1+1, x0:x1+1] = (255, 255, 255)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)

    cv2.imshow(win_name, img)

# ============================================================
# 9) STABILIZERS
# ============================================================
def board_sig(b):
    return (b.shape, b.tobytes())

def piece_sig(p):
    if p is None or p.size == 0 or int(p.sum()) == 0:
        return ("EMPTY",)
    return (p.shape, p.tobytes())

def stable_board_mat(sct):
    start = time.time()
    last = None
    same = 0
    b_last = None
    while True:
        img = np.array(sct.grab(BOARD_MON))[:, :, :3]
        b = board_to_matrix(img)
        sig = board_sig(b)
        b_last = b
        if sig == last:
            same += 1
        else:
            last = sig
            same = 1
        if same >= BOARD_STABLE_FRAMES:
            return b
        if time.time() - start > STABLE_TIMEOUT_SEC:
            return b_last

def stable_bank_pieces(sct):
    start = time.time()
    last = None
    same = 0
    last_pieces = None

    while True:
        bank_frame = np.array(sct.grab(BANK_MON))[:, :, :3]
        H, W = bank_frame.shape[:2]
        slot_w = W // N_SLOTS

        pieces = []
        for i in range(N_SLOTS):
            x0 = max(0, i * slot_w - SLOT_PAD)
            x1 = min(W, (i + 1) * slot_w + SLOT_PAD)
            slot = bank_frame[:, x0:x1]
            mat = matrix_from_slot_bgr(slot)
            pieces.append(mat)

        sig = tuple(piece_sig(p) for p in pieces)
        last_pieces = pieces

        if sig == last:
            same += 1
        else:
            last = sig
            same = 1

        if same >= BANK_STABLE_FRAMES:
            return last_pieces, sig

        if time.time() - start > STABLE_TIMEOUT_SEC:
            return last_pieces, sig

# ============================================================
# 10) MAIN LOOP
# ============================================================
turns_since_clear = 0
last_bank_sig = None
last_move = None

with mss.mss() as sct:
    while True:
        board_mat = stable_board_mat(sct)
        slot_pieces, bank_sig = stable_bank_pieces(sct)

        if SHOW_BANK_DETECTED_WINDOWS:
            for i, p in enumerate(slot_pieces):
                win = f"BANK_SLOT_{i}"
                render_piece_matrix_window(p, i, win)
                cv2.moveWindow(win, BANK_WIN_X0 + i * BANK_WIN_DX, BANK_WIN_Y0)

        # update combo based on bank change (piece consumed)
        if last_bank_sig is None:
            last_bank_sig = bank_sig
        elif bank_sig != last_bank_sig:
            if last_move is not None:
                cleared = int(last_move.get("cleared", 0))
                turns_since_clear = 0 if cleared > 0 else min(2, turns_since_clear + 1)
            last_bank_sig = bank_sig
            last_move = None

        move = solve_best_one_move(board_mat, slot_pieces, turns_since_clear)

        if DEBUG_PRINT:
            print("tsc:", turns_since_clear, "board ones:", int(board_mat.sum()),
                  "move:", None if move is None else (move["slot"], move["top"], move["left"], move["cleared"]))
            for i, p in enumerate(slot_pieces):
                if p is None:
                    print("  slot", i, "None")
                else:
                    print("  slot", i, "shape", p.shape, "sum", int(p.sum()))

        render_one_move(board_mat, move, turns_since_clear)
        last_move = move

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        time.sleep(1.0 / SCAN_FPS)

cv2.destroyAllWindows()
