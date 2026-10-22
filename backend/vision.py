"""
Screenshot -> board + tray pieces.

Ported from v1blockblast.py. v1 read fixed screen coordinates; here the board
and tray boxes come from the client (or detect_regions), and the tray cell size
is derived from the board cell size so any screenshot resolution works.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

N = 8
N_SLOTS = 3

# --- Board occupancy ---
V_THRESH = 165
CELL_INNER_PAD_FRAC = 0.30
BRIGHT_FRAC = 0.05

# --- Tray / piece extraction ---
BLUR_K = 3
SLOT_PAD_FRAC = 0.05  # of tray width
BBOX_PAD = 2
FILL_THR = 0.40
TRAY_CELL_RATIO = 0.47  # tray blocks are drawn ~half the size of board cells (21px vs 45px in v1)
V_MIN = 40
MAX_DIM = 5
MAX_BLOCKS = 9
MIN_AREA_CELLS = 0.6  # reject blobs smaller than ~0.6 of a tray cell (phantom 1x1)


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def crop(img: np.ndarray, box: dict) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, int(round(box["x"])))
    y0 = max(0, int(round(box["y"])))
    x1 = min(w, int(round(box["x"] + box["w"])))
    y1 = min(h, int(round(box["y"] + box["h"])))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("Selected region is too small")
    return img[y0:y1, x0:x1]


# ============================================================
# Board -> 8x8 matrix
# ============================================================
def board_to_matrix(board_bgr: np.ndarray) -> list[list[int]]:
    V = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    h, w = V.shape
    xs = [int(round(i * w / N)) for i in range(N + 1)]
    ys = [int(round(i * h / N)) for i in range(N + 1)]

    grid = [[0] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            cell = V[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            ch, cw = cell.shape
            py, px = int(ch * CELL_INNER_PAD_FRAC), int(cw * CELL_INNER_PAD_FRAC)
            core = cell[py:ch - py, px:cw - px]
            if core.size == 0:
                core = cell
            grid[r][c] = int(np.mean(core > V_THRESH) > BRIGHT_FRAC)
    return grid


# ============================================================
# Tray -> piece matrices
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


def matrix_from_mask(mask01: np.ndarray, cell: float) -> np.ndarray | None:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None
    crop_ = mask01[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop_.shape
    if h < 2 or w < 2:
        return None

    # Snap the blob to a whole number of cells, then measure fill per cell.
    rows = max(1, round(h / cell))
    cols = max(1, round(w / cell))
    resized = cv2.resize(crop_.astype(np.float32), (cols * 16, rows * 16), interpolation=cv2.INTER_AREA)
    fill = resized.reshape(rows, 16, cols, 16).mean(axis=(1, 3))
    return trim((fill >= FILL_THR).astype(np.uint8))


def sanitize_piece(mat: np.ndarray | None) -> list[list[int]] | None:
    if mat is None or mat.size == 0:
        return None
    mat = trim((mat > 0).astype(np.uint8))
    h, w = mat.shape
    s = int(mat.sum())
    if h > MAX_DIM or w > MAX_DIM or s <= 0 or s > MAX_BLOCKS:
        return None
    return mat.tolist()


def piece_from_slot(slot_bgr: np.ndarray, cell: float) -> list[list[int]] | None:
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    if BLUR_K > 1:
        S = cv2.GaussianBlur(S, (BLUR_K, BLUR_K), 0)
        V = cv2.GaussianBlur(V, (BLUR_K, BLUR_K), 0)

    # Otsu on saturation separates colorful blocks from the dark tray.
    _, s_mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, v_mask = cv2.threshold(V, V_MIN, 255, cv2.THRESH_BINARY)
    mask01 = (cv2.bitwise_and(s_mask, v_mask) > 0).astype(np.uint8)

    k = np.ones((3, 3), np.uint8)
    mask01 = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, k)
    mask01 = cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, k)
    mask01 = keep_largest_component(mask01)

    if mask01.sum() < MIN_AREA_CELLS * cell * cell:
        return None
    # A slot that is mostly "saturated" is background, not a piece (Otsu on a flat slot).
    if mask01.mean() > 0.6:
        return None

    ys, xs = np.where(mask01 > 0)
    y0, y1 = max(0, ys.min() - BBOX_PAD), min(mask01.shape[0] - 1, ys.max() + BBOX_PAD)
    x0, x1 = max(0, xs.min() - BBOX_PAD), min(mask01.shape[1] - 1, xs.max() + BBOX_PAD)
    return sanitize_piece(matrix_from_mask(mask01[y0:y1 + 1, x0:x1 + 1], cell))


def tray_to_pieces(tray_bgr: np.ndarray, board_cell_px: float) -> list[list[list[int]] | None]:
    cell = board_cell_px * TRAY_CELL_RATIO
    H, W = tray_bgr.shape[:2]
    slot_w = W / N_SLOTS
    pad = int(W * SLOT_PAD_FRAC)
    pieces = []
    for i in range(N_SLOTS):
        x0 = max(0, int(i * slot_w) - pad)
        x1 = min(W, int((i + 1) * slot_w) + pad)
        pieces.append(piece_from_slot(tray_bgr[:, x0:x1], cell))
    return pieces


# ============================================================
# Region auto-detection
# ============================================================
def _line_strength(profile: np.ndarray, k: int) -> np.ndarray:
    """High-pass a 1D intensity profile: thin grid lines and border strips stand out
    from their neighborhood whether they are lighter or darker than the cells."""
    k = k | 1
    pad = np.pad(profile, k // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, k)
    return np.abs(profile - np.median(windows, axis=1))


def _best_lattice(strength: np.ndarray, step: float) -> tuple[float, int]:
    """Best start offset for 9 evenly spaced lines at the given step."""
    n = len(strength)
    span = int(round(8 * step))
    if span >= n:
        return -1.0, 0
    # Max-pool by 1px so slightly-off rounding still lands on the line.
    pooled = np.maximum(strength, np.maximum(np.roll(strength, 1), np.roll(strength, -1)))
    offs = np.round(np.arange(N + 1) * step).astype(int)
    mids = np.round((np.arange(N) + 0.5) * step).astype(int)
    starts = np.arange(0, n - span)
    vals = pooled[starts[:, None] + offs[None, :]]           # (starts, 9)
    gaps = strength[starts[:, None] + mids[None, :]]         # (starts, 8) cell centers
    # Median: most of the 9 lines must be present, so one strong decorative edge
    # can't anchor a lattice. The weaker outer edge breaks ties against a window
    # shifted by one cell (inner lines match, one edge doesn't). Cell centers
    # should be flat, so their strength counts against the lattice.
    score = (np.median(vals, axis=1) + 0.5 * np.minimum(vals[:, 0], vals[:, -1])
             - np.median(gaps, axis=1))
    i = int(np.argmax(score))
    return float(score[i]), int(starts[i])


def detect_regions(img: np.ndarray) -> dict:
    """
    Best-effort guess at the board and tray boxes: find one step size where 9 evenly
    spaced lines show up in both the row and column intensity profiles. The tray is
    assumed to sit directly under the board. The UI lets the user drag both boxes.
    """
    h, w = img.shape[:2]
    scale = min(1.0, 700 / max(h, w))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sh, sw = gray.shape

    k = max(5, int(min(sh, sw) / 40))
    col_strength = _line_strength(gray.mean(axis=0), k)
    row_strength = _line_strength(gray.mean(axis=1), k)
    noise = np.median(np.concatenate([col_strength, row_strength])) + 1e-3

    best = (-1.0, 0, 0, 0.0)
    min_step = min(sh, sw) * 0.35 / N
    max_step = min(sh, sw) * 0.99 / N
    for step in np.arange(min_step, max_step, 0.25):
        cs, x = _best_lattice(col_strength, step)
        rs, y = _best_lattice(row_strength, step)
        if cs + rs > best[0]:
            best = (cs + rs, x, y, step)

    score, x, y, step = best
    confident = score / noise > 6.0
    size = 8 * step
    if not confident:  # fall back to a typical phone layout
        size = sw * 0.9
        x = (sw - size) / 2
        y = sh * 0.3

    inv = 1 / scale
    board = {"x": x * inv, "y": y * inv, "w": size * inv, "h": size * inv}
    tray_y = min(h - 8, board["y"] + board["h"] + board["w"] * 0.04)
    tray = {"x": board["x"], "y": tray_y, "w": board["w"], "h": min(board["w"] * 0.42, h - tray_y)}
    return {"board": board, "tray": tray, "confident": bool(confident)}


def analyze(img: np.ndarray, board_box: dict, tray_box: dict) -> dict:
    board = board_to_matrix(crop(img, board_box))
    cell_px = board_box["w"] / N
    pieces = tray_to_pieces(crop(img, tray_box), cell_px)
    return {"board": board, "pieces": pieces}
