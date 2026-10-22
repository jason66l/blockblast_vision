"""
Block Blast solver on 64-bit bitboards.

Bit (r * 8 + c) is set when cell (r, c) is filled. Pieces are precomputed as a
list of placed masks (one per legal offset), so "can I place here?" is a single
AND and line clears are 16 mask compares.

Unlike v1 (best single move + 2-ply), this searches every order and placement
of all pieces in the tray and returns the best full sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

N = 8
FULL = (1 << 64) - 1
ROW_MASKS = [0xFF << (8 * r) for r in range(N)]
COL_MASKS = [sum(1 << (r * 8 + c) for r in range(N)) for c in range(N)]
NOT_COL0 = FULL & ~COL_MASKS[0]
NOT_COL7 = FULL & ~COL_MASKS[7]
BORDER = ROW_MASKS[0] | ROW_MASKS[7] | COL_MASKS[0] | COL_MASKS[7]

# ============================================================
# Heuristic weights (carried over from v1, plus lookahead terms)
# ============================================================
W_CLEAR = 220.0        # per line cleared
W_MULTI = 60.0         # bonus per extra line in a single move (2+ lines)
W_EMPTY = 1.0          # per empty cell at the end
W_HOLES = 14.0         # per empty cell sealed off from the border
W_FRAG = 10.0          # per separate empty region
W_MOB = 0.9            # per placement available to "reference" pieces
W_NOFIT = 120.0        # per big reference piece that no longer fits anywhere
W_COMBO_BREAK = 150.0  # going 3 moves without a clear
W_UNPLACED = 1e6       # per tray piece that can't be placed (game over)

# Shapes that tend to end runs. Keeping room for them makes the board resilient
# to whatever the next tray brings.
REFERENCE_PIECES = [
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[1, 1, 1, 1, 1]],
    [[1], [1], [1], [1], [1]],
    [[1, 1], [1, 1]],
    [[1, 1, 1, 1]],
    [[1], [1], [1], [1]],
    [[1, 0, 0], [1, 0, 0], [1, 1, 1]],
]
BIG_REFERENCE = 3  # the first 3 reference pieces are the "game enders"


def popcount(x: int) -> int:
    return x.bit_count()


# ============================================================
# Conversions
# ============================================================
def board_from_matrix(mat) -> int:
    b = 0
    for r in range(N):
        for c in range(N):
            if mat[r][c]:
                b |= 1 << (r * 8 + c)
    return b


def board_to_matrix(b: int) -> list[list[int]]:
    return [[(b >> (r * 8 + c)) & 1 for c in range(N)] for r in range(N)]


def trim(mat) -> list[list[int]]:
    rows = [r for r, row in enumerate(mat) if any(row)]
    cols = [c for c in range(len(mat[0])) if any(row[c] for row in mat)] if mat else []
    if not rows or not cols:
        return []
    return [[1 if mat[r][c] else 0 for c in range(cols[0], cols[-1] + 1)]
            for r in range(rows[0], rows[-1] + 1)]


@dataclass(frozen=True)
class Piece:
    mat: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]  # bit offset of each filled cell from the top-left anchor
    base: int                 # piece mask anchored at (0, 0)
    anchors: int              # bit set for every in-bounds top-left anchor

    @staticmethod
    def from_matrix(mat) -> "Piece | None":
        mat = trim(mat)
        if not mat:
            return None
        h, w = len(mat), len(mat[0])
        if h > N or w > N:
            return None
        offsets = tuple(r * 8 + c for r, c in product(range(h), range(w)) if mat[r][c])
        anchors = 0
        for top, left in product(range(N - h + 1), range(N - w + 1)):
            anchors |= 1 << (top * 8 + left)
        return Piece(tuple(tuple(row) for row in mat), offsets, sum(1 << o for o in offsets), anchors)

    def legal_anchors(self, b: int) -> int:
        """Bit-parallel: anchor a is legal iff every cell a + offset is empty."""
        empty = FULL & ~b
        legal = self.anchors
        for off in self.offsets:
            legal &= empty >> off
        return legal


REF = [Piece.from_matrix(m) for m in REFERENCE_PIECES]


# ============================================================
# Game mechanics
# ============================================================
def place_and_clear(b: int, mask: int) -> tuple[int, int, int]:
    """Returns (new_board, row_bits, col_bits); bit i of row_bits/col_bits is set
    when row/col i was cleared."""
    nb = b | mask
    clear = 0
    row_bits = col_bits = 0
    for i in range(N):
        rm = ROW_MASKS[i]
        if nb & rm == rm:
            clear |= rm
            row_bits |= 1 << i
        cm = COL_MASKS[i]
        if nb & cm == cm:
            clear |= cm
            col_bits |= 1 << i
    return nb & ~clear, row_bits, col_bits


def flood(seed: int, empty: int) -> int:
    reach = seed & empty
    while True:
        grown = reach | (
            ((reach << 1) & NOT_COL0) | ((reach >> 1) & NOT_COL7) |
            ((reach << 8) & FULL) | (reach >> 8)
        ) & empty
        if grown == reach:
            return reach
        reach = grown


def count_holes(b: int) -> int:
    empty = FULL & ~b
    return popcount(empty & ~flood(BORDER, empty))


def count_empty_regions(b: int) -> int:
    empty = FULL & ~b
    regions = 0
    while empty:
        seed = empty & -empty
        empty &= ~flood(seed, empty)
        regions += 1
    return regions


def count_fits(b: int, piece: Piece) -> int:
    return popcount(piece.legal_anchors(b))


_eval_cache: dict[int, float] = {}


def eval_board(b: int) -> float:
    """Static score of a board after the tray is used up."""
    hit = _eval_cache.get(b)
    if hit is not None:
        return hit
    score = W_EMPTY * (64 - popcount(b)) - W_HOLES * count_holes(b) - W_FRAG * count_empty_regions(b)
    for i, ref in enumerate(REF):
        fits = count_fits(b, ref)
        score += W_MOB * fits
        if i < BIG_REFERENCE and fits == 0:
            score -= W_NOFIT
    if len(_eval_cache) > 200_000:
        _eval_cache.clear()
    _eval_cache[b] = score
    return score


def move_score(lines: int, moves_since_clear: int) -> tuple[float, int]:
    """Score for the clear itself and the updated streak counter."""
    if lines:
        return W_CLEAR * lines + W_MULTI * max(0, lines - 1), 0
    msc = moves_since_clear + 1
    return (-W_COMBO_BREAK if msc == 3 else 0.0), min(msc, 3)


# ============================================================
# Search
# ============================================================
def solve(board_mat, piece_mats, moves_since_clear: int = 0) -> dict:
    """
    board_mat: 8x8 0/1 matrix. piece_mats: up to 3 piece matrices (None = empty slot).
    Returns the best sequence of placements over all orders.
    """
    b0 = board_from_matrix(board_mat)
    pieces = [Piece.from_matrix(m) if m is not None else None for m in piece_mats]
    slots = tuple(i for i, p in enumerate(pieces) if p is not None)

    memo: dict[tuple[int, tuple[int, ...], int], tuple[float, tuple]] = {}
    nodes = 0

    def search(b: int, remaining: tuple[int, ...], msc: int) -> tuple[float, tuple]:
        nonlocal nodes
        key = (b, remaining, msc)
        if key in memo:
            return memo[key]
        nodes += 1
        if not remaining:
            res = (eval_board(b), ())
            memo[key] = res
            return res

        best: tuple[float, tuple] | None = None
        seen_pieces = set()
        for slot in remaining:
            p = pieces[slot]
            if p.mat in seen_pieces:  # identical pieces: order doesn't matter
                continue
            seen_pieces.add(p.mat)
            rest = tuple(s for s in remaining if s != slot)
            legal = p.legal_anchors(b)
            while legal:
                low = legal & -legal
                legal ^= low
                anchor = low.bit_length() - 1
                top, left = divmod(anchor, 8)
                nb, rows, cols = place_and_clear(b, p.base << anchor)
                gained, nmsc = move_score(popcount(rows) + popcount(cols), msc)
                sub_score, sub_path = search(nb, rest, nmsc)
                total = gained + sub_score
                if best is None or total > best[0]:
                    best = (total, ((slot, top, left, rows, cols, nb),) + sub_path)

        if best is None:  # nothing fits: game over with len(remaining) unplaced
            best = (eval_board(b) - W_UNPLACED * len(remaining), ())
        memo[key] = best
        return best

    score, path = search(b0, slots, moves_since_clear)

    steps = []
    for slot, top, left, rows, cols, nb in path:
        mat = pieces[slot].mat
        steps.append({
            "slot": slot,
            "top": top,
            "left": left,
            "cells": [[top + r, left + c] for r in range(len(mat)) for c in range(len(mat[0])) if mat[r][c]],
            "cleared_rows": [i for i in range(N) if rows >> i & 1],
            "cleared_cols": [i for i in range(N) if cols >> i & 1],
            "board_after": board_to_matrix(nb),
        })

    return {
        "steps": steps,
        "score": round(score, 1),
        "placed_all": len(steps) == len(slots),
        "unplaced_slots": [s for s in slots if s not in {st["slot"] for st in steps}],
        "nodes_searched": nodes,
    }
