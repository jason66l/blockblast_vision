import solver
from solver import Piece, board_from_matrix, count_empty_regions, count_holes, place_and_clear, solve

EMPTY = [[0] * 8 for _ in range(8)]


def board(rows):
    """Build a board from strings like '##......'."""
    return [[1 if ch == "#" else 0 for ch in row] for row in rows]


def test_legal_anchors_match_brute_force():
    b = board([
        "#.......",
        "..##....",
        "........",
        "....#...",
        "........",
        ".#......",
        "........",
        ".......#",
    ])
    bits = board_from_matrix(b)
    for mat in ([[1, 1, 1], [1, 0, 0]], [[1], [1], [1], [1], [1]], [[1, 1], [1, 1]]):
        p = Piece.from_matrix(mat)
        legal = p.legal_anchors(bits)
        h, w = len(mat), len(mat[0])
        for top in range(8):
            for left in range(8):
                fits = top + h <= 8 and left + w <= 8 and all(
                    not (mat[r][c] and b[top + r][left + c]) for r in range(h) for c in range(w)
                )
                assert bool(legal >> (top * 8 + left) & 1) == fits, (mat, top, left)


def test_place_clears_row_and_column():
    b = board(["#######."] + ["#......."] * 6 + ["........"])
    bits = board_from_matrix(b)
    # Filling (0,7) completes row 0; then (7,0) would complete column 0.
    nb, rows, cols = place_and_clear(bits, 1 << 7)
    assert rows == 0b1 and cols == 0
    nb2, rows2, cols2 = place_and_clear(bits | (1 << 7), 1 << 56)
    assert rows2 == 0b1 and cols2 == 0b1
    assert nb2 == 0


def test_holes_and_regions():
    b = board([
        "........",
        ".###....",
        ".#.#....",
        ".###....",
        "........",
        "........",
        "........",
        "........",
    ])
    bits = board_from_matrix(b)
    assert count_holes(bits) == 1
    assert count_empty_regions(bits) == 2


def test_prefers_clearing_a_line():
    b = board(["####.###"] + ["........"] * 7)
    res = solve(b, [[[1]], None, None])
    assert res["steps"][0]["cells"] == [[0, 4]]
    assert res["steps"][0]["cleared_rows"] == [0]


def test_order_matters_and_all_pieces_placed():
    # Only a 1x1 hole at (0,7) and a 5-long vertical slot in column 0 rows 3-7.
    # Board is otherwise full except those cells; a bad order would block a piece.
    rows = ["#######."] + ["########"] * 2 + [".#######"] * 5
    res = solve(board(rows), [[[1], [1], [1], [1], [1]], [[1]], None])
    assert res["placed_all"]
    assert {s["slot"] for s in res["steps"]} == {0, 1}


def test_reports_unplaceable_piece():
    full_but_one = ["########"] * 7 + ["#######."]
    res = solve(board(full_but_one), [[[1, 1]], None, None])
    assert not res["placed_all"]
    assert res["unplaced_slots"] == [0]


def test_combo_rule_forces_clear_on_third_move():
    # With 2 moves already without a clear, a clearing move should win even if a
    # non-clearing spot looks tidier.
    b = board(["###.####"] + ["........"] * 7)
    res = solve(b, [[[1]], None, None], moves_since_clear=2)
    assert res["steps"][0]["cleared_rows"] == [0]


def test_board_after_is_consistent():
    res = solve(EMPTY, [[[1, 1, 1]], [[1], [1]], [[1, 1], [1, 1]]])
    b = EMPTY
    for step in res["steps"]:
        b = [row[:] for row in b]
        for r, c in step["cells"]:
            assert b[r][c] == 0
            b[r][c] = 1
        for r in step["cleared_rows"]:
            b[r] = [0] * 8
        for c in step["cleared_cols"]:
            for r in range(8):
                b[r][c] = 0
        assert b == step["board_after"]


def test_identical_pieces_are_deduplicated():
    res = solve(EMPTY, [[[1, 1]], [[1, 1]], [[1, 1]]])
    assert res["placed_all"]
    assert sorted(s["slot"] for s in res["steps"]) == [0, 1, 2]
    assert solver.W_UNPLACED > 0
