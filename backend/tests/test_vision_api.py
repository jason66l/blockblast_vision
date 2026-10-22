from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

import vision
from app import app

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)

# Ground truth read off the demo-video frames by eye.
EXPECTED = {
    "f02_phone.png": {
        "board": ["###..###", ".......#"] + ["........"] * 6,
        "pieces": [None, [[1], [1]], None],
    },
    "f05_phone.png": {
        "board": ["########", "##.##..#", "##..#..."] + ["........"] * 3 + [".......#", "......##"],
        "pieces": [None, None, None],  # tray is cropped out of this frame
    },
}


def as_rows(m):
    return ["".join("#" if v else "." for v in row) for row in m]


@pytest.mark.parametrize("name", EXPECTED)
def test_detect_and_read(name):
    img = cv2.imread(str(FIXTURES / name))
    regions = vision.detect_regions(img)
    assert regions["confident"]
    read = vision.analyze(img, regions["board"], regions["tray"])
    assert as_rows(read["board"]) == EXPECTED[name]["board"]
    assert read["pieces"] == EXPECTED[name]["pieces"]


def test_analyze_endpoint():
    with open(FIXTURES / "f02_phone.png", "rb") as f:
        res = client.post("/api/analyze", files={"image": ("s.png", f, "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert as_rows(body["board"]) == EXPECTED["f02_phone.png"]["board"]
    assert body["plan"]["placed_all"]


def test_analyze_rejects_non_image():
    res = client.post("/api/analyze", files={"image": ("x.png", b"not an image", "image/png")})
    assert res.status_code == 400


def test_solve_endpoint_validates_board():
    res = client.post("/api/solve", json={"board": [[0] * 8] * 7, "pieces": [[[1]]]})
    assert res.status_code == 422


def test_solve_endpoint():
    res = client.post("/api/solve", json={"board": [[0] * 8 for _ in range(8)], "pieces": [[[1, 1]], None, [[1]]]})
    assert res.status_code == 200
    assert len(res.json()["steps"]) == 2
