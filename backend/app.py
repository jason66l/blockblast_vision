"""
FastAPI server.

  POST /api/analyze  screenshot (+ optional board/tray boxes) -> board, pieces, plan
  POST /api/solve    board + pieces JSON -> plan   (used after the user edits cells)

In production it also serves the built React app from ../frontend/dist.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import solver
import vision

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Block Blast Solver")


class Box(BaseModel):
    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)


Matrix = list[list[int]]


class SolveRequest(BaseModel):
    board: Matrix
    pieces: list[Matrix | None] = Field(min_length=1, max_length=3)
    moves_since_clear: int = Field(0, ge=0, le=2)

    @field_validator("board")
    @classmethod
    def check_board(cls, b: Matrix) -> Matrix:
        if len(b) != 8 or any(len(row) != 8 for row in b):
            raise ValueError("board must be 8x8")
        if any(v not in (0, 1) for row in b for v in row):
            raise ValueError("board cells must be 0 or 1")
        return b

    @field_validator("pieces")
    @classmethod
    def check_pieces(cls, ps: list[Matrix | None]) -> list[Matrix | None]:
        for p in ps:
            if p is None:
                continue
            if not p or len(p) > 5 or any(len(row) != len(p[0]) or len(row) > 5 for row in p):
                raise ValueError("pieces must be rectangular and at most 5x5")
            if any(v not in (0, 1) for row in p for v in row):
                raise ValueError("piece cells must be 0 or 1")
        return ps


def _parse_box(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        return Box(**json.loads(raw)).model_dump()
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"Invalid box: {e}")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/solve")
def solve(req: SolveRequest):
    return solver.solve(req.board, req.pieces, req.moves_since_clear)


@app.post("/api/analyze")
def analyze(
    image: Annotated[UploadFile, File()],
    board_box: Annotated[str | None, Form()] = None,
    tray_box: Annotated[str | None, Form()] = None,
    moves_since_clear: Annotated[int, Form(ge=0, le=2)] = 0,
):
    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB)")
    try:
        img = vision.decode_image(data)
    except ValueError as e:
        raise HTTPException(400, str(e))

    board, tray = _parse_box(board_box), _parse_box(tray_box)
    detected = vision.detect_regions(img)
    regions = {
        "board": board or detected["board"],
        "tray": tray or detected["tray"],
        "confident": detected["confident"] if not (board and tray) else True,
    }
    try:
        read = vision.analyze(img, regions["board"], regions["tray"])
    except ValueError as e:
        raise HTTPException(422, str(e))

    return {
        "width": img.shape[1],
        "height": img.shape[0],
        "regions": regions,
        **read,
        "plan": solver.solve(read["board"], read["pieces"], moves_since_clear),
    }


DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
