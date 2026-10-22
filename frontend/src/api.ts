import type { AnalyzeResponse, Box, Matrix, Plan } from "./types";

async function request<T>(url: string, init: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") msg = body.detail;
      else if (Array.isArray(body.detail)) msg = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* non-JSON error body */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export function analyze(
  image: File,
  movesSinceClear: number,
  boxes?: { board: Box; tray: Box },
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("image", image);
  form.append("moves_since_clear", String(movesSinceClear));
  if (boxes) {
    form.append("board_box", JSON.stringify(boxes.board));
    form.append("tray_box", JSON.stringify(boxes.tray));
  }
  return request("/api/analyze", { method: "POST", body: form });
}

export function solve(board: Matrix, pieces: (Matrix | null)[], movesSinceClear: number, signal?: AbortSignal): Promise<Plan> {
  return request("/api/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ board, pieces, moves_since_clear: movesSinceClear }),
    signal,
  });
}
