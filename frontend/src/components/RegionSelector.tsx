import { useEffect, useRef, useState } from "react";
import type { Box } from "../types";

type Key = "board" | "tray";
interface Boxes {
  board: Box;
  tray: Box;
}

interface Props {
  src: string;
  naturalWidth: number;
  naturalHeight: number;
  boxes: Boxes;
  onCommit: (boxes: Boxes) => void; // called when a drag ends
}

interface Drag {
  key: Key;
  mode: "move" | "resize";
  startX: number;
  startY: number;
  orig: Box;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Screenshot with draggable/resizable board and tray boxes (in image pixel coords). */
export default function RegionSelector({ src, naturalWidth, naturalHeight, boxes, onCommit }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [local, setLocal] = useState(boxes);
  const drag = useRef<Drag | null>(null);

  useEffect(() => setLocal(boxes), [boxes]);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setScale(el.clientWidth / naturalWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, [naturalWidth]);

  const onPointerDown = (key: Key, mode: Drag["mode"]) => (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    drag.current = { key, mode, startX: e.clientX, startY: e.clientY, orig: local[key] };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = (e.clientX - d.startX) / scale;
    const dy = (e.clientY - d.startY) / scale;
    const o = d.orig;
    let next: Box;
    if (d.mode === "move") {
      next = { ...o, x: clamp(o.x + dx, 0, naturalWidth - o.w), y: clamp(o.y + dy, 0, naturalHeight - o.h) };
    } else if (d.key === "board") {
      // The board is always square.
      const size = clamp(Math.max(o.w + dx, o.h + dy), 40, Math.min(naturalWidth - o.x, naturalHeight - o.y));
      next = { ...o, w: size, h: size };
    } else {
      next = { ...o, w: clamp(o.w + dx, 40, naturalWidth - o.x), h: clamp(o.h + dy, 20, naturalHeight - o.y) };
    }
    setLocal((prev) => ({ ...prev, [d.key]: next }));
  };

  const onPointerUp = () => {
    if (drag.current) onCommit(local);
    drag.current = null;
  };

  const renderBox = (key: Key, label: string) => {
    const b = local[key];
    return (
      <div
        className={`region region-${key}`}
        style={{ left: b.x * scale, top: b.y * scale, width: b.w * scale, height: b.h * scale }}
        onPointerDown={onPointerDown(key, "move")}
      >
        <span className="region-label">{label}</span>
        <span className="region-handle" onPointerDown={onPointerDown(key, "resize")} aria-label={`Resize ${label}`} />
      </div>
    );
  };

  return (
    <div className="regions" ref={wrap} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
      <img src={src} alt="Uploaded screenshot" draggable={false} />
      {renderBox("board", "Board")}
      {renderBox("tray", "Pieces")}
    </div>
  );
}
