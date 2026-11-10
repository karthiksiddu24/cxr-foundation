"use client";

import * as React from "react";

import type { HeatmapGeometry } from "@/lib/api/client";
import { heatmapRect, rampColor, windowLut } from "@/lib/heatmap";
import { cn } from "@/lib/utils";

export type Tool = "pan" | "window";

export interface ViewState {
  zoom: number; // 1 = fit
  panX: number; // CSS px
  panY: number;
  center: number; // window center, 0..255 display units
  width: number; // window width, 1..512
  invert: boolean;
}

export const DEFAULT_VIEW: ViewState = {
  zoom: 1,
  panX: 0,
  panY: 0,
  center: 128,
  width: 256,
  invert: false,
};
export const MIN_ZOOM = 0.5;
export const MAX_ZOOM = 12;

export interface ImageViewerProps {
  src: string; // data URL of the grayscale display PNG
  original: { width: number; height: number };
  display: { width: number; height: number };
  geometry: HeatmapGeometry;
  heatmap: number[][] | null; // normalized [0,1] grid, or null to hide
  heatmapOpacity: number; // 0..1
  view: ViewState;
  onViewChange: (v: ViewState | ((v: ViewState) => ViewState)) => void;
  tool: Tool;
  label: string;
  className?: string;
}

function clampZoom(z: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));
}

export function ImageViewer({
  src,
  original,
  display,
  geometry,
  heatmap,
  heatmapOpacity,
  view,
  onViewChange,
  tool,
  label,
  className,
}: ImageViewerProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const baseRef = React.useRef<ImageData | null>(null); // original grayscale pixels
  const processedRef = React.useRef<HTMLCanvasElement | null>(null); // after window/level
  const heatRef = React.useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = React.useState({ w: 0, h: 0 });
  // Which src has finished loading; derived (not reset in an effect) so a new src reads as loading.
  const [loadedSrc, setLoadedSrc] = React.useState<string | null>(null);
  const loaded = loadedSrc === src;
  const pointers = React.useRef(new Map<number, { x: number; y: number }>());
  const drag = React.useRef<{ x: number; y: number; view: ViewState; pinch?: number } | null>(null);

  // Load the image once per src into an ImageData buffer.
  React.useEffect(() => {
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      baseRef.current = ctx.getImageData(0, 0, c.width, c.height);
      processedRef.current = document.createElement("canvas");
      processedRef.current.width = c.width;
      processedRef.current.height = c.height;
      setLoadedSrc(src);
    };
    img.src = src;
    return () => {
      cancelled = true;
    };
  }, [src]);

  // Track container size (CSS px).
  React.useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setSize({ w: Math.round(width), h: Math.round(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Window/level: apply LUT to the base pixels (only when W/L or inversion changes).
  React.useEffect(() => {
    const base = baseRef.current;
    const out = processedRef.current;
    if (!loaded || !base || !out) return;
    const lut = windowLut(view.center, view.width, view.invert);
    const px = new ImageData(base.width, base.height);
    const s = base.data;
    const d = px.data;
    for (let i = 0; i < s.length; i += 4) {
      const v = lut[s[i]!]!;
      d[i] = v;
      d[i + 1] = v;
      d[i + 2] = v;
      d[i + 3] = 255;
    }
    out.getContext("2d")?.putImageData(px, 0, 0);
  }, [loaded, view.center, view.width, view.invert]);

  // Heatmap: small RGBA canvas at grid resolution, upsampled with smoothing at draw time.
  React.useEffect(() => {
    if (!heatmap || heatmap.length === 0) {
      heatRef.current = null;
      return;
    }
    const gh = heatmap.length;
    const gw = heatmap[0]!.length;
    const c = document.createElement("canvas");
    c.width = gw;
    c.height = gh;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(gw, gh);
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        const [r, g, b, a] = rampColor(heatmap[y]![x]!);
        const o = (y * gw + x) * 4;
        img.data[o] = r;
        img.data[o + 1] = g;
        img.data[o + 2] = b;
        img.data[o + 3] = a;
      }
    }
    ctx.putImageData(img, 0, 0);
    heatRef.current = c;
  }, [heatmap]);

  const fit = React.useMemo(() => {
    if (!size.w || !size.h) return 1;
    return Math.min(size.w / display.width, size.h / display.height) * 0.96;
  }, [size, display]);

  // Draw.
  React.useEffect(() => {
    const canvas = canvasRef.current;
    const proc = processedRef.current;
    if (!canvas || !loaded || !proc || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.w * dpr);
    canvas.height = Math.round(size.h * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const s = fit * view.zoom;
    const ox = size.w / 2 + view.panX - (display.width * s) / 2;
    const oy = size.h / 2 + view.panY - (display.height * s) / 2;
    ctx.setTransform(s * dpr, 0, 0, s * dpr, ox * dpr, oy * dpr);
    ctx.imageSmoothingEnabled = view.zoom < 2;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(proc, 0, 0, display.width, display.height);
    const heat = heatRef.current;
    if (heat && heatmapOpacity > 0) {
      const r = heatmapRect(geometry, original, display);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, display.width, display.height); // never paint outside the image
      ctx.clip();
      ctx.globalAlpha = heatmapOpacity;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(heat, r.x, r.y, r.w, r.h);
      ctx.restore();
    }
  }, [loaded, size, fit, view, heatmap, heatmapOpacity, geometry, original, display]);

  // ------------------------------------------------------------------ interaction
  const zoomAt = React.useCallback(
    (factor: number, cx: number, cy: number) => {
      onViewChange((v) => {
        const z = clampZoom(v.zoom * factor);
        const k = z / v.zoom;
        // keep the point under the cursor fixed
        const dx = cx - size.w / 2;
        const dy = cy - size.h / 2;
        return { ...v, zoom: z, panX: dx - (dx - v.panX) * k, panY: dy - (dy - v.panY) * k };
      });
    },
    [onViewChange, size],
  );

  React.useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - rect.left, e.clientY - rect.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const pts = [...pointers.current.values()];
    if (pts.length === 2) {
      const [a, b] = pts as [{ x: number; y: number }, { x: number; y: number }];
      drag.current = {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
        view,
        pinch: Math.hypot(a.x - b.x, a.y - b.y),
      };
    } else {
      drag.current = { x: e.clientX, y: e.clientY, view };
    }
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!pointers.current.has(e.pointerId) || !drag.current) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const start = drag.current;
    const pts = [...pointers.current.values()];
    if (pts.length === 2 && start.pinch) {
      const [a, b] = pts as [{ x: number; y: number }, { x: number; y: number }];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const z = clampZoom(start.view.zoom * (dist / start.pinch));
      onViewChange({ ...start.view, zoom: z });
      return;
    }
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    const useWindow = tool === "window" || e.buttons === 2 || e.shiftKey;
    if (useWindow) {
      onViewChange({
        ...start.view,
        width: Math.min(512, Math.max(1, start.view.width + dx)),
        center: Math.min(255, Math.max(0, start.view.center + dy * 0.6)),
      });
    } else {
      onViewChange({ ...start.view, panX: start.view.panX + dx, panY: start.view.panY + dy });
    }
  };

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    pointers.current.delete(e.pointerId);
    drag.current = null;
  };

  return (
    <div
      ref={containerRef}
      className={cn("relative h-full w-full overflow-hidden bg-viewer-bg", className)}
    >
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={label}
        className={cn(
          "absolute inset-0 h-full w-full touch-none select-none",
          tool === "window" ? "cursor-ns-resize" : "cursor-grab active:cursor-grabbing",
        )}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={() => onViewChange((v) => ({ ...v, zoom: 1, panX: 0, panY: 0 }))}
        onContextMenu={(e) => e.preventDefault()}
      />
      {!loaded && (
        <div className="absolute inset-0 grid place-items-center text-sm text-neutral-400">
          Loading image…
        </div>
      )}
    </div>
  );
}
