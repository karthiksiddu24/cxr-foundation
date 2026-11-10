import type { HeatmapGeometry } from "@/lib/api/client";

/**
 * Sequential warm ramp for heatmap overlays: transparent at low values, orange to amber at high.
 * Single warm hue family so it reads as "more" on a grayscale image and survives color-vision
 * deficiency; magnitude is also carried by alpha.
 */
const STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [235, 104, 52]],
  [0.6, [242, 140, 40]],
  [1.0, [255, 214, 102]],
];

export function rampColor(v: number): [number, number, number, number] {
  const t = Math.min(1, Math.max(0, v));
  let i = 0;
  while (i < STOPS.length - 2 && t > STOPS[i + 1]![0]) i++;
  const [t0, c0] = STOPS[i]!;
  const [t1, c1] = STOPS[i + 1]!;
  const f = (t - t0) / (t1 - t0 || 1);
  const rgb = c0.map((c, k) => Math.round(c + (c1[k]! - c) * f)) as [number, number, number];
  // Below 0.25 fully transparent so low evidence never tints the image.
  const alpha = t < 0.25 ? 0 : Math.round(255 * Math.min(1, (t - 0.25) / 0.6));
  return [...rgb, alpha];
}

/**
 * Where the heatmap square (model-input space) lands in display-image pixel coordinates.
 * original_x = (model_x - pad_x) / scale; display = original * (displayW / originalW).
 */
export function heatmapRect(
  g: HeatmapGeometry,
  original: { width: number; height: number },
  display: { width: number; height: number },
): { x: number; y: number; w: number; h: number } {
  const k = display.width / original.width;
  return {
    x: (-g.pad_x / g.scale) * k,
    y: (-g.pad_y / g.scale) * k,
    w: (g.input_size / g.scale) * k,
    h: (g.input_size / g.scale) * k,
  };
}

/** Window/level lookup table on 8-bit display values. center/width in 0..255. */
export function windowLut(center: number, width: number, invert: boolean): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256);
  const w = Math.max(2, width);
  const lo = center - w / 2;
  // Map [lo, lo + w - 1] onto [0, 255]; center 128 / width 256 is exactly the identity.
  for (let v = 0; v < 256; v++) {
    const y = ((v - lo) / (w - 1)) * 255;
    lut[v] = invert ? 255 - y : y;
  }
  return lut;
}
