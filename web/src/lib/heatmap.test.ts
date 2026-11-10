import { describe, expect, it } from "vitest";

import { heatmapRect, rampColor, windowLut } from "./heatmap";

describe("heatmapRect", () => {
  it("covers the full image for square inputs", () => {
    const r = heatmapRect(
      { input_size: 224, grid_h: 7, grid_w: 7, scale: 224 / 1024, pad_x: 0, pad_y: 0 },
      { width: 1024, height: 1024 },
      { width: 1024, height: 1024 },
    );
    expect(r.x).toBeCloseTo(0);
    expect(r.w).toBeCloseTo(1024);
  });

  it("extends past the image horizontally for portrait (letterboxed) inputs", () => {
    // 400x800 portrait -> scale 0.28, content width 112, pad_x 56 in a 224 square
    const r = heatmapRect(
      { input_size: 224, grid_h: 7, grid_w: 7, scale: 0.28, pad_x: 56, pad_y: 0 },
      { width: 400, height: 800 },
      { width: 400, height: 800 },
    );
    expect(r.x).toBeCloseTo(-200);
    expect(r.w).toBeCloseTo(800);
    expect(r.y).toBeCloseTo(0);
    expect(r.h).toBeCloseTo(800);
  });

  it("scales into display coordinates", () => {
    const r = heatmapRect(
      { input_size: 224, grid_h: 7, grid_w: 7, scale: 224 / 2048, pad_x: 0, pad_y: 0 },
      { width: 2048, height: 2048 },
      { width: 1024, height: 1024 },
    );
    expect(r.w).toBeCloseTo(1024);
  });
});

describe("rampColor", () => {
  it("is fully transparent for weak evidence and opaque at the top", () => {
    expect(rampColor(0.1)[3]).toBe(0);
    expect(rampColor(1)[3]).toBe(255);
    expect(rampColor(-5)[3]).toBe(0);
  });
});

describe("windowLut", () => {
  it("is identity for the full window", () => {
    const lut = windowLut(128, 256, false);
    expect(lut[0]).toBe(0);
    expect(lut[128]).toBe(128);
    expect(lut[255]).toBe(255);
  });
  it("clips outside a narrow window and inverts", () => {
    const lut = windowLut(100, 20, false);
    expect(lut[50]).toBe(0);
    expect(lut[150]).toBe(255);
    expect(windowLut(128, 256, true)[0]).toBe(255);
  });
});
