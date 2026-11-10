import type { FindingResult, PredictResponse } from "@/lib/api/client";

export function finding(name: string, probability: number, threshold = 0.5): FindingResult {
  return {
    name,
    display_name: name,
    description: `About ${name}`,
    probability,
    threshold,
    flagged: probability >= threshold,
    test_auroc: { point: 0.81, ci_low: 0.8, ci_high: 0.82 },
    heatmap: {
      grid: [
        [0, 1],
        [0.5, 0.25],
      ],
      logit_min: -1,
      logit_max: 2,
    },
  };
}

export function prediction(overrides: Partial<PredictResponse> = {}): PredictResponse {
  return {
    request_id: "req-1",
    model: { model_id: "m", version: "1.0.0", placeholder: false },
    image: {
      format: "png",
      width: 4,
      height: 4,
      display_png_base64: "",
      display_width: 4,
      display_height: 4,
    },
    geometry: { input_size: 224, grid_h: 2, grid_w: 2, scale: 1, pad_x: 0, pad_y: 0 },
    quality: { plausible: true, warnings: [], ood_score: 1, ood_threshold: 10 },
    findings: [finding("Effusion", 0.8), finding("Mass", 0.3), finding("Hernia", 0.01)],
    timing: { decode_ms: 1, inference_ms: 2, total_ms: 3 },
    disclaimer: "Research demonstration only. Not a medical device. Not for clinical use.",
    ...overrides,
  };
}
