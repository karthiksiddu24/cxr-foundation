import raw from "@/content/results.json";

export interface Point {
  fraction: number;
  mean: number;
  ci_low: number;
  ci_high: number;
  seeds: number[];
}
export interface Series {
  init: "random" | "imagenet" | "ssl";
  label: string;
  points: Point[];
}
export interface Results {
  status: "ready";
  generated_at: string;
  backbone: string;
  label_efficiency: Series[];
  reference?: { label: string; points: Point[] };
  baselines: Array<{ name: string; mean_auroc: number; ci_low: number; ci_high: number }>;
}
export type ResultsFile = Results | { status: "pending"; note: string };

export const RESULTS = raw as ResultsFile;
