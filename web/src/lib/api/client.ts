import type { components } from "./schema";

export type PredictResponse = components["schemas"]["PredictResponse"];
export type FindingResult = components["schemas"]["FindingResult"];
export type ModelInfo = components["schemas"]["ModelInfo"];
export type Problem = components["schemas"]["Problem"];
export type QualityWarning = components["schemas"]["QualityWarning"];
export type HeatmapGeometry = components["schemas"]["HeatmapGeometry"];
export type Interval = components["schemas"]["Interval"];

/** Browser requests go through the Next.js same-origin proxy (see next.config.ts). */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;

  constructor(status: number, code: string, message: string, requestId: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }

  /** Whether retrying the same request might succeed. */
  get retryable(): boolean {
    return this.status === 0 || this.status === 429 || this.status >= 500;
  }
}

function isProblem(x: unknown): x is Problem {
  return typeof x === "object" && x !== null && "code" in x && "detail" in x;
}

function toApiError(status: number, body: unknown, requestId: string | null): ApiError {
  if (isProblem(body))
    return new ApiError(status, body.code, body.detail, body.request_id ?? requestId);
  if (status === 0) return new ApiError(0, "network_error", "Could not reach the analysis server.");
  if (status === 502 || status === 503 || status === 504)
    return new ApiError(
      status,
      "unavailable",
      "The analysis server is unavailable. Try again shortly.",
      requestId,
    );
  return new ApiError(status, "unexpected", `Unexpected server response (${status}).`, requestId);
}

export interface PredictOptions {
  signal?: AbortSignal;
  /** Fraction of the upload sent, 0..1. */
  onUploadProgress?: (fraction: number) => void;
}

/**
 * Upload a file for analysis. Uses XMLHttpRequest (not fetch) so upload progress can be shown.
 */
export function predict(file: File, opts: PredictOptions = {}): Promise<PredictResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/v1/predict`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) opts.onUploadProgress?.(e.loaded / e.total);
    };
    xhr.onload = () => {
      const rid = xhr.getResponseHeader("x-request-id");
      if (xhr.status === 200 && xhr.response) resolve(xhr.response as PredictResponse);
      else reject(toApiError(xhr.status, xhr.response, rid));
    };
    xhr.onerror = () => reject(toApiError(0, null, null));
    xhr.ontimeout = () => reject(new ApiError(0, "timeout", "The request timed out."));
    xhr.timeout = 120_000;
    opts.signal?.addEventListener("abort", () => {
      xhr.abort();
      reject(new DOMException("Aborted", "AbortError"));
    });
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
}

export async function getModelInfo(base: string, init?: RequestInit): Promise<ModelInfo> {
  let res: Response;
  try {
    res = await fetch(`${base}/v1/model-info`, init);
  } catch {
    throw toApiError(0, null, null);
  }
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) throw toApiError(res.status, body, res.headers.get("x-request-id"));
  return body as ModelInfo;
}
