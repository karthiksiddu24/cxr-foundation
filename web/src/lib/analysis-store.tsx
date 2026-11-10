"use client";

import * as React from "react";

import { predict, type ApiError, type PredictResponse } from "@/lib/api/client";
import { precheck } from "@/lib/upload";

export type AnalysisState =
  | { status: "idle" }
  | { status: "uploading"; fileName: string; progress: number }
  | { status: "analyzing"; fileName: string }
  | { status: "done"; fileName: string; result: PredictResponse }
  | { status: "error"; fileName: string | null; error: ApiError | Error; file: File | null };

interface AnalysisContextValue {
  state: AnalysisState;
  analyze: (file: File) => Promise<PredictResponse | null>;
  retry: () => void;
  reset: () => void;
}

const AnalysisContext = React.createContext<AnalysisContextValue | null>(null);

/**
 * Holds the current analysis in memory only. Nothing is written to localStorage, sessionStorage
 * or IndexedDB: a page reload discards the image and results, by design.
 */
export function AnalysisProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = React.useState<AnalysisState>({ status: "idle" });
  const abortRef = React.useRef<AbortController | null>(null);
  const lastFile = React.useRef<File | null>(null);

  const analyze = React.useCallback(async (file: File) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    lastFile.current = file;
    const check = await precheck(file);
    if (!check.ok) {
      setState({
        status: "error",
        fileName: file.name,
        error: new Error(check.message),
        file: null,
      });
      return null;
    }
    setState({ status: "uploading", fileName: file.name, progress: 0 });
    try {
      const result = await predict(file, {
        signal: ctrl.signal,
        onUploadProgress: (p) =>
          setState((s) =>
            p >= 1
              ? { status: "analyzing", fileName: file.name }
              : s.status === "uploading"
                ? { ...s, progress: p }
                : s,
          ),
      });
      setState({ status: "done", fileName: file.name, result });
      return result;
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return null;
      const error = e instanceof Error ? e : new Error(String(e));
      setState({ status: "error", fileName: file.name, error, file });
      return null;
    }
  }, []);

  const retry = React.useCallback(() => {
    if (lastFile.current) void analyze(lastFile.current);
  }, [analyze]);

  const reset = React.useCallback(() => {
    abortRef.current?.abort();
    lastFile.current = null;
    setState({ status: "idle" });
  }, []);

  const value = React.useMemo(
    () => ({ state, analyze, retry, reset }),
    [state, analyze, retry, reset],
  );
  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
}

export function useAnalysis(): AnalysisContextValue {
  const ctx = React.useContext(AnalysisContext);
  if (!ctx) throw new Error("useAnalysis must be used inside <AnalysisProvider>");
  return ctx;
}
