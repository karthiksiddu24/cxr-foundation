"use client";

import { LockIcon, RotateCwIcon } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import * as React from "react";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { useAnalysis } from "@/lib/analysis-store";
import { ApiError } from "@/lib/api/client";

import { Dropzone } from "./dropzone";
import { SampleGallery, type Sample } from "./sample-gallery";

export function UploadScreen() {
  const { state, analyze, retry, reset } = useAnalysis();
  const router = useRouter();
  const busy = state.status === "uploading" || state.status === "analyzing";

  const run = React.useCallback(
    async (file: File) => {
      const result = await analyze(file);
      if (result) {
        toast.success("Analysis complete", {
          description: `${result.findings.filter((f) => f.flagged).length} findings above threshold · ${result.timing.total_ms.toFixed(0)} ms`,
        });
        router.push("/viewer");
      }
    },
    [analyze, router],
  );

  const pickSample = async (s: Sample) => {
    try {
      const res = await fetch(s.file);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      await run(new File([blob], `${s.id}.png`, { type: "image/png" }));
    } catch {
      toast.error("Could not load the sample image.");
    }
  };

  const statusText =
    state.status === "uploading"
      ? `Uploading ${state.fileName}…`
      : state.status === "analyzing"
        ? "Analyzing image…"
        : undefined;

  return (
    <div className="mx-auto grid w-full max-w-5xl gap-8 px-4 py-10 sm:px-6 lg:py-14">
      <div className="max-w-2xl space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Chest X-ray finding analysis
        </h1>
        <p className="text-base leading-relaxed text-muted-foreground">
          Upload a frontal chest radiograph to see calibrated scores for 14 findings and a heatmap
          of where the model found evidence. The model was pretrained without labels on chest
          X-rays, then fine-tuned on NIH ChestX-ray14.{" "}
          <Link
            href="/model"
            prefetch={false}
            className="text-primary underline underline-offset-4"
          >
            Read the model card
          </Link>
          .
        </p>
      </div>

      <Dropzone
        onFile={(f) => void run(f)}
        busy={busy}
        statusText={statusText}
        progress={state.status === "uploading" ? state.progress : null}
      />

      {state.status === "error" && (
        <Alert variant="destructive">
          <AlertTitle>
            {state.error instanceof ApiError && state.error.status === 429
              ? "Too many requests"
              : "That didn't work"}
          </AlertTitle>
          <AlertDescription className="space-y-3">
            <p>{state.error.message}</p>
            <div className="flex gap-2">
              {state.error instanceof ApiError && state.error.retryable && (
                <Button size="sm" variant="outline" onClick={retry}>
                  <RotateCwIcon /> Try again
                </Button>
              )}
              <Button size="sm" variant="ghost" onClick={reset}>
                Choose a different file
              </Button>
            </div>
            {state.error instanceof ApiError && state.error.requestId && (
              <p className="font-mono text-xs">Request ID: {state.error.requestId}</p>
            )}
          </AlertDescription>
        </Alert>
      )}

      <SampleGallery onPick={(s) => void pickSample(s)} disabled={busy} />

      <div className="flex items-start gap-3 rounded-lg border bg-card p-4 text-sm">
        <LockIcon className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden />
        <div className="space-y-1">
          <p className="font-medium">Your image stays private</p>
          <p className="text-muted-foreground">
            Images are processed in memory and discarded after the response. Nothing is saved on the
            server or in your browser, and DICOM patient fields are never read or logged. Don&apos;t
            upload images with patient details burned into the pixels.
          </p>
        </div>
      </div>
    </div>
  );
}
