"use client";

import { ArrowLeftIcon, ImageOffIcon, UploadIcon } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useAnalysis } from "@/lib/analysis-store";

import { Workspace } from "./workspace";

export function ViewerScreen() {
  const { state, reset } = useAnalysis();

  if (state.status === "uploading" || state.status === "analyzing") return <ViewerSkeleton />;

  if (state.status !== "done") {
    return (
      <div className="mx-auto flex max-w-md flex-1 flex-col items-center justify-center gap-4 px-4 py-24 text-center">
        <div className="grid size-12 place-items-center rounded-full bg-secondary">
          <ImageOffIcon className="size-6 text-muted-foreground" aria-hidden />
        </div>
        <h1 className="text-xl font-semibold">No image loaded</h1>
        <p className="text-sm text-muted-foreground">
          Results are kept only in this tab&apos;s memory, so they disappear when the page reloads.
          Upload an image or pick a sample to start.
        </p>
        <Button asChild>
          <Link href="/">
            <UploadIcon /> Upload an image
          </Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-[1600px] flex-1 flex-col gap-3 px-4 py-4 sm:px-6">
      <div className="flex flex-wrap items-center gap-3">
        <Button asChild variant="ghost" size="sm" onClick={reset}>
          <Link href="/">
            <ArrowLeftIcon /> New image
          </Link>
        </Button>
        <h1 className="truncate text-sm font-medium text-muted-foreground">
          <span className="sr-only">Results for </span>
          {state.fileName}
        </h1>
        {state.result.model.placeholder && (
          <span className="rounded-sm border border-destructive/50 px-2 py-0.5 text-xs font-medium text-destructive">
            Untrained placeholder model
          </span>
        )}
      </div>
      <Workspace result={state.result} />
    </div>
  );
}

function ViewerSkeleton() {
  return (
    <div
      className="mx-auto grid w-full max-w-[1600px] flex-1 gap-4 px-4 py-4 sm:px-6 lg:grid-cols-[minmax(0,1fr)_360px]"
      aria-busy="true"
      aria-label="Loading results"
    >
      <Skeleton className="min-h-[480px]" />
      <div className="space-y-3 rounded-lg border bg-card p-4">
        {Array.from({ length: 8 }, (_, i) => (
          <Skeleton key={i} className="h-9" />
        ))}
      </div>
    </div>
  );
}
