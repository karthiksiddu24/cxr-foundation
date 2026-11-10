"use client";

import { FileImageIcon, Loader2Icon, UploadIcon } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface DropzoneProps {
  onFile: (file: File) => void;
  busy: boolean;
  statusText?: string;
  progress?: number | null; // 0..1, or null for indeterminate
}

export function Dropzone({ onFile, busy, statusText, progress }: DropzoneProps) {
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [over, setOver] = React.useState(false);

  const take = (files: FileList | null) => {
    const f = files?.[0];
    if (f && !busy) onFile(f);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!busy) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        take(e.dataTransfer.files);
      }}
      data-testid="dropzone"
      className={cn(
        "relative flex min-h-64 flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed border-input bg-card px-6 py-10 text-center transition-colors",
        over && "border-primary bg-accent",
        busy && "cursor-progress",
      )}
    >
      {busy ? (
        <div className="flex w-full max-w-sm flex-col items-center gap-3" aria-live="polite">
          <Loader2Icon className="size-8 animate-spin text-primary" aria-hidden />
          <p className="text-sm font-medium">{statusText}</p>
          <div
            className="h-1.5 w-full overflow-hidden rounded-full bg-bar-track"
            role="progressbar"
            aria-label={statusText}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress == null ? undefined : Math.round(progress * 100)}
          >
            {progress == null ? (
              <div className="h-full w-1/3 animate-[indeterminate_1.2s_ease-in-out_infinite] rounded-full bg-primary" />
            ) : (
              <div
                className="h-full rounded-full bg-primary transition-[width]"
                style={{ width: `${progress * 100}%` }}
              />
            )}
          </div>
        </div>
      ) : (
        <>
          <div className="grid size-12 place-items-center rounded-full bg-accent text-accent-foreground">
            <FileImageIcon className="size-6" aria-hidden />
          </div>
          <div className="space-y-1">
            <p className="text-base font-medium">Drop a chest X-ray here</p>
            <p className="text-sm text-muted-foreground">
              PNG, JPEG, or DICOM · frontal view · up to 25 MB
            </p>
          </div>
          <Button onClick={() => inputRef.current?.click()}>
            <UploadIcon /> Choose file
          </Button>
          <input
            ref={inputRef}
            type="file"
            className="sr-only"
            tabIndex={-1}
            aria-label="Upload chest X-ray"
            accept=".png,.jpg,.jpeg,.dcm,.dicom,image/png,image/jpeg,application/dicom"
            onChange={(e) => {
              take(e.target.files);
              e.target.value = "";
            }}
          />
        </>
      )}
    </div>
  );
}
