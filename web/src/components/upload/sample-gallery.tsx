"use client";

import Image from "next/image";

import samples from "../../../public/samples/samples.json";

export interface Sample {
  id: string;
  file: string;
  title: string;
  dataset_labels: string[];
}

export const SAMPLES: Sample[] = samples.samples;

export function SampleGallery({
  onPick,
  disabled,
}: {
  onPick: (s: Sample) => void;
  disabled: boolean;
}) {
  return (
    <section aria-labelledby="samples-heading" className="space-y-3">
      <div className="flex items-baseline justify-between gap-3">
        <h2 id="samples-heading" className="text-sm font-semibold">
          Or try a sample
        </h2>
        <p className="text-xs text-muted-foreground">NIH ChestX-ray14 test set</p>
      </div>
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4" role="list">
        {SAMPLES.map((s) => (
          <li key={s.id}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(s)}
              className="group w-full overflow-hidden rounded-lg border bg-card text-left transition-colors hover:border-primary/60 disabled:opacity-60"
            >
              <div className="relative aspect-square bg-black">
                <Image
                  src={s.file}
                  alt=""
                  fill
                  sizes="(min-width: 640px) 200px, 45vw"
                  className="object-contain opacity-90 transition-opacity group-hover:opacity-100"
                />
              </div>
              <div className="space-y-0.5 p-2.5">
                <p className="truncate text-sm font-medium">
                  <span className="sr-only">Analyze sample: </span>
                  {s.title}
                </p>
                <p className="truncate text-xs text-muted-foreground">
                  Dataset label: {s.dataset_labels.join(", ")}
                </p>
              </div>
            </button>
          </li>
        ))}
      </ul>
      <p className="text-xs text-muted-foreground">{samples.label_note}</p>
    </section>
  );
}
