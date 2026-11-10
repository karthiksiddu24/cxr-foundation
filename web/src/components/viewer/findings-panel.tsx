"use client";

import { EyeIcon, EyeOffIcon, InfoIcon, TriangleAlertIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { FindingResult } from "@/lib/api/client";
import { cn, formatPercent } from "@/lib/utils";

export interface FindingsPanelProps {
  findings: FindingResult[];
  selected: string | null;
  onSelect: (name: string | null) => void;
  plausible: boolean;
  revealed: boolean;
  onRevealedChange: (revealed: boolean) => void;
}

export function FindingsPanel({
  findings,
  selected,
  onSelect,
  plausible,
  revealed,
  onRevealedChange: setRevealed,
}: FindingsPanelProps) {
  const hidden = !plausible && !revealed;
  const flagged = findings.filter((f) => f.flagged).length;

  return (
    <section aria-labelledby="findings-heading" className="flex min-h-0 flex-col">
      <header className="flex items-baseline justify-between gap-3 px-4 pb-2 pt-4">
        <h2 id="findings-heading" className="text-sm font-semibold">
          Model findings
        </h2>
        <p className="text-xs text-muted-foreground" aria-live="polite">
          {hidden ? "Scores hidden" : `${flagged} of ${findings.length} above threshold`}
        </p>
      </header>
      <p className="px-4 pb-3 text-xs leading-relaxed text-muted-foreground">
        Calibrated model scores, sorted high to low. The tick marks each finding&apos;s decision
        threshold. Select a finding to show where the model found evidence.
      </p>

      {hidden ? (
        <div className="mx-4 mb-4 rounded-lg border border-flag/40 bg-flag-soft p-4 text-sm">
          <div className="flex items-start gap-2">
            <TriangleAlertIcon className="mt-0.5 size-4 shrink-0 text-flag" aria-hidden />
            <div className="space-y-2">
              <p className="font-medium">Scores hidden for this image</p>
              <p className="text-muted-foreground">
                This image failed the plausibility check (see the warning above), so the scores are
                likely meaningless.
              </p>
              <Button size="sm" variant="outline" onClick={() => setRevealed(true)}>
                <EyeIcon /> Show scores anyway
              </Button>
            </div>
          </div>
        </div>
      ) : (
        <ul className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2 pb-3" role="list">
          {findings.map((f, i) => (
            <FindingRow
              key={f.name}
              finding={f}
              index={i}
              selected={selected === f.name}
              onSelect={() => onSelect(selected === f.name ? null : f.name)}
            />
          ))}
        </ul>
      )}
      {!plausible && revealed && (
        <div className="px-4 pb-3">
          <Button size="sm" variant="ghost" onClick={() => setRevealed(false)}>
            <EyeOffIcon /> Hide scores
          </Button>
        </div>
      )}
    </section>
  );
}

function FindingRow({
  finding: f,
  index,
  selected,
  onSelect,
}: {
  finding: FindingResult;
  index: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const pct = formatPercent(f.probability);
  const thr = formatPercent(f.threshold, 0);
  return (
    <li>
      <div
        className={cn(
          "group relative rounded-md border border-transparent px-2 py-2 transition-colors",
          selected ? "border-primary/60 bg-accent" : "hover:bg-surface-2",
        )}
      >
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onSelect}
            aria-pressed={selected}
            data-finding={f.name}
            data-index={index}
            className="flex min-w-0 flex-1 items-center gap-2 rounded-sm text-left text-sm after:absolute after:inset-0 after:content-['']"
          >
            <span className="truncate font-medium">{f.display_name}</span>{" "}
            {f.flagged && <Badge variant="flag">Above threshold</Badge>}
            <span className="sr-only">
              , score {pct}, threshold {thr}. {selected ? "Heatmap shown." : "Show heatmap."}
            </span>
          </button>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                className="relative z-10 rounded-sm p-0.5 text-muted-foreground hover:text-foreground"
                aria-label={`What is ${f.display_name}?`}
              >
                <InfoIcon className="size-3.5" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="left">
              <p className="font-medium">{f.display_name}</p>
              <p className="mt-1 text-muted-foreground">{f.description}</p>
              {f.test_auroc?.point != null && (
                <p className="mt-2 text-muted-foreground">
                  Test AUROC {f.test_auroc.point.toFixed(3)}
                  {f.test_auroc.ci_low != null && f.test_auroc.ci_high != null
                    ? ` (95% CI ${f.test_auroc.ci_low.toFixed(3)}–${f.test_auroc.ci_high.toFixed(3)})`
                    : ""}
                </p>
              )}
            </TooltipContent>
          </Tooltip>
          <span className="tabular w-14 text-right font-mono text-sm">{pct}</span>
        </div>
        <ProbabilityBar value={f.probability} threshold={f.threshold} flagged={f.flagged} />
      </div>
    </li>
  );
}

export function ProbabilityBar({
  value,
  threshold,
  flagged,
}: {
  value: number;
  threshold: number;
  flagged: boolean;
}) {
  return (
    <div className="relative mt-1.5 h-1.5 rounded-full bg-bar-track" aria-hidden="true">
      <div
        className={cn(
          "absolute inset-y-0 left-0 rounded-full",
          flagged ? "bg-flag" : "bg-bar-fill",
        )}
        style={{ width: `${Math.max(1, value * 100)}%` }}
      />
      <div
        className="absolute -top-1 h-3.5 w-0.5 rounded-full bg-foreground"
        style={{ left: `calc(${threshold * 100}% - 1px)` }}
        data-testid="threshold-marker"
      />
    </div>
  );
}
