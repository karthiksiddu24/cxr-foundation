import { AlertTriangleIcon, CheckCircle2Icon, XCircleIcon } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { Interval, ModelInfo } from "@/lib/api/client";
import { formatPercent } from "@/lib/utils";

function fmtCI(iv: Interval | null | undefined, digits = 3): string {
  if (!iv || iv.point == null) return "—";
  const ci =
    iv.ci_low != null && iv.ci_high != null
      ? ` (${iv.ci_low.toFixed(digits)}–${iv.ci_high.toFixed(digits)})`
      : "";
  return `${iv.point.toFixed(digits)}${ci}`;
}

export function ModelCardView({ info }: { info: ModelInfo }) {
  const c = info.card;
  return (
    <div className="mx-auto w-full max-w-5xl space-y-6 px-4 py-10 sm:px-6">
      <header className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="accent">Model card</Badge>
          <Badge variant="outline">v{info.version}</Badge>
          <Badge variant="outline">{info.arch}</Badge>
          <Badge variant="outline">init: {info.init}</Badge>
        </div>
        <h1 className="text-3xl font-semibold tracking-tight">{info.model_id}</h1>
        <p className="max-w-3xl text-muted-foreground">
          Scores for 14 chest X-ray findings with class activation heatmaps. Every number on this
          page is read from the deployed model bundle, which is generated from evaluation artifacts.
        </p>
      </header>

      {info.placeholder && (
        <Alert variant="destructive">
          <AlertTriangleIcon aria-hidden />
          <AlertTitle>Untrained placeholder model</AlertTitle>
          <AlertDescription>
            The server is running a development placeholder. No performance numbers apply.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        <Stat label="Mean test AUROC (95% CI)" value={fmtCI(c.mean_test_auroc)} />
        <Stat
          label="Calibration error (ECE), raw → calibrated"
          value={
            c.mean_ece_raw != null && c.mean_ece_calibrated != null
              ? `${c.mean_ece_raw.toFixed(3)} → ${c.mean_ece_calibrated.toFixed(3)}`
              : "—"
          }
        />
        <Stat
          label="Input / heatmap grid"
          value={`${info.input_size}px / ${info.heatmap_grid.join("×")}`}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2Icon className="size-4 text-primary" aria-hidden /> Intended use
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">{c.intended_use}</CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <XCircleIcon className="size-4 text-destructive" aria-hidden /> Not intended for
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc space-y-1 pl-4 text-sm text-muted-foreground">
              {c.not_intended_for.map((x) => (
                <li key={x}>{x}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Per-finding performance (NIH official test split)</CardTitle>
          <CardDescription>
            AUROC and AUPRC with patient-level bootstrap 95% confidence intervals. Thresholds were
            chosen on the validation split (Youden&apos;s J) and their test sensitivity/specificity
            are shown. AUPRC depends on prevalence.
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <caption className="sr-only">Per-finding test metrics</caption>
            <thead>
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th scope="col" className="py-2 pr-3 font-medium">
                  Finding
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  AUROC (95% CI)
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  AUPRC (95% CI)
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  Prevalence
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  Threshold
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  Sens.
                </th>
                <th scope="col" className="py-2 text-right font-medium">
                  Spec.
                </th>
              </tr>
            </thead>
            <tbody className="tabular font-mono text-[13px]">
              {info.findings.map((f) => (
                <tr key={f.name} className="border-b last:border-0">
                  <th scope="row" className="py-2 pr-3 text-left font-sans font-medium">
                    {f.display_name}
                  </th>
                  <td className="py-2 pr-3 text-right">{fmtCI(f.test_auroc)}</td>
                  <td className="py-2 pr-3 text-right">{fmtCI(f.test_auprc)}</td>
                  <td className="py-2 pr-3 text-right">
                    {f.test_prevalence != null ? formatPercent(f.test_prevalence) : "—"}
                  </td>
                  <td className="py-2 pr-3 text-right">{f.threshold.toFixed(3)}</td>
                  <td className="py-2 pr-3 text-right">
                    {f.test_sensitivity != null ? f.test_sensitivity.toFixed(2) : "—"}
                  </td>
                  <td className="py-2 text-right">
                    {f.test_specificity != null ? f.test_specificity.toFixed(2) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {(c.external ?? []).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>External validation</CardTitle>
            <CardDescription>
              Performance on data from other institutions (domain shift).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {(c.external ?? []).map((e) => (
              <div key={`${e.dataset}-${e.split}`} className="space-y-2">
                <p className="text-sm font-medium">
                  {e.dataset} · {e.split} · {e.n_images.toLocaleString()} images · mean AUROC{" "}
                  {fmtCI(e.mean_auroc)}
                </p>
                <ul className="grid gap-1 text-sm sm:grid-cols-2">
                  {Object.entries(e.per_finding).map(([k, v]) => (
                    <li key={k} className="flex justify-between gap-3 border-b py-1">
                      <span>{k}</span>
                      <span className="tabular font-mono text-[13px]">{fmtCI(v)}</span>
                    </li>
                  ))}
                </ul>
                {e.notes && <p className="text-xs text-muted-foreground">{e.notes}</p>}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Training data</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">{c.training_data}</CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Evaluation data</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">{c.evaluation_data}</CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Known limitations and failure modes</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc space-y-1.5 pl-4 text-sm text-muted-foreground">
            {c.limitations.map((x) => (
              <li key={x}>{x}</li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground">
        Built {info.created_at} from commit {info.git_commit?.slice(0, 10) ?? "unknown"} · running
        on {info.execution_provider}
        {c.eval_artifact && (
          <>
            {" "}
            · numbers from <code className="font-mono">{c.eval_artifact}</code>
          </>
        )}
      </p>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-card p-4">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="tabular mt-1 font-mono text-lg font-medium">{value}</p>
    </div>
  );
}
