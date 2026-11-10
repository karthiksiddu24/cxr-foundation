import { FlaskConicalIcon } from "lucide-react";
import type { Metadata } from "next";

import { LabelEfficiencyChart } from "@/components/about/label-efficiency-chart";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { RESULTS } from "@/lib/results";

export const metadata: Metadata = { title: "Methodology" };

const STEPS = [
  {
    title: "1. Data",
    body: "112,120 frontal chest X-rays from NIH ChestX-ray14. Patient-level train/validation split; the official test list is held out. Images are decoded once into uint8 arrays so training is never I/O-bound.",
  },
  {
    title: "2. Self-supervised pretraining",
    body: "A Vision Transformer learns chest anatomy by reconstructing randomly masked image patches (Masked Autoencoder), using only unlabeled images — no ImageNet, no diagnoses.",
  },
  {
    title: "3. Fine-tuning",
    body: "The pretrained encoder is fine-tuned to predict 14 findings using 1%, 10% or 100% of the labeled training patients, and compared to random and ImageNet initializations.",
  },
  {
    title: "4. Evaluation",
    body: "Per-finding AUROC with patient-level bootstrap confidence intervals, calibration, external validation on CheXpert, heatmap localization against radiologist boxes, and a shortcut-learning audit.",
  },
];

export default function AboutPage() {
  return (
    <div className="mx-auto w-full max-w-5xl space-y-10 px-4 py-10 sm:px-6">
      <header className="max-w-3xl space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">Methodology</h1>
        <p className="text-muted-foreground">
          Can a model pretrained only on unlabeled chest X-rays learn useful features with fewer
          labels than one pretrained on everyday photos? This project measures that with a
          controlled label-efficiency study, then serves the best model with calibrated scores and
          heatmaps.
        </p>
      </header>

      <ol className="grid gap-4 sm:grid-cols-2">
        {STEPS.map((s) => (
          <li key={s.title}>
            <Card className="h-full">
              <CardHeader>
                <CardTitle>{s.title}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm leading-relaxed text-muted-foreground">
                {s.body}
              </CardContent>
            </Card>
          </li>
        ))}
      </ol>

      <Card>
        <CardHeader>
          <CardTitle>Label efficiency</CardTitle>
          <CardDescription>
            Mean test AUROC over 14 findings vs. fraction of labeled training patients, for each
            initialization of the same ViT-Base backbone.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {RESULTS.status === "ready" ? (
            <LabelEfficiencyChart
              series={RESULTS.label_efficiency}
              title="Mean AUROC vs. labeled fraction"
            />
          ) : (
            <div className="flex items-start gap-3 rounded-lg border border-dashed p-6 text-sm">
              <FlaskConicalIcon
                className="mt-0.5 size-5 shrink-0 text-muted-foreground"
                aria-hidden
              />
              <div>
                <p className="font-medium">Results pending</p>
                <p className="text-muted-foreground">
                  The experiments are still running. This chart is generated from evaluation
                  artifacts and will appear once they are complete; no numbers are shown before
                  then.
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Design choices worth noting</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc space-y-2 pl-4 text-sm leading-relaxed text-muted-foreground">
            <li>
              <span className="text-foreground">No horizontal flips.</span> Chest anatomy is
              left-right asymmetric (heart, aortic arch, stomach bubble), so mirrored images are
              anatomically implausible.
            </li>
            <li>
              <span className="text-foreground">Exact heatmaps.</span> Both CNNs and ViTs end in
              average pooling and a linear layer, so the per-region class scores average exactly to
              the prediction and are computed in the same forward pass.
            </li>
            <li>
              <span className="text-foreground">Patient-level everything.</span> Splits, label
              subsets and bootstrap confidence intervals all resample patients, not images, because
              some patients have dozens of studies.
            </li>
            <li>
              <span className="text-foreground">Calibrated scores.</span> Per-finding temperature
              scaling and decision thresholds are fit on validation data only.
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
