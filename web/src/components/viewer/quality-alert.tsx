import { InfoIcon, ShieldAlertIcon, TriangleAlertIcon } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { QualityWarning } from "@/lib/api/client";

export function QualityAlerts({ warnings }: { warnings: QualityWarning[] }) {
  if (warnings.length === 0) return null;
  const model = warnings.filter((w) => w.code === "placeholder_model");
  const high = warnings.filter((w) => w.severity === "high" && w.code !== "placeholder_model");
  const rest = warnings.filter((w) => w.severity !== "high");
  return (
    <div className="space-y-2">
      {model.length > 0 && (
        <Alert variant="destructive">
          <ShieldAlertIcon aria-hidden />
          <AlertTitle>Untrained placeholder model</AlertTitle>
          <AlertDescription>{model[0]!.message}</AlertDescription>
        </Alert>
      )}
      {high.length > 0 && (
        <Alert variant="destructive">
          <ShieldAlertIcon aria-hidden />
          <AlertTitle>This image may not be a frontal chest X-ray</AlertTitle>
          <AlertDescription>
            <ul className="mt-1 list-disc space-y-0.5 pl-4">
              {high.map((w) => (
                <li key={w.code}>{w.message}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
      {rest.length > 0 && (
        <Alert variant="warning">
          {high.length ? <InfoIcon aria-hidden /> : <TriangleAlertIcon aria-hidden />}
          <AlertTitle>Image quality notes</AlertTitle>
          <AlertDescription>
            <ul className="mt-1 list-disc space-y-0.5 pl-4">
              {rest.map((w) => (
                <li key={w.code}>{w.message}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
