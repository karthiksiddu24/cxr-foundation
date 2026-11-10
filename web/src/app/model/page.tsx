import { ServerCrashIcon } from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";

import { ModelCardView } from "@/components/model/model-card-view";
import { Button } from "@/components/ui/button";
import { getModelInfo, type ModelInfo } from "@/lib/api/client";

export const metadata: Metadata = { title: "Model card" };
export const dynamic = "force-dynamic";

export default async function ModelPage() {
  let info: ModelInfo | null = null;
  let error: string | null = null;
  try {
    info = await getModelInfo(process.env.API_INTERNAL_URL ?? "http://localhost:8000", {
      cache: "no-store",
    });
  } catch (e) {
    error = e instanceof Error ? e.message : "Unknown error";
  }

  if (!info) {
    return (
      <div className="mx-auto flex max-w-md flex-1 flex-col items-center justify-center gap-4 px-4 py-24 text-center">
        <ServerCrashIcon className="size-8 text-muted-foreground" aria-hidden />
        <h1 className="text-xl font-semibold">Model card unavailable</h1>
        <p className="text-sm text-muted-foreground">
          The analysis server could not be reached ({error}). The model card is served by the API so
          it always matches the deployed model.
        </p>
        <Button asChild variant="outline">
          <Link href="/model">Try again</Link>
        </Button>
      </div>
    );
  }
  return <ModelCardView info={info} />;
}
