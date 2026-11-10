import type { Metadata } from "next";

import { ViewerScreen } from "@/components/viewer/viewer-screen";

export const metadata: Metadata = { title: "Viewer" };

export default function ViewerPage() {
  return <ViewerScreen />;
}
