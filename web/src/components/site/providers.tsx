"use client";

import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";

import { TooltipProvider } from "@/components/ui/tooltip";
import { AnalysisProvider } from "@/lib/analysis-store";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem={false}
      disableTransitionOnChange
    >
      <TooltipProvider delayDuration={250}>
        <AnalysisProvider>
          {children}
          <Toaster
            position="bottom-right"
            closeButton
            toastOptions={{
              // Use our contrast-checked tokens rather than the library's rich colors.
              classNames: {
                toast: "!bg-popover !text-popover-foreground !border-border",
                description: "!text-muted-foreground",
                success: "[&_[data-icon]]:!text-primary",
                error: "[&_[data-icon]]:!text-destructive",
              },
            }}
          />
        </AnalysisProvider>
      </TooltipProvider>
    </ThemeProvider>
  );
}
