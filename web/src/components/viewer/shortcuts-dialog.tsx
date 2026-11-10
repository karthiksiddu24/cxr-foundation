"use client";

import { KeyboardIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Kbd } from "@/components/ui/kbd";

export const SHORTCUTS: Array<[string[], string]> = [
  [["+", "−"], "Zoom in / out"],
  [["0"], "Reset view (zoom, pan, window/level)"],
  [["W"], "Toggle window/level drag tool"],
  [["I"], "Invert grayscale"],
  [["H"], "Show / hide heatmap"],
  [["↑", "↓"], "Previous / next finding"],
  [["Esc"], "Clear selected finding"],
  [["F"], "Fullscreen"],
  [["?"], "Show this help"],
];

export function ShortcutsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label="Keyboard shortcuts">
          <KeyboardIcon />
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle>Keyboard shortcuts</DialogTitle>
        <DialogDescription>
          Drag to pan, scroll to zoom. Hold Shift or right-drag to adjust window/level.
        </DialogDescription>
        <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
          {SHORTCUTS.map(([keys, desc]) => (
            <div key={desc} className="contents">
              <dt className="flex gap-1">
                {keys.map((k) => (
                  <Kbd key={k}>{k}</Kbd>
                ))}
              </dt>
              <dd className="text-muted-foreground">{desc}</dd>
            </div>
          ))}
        </dl>
      </DialogContent>
    </Dialog>
  );
}
