"use client";

import {
  ContrastIcon,
  FlameIcon,
  HandIcon,
  MaximizeIcon,
  MinimizeIcon,
  RotateCcwIcon,
  SunMediumIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useShortcuts } from "@/hooks/use-shortcuts";
import type { PredictResponse } from "@/lib/api/client";
import { cn, formatPercent } from "@/lib/utils";

import { FindingsPanel } from "./findings-panel";
import {
  DEFAULT_VIEW,
  ImageViewer,
  MAX_ZOOM,
  MIN_ZOOM,
  type Tool,
  type ViewState,
} from "./image-viewer";
import { QualityAlerts } from "./quality-alert";
import { ShortcutsDialog } from "./shortcuts-dialog";

function ToolButton({
  label,
  shortcut,
  pressed,
  onClick,
  children,
}: {
  label: string;
  shortcut?: string;
  pressed?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={label}
          aria-pressed={pressed}
          aria-keyshortcuts={shortcut}
          onClick={onClick}
          className={cn(pressed && "bg-accent text-accent-foreground")}
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>
        {label}
        {shortcut && <span className="ml-2 font-mono text-muted-foreground">{shortcut}</span>}
      </TooltipContent>
    </Tooltip>
  );
}

export function Workspace({ result }: { result: PredictResponse }) {
  const findings = result.findings;
  const [selected, setSelected] = React.useState<string | null>(
    () => findings.find((f) => f.flagged)?.name ?? null,
  );
  const [view, setView] = React.useState<ViewState>(DEFAULT_VIEW);
  const [tool, setTool] = React.useState<Tool>("pan");
  const [showHeatmap, setShowHeatmap] = React.useState(true);
  const [opacity, setOpacity] = React.useState(0.55);
  const [helpOpen, setHelpOpen] = React.useState(false);
  const [fullscreen, setFullscreen] = React.useState(false);
  const stageRef = React.useRef<HTMLDivElement>(null);
  const plausible = result.quality.plausible;
  const [revealed, setRevealed] = React.useState(false);
  const usable = plausible || revealed;

  const current = findings.find((f) => f.name === selected) ?? null;
  const heatmap = showHeatmap && current && usable ? current.heatmap.grid : null;

  React.useEffect(() => {
    const onFs = () => setFullscreen(document.fullscreenElement === stageRef.current);
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  const toggleFullscreen = React.useCallback(() => {
    const el = stageRef.current;
    if (!el) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void el.requestFullscreen?.();
  }, []);

  const zoomBy = (f: number) =>
    setView((v) => ({ ...v, zoom: Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.zoom * f)) }));
  const step = (d: number) => {
    const i = findings.findIndex((f) => f.name === selected);
    const next = findings[Math.min(findings.length - 1, Math.max(0, (i < 0 ? -1 : i) + d))];
    if (next) {
      setSelected(next.name);
      document
        .querySelector<HTMLElement>(`[data-finding="${next.name}"]`)
        ?.scrollIntoView({ block: "nearest" });
    }
  };

  useShortcuts({
    "+": () => zoomBy(1.25),
    "=": () => zoomBy(1.25),
    "-": () => zoomBy(0.8),
    "0": () => setView(DEFAULT_VIEW),
    w: () => setTool((t) => (t === "window" ? "pan" : "window")),
    W: () => setTool((t) => (t === "window" ? "pan" : "window")),
    i: () => setView((v) => ({ ...v, invert: !v.invert })),
    I: () => setView((v) => ({ ...v, invert: !v.invert })),
    h: () => setShowHeatmap((s) => !s),
    H: () => setShowHeatmap((s) => !s),
    f: toggleFullscreen,
    F: toggleFullscreen,
    ArrowDown: () => step(1),
    ArrowUp: () => step(-1),
    Escape: () => setSelected(null),
    "?": () => setHelpOpen(true),
  });

  const label = `Chest X-ray, ${result.image.width} by ${result.image.height} pixels${
    heatmap && current ? `, with heatmap for ${current.display_name}` : ""
  }`;

  return (
    <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
      <div className="flex min-h-0 flex-col gap-3">
        <QualityAlerts warnings={result.quality.warnings} />
        <div
          ref={stageRef}
          className="flex min-h-[420px] flex-1 flex-col overflow-hidden rounded-lg border bg-card lg:min-h-0"
        >
          <div
            role="toolbar"
            aria-label="Viewer tools"
            className="flex flex-wrap items-center gap-1 border-b px-2 py-1.5"
          >
            <ToolButton
              label="Pan"
              shortcut="W"
              pressed={tool === "pan"}
              onClick={() => setTool("pan")}
            >
              <HandIcon />
            </ToolButton>
            <ToolButton
              label="Window / level (drag)"
              shortcut="W"
              pressed={tool === "window"}
              onClick={() => setTool("window")}
            >
              <SunMediumIcon />
            </ToolButton>
            <Separator orientation="vertical" className="mx-1 h-5" />
            <ToolButton label="Zoom in" shortcut="+" onClick={() => zoomBy(1.25)}>
              <ZoomInIcon />
            </ToolButton>
            <ToolButton label="Zoom out" shortcut="-" onClick={() => zoomBy(0.8)}>
              <ZoomOutIcon />
            </ToolButton>
            <ToolButton
              label="Invert"
              shortcut="I"
              pressed={view.invert}
              onClick={() => setView((v) => ({ ...v, invert: !v.invert }))}
            >
              <ContrastIcon />
            </ToolButton>
            <ToolButton label="Reset view" shortcut="0" onClick={() => setView(DEFAULT_VIEW)}>
              <RotateCcwIcon />
            </ToolButton>
            <ToolButton
              label={fullscreen ? "Exit fullscreen" : "Fullscreen"}
              shortcut="F"
              onClick={toggleFullscreen}
            >
              {fullscreen ? <MinimizeIcon /> : <MaximizeIcon />}
            </ToolButton>
            <span className="tabular ml-auto hidden font-mono text-xs text-muted-foreground sm:inline">
              {Math.round(view.zoom * 100)}% · W {Math.round(view.width)} L{" "}
              {Math.round(view.center)}
            </span>
            <ShortcutsDialog open={helpOpen} onOpenChange={setHelpOpen} />
          </div>

          <div className="relative min-h-[360px] flex-1">
            <ImageViewer
              src={`data:image/png;base64,${result.image.display_png_base64}`}
              original={{ width: result.image.width, height: result.image.height }}
              display={{ width: result.image.display_width, height: result.image.display_height }}
              geometry={result.geometry}
              heatmap={heatmap}
              heatmapOpacity={opacity}
              view={view}
              onViewChange={setView}
              tool={tool}
              label={label}
            />
            {current && heatmap && (
              <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/70 px-2.5 py-1.5 text-xs text-white">
                Heatmap: <span className="font-medium">{current.display_name}</span>{" "}
                <span className="tabular font-mono">{formatPercent(current.probability)}</span>
              </div>
            )}
          </div>

          <div className="grid gap-3 border-t px-3 py-2.5 sm:grid-cols-2">
            <div className="flex items-center gap-3">
              <FlameIcon className="size-4 text-muted-foreground" aria-hidden />
              <label htmlFor="heatmap-toggle" className="text-xs font-medium">
                Heatmap
              </label>
              <Switch
                id="heatmap-toggle"
                checked={showHeatmap}
                onCheckedChange={setShowHeatmap}
                aria-keyshortcuts="H"
                disabled={!usable}
              />
              <Slider
                thumbLabel="Heatmap opacity"
                min={0}
                max={1}
                step={0.05}
                value={[opacity]}
                onValueChange={([v]) => setOpacity(v ?? 0)}
                disabled={!showHeatmap || !usable}
                className="max-w-40"
              />
              <span className="tabular w-9 font-mono text-xs text-muted-foreground">
                {Math.round(opacity * 100)}%
              </span>
            </div>
            <div className="flex items-center gap-3">
              <SunMediumIcon className="size-4 text-muted-foreground" aria-hidden />
              <span className="text-xs font-medium" id="wl-label">
                Window / level
              </span>
              <span className="text-[11px] text-muted-foreground" aria-hidden>
                W
              </span>
              <Slider
                thumbLabel="Window width (contrast)"
                min={8}
                max={512}
                step={1}
                value={[view.width]}
                onValueChange={([v]) => setView((s) => ({ ...s, width: v ?? s.width }))}
                className="max-w-28"
              />
              <span className="text-[11px] text-muted-foreground" aria-hidden>
                L
              </span>
              <Slider
                thumbLabel="Window level (brightness)"
                min={0}
                max={255}
                step={1}
                value={[view.center]}
                onValueChange={([v]) => setView((s) => ({ ...s, center: v ?? s.center }))}
                className="max-w-28"
              />
            </div>
          </div>
          {showHeatmap && usable && (
            <div className="flex items-center gap-2 border-t px-3 py-1.5 text-[11px] text-muted-foreground">
              <span>Weaker evidence</span>
              <span
                aria-hidden
                className="h-1.5 w-28 rounded-full"
                style={{
                  background:
                    "linear-gradient(90deg, rgba(235,104,52,0) 0%, rgba(235,104,52,0.6) 40%, rgb(242,140,40) 70%, rgb(255,214,102) 100%)",
                }}
              />
              <span>Stronger evidence</span>
              <span className="ml-auto hidden md:inline">
                Coarse ({result.geometry.grid_h}×{result.geometry.grid_w}) map, not a segmentation.
              </span>
            </div>
          )}
          {showHeatmap && usable && current && !current.flagged && (
            <p
              data-testid="heatmap-low-score-note"
              className="border-t px-3 py-1.5 text-[11px] text-muted-foreground"
            >
              {current.display_name} is below its threshold. Each map is scaled within its own
              finding, so the colors show where this weak evidence is relatively strongest, not that
              it is strong.
            </p>
          )}
        </div>
      </div>

      <aside className="flex min-h-0 flex-col overflow-hidden rounded-lg border bg-card lg:max-h-[calc(100dvh-10rem)]">
        <FindingsPanel
          findings={findings}
          selected={selected}
          onSelect={setSelected}
          plausible={plausible}
          revealed={revealed}
          onRevealedChange={setRevealed}
        />
        <footer className="space-y-1 border-t px-4 py-3 text-[11px] leading-relaxed text-muted-foreground">
          <p>
            Model {result.model.model_id} v{result.model.version} · inference{" "}
            {result.timing.inference_ms.toFixed(0)} ms
          </p>
          <p>{result.disclaimer} These are model scores, not a diagnosis.</p>
        </footer>
      </aside>
    </div>
  );
}
