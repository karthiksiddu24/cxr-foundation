"use client";

import * as React from "react";

import { Button } from "@/components/ui/button";
import type { Point, Series } from "@/lib/results";

// Fixed identity -> color (CVD-validated categorical slots 1-3); matches docs/figures.
const COLORS: Record<Series["init"], string> = {
  imagenet: "#3987e5",
  random: "#d95926",
  ssl: "#199e70",
};
const W = 640;
const H = 360;
const M = { top: 20, right: 120, bottom: 44, left: 56 };

export function LabelEfficiencyChart({ series, title }: { series: Series[]; title: string }) {
  const [table, setTable] = React.useState(false);
  const [hover, setHover] = React.useState<{ s: Series; p: Point } | null>(null);

  const all = series.flatMap((s) => s.points);
  const yMin = Math.floor(Math.min(...all.map((p) => p.ci_low)) * 20) / 20;
  const yMax = Math.ceil(Math.max(...all.map((p) => p.ci_high)) * 20) / 20;
  const fr = [...new Set(all.map((p) => p.fraction))].sort((a, b) => a - b);
  const lx = (f: number) => Math.log10(f);
  const x = (f: number) =>
    M.left +
    ((lx(f) - lx(fr[0]!)) / (lx(fr[fr.length - 1]!) - lx(fr[0]!) || 1)) * (W - M.left - M.right);
  const y = (v: number) => M.top + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - M.top - M.bottom);
  const ticks: number[] = [];
  for (let v = yMin; v <= yMax + 1e-9; v += 0.05) ticks.push(Number(v.toFixed(2)));

  return (
    <figure className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <figcaption className="text-sm font-medium">{title}</figcaption>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setTable((t) => !t)}
          aria-pressed={table}
        >
          {table ? "Show chart" : "Show table"}
        </Button>
      </div>
      {table ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th scope="col" className="py-2 pr-3 font-medium">
                  Initialization
                </th>
                {fr.map((f) => (
                  <th key={f} scope="col" className="py-2 pr-3 text-right font-medium">
                    {f * 100}% labels
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tabular font-mono text-[13px]">
              {series.map((s) => (
                <tr key={s.init} className="border-b">
                  <th scope="row" className="py-2 pr-3 text-left font-sans font-medium">
                    {s.label}
                  </th>
                  {fr.map((f) => {
                    const p = s.points.find((q) => q.fraction === f);
                    return (
                      <td key={f} className="py-2 pr-3 text-right">
                        {p
                          ? `${p.mean.toFixed(3)} (${p.ci_low.toFixed(3)}–${p.ci_high.toFixed(3)})`
                          : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="relative">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full"
            role="img"
            aria-label={`${title}. Use "Show table" for values.`}
          >
            {ticks.map((t) => (
              <g key={t}>
                <line
                  x1={M.left}
                  x2={W - M.right}
                  y1={y(t)}
                  y2={y(t)}
                  className="stroke-border"
                  strokeWidth={1}
                />
                <text
                  x={M.left - 8}
                  y={y(t)}
                  dy="0.32em"
                  textAnchor="end"
                  className="fill-muted-foreground text-[11px]"
                >
                  {t.toFixed(2)}
                </text>
              </g>
            ))}
            {fr.map((f) => (
              <text
                key={f}
                x={x(f)}
                y={H - M.bottom + 18}
                textAnchor="middle"
                className="fill-muted-foreground text-[11px]"
              >
                {f * 100}%
              </text>
            ))}
            <text
              x={(M.left + W - M.right) / 2}
              y={H - 6}
              textAnchor="middle"
              className="fill-muted-foreground text-[11px]"
            >
              Labeled training data (log scale)
            </text>
            <text
              transform={`translate(14 ${(M.top + H - M.bottom) / 2}) rotate(-90)`}
              textAnchor="middle"
              className="fill-muted-foreground text-[11px]"
            >
              Mean test AUROC
            </text>
            {series.map((s) => {
              const c = COLORS[s.init];
              const pts = [...s.points].sort((a, b) => a.fraction - b.fraction);
              const last = pts[pts.length - 1]!;
              return (
                <g key={s.init}>
                  <path
                    d={`${pts.map((p, i) => `${i ? "L" : "M"}${x(p.fraction)},${y(p.ci_high)}`).join("")}${[
                      ...pts,
                    ]
                      .reverse()
                      .map((p) => `L${x(p.fraction)},${y(p.ci_low)}`)
                      .join("")}Z`}
                    fill={c}
                    opacity={0.12}
                  />
                  <path
                    d={pts.map((p, i) => `${i ? "L" : "M"}${x(p.fraction)},${y(p.mean)}`).join("")}
                    stroke={c}
                    strokeWidth={2}
                    fill="none"
                  />
                  {pts.map((p) => (
                    <circle
                      key={p.fraction}
                      cx={x(p.fraction)}
                      cy={y(p.mean)}
                      r={5}
                      fill={c}
                      className="stroke-card"
                      strokeWidth={2}
                      // Inside role="img" children are presentational: points are a mouse
                      // affordance only; keyboard and screen-reader users get the "Show table" view.
                      onMouseEnter={() => setHover({ s, p })}
                      onMouseLeave={() => setHover(null)}
                    />
                  ))}
                  <text
                    x={x(last.fraction) + 10}
                    y={y(last.mean)}
                    dy="0.32em"
                    className="fill-foreground text-[12px]"
                  >
                    {s.label}
                  </text>
                </g>
              );
            })}
          </svg>
          {hover && (
            <div
              className="pointer-events-none absolute rounded-md border bg-popover px-3 py-2 text-xs shadow-lg"
              style={{
                left: `${(x(hover.p.fraction) / W) * 100}%`,
                top: `${(y(hover.p.mean) / H) * 100}%`,
                transform: "translate(-50%, calc(-100% - 12px))",
              }}
            >
              <p className="font-medium">
                {hover.s.label} · {hover.p.fraction * 100}% labels
              </p>
              <p className="tabular font-mono">
                {hover.p.mean.toFixed(3)} ({hover.p.ci_low.toFixed(3)}–{hover.p.ci_high.toFixed(3)})
              </p>
              {hover.p.seeds.length > 1 && (
                <p className="text-muted-foreground">{hover.p.seeds.length} seeds</p>
              )}
            </div>
          )}
          <ul className="mt-2 flex flex-wrap gap-4 text-xs" aria-hidden>
            {series.map((s) => (
              <li key={s.init} className="flex items-center gap-1.5">
                <span className="size-2.5 rounded-full" style={{ background: COLORS[s.init] }} />
                {s.label}
              </li>
            ))}
            <li className="text-muted-foreground">Shaded: patient-bootstrap 95% CI</li>
          </ul>
        </div>
      )}
    </figure>
  );
}
