// Records the README demo: a scripted walk through the running app, saved as WebM, then
// converted to docs/demo.gif with ffmpeg (two-pass palette for small, clean output).
//
//   node scripts/record-demo.mjs [baseUrl=http://localhost:3000]
//
// Needs the API + web app running with a real (non-placeholder) bundle.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const BASE = process.argv[2] ?? "http://localhost:3000";
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const OUT = path.join(ROOT, "docs", "demo.gif");
const SIZE = { width: 1280, height: 760 };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "cxr-demo-"));
const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: SIZE,
  colorScheme: "dark",
  recordVideo: { dir: tmp, size: SIZE },
});
const page = await context.newPage();

const info = await (await fetch(`${BASE}/api/v1/model-info`)).json();
if (info.placeholder) throw new Error("refusing to record a demo of the placeholder model");

await page.goto(BASE);
await page.waitForLoadState("networkidle");
await pause(1800);

await page.getByRole("button", { name: /Analyze sample: Pleural effusion/ }).click();
await page.waitForURL("**/viewer");
const panel = page.getByRole("region", { name: "Model findings" });
await panel.getByRole("listitem").first().waitFor();
await pause(1800);

// Step through the three highest-scoring findings so their heatmaps show in turn.
const findings = panel.locator("[data-finding]");
for (let i = 0; i < 3; i++) {
  await findings.nth(i).click();
  await pause(1500);
}
await page.keyboard.press("+");
await pause(500);
await page.keyboard.press("+");
await pause(1200);
await page.keyboard.press("0");
await pause(800);

await page.goto(`${BASE}/model`);
await page.waitForLoadState("networkidle");
await pause(2200);
await page.mouse.wheel(0, 500);
await pause(1500);

await page.goto(`${BASE}/about`);
await page.waitForLoadState("networkidle");
await pause(2200);

const video = page.video();
await context.close();
await browser.close();
const webm = await video.path();

const palette = path.join(tmp, "palette.png");
const filters = "fps=8,scale=880:-1:flags=lanczos";
execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", webm, "-vf", `${filters},palettegen=max_colors=96:stats_mode=diff`, palette]);
execFileSync("ffmpeg", [
  "-y", "-loglevel", "error", "-i", webm, "-i", palette,
  "-lavfi", `${filters} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5`,
  OUT,
]);
fs.rmSync(tmp, { recursive: true, force: true });
console.log(`wrote ${OUT} (${(fs.statSync(OUT).size / 1e6).toFixed(1)} MB) for model ${info.model_id}`);
