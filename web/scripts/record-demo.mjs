// Records the demo from the running app: a scripted, captioned walk-through.
//
//   node scripts/record-demo.mjs [baseUrl=http://localhost:3000]
//
// Outputs (repo root):
//   docs/demo.mp4  full walk-through with captions (H.264, 1280x760, ~1 min)
//   docs/demo.gif  first part only (upload -> findings -> heatmaps), small enough for the README
// Needs the API + web app running with a real (non-placeholder) bundle, and ffmpeg on PATH.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const BASE = process.argv[2] ?? "http://localhost:3000";
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const MP4 = path.join(ROOT, "docs", "demo.mp4");
const GIF = path.join(ROOT, "docs", "demo.gif");
const SIZE = { width: 1280, height: 760 };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

const info = await (await fetch(`${BASE}/api/v1/model-info`)).json();
if (info.placeholder) throw new Error("refusing to record a demo of the placeholder model");

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "cxr-demo-"));
const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: SIZE,
  colorScheme: "dark",
  recordVideo: { dir: tmp, size: SIZE },
});
const page = await context.newPage();
const t0 = Date.now();
let gifEnd = 0;

// Caption bubble, injected into the page (recording only; not part of the app).
async function caption(text) {
  await page.evaluate((t) => {
    let el = document.getElementById("__demo_caption");
    if (!el) {
      el = document.createElement("div");
      el.id = "__demo_caption";
      Object.assign(el.style, {
        position: "fixed", left: "50%", bottom: "28px", transform: "translateX(-50%)",
        zIndex: "2147483647", maxWidth: "860px", padding: "10px 18px", borderRadius: "10px",
        background: "rgba(10,14,20,0.97)", color: "#f4f6f8", font: "500 17px/1.35 system-ui, sans-serif",
        boxShadow: "0 6px 24px rgba(0,0,0,0.45)", textAlign: "center", pointerEvents: "none",
      });
      document.body.appendChild(el);
    }
    el.textContent = t;
  }, text);
}

async function go(url) {
  await page.goto(url);
  await page.waitForLoadState("networkidle");
}

// 1. Upload
await go(BASE);
await caption("CXR-Foundation: upload a chest X-ray (PNG, JPEG or DICOM), or try a sample.");
await pause(2600);
await page.getByRole("button", { name: /Analyze sample: Pleural effusion/ }).click();
await page.waitForURL("**/viewer");
const panel = page.getByRole("region", { name: "Model findings" });
await panel.getByRole("listitem").first().waitFor();
await caption(`14 findings with calibrated scores and validation-chosen thresholds (model: ${info.model_id}).`);
await pause(3000);

// 2. Heatmaps
const findings = panel.locator("[data-finding]");
await caption("Select a finding to see where the model found evidence for it.");
for (let i = 0; i < 3; i++) {
  await findings.nth(i).click();
  await pause(1600);
}
gifEnd = (Date.now() - t0) / 1000;

// 3. Viewer tools
await caption("A PACS-style viewer: zoom, pan, window/level and invert, with keyboard shortcuts.");
await page.keyboard.press("+");
await pause(500);
await page.keyboard.press("+");
await pause(900);
await page.keyboard.press("i");
await pause(1300);
await page.keyboard.press("i");
await page.keyboard.press("0");
await pause(700);
await findings.last().click();
await caption("Low-scoring findings say so: their heatmaps show weak evidence, scaled per finding.");
await pause(3200);

// 4. Model card
await go(`${BASE}/model`);
await caption("The model card is served live from the deployed bundle: data, metrics, limitations.");
await pause(2600);
await page.mouse.wheel(0, 650);
await pause(2400);

// 5. Methodology / results
await go(`${BASE}/about`);
await caption("Headline result: MAE pretraining on unlabeled X-rays beats ImageNet at 1%, 10% and 100% labels.");
await pause(1800);
// Bring the label-efficiency chart (the SVG with role="img") to the middle of the screen.
await page.locator('svg[role="img"]').first().evaluate((el) =>
  el.scrollIntoView({ behavior: "smooth", block: "center" }),
);
await pause(3600);
const tableBtn = page.getByRole("button", { name: "Show table" });
if (await tableBtn.count()) {
  await tableBtn.first().click();
  await page.locator("table").first().evaluate((el) =>
    el.scrollIntoView({ behavior: "smooth", block: "center" }),
  );
  await caption("Every number comes from saved evaluation artifacts, with patient-level 95% CIs.");
  await pause(3000);
}

// 6. Light theme
await page.getByRole("button", { name: "Switch to light theme" }).click();
await caption("Light and dark themes, WCAG 2.1 AA checked in CI. Research demo only, not for clinical use.");
await pause(3200);

const video = page.video();
await context.close();
await browser.close();
const webm = await video.path();

const run = (args) => execFileSync("ffmpeg", ["-y", "-loglevel", "error", ...args]);
run(["-i", webm, "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
  "-movflags", "+faststart", MP4]);

const palette = path.join(tmp, "palette.png");
const filters = "fps=8,scale=880:-1:flags=lanczos";
const cut = ["-t", gifEnd.toFixed(1)];
run([...cut, "-i", webm, "-vf", `${filters},palettegen=max_colors=96:stats_mode=diff`, palette]);
run([...cut, "-i", webm, "-i", palette, "-lavfi",
  `${filters} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5`, GIF]);
fs.rmSync(tmp, { recursive: true, force: true });
const mb = (p) => (fs.statSync(p).size / 1e6).toFixed(1);
console.log(`wrote ${MP4} (${mb(MP4)} MB) and ${GIF} (${mb(GIF)} MB, first ${gifEnd.toFixed(0)} s) for ${info.model_id}`);
