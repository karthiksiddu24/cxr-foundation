import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SAMPLE = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "public", "samples");

async function expectNoA11yViolations(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  const serious = results.violations.filter((v) => ["serious", "critical"].includes(v.impact ?? ""));
  expect(serious, JSON.stringify(serious.map((v) => [v.id, v.nodes.map((n) => n.target)]), null, 1)).toEqual([]);
}

test("upload a PNG and inspect results", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Chest X-ray finding analysis" })).toBeVisible();
  await expect(page.getByRole("note")).toContainText("Not for clinical use");

  const fs = await import("node:fs");
  const sample = fs.readdirSync(SAMPLE).find((f) => f.endsWith(".png"))!;
  await page.getByLabel("Upload chest X-ray").setInputFiles(path.join(SAMPLE, sample));

  await page.waitForURL("**/viewer");
  await expect(page.getByRole("img", { name: /Chest X-ray, \d+ by \d+ pixels/ })).toBeVisible();
  const panel = page.getByRole("region", { name: "Model findings" });
  // the dev placeholder model hides scores until explicitly revealed
  const reveal = page.getByRole("button", { name: "Show scores anyway" });
  if (await reveal.isVisible()) await reveal.click();
  const rows = panel.getByRole("listitem");
  await expect(rows).toHaveCount(14);

  // probabilities are sorted descending
  const pcts = await panel.locator("span.tabular").allTextContents();
  const values = pcts.map((t) => parseFloat(t));
  expect(values).toEqual([...values].sort((a, b) => b - a));

  // selecting a finding shows its heatmap label on the canvas
  const first = panel.locator("[data-finding]").first();
  const name = await first.getAttribute("data-finding");
  if ((await first.getAttribute("aria-pressed")) !== "true") await first.click();
  await expect(page.getByText(`Heatmap:`)).toBeVisible();
  await expect(page.getByRole("img", { name: new RegExp(`heatmap for`) })).toBeVisible();
  expect(name).toBeTruthy();
});

test("sample image flow and viewer keyboard shortcuts", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Analyze sample: Pleural effusion/ }).click();
  await page.waitForURL("**/viewer");
  const reveal = page.getByRole("button", { name: "Show scores anyway" });
  if (await reveal.isVisible()) await reveal.click();

  const zoom = page.getByText(/^\d+% · W/);
  await expect(zoom).toContainText("100%");
  await page.keyboard.press("+");
  await expect(zoom).toContainText("125%");
  await page.keyboard.press("0");
  await expect(zoom).toContainText("100%");

  await page.keyboard.press("i");
  await expect(page.getByRole("button", { name: "Invert" })).toHaveAttribute("aria-pressed", "true");

  await page.keyboard.press("?");
  await expect(page.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeVisible();
  await page.keyboard.press("Escape");

  await page.keyboard.press("ArrowDown");
  await expect(page.locator('[data-finding][aria-pressed="true"]')).toHaveCount(1);
  await expectNoA11yViolations(page);
});

test("rejects a non-image file with a clear message", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Upload chest X-ray").setInputFiles({
    name: "notes.png",
    mimeType: "image/png",
    buffer: Buffer.from("definitely not an image"),
  });
  await expect(page.getByRole("alert").filter({ hasText: "didn't work" })).toContainText(
    "Unsupported file",
  );
  await expect(page).toHaveURL(/\/$/);
});

test("viewer without a result shows an empty state", async ({ page }) => {
  await page.goto("/viewer");
  await expect(page.getByRole("heading", { name: "No image loaded" })).toBeVisible();
  await page.getByRole("link", { name: "Upload an image" }).click();
  await expect(page).toHaveURL(/\/$/);
});

for (const route of ["/", "/model", "/about", "/viewer"]) {
  test(`no serious accessibility violations on ${route}`, async ({ page }) => {
    await page.goto(route);
    await page.waitForLoadState("networkidle");
    await expectNoA11yViolations(page);
  });
}

test("theme toggle switches to light mode", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.locator("html")).not.toHaveClass(/dark/);
  await expectNoA11yViolations(page);
});
