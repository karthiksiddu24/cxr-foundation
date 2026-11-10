import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as React from "react";
import { describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";
import { finding } from "@/test/fixtures";

import { FindingsPanel } from "./findings-panel";

function setup(props: Partial<React.ComponentProps<typeof FindingsPanel>> = {}) {
  const onSelect = vi.fn();
  const onRevealedChange = vi.fn();
  const findings = [finding("Effusion", 0.8), finding("Mass", 0.3, 0.2), finding("Hernia", 0.01)];
  render(
    <TooltipProvider>
      <FindingsPanel
        findings={findings}
        selected={null}
        onSelect={onSelect}
        plausible
        revealed={false}
        onRevealedChange={onRevealedChange}
        {...props}
      />
    </TooltipProvider>,
  );
  return { onSelect, onRevealedChange };
}

describe("FindingsPanel", () => {
  it("renders findings in the given (probability) order with formatted scores", () => {
    setup();
    const rows = screen.getAllByRole("listitem");
    expect(rows).toHaveLength(3);
    expect(within(rows[0]!).getByText("80.0%")).toBeInTheDocument();
    expect(within(rows[2]!).getByText("1.0%")).toBeInTheDocument();
  });

  it("marks findings above threshold with a text badge, not color alone", () => {
    setup();
    expect(screen.getAllByText("Above threshold")).toHaveLength(2);
    expect(screen.getByText("2 of 3 above threshold")).toBeInTheDocument();
  });

  it("selects a finding on click and exposes pressed state", async () => {
    const { onSelect } = setup();
    await userEvent.click(screen.getByRole("button", { name: /^Effusion\b.*score 80\.0%/ }));
    expect(onSelect).toHaveBeenCalledWith("Effusion");
  });

  it("toggles off when the selected finding is clicked again", async () => {
    const { onSelect } = setup({ selected: "Mass" });
    const btn = screen.getByRole("button", { name: /^Mass .*score 30.0%/ });
    expect(btn).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(btn);
    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it("hides scores behind an explicit choice when the image is implausible", async () => {
    const { onRevealedChange } = setup({ plausible: false });
    expect(screen.queryByText("80.0%")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Show scores anyway/ }));
    expect(onRevealedChange).toHaveBeenCalledWith(true);
  });

  it("draws one threshold marker per finding", () => {
    setup();
    expect(screen.getAllByTestId("threshold-marker")).toHaveLength(3);
  });
});
