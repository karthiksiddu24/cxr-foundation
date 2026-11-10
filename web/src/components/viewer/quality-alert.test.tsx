import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { QualityAlerts } from "./quality-alert";

describe("QualityAlerts", () => {
  it("renders nothing without warnings", () => {
    const { container } = render(<QualityAlerts warnings={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("separates model problems from image problems", () => {
    render(
      <QualityAlerts
        warnings={[
          { code: "placeholder_model", severity: "high", message: "placeholder msg" },
          { code: "color_image", severity: "high", message: "color msg" },
          { code: "low_contrast", severity: "caution", message: "contrast msg" },
        ]}
      />,
    );
    expect(screen.getByText("Untrained placeholder model")).toBeInTheDocument();
    expect(screen.getByText("This image may not be a frontal chest X-ray")).toBeInTheDocument();
    expect(screen.getByText("Image quality notes")).toBeInTheDocument();
    expect(screen.getAllByRole("alert")).toHaveLength(3);
  });

  it("does not claim the image is wrong when only the model is a placeholder", () => {
    render(
      <QualityAlerts warnings={[{ code: "placeholder_model", severity: "high", message: "m" }]} />,
    );
    expect(screen.queryByText(/may not be a frontal chest X-ray/)).not.toBeInTheDocument();
  });
});
