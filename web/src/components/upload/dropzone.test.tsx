import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { Dropzone } from "./dropzone";

describe("Dropzone", () => {
  it("passes a chosen file to onFile", async () => {
    const onFile = vi.fn();
    render(<Dropzone onFile={onFile} busy={false} />);
    const f = new File(["x"], "scan.png", { type: "image/png" });
    await userEvent.upload(screen.getByLabelText("Upload chest X-ray"), f);
    expect(onFile).toHaveBeenCalledWith(f);
  });

  it("accepts dropped files", () => {
    const onFile = vi.fn();
    render(<Dropzone onFile={onFile} busy={false} />);
    const f = new File(["x"], "scan.dcm");
    fireEvent.drop(screen.getByTestId("dropzone"), { dataTransfer: { files: [f] } });
    expect(onFile).toHaveBeenCalledWith(f);
  });

  it("shows determinate progress and ignores drops while busy", () => {
    const onFile = vi.fn();
    render(<Dropzone onFile={onFile} busy statusText="Uploading scan.png…" progress={0.4} />);
    expect(screen.getByRole("progressbar", { name: "Uploading scan.png…" })).toHaveAttribute(
      "aria-valuenow",
      "40",
    );
    fireEvent.drop(screen.getByTestId("dropzone"), {
      dataTransfer: { files: [new File(["x"], "a.png")] },
    });
    expect(onFile).not.toHaveBeenCalled();
  });
});
