import { describe, expect, it } from "vitest";

import { MAX_UPLOAD_BYTES, precheck, sniffFile } from "./upload";

const file = (bytes: number[] | Uint8Array, name = "x") => new File([new Uint8Array(bytes)], name);

describe("sniffFile", () => {
  it("detects PNG, JPEG and DICOM by magic bytes regardless of name", async () => {
    expect(
      await sniffFile(file([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0], "a.dcm")),
    ).toBe("png");
    expect(await sniffFile(file([0xff, 0xd8, 0xff, 0xe0], "a.png"))).toBe("jpeg");
    const d = new Uint8Array(140);
    d.set([0x44, 0x49, 0x43, 0x4d], 128);
    expect(await sniffFile(file(d, "a.jpg"))).toBe("dicom");
    expect(await sniffFile(file([1, 2, 3], "a.png"))).toBeNull();
  });
});

describe("precheck", () => {
  it("rejects empty, oversized and unknown files with specific messages", async () => {
    expect(await precheck(file([]))).toEqual({ ok: false, message: "That file is empty." });
    const big = { size: MAX_UPLOAD_BYTES + 1, slice: () => new Blob() } as unknown as File;
    const r = await precheck(big);
    expect(r.ok).toBe(false);
    expect(!r.ok && r.message).toMatch(/limit is 25 MB/);
    expect((await precheck(file([1, 2, 3]))).ok).toBe(false);
  });
});
