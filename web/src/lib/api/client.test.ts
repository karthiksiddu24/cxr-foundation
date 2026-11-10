import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, getModelInfo } from "./client";

afterEach(() => vi.restoreAllMocks());

describe("getModelInfo", () => {
  it("maps RFC 7807 problem responses to ApiError", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          code: "not_ready",
          detail: "loading",
          title: "x",
          status: 503,
          request_id: "r1",
        }),
        {
          status: 503,
        },
      ),
    );
    const err = await getModelInfo("http://api").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 503, code: "not_ready", requestId: "r1", retryable: true });
  });

  it("reports network failures as retryable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"));
    const err = (await getModelInfo("http://api").catch((e: unknown) => e)) as ApiError;
    expect(err.code).toBe("network_error");
    expect(err.retryable).toBe(true);
  });

  it("treats 4xx validation errors as not retryable", () => {
    expect(new ApiError(422, "image_too_small", "m").retryable).toBe(false);
  });
});
