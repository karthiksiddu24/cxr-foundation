"""Decode-once preprocessing into uint8 memmaps ([N, S, S]) for each target size.

Workers open the memmaps in r+ mode and write disjoint rows, so no data passes through pipes
except row indices. A `.done` marker is written only after every row succeeded.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from cxr_common.preprocess import letterbox, to_gray_uint8

_STATE: dict[str, object] = {}


def _init(paths: Sequence[str], out_files: dict[int, str], n: int) -> None:
    cv2.setNumThreads(1)
    _STATE["paths"] = paths
    _STATE["mms"] = {
        s: np.memmap(f, dtype=np.uint8, mode="r+", shape=(n, s, s)) for s, f in out_files.items()
    }


def _work(rows: list[int]) -> tuple[int, list[str]]:
    paths: Sequence[str] = _STATE["paths"]  # type: ignore[assignment]
    mms: dict[int, np.memmap] = _STATE["mms"]  # type: ignore[assignment]
    errors: list[str] = []
    for r in rows:
        img = cv2.imread(paths[r], cv2.IMREAD_UNCHANGED)
        if img is None:
            errors.append(paths[r])
            continue
        g = to_gray_uint8(img)
        for s, mm in mms.items():
            mm[r] = letterbox(g, s)[0]
    for mm in mms.values():
        mm.flush()
    return len(rows), errors


def write_memmaps(
    image_paths: Sequence[Path],
    out_dir: Path,
    sizes: Sequence[int],
    workers: int = 8,
    chunk: int = 256,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    out_files = {s: str(out_dir / f"img_{s}.u8") for s in sizes}
    for s, f in out_files.items():
        np.memmap(f, dtype=np.uint8, mode="w+", shape=(n, s, s)).flush()
    jobs = [list(range(i, min(i + chunk, n))) for i in range(0, n, chunk)]
    t0 = time.time()
    done, errors = 0, []
    with mp.get_context("fork").Pool(
        workers, initializer=_init, initargs=([str(p) for p in image_paths], out_files, n)
    ) as pool:
        for k, errs in pool.imap_unordered(_work, jobs):
            done += k
            errors.extend(errs)
            if done % (chunk * 40) < chunk or done == n:
                rate = done / (time.time() - t0)
                print(f"  {done}/{n} images  {rate:.0f} img/s", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} images failed to decode, e.g. {errors[:3]}")
    meta = {
        "n": n,
        "sizes": list(sizes),
        "dtype": "uint8",
        "layout": "[N, S, S] letterboxed (long side = S, centered, zero pad)",
        "seconds": round(time.time() - t0, 1),
    }
    for s in sizes:
        (out_dir / f"img_{s}.json").write_text(json.dumps({**meta, "size": s}, indent=2))
    return meta


def open_images(path: Path, n: int, size: int) -> np.memmap:
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(n, size, size))
