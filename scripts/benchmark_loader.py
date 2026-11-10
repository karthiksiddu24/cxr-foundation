#!/usr/bin/env python3
"""Measure input-pipeline throughput (images/sec) for each stage, and full training steps.

Writes docs/benchmarks/loader.json (measured on this machine; rerun to refresh).

Usage: uv run scripts/benchmark_loader.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch

from cxr_ml.data.augment import AugmentConfig, augment, u8_to_float
from cxr_ml.data.loaders import GpuResidentLoader, MemmapLoader
from cxr_ml.models.classifier import ModelConfig, build_classifier
from cxr_ml.paths import DataPaths

REPO = Path(__file__).resolve().parents[1]


def _decode(p: str) -> int:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return int(img.shape[0])


def bench_png(paths: list[str], workers: int) -> float:
    t = time.perf_counter()
    if workers == 1:
        for p in paths:
            _decode(p)
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            pool.map(_decode, paths, chunksize=16)
    return len(paths) / (time.perf_counter() - t)


def bench_iter(loader, n_batches: int, fn=None) -> float:  # type: ignore[no-untyped-def]
    it = iter(loader)
    next(it)  # warm
    torch.cuda.synchronize()
    t, n = time.perf_counter(), 0
    for _ in range(n_batches):
        b = next(it)
        if fn:
            fn(b)
        n += b.x.shape[0]
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t)


def bench_train(
    arch: str, size: int, bs: int, steps: int, images: torch.Tensor, labels: np.ndarray
) -> float:
    model = build_classifier(ModelConfig(arch=arch, init="random", img_size=size)).cuda()
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    cmodel = torch.compile(model)
    loader = GpuResidentLoader(images, labels, np.arange(len(labels)), bs, True, True)
    cfg = AugmentConfig()
    it = iter(loader)

    def step() -> None:
        b = next(it)
        x = augment(u8_to_float(b.x), cfg).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(cmodel(x).float(), b.y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(15):  # compile + warmup
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    return steps * bs / (time.perf_counter() - t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    paths = DataPaths.from_env()
    df = pl.read_parquet(paths.index("nih"))
    n = df.height
    labels = df.select(
        df.columns[df.columns.index("Atelectasis") : df.columns.index("Hernia") + 1]
    ).to_numpy()
    res: dict[str, object] = {"gpu": torch.cuda.get_device_name(0), "cpu_count": mp.cpu_count()}

    rng = np.random.default_rng(0)
    sample = [
        str(paths.nih_raw / "images" / df["image_id"][int(i)])
        for i in rng.choice(n, 800, replace=False)
    ]
    res["png_decode_1proc"] = bench_png(sample[:200], 1)
    res["png_decode_8proc"] = bench_png(sample, 8)

    dev = torch.device("cuda")
    t = time.perf_counter()
    imgs224 = GpuResidentLoader.load_to_device(paths.images("nih", 224), n, 224, dev)
    res["load_224_to_gpu_seconds"] = time.perf_counter() - t
    gl = GpuResidentLoader(imgs224, labels, np.arange(n), 256, True, True)
    res["gpu_resident_224_gather"] = bench_iter(gl, 100)
    cfg = AugmentConfig()
    res["gpu_resident_224_gather_plus_augment"] = bench_iter(
        gl, 100, lambda b: augment(u8_to_float(b.x), cfg)
    )

    ml = MemmapLoader(
        paths.images("nih", 512), n, 512, labels, np.arange(n), 64, True, True, dev, workers=6
    )
    res["memmap_512_6workers_random_batches"] = bench_iter(ml, 40 if args.quick else 150)

    steps = 20 if args.quick else 60
    res["train_step_densenet121_224_bs128"] = bench_train(
        "densenet121", 224, 128, steps, imgs224, labels
    )
    res["train_step_vit_base_224_bs128"] = bench_train(
        "vit_base_patch16_224", 224, 128, steps, imgs224, labels
    )
    res = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items()}
    out = REPO / "docs" / "benchmarks" / "loader.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
