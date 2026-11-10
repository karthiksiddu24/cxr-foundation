"""Heatmap localization against radiologist bounding boxes.

    uv run python -m cxr_ml.explain.localization <run> [--dataset nih|vindr] [--method cam|rollout]

Boxes are mapped from original-image pixels into model-input space through the same letterbox
geometry used for preprocessing. Metrics per finding (with image-level bootstrap 95% CIs):

- pointing game: is the heatmap's maximum inside a box of that finding?
  Baselines: (a) image center, (b) a uniformly random point over the image content
  (expected hit rate = box area / content area). A heatmap is only informative if it beats both.
- IoU: heatmap thresholded at 50% of its per-image range vs. the union of that finding's boxes.

Heatmaps for CAM are class-specific; attention rollout (ViT only) is class-agnostic and is
reported for comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from cxr_common.labels import NIH_BBOX_NAME_TO_LABEL, NIH_LABELS, VINDR_MAP
from cxr_ml.eval.metrics import F64
from cxr_ml.models.classifier import CXRClassifier, ModelConfig, build_classifier
from cxr_ml.paths import DataPaths


@dataclass
class BoxImage:
    image_id: str
    finding: str  # NIH output label scored for this box set
    boxes: list[tuple[float, float, float, float]]  # x0,y0,x1,y1 in model-input pixels
    pixels: np.ndarray  # uint8 [S,S] letterboxed


def load_run_model(run: str) -> tuple[CXRClassifier, dict[str, Any]]:
    blob = torch.load(
        DataPaths.from_env().runs / run / "best.pt", map_location="cpu", weights_only=False
    )
    mcfg = blob["config"]["model"]
    model = build_classifier(ModelConfig(**{**mcfg, "init": "random", "ssl_checkpoint": None}))
    model.load_state_dict(blob["model"])
    return model.eval(), blob["config"]


def nih_box_images(size: int) -> list[BoxImage]:
    """NIH BBox_List_2017 boxes on test-split images (original images are 1024x1024)."""
    paths = DataPaths.from_env()
    idx = pl.read_parquet(paths.index("nih"))
    raw = pl.read_csv(paths.nih_raw / "BBox_List_2017.csv", infer_schema_length=0)
    raw = raw.select(
        pl.col("Image Index").alias("image_id"),
        pl.col("Finding Label").alias("name"),
        pl.col("Bbox [x").cast(pl.Float64).alias("x"),
        pl.col("y").cast(pl.Float64).alias("y"),
        pl.col("w").cast(pl.Float64).alias("w"),
        pl.col("h]").cast(pl.Float64).alias("h"),
    )
    raw = raw.join(idx.select("image_id", "row", "split", "orig_w", "orig_h"), on="image_id")
    # Safeguard: localization must only use held-out images (all 984 boxes are on test images).
    raw = raw.filter(pl.col("split") == "test")
    mm = np.memmap(
        paths.images("nih", size), dtype=np.uint8, mode="r", shape=(idx.height, size, size)
    )
    out: list[BoxImage] = []
    for (img_id, name), g in raw.group_by(["image_id", "name"], maintain_order=True):
        label = NIH_BBOX_NAME_TO_LABEL.get(str(name))
        if label is None:
            continue
        # BBox coordinates are given on the 1024x1024 released images.
        s = size / 1024.0
        boxes = [
            (r["x"] * s, r["y"] * s, (r["x"] + r["w"]) * s, (r["y"] + r["h"]) * s)
            for r in g.iter_rows(named=True)
        ]
        out.append(BoxImage(str(img_id), label, boxes, np.array(mm[int(g["row"][0])])))
    return out


def vindr_box_images(size: int) -> list[BoxImage]:
    """VinDr-CXR test boxes (requires prepared data; see scripts/prep_vindr.py)."""
    paths = DataPaths.from_env()
    proc = paths.processed("vindr")
    meta = pl.read_parquet(proc / "boxes.parquet")
    idx = pl.read_parquet(proc / "index.parquet")
    mm = np.memmap(
        proc / f"img_{size}.u8", dtype=np.uint8, mode="r", shape=(idx.height, size, size)
    )
    vmap = {m.target: m.nih for m in VINDR_MAP}
    out = []
    for (img_id, cls), g in meta.group_by(["image_id", "class_name"], maintain_order=True):
        if cls not in vmap:
            continue
        row = idx.filter(pl.col("image_id") == img_id).row(0, named=True)
        k = size / row["letterbox_size"]
        boxes = [
            (r["mx0"] * k, r["my0"] * k, r["mx1"] * k, r["my1"] * k)
            for r in g.iter_rows(named=True)
        ]
        for nih in vmap[str(cls)]:  # approximate mappings score each mapped output
            out.append(BoxImage(str(img_id), nih, boxes, np.array(mm[row["row"]])))
    return out


@torch.no_grad()
def cam_maps(model: CXRClassifier, items: list[BoxImage], dev: torch.device) -> list[np.ndarray]:
    m = model.to(dev)
    maps = []
    for i in range(0, len(items), 64):
        batch = items[i : i + 64]
        x = torch.from_numpy(np.stack([b.pixels for b in batch])).to(dev).unsqueeze(1).float() / 255
        _, cams, _ = m.forward_with_cam(x)
        for j, b in enumerate(batch):
            maps.append(cams[j, NIH_LABELS.index(b.finding)].float().cpu().numpy())
    return maps


@torch.no_grad()
def rollout_maps(
    model: CXRClassifier, items: list[BoxImage], dev: torch.device
) -> list[np.ndarray]:
    """Attention rollout (Abnar & Zuidema 2020): head-averaged attention + identity, multiplied
    through layers; the CLS row over patch tokens gives a class-agnostic map."""
    if not model.vit:
        raise ValueError("attention rollout needs a ViT")
    vit = model.backbone
    attn_store: list[torch.Tensor] = []
    hooks = []
    for blk in vit.blocks:
        blk.attn.fused_attn = False  # materialize attention weights
        hooks.append(
            blk.attn.attn_drop.register_forward_hook(lambda _m, _i, o: attn_store.append(o))
        )
    m = model.to(dev)
    maps = []
    try:
        for i in range(0, len(items), 32):
            batch = items[i : i + 32]
            attn_store.clear()
            x = (
                torch.from_numpy(np.stack([b.pixels for b in batch])).to(dev).unsqueeze(1).float()
                / 255
            )
            m(x)
            n = attn_store[0].shape[-1]
            eye = torch.eye(n, device=dev)
            joint = eye.expand(len(batch), n, n)
            for a in attn_store:
                a = a.mean(dim=1) + eye
                a = a / a.sum(dim=-1, keepdim=True)
                joint = a @ joint
            npt = vit.num_prefix_tokens
            g = round((n - npt) ** 0.5)
            cls_to_patch = joint[:, 0, npt:].reshape(len(batch), g, g)
            maps.extend(cls_to_patch.float().cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
        for blk in vit.blocks:
            blk.attn.fused_attn = True
    return maps


def _upsample(cam: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(cam.astype(np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def _box_mask(boxes: list[tuple[float, float, float, float]], size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        mask[
            max(0, int(np.floor(y0))) : min(size, int(np.ceil(y1))),
            max(0, int(np.floor(x0))) : min(size, int(np.ceil(x1))),
        ] = True
    return mask


def score_item(cam: np.ndarray, item: BoxImage, size: int) -> dict[str, float]:
    up = _upsample(cam, size)
    mask = _box_mask(item.boxes, size)
    y, x = np.unravel_index(int(np.argmax(up)), up.shape)
    hit = float(mask[y, x])
    center_hit = float(mask[size // 2, size // 2])
    content = item.pixels > 0
    random_hit = float(mask[content].mean()) if content.any() else float(mask.mean())
    lo, hi = float(up.min()), float(up.max())
    pred = up >= lo + 0.5 * (hi - lo) if hi > lo else np.zeros_like(mask)
    inter = float((pred & mask).sum())
    union = float((pred | mask).sum())
    return {
        "hit": hit,
        "center_hit": center_hit,
        "random_hit": random_hit,
        "iou": inter / union if union else 0.0,
    }


def _boot_ci(v: F64, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(run_name: str, dataset: str, method: str) -> Path:
    model, cfg = load_run_model(run_name)
    size = cfg["data"]["size"]
    items = nih_box_images(size) if dataset == "nih" else vindr_box_images(size)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps = cam_maps(model, items, dev) if method == "cam" else rollout_maps(model, items, dev)
    rows = [
        {"image_id": it.image_id, "finding": it.finding, **score_item(mp, it, size)}
        for it, mp in zip(items, maps, strict=True)
    ]
    df = pl.DataFrame(rows)
    out_dir = DataPaths.from_env().eval_artifacts / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / f"localization_{dataset}_{method}.parquet")
    summary: dict[str, Any] = {
        "run": run_name,
        "dataset": dataset,
        "method": method,
        "input_size": size,
        "n_items": len(rows),
        "per_finding": {},
    }
    for (finding,), g in df.group_by(["finding"], maintain_order=True):
        hits = g["hit"].to_numpy().astype(np.float64)
        ious = g["iou"].to_numpy().astype(np.float64)
        summary["per_finding"][str(finding)] = {
            "n": g.height,
            "pointing_game": float(hits.mean()),
            "pointing_game_ci": _boot_ci(hits),
            "center_baseline": float(g["center_hit"].mean()),
            "random_baseline": float(g["random_hit"].mean()),
            "mean_iou": float(ious.mean()),
            "mean_iou_ci": _boot_ci(ious),
        }
    allh = df["hit"].to_numpy().astype(np.float64)
    summary["overall"] = {
        "pointing_game": float(allh.mean()),
        "pointing_game_ci": _boot_ci(allh),
        "center_baseline": float(df["center_hit"].mean()),
        "random_baseline": float(df["random_hit"].mean()),
        "mean_iou": float(df["iou"].mean()),
    }
    path = out_dir / f"localization_{dataset}_{method}.json"
    path.write_text(json.dumps(summary, indent=2))
    o = summary["overall"]
    print(
        f"{run_name} {dataset}/{method}: pointing {o['pointing_game']:.3f} "
        f"(center {o['center_baseline']:.3f}, random {o['random_baseline']:.3f}), IoU {o['mean_iou']:.3f}, n={len(rows)}"
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dataset", default="nih", choices=["nih", "vindr"])
    ap.add_argument("--method", default="cam", choices=["cam", "rollout"])
    a = ap.parse_args()
    for r in a.runs:
        run(r, a.dataset, a.method)


if __name__ == "__main__":
    main()
