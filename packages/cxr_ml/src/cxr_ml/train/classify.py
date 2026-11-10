"""Supervised training / fine-tuning of the multi-label classifier.

    uv run python -m cxr_ml.train.classify --config configs/sup/<cfg>.yaml [key=value ...]

Writes to $CXR_DATA_ROOT/runs/<name>/:
    config.yaml, best.pt, last.pt (resumable), metrics.jsonl, preds_{val,test}.parquet, DONE
Resumes automatically from last.pt if present and the resolved config hash matches.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from cxr_common.labels import NIH_LABELS
from cxr_ml.data.augment import AugmentConfig, augment, u8_to_float
from cxr_ml.data.loaders import GpuResidentLoader, MemmapLoader
from cxr_ml.eval.metrics import weighted_auroc
from cxr_ml.models.classifier import CXRClassifier, ModelConfig, build_classifier, is_vit
from cxr_ml.paths import DataPaths
from cxr_ml.train.common import (
    Tracker,
    atomic_save,
    config_hash,
    cosine_lr,
    env_info,
    git_info,
    param_groups,
    seed_everything,
    vit_layer_id,
)


@dataclass
class DataConfig:
    dataset: str = "nih"
    size: int = 224
    fraction: float = 1.0  # 1.0 | 0.1 | 0.01 (patient-level, precomputed nested subsets)
    batch_size: int = 128
    eval_batch_size: int = 256
    loader: str = "auto"  # auto | gpu | memmap


@dataclass
class OptimConfig:
    lr: float = 5e-4
    weight_decay: float = 0.05
    beta2: float = 0.999
    layer_decay: float | None = None  # ViT layer-wise LR decay (e.g. 0.75)
    warmup_frac: float = 0.05
    min_lr_ratio: float = 0.01
    grad_clip: float | None = 3.0


@dataclass
class ScheduleConfig:
    epochs: int = 30
    min_steps: int = 0  # floor on total steps (keeps tiny-fraction runs long enough)
    eval_every_steps: int = 0  # 0 = once per epoch
    early_stop_patience: int = 6  # in evaluations


@dataclass
class TrainConfig:
    name: str = "debug"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    aug: AugmentConfig = field(default_factory=AugmentConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    pos_weight: str = "sqrt"  # none | sqrt | full  (neg/pos ratio per class, clamped)
    compile: bool = True
    tracking: str = "mlflow"  # mlflow | wandb | none
    tags: dict[str, str] = field(default_factory=dict)


def frac_col(f: float) -> str:
    return f"in_frac_{f:g}".replace(".", "p")


def load_config(argv: list[str]) -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, action="append", default=[])
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = OmegaConf.structured(TrainConfig)
    for c in a.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(c))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]


def compute_pos_weight(y: np.ndarray, mode: str) -> torch.Tensor:
    pos = y.sum(0).astype(float)
    neg = len(y) - pos
    ratio = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    if mode == "none":
        ratio = np.ones_like(ratio)
    elif mode == "sqrt":
        ratio = np.sqrt(ratio)
    elif mode != "full":
        raise ValueError(mode)
    return torch.tensor(np.clip(ratio, 1.0, 100.0), dtype=torch.float32)


@torch.no_grad()
def predict(model: torch.nn.Module, loader: Any, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Logits for every row (in loader order) without augmentation. Pads the last batch so the
    compiled graph sees a single static shape."""
    model.eval()
    outs, rows = [], []
    bs = None
    for b in loader:
        x = u8_to_float(b.x)
        if x.shape[-1] != size:
            x = augment(x, AugmentConfig(enabled=False), out_size=size)
        n = x.shape[0]
        bs = bs or n
        if n < bs:
            x = torch.cat([x, x[-1:].expand(bs - n, -1, -1, -1)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(x.contiguous(memory_format=torch.channels_last)).float()
        outs.append(lo[:n].cpu())
        rows.append(b.rows)
    model.train()
    return torch.cat(outs).numpy(), torch.cat(rows).numpy()


def mean_auroc(logits: np.ndarray, y: np.ndarray) -> tuple[float, list[float]]:
    ones = np.ones((1, len(y)))
    per = [
        float(weighted_auroc(logits[:, k].astype(np.float64), y[:, k], ones)[0])
        for k in range(y.shape[1])
    ]
    return float(np.nanmean(per)), per


def save_preds(path: Path, df: pl.DataFrame, rows: np.ndarray, logits: np.ndarray) -> None:
    sub = df[rows.tolist()].select(
        ["row", "image_id", "patient_id", "view", "sex", "age", *NIH_LABELS]
    )
    sub = sub.with_columns(
        [pl.Series(f"logit_{n}", logits[:, k]) for k, n in enumerate(NIH_LABELS)]
    )
    sub.write_parquet(path)


def run(cfg: TrainConfig) -> Path:
    paths = DataPaths.from_env()
    run_dir = paths.runs / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE").exists():
        print(f"{cfg.name}: already done")
        return run_dir
    cfg_dict = asdict(cfg)
    chash = config_hash(cfg_dict)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(cfg_dict)))
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    # ---------------- data
    df = pl.read_parquet(paths.index(cfg.data.dataset))
    splits_meta = json.loads((paths.processed(cfg.data.dataset) / "splits.json").read_text())
    y_all = df.select(list(NIH_LABELS)).to_numpy().astype(np.uint8)
    train_mask = df["split"] == "train"
    if cfg.data.fraction < 1.0:
        train_mask = train_mask & df[frac_col(cfg.data.fraction)]
    rows_train = df.filter(train_mask)["row"].to_numpy()
    rows_val = df.filter(pl.col("split") == "val")["row"].to_numpy()
    rows_test = df.filter(pl.col("split") == "test")["row"].to_numpy()
    n, s = df.height, cfg.data.size
    img_path = paths.images(cfg.data.dataset, s)
    use_gpu = cfg.data.loader == "gpu" or (cfg.data.loader == "auto" and s <= 256)
    if use_gpu:
        images = GpuResidentLoader.load_to_device(img_path, n, s, dev)

        def make(rows: np.ndarray, bs: int, train: bool) -> Any:
            return GpuResidentLoader(images, y_all, rows, bs, train, train, seed=cfg.seed)
    else:

        def make(rows: np.ndarray, bs: int, train: bool) -> Any:
            return MemmapLoader(img_path, n, s, y_all, rows, bs, train, train, dev, seed=cfg.seed)

    train_loader = make(rows_train, cfg.data.batch_size, True)
    val_loader = make(rows_val, cfg.data.eval_batch_size, False)
    steps_per_epoch = len(train_loader)
    total_steps = max(cfg.schedule.epochs * steps_per_epoch, cfg.schedule.min_steps)
    eval_every = cfg.schedule.eval_every_steps or steps_per_epoch
    warmup = int(cfg.optim.warmup_frac * total_steps)

    # ---------------- model / optim
    model: CXRClassifier = build_classifier(cfg.model).to(dev)
    if not model.vit:
        model = model.to(memory_format=torch.channels_last)
    layer_fn = None
    n_layers = None
    if cfg.optim.layer_decay is not None and is_vit(cfg.model.arch):
        n_layers = len(model.backbone.blocks)  # type: ignore[arg-type]
        layer_fn = lambda name: vit_layer_id(name, n_layers)  # noqa: E731
    groups = param_groups(model, cfg.optim.weight_decay, cfg.optim.layer_decay, n_layers, layer_fn)
    opt = torch.optim.AdamW(groups, lr=cfg.optim.lr, betas=(0.9, cfg.optim.beta2), fused=True)
    pos_weight = compute_pos_weight(y_all[rows_train], cfg.pos_weight).to(dev)
    fmodel = torch.compile(model) if cfg.compile else model

    step, best, best_step, evals_since_best = 0, -math.inf, -1, 0
    last_ckpt = run_dir / "last.pt"
    if last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        if ck["config_hash"] != chash:
            raise SystemExit(f"{last_ckpt} was written with a different config; refusing to resume")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        step, best, best_step, evals_since_best = (
            ck["step"],
            ck["best"],
            ck["best_step"],
            ck["evals_since_best"],
        )
        torch.set_rng_state(ck["rng_cpu"])
        torch.cuda.set_rng_state(ck["rng_cuda"])
        train_loader.gen.set_state(ck["loader_rng"]) if hasattr(train_loader, "gen") else None
        print(f"resumed at step {step} (best val mAUROC {best:.4f} @ {best_step})")

    provenance = {
        "config": cfg_dict,
        "config_hash": chash,
        "git": git_info(),
        "env": env_info(),
        "data": {
            "split_digest": splits_meta["split_digest"],
            "index_digest": splits_meta["index_digest"],
            "n_train": len(rows_train),
            "n_val": len(rows_val),
            "train_positives": dict(
                zip(NIH_LABELS, y_all[rows_train].sum(0).tolist(), strict=True)
            ),
        },
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    tracker = Tracker(run_dir, cfg.name, cfg_dict, backend=cfg.tracking)
    print(
        f"{cfg.name}: {len(rows_train)} train imgs, {steps_per_epoch} steps/epoch, "
        f"{total_steps} total steps, eval every {eval_every}, warmup {warmup}"
    )

    t0, seen = time.time(), 0
    stop = False
    while step < total_steps and not stop:
        for b in train_loader:
            lr = cosine_lr(step, total_steps, warmup, cfg.optim.lr, cfg.optim.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_scale"]
            x = augment(u8_to_float(b.x), cfg.aug, out_size=s)
            if not model.vit:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fmodel(x).float()
            loss = F.binary_cross_entropy_with_logits(logits, b.y, pos_weight=pos_weight)
            loss.backward()
            gnorm = (
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                if cfg.optim.grad_clip
                else torch.tensor(0.0)
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            seen += x.shape[0]
            if step % 50 == 0:
                tracker.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/grad_norm": float(gnorm),
                        "train/img_per_s": seen / (time.time() - t0),
                    },
                    step,
                )
            if step % eval_every == 0 or step == total_steps:
                logits_val, vrows = predict(fmodel, val_loader, s)
                m, per = mean_auroc(logits_val, y_all[vrows])
                improved = m > best
                if improved:
                    best, best_step, evals_since_best = m, step, 0
                    atomic_save(
                        {
                            "model": model.state_dict(),
                            "step": step,
                            "val_mean_auroc": m,
                            **provenance,
                        },
                        run_dir / "best.pt",
                    )
                else:
                    evals_since_best += 1
                tracker.log(
                    {
                        "val/mean_auroc": m,
                        **{f"val/auroc_{k}": v for k, v in zip(NIH_LABELS, per, strict=True)},
                    },
                    step,
                )
                print(
                    f"step {step}/{total_steps} loss {loss.item():.4f} val mAUROC {m:.4f} "
                    f"best {best:.4f}@{best_step} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
                atomic_save(
                    {
                        "model": model.state_dict(),
                        "optim": opt.state_dict(),
                        "step": step,
                        "best": best,
                        "best_step": best_step,
                        "evals_since_best": evals_since_best,
                        "config_hash": chash,
                        "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state(),
                        "loader_rng": train_loader.gen.get_state()
                        if hasattr(train_loader, "gen")
                        else None,
                    },
                    last_ckpt,
                )
                if evals_since_best >= cfg.schedule.early_stop_patience:
                    print(f"early stop at step {step}")
                    stop = True
            if step >= total_steps or stop:
                break

    # ---------------- final predictions with the best checkpoint
    model.load_state_dict(
        torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["model"]
    )
    test_loader = make(rows_test, cfg.data.eval_batch_size, False)
    tm = float("nan")
    for split, loader in (("val", val_loader), ("test", test_loader)):
        lg, rows = predict(fmodel, loader, s)
        save_preds(run_dir / f"preds_{split}.parquet", df, rows, lg)
        if split == "test":
            tm, _ = mean_auroc(lg, y_all[rows])
    tracker.log({"best/val_mean_auroc": best, "best/step": best_step, "test/mean_auroc": tm}, step)
    tracker.close()
    summary = {
        "best_val_mean_auroc": best,
        "best_step": best_step,
        "test_mean_auroc": tm,
        "steps": step,
        "wall_seconds": time.time() - t0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    last_ckpt.unlink(missing_ok=True)  # optimizer state no longer needed
    (run_dir / "DONE").write_text("ok\n")
    print(json.dumps(summary))
    return run_dir


def main() -> None:
    run(load_config(sys.argv[1:]))


if __name__ == "__main__":
    main()
