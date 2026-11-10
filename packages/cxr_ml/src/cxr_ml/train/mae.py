"""MAE pretraining on all non-test NIH images (labels discarded). Resumable.

    uv run python -m cxr_ml.train.mae --config configs/mae/vit_base.yaml [key=value ...]
    uv run python -m cxr_ml.train.mae --config configs/mae/vit_base.yaml --benchmark

Writes to $CXR_DATA_ROOT/runs/<name>/:
    last.pt (full state, every epoch), encoder_epXXXX.pt snapshots, encoder.pt (final),
    recon/epoch_XXXX.png reconstruction grids, metrics.jsonl, provenance.json, DONE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf

from cxr_ml.data.augment import AugmentConfig, augment, u8_to_float
from cxr_ml.data.loaders import GpuResidentLoader
from cxr_ml.models.mae import MAE, MAEConfig
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
)


def _mae_aug() -> AugmentConfig:
    return AugmentConfig(
        crop_scale_min=0.4,
        rotate_deg=5.0,
        translate=0.0,
        brightness=0.05,
        contrast=0.1,
        gamma=0.1,
    )


@dataclass
class MAETrainConfig:
    name: str = "mae_debug"
    seed: int = 0
    mae: MAEConfig = field(default_factory=MAEConfig)
    aug: AugmentConfig = field(default_factory=_mae_aug)
    include_splits: list[str] = field(default_factory=lambda: ["train", "val"])  # never "test"
    batch_size: int = 256
    epochs: int = 800
    warmup_epochs: int = 40
    blr: float = 1.5e-4  # lr = blr * batch_size / 256
    weight_decay: float = 0.05
    snapshot_epochs: list[int] = field(default_factory=lambda: [100, 200, 400, 800])
    recon_every: int = 20
    compile: bool = True
    tracking: str = "mlflow"


def load_config(argv: list[str]) -> tuple[MAETrainConfig, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, action="append", default=[])
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = OmegaConf.structured(MAETrainConfig)
    for c in a.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(c))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return OmegaConf.to_object(cfg), a.benchmark  # type: ignore[return-value]


def save_recon_grid(model: MAE, x: torch.Tensor, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = torch.Generator(device=x.device).manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        masked, recon = model.reconstruct(x, generator=g)
    n = x.shape[0]
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6.3))
    for i in range(n):
        for r, (img, title) in enumerate(
            ((x, "original"), (masked, "masked 75%"), (recon, "reconstruction"))
        ):
            ax = axes[r, i]
            ax.imshow(img[i, 0].float().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def run(cfg: MAETrainConfig, benchmark: bool = False) -> Path | None:
    if "test" in cfg.include_splits:
        raise ValueError("MAE pretraining must not see the test split")
    paths = DataPaths.from_env()
    run_dir = paths.runs / cfg.name
    if (run_dir / "DONE").exists() and not benchmark:
        print(f"{cfg.name}: already done")
        return run_dir
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")

    df = pl.read_parquet(paths.index("nih"))
    rows = df.filter(pl.col("split").is_in(cfg.include_splits))["row"].to_numpy()
    val_rows = df.filter(pl.col("split") == "val")["row"].to_numpy()
    s = cfg.mae.img_size
    images = GpuResidentLoader.load_to_device(paths.images("nih", s), df.height, s, dev)
    dummy = np.zeros((df.height, 1), dtype=np.uint8)
    loader = GpuResidentLoader(images, dummy, rows, cfg.batch_size, True, True, seed=cfg.seed)
    vis_rows = val_rows[:: max(1, len(val_rows) // 8)][:8].astype(np.int64)
    vis = u8_to_float(images[torch.as_tensor(vis_rows, device=dev)])

    model = MAE(cfg.mae).to(dev)
    lr = cfg.blr * cfg.batch_size / 256
    opt = torch.optim.AdamW(
        param_groups(model, cfg.weight_decay), lr=lr, betas=(0.9, 0.95), fused=True
    )
    fmodel = torch.compile(model) if cfg.compile else model
    spe = len(loader)
    total = cfg.epochs * spe
    warmup = cfg.warmup_epochs * spe

    def train_step(step: int) -> torch.Tensor:
        cur = cosine_lr(step, total, warmup, lr, min_ratio=0.0)
        for g in opt.param_groups:
            g["lr"] = cur * g["lr_scale"]
        b = next(it)
        x = augment(u8_to_float(b.x), cfg.aug)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _ = fmodel(x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    if benchmark:
        it = iter(loader)
        for i in range(20):
            train_step(i)
        torch.cuda.synchronize()
        t, k = time.perf_counter(), 100
        for i in range(k):
            train_step(i)
        torch.cuda.synchronize()
        ips = k * cfg.batch_size / (time.perf_counter() - t)
        est_h = cfg.epochs * len(rows) / ips / 3600
        mem = torch.cuda.max_memory_allocated() / 2**30
        res = {
            "arch": cfg.mae.arch,
            "batch_size": cfg.batch_size,
            "img_per_s": round(ips, 1),
            "images_per_epoch": len(rows),
            "epochs": cfg.epochs,
            "estimated_hours": round(est_h, 2),
            "peak_mem_gib": round(mem, 1),
        }
        print(json.dumps(res))
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    chash = config_hash(cfg_dict)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(cfg_dict)))
    start_epoch = 0
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["config_hash"] != chash:
            raise SystemExit("last.pt was written with a different config; refusing to resume")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"]
        torch.set_rng_state(ck["rng_cpu"])
        torch.cuda.set_rng_state(ck["rng_cuda"])
        loader.gen.set_state(ck["loader_rng"])
        print(f"resumed at epoch {start_epoch}")
    provenance = {
        "config": cfg_dict,
        "config_hash": chash,
        "git": git_info(),
        "env": env_info(),
        "n_images": len(rows),
        "splits": cfg.include_splits,
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    tracker = Tracker(run_dir, cfg.name, cfg_dict, backend=cfg.tracking, experiment="cxr-mae")
    print(f"{cfg.name}: {len(rows)} images, {spe} steps/epoch, {cfg.epochs} epochs, lr {lr:.2e}")

    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        it = iter(loader)
        losses = torch.zeros((), device=dev)
        te = time.time()
        for i in range(spe):
            losses += train_step(epoch * spe + i).detach()
        ep_loss = (losses / spe).item()
        ep = epoch + 1
        ips = spe * cfg.batch_size / (time.time() - te)
        tracker.log(
            {"train/loss": ep_loss, "train/img_per_s": ips, "train/lr": opt.param_groups[0]["lr"]},
            ep * spe,
        )
        eta = (cfg.epochs - ep) * (time.time() - te) / 3600
        print(
            f"epoch {ep}/{cfg.epochs} loss {ep_loss:.4f} {ips:.0f} img/s eta {eta:.1f}h", flush=True
        )
        if ep % cfg.recon_every == 0 or ep == 1 or ep == cfg.epochs:
            model.eval()
            save_recon_grid(model, vis, run_dir / "recon" / f"epoch_{ep:04d}.png")
            model.train()
        if ep in cfg.snapshot_epochs:
            atomic_save(
                {"encoder": model.encoder_state_dict(), "epoch": ep, **provenance},
                run_dir / f"encoder_ep{ep:04d}.pt",
            )
        atomic_save(
            {
                "model": model.state_dict(),
                "optim": opt.state_dict(),
                "epoch": ep,
                "config_hash": chash,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "loader_rng": loader.gen.get_state(),
            },
            last,
        )
    atomic_save(
        {"encoder": model.encoder_state_dict(), "epoch": cfg.epochs, **provenance},
        run_dir / "encoder.pt",
    )
    tracker.close()
    (run_dir / "summary.json").write_text(
        json.dumps({"wall_seconds_this_session": time.time() - t0})
    )
    (run_dir / "DONE").write_text("ok\n")
    return run_dir


def main() -> None:
    cfg, bench = load_config(sys.argv[1:])
    run(cfg, bench)


if __name__ == "__main__":
    main()
