"""Training utilities shared by supervised, fine-tuning and MAE runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[5]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (seeds third-party code using the global RNG)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain")),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
    }


def env_info() -> dict[str, Any]:
    import timm

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.node(),
    }


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def atomic_save(obj: Any, path: Path) -> None:
    """torch.save to a temp file then rename, so a crash never leaves a truncated checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def cosine_lr(step: int, total: int, warmup: int, base: float, min_ratio: float = 0.01) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def param_groups(
    model: nn.Module,
    weight_decay: float,
    layer_decay: float | None = None,
    num_layers: int | None = None,
    layer_id_fn: Any = None,
) -> list[dict[str, Any]]:
    """AdamW param groups: no weight decay on biases/norms/embeddings; optional layer-wise LR decay.

    Each group carries `lr_scale`, multiplied into the scheduled LR every step.
    """
    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = p.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name
        lid = layer_id_fn(name) if (layer_decay is not None and layer_id_fn) else 0
        scale = layer_decay ** ((num_layers or 0) + 1 - lid) if layer_decay is not None else 1.0
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = {
                "params": [],
                "weight_decay": 0.0 if no_wd else weight_decay,
                "lr_scale": scale,
            }
        groups[key]["params"].append(p)
    return list(groups.values())


def vit_layer_id(name: str, num_layers: int) -> int:
    """Layer index for layer-wise LR decay (BEiT/MAE convention). Head = num_layers + 1."""
    name = name.removeprefix("backbone.")
    if name.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_layers + 1


class Tracker:
    """Minimal experiment tracker: always writes metrics.jsonl; optionally MLflow and/or W&B."""

    def __init__(
        self,
        run_dir: Path,
        run_name: str,
        config: dict[str, Any],
        backend: str = "mlflow",
        experiment: str = "cxr-foundation",
    ) -> None:
        self.run_dir = run_dir
        self.backend = backend
        self.jsonl = (run_dir / "metrics.jsonl").open("a")
        self._mlflow: Any = None
        self._wandb: Any = None
        if backend == "mlflow":
            # The file store is in "maintenance mode" in MLflow 3.x but works fine on one machine
            # and avoids running a database server.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            import mlflow

            root = Path(os.environ.get("CXR_DATA_ROOT", "/data/cxr"))
            mlflow.set_tracking_uri(
                os.environ.get("MLFLOW_TRACKING_URI", f"file://{root}/mlflow/runs")
            )
            (root / "mlflow").mkdir(parents=True, exist_ok=True)
            if mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(
                    experiment, artifact_location=f"file://{root}/mlflow/artifacts/{experiment}"
                )
            mlflow.set_experiment(experiment)
            mlflow.start_run(run_name=run_name)
            flat = _flatten(config)
            mlflow.log_params({k: str(v)[:500] for k, v in list(flat.items())[:400]})
            self._mlflow = mlflow
        elif backend == "wandb":
            import wandb

            self._wandb = wandb.init(
                project=experiment,
                name=run_name,
                config=config,
                dir=str(run_dir),
                mode=os.environ.get("WANDB_MODE", "offline"),  # type: ignore[arg-type]
            )
        elif backend != "none":
            raise ValueError(f"unknown tracking backend {backend!r}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        rec = {"step": step, "time": time.time(), **metrics}
        self.jsonl.write(json.dumps(rec) + "\n")
        self.jsonl.flush()
        if self._mlflow:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
        if self._wandb:
            self._wandb.log(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        if self._mlflow:
            self._mlflow.log_artifact(str(path))
        if self._wandb:
            self._wandb.save(str(path), base_path=str(path.parent))

    def close(self) -> None:
        self.jsonl.close()
        if self._mlflow:
            self._mlflow.end_run()
        if self._wandb:
            self._wandb.finish()


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def count_params(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)
