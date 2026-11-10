"""Multi-label CXR classifier with in-graph, class-specific activation maps.

Contract: input float32 [B, 1, S, S] in [0, 1] (intensity normalization happens inside).

Both backbone families end in `global average pool -> linear`, which makes class activation maps
exact and gradient-free:

- CNN (DenseNet/ResNet):  logits_k = mean_{h,w} (W_k . F[:, h, w]) + b_k
  so CAM_k = W_k . F + b_k, and Grad-CAM at the last conv layer equals CAM up to scale.
- ViT (avg-pool head, final LayerNorm applied per token *before* pooling, i.e. fc_norm=False):
  logits_k = mean_i (W_k . norm(t_i)) + b_k, so per-patch logits form a 14x14 map whose mean is
  exactly the logit. (With the usual fc_norm after pooling this decomposition would not hold.)

Hence `mean(cams, (2,3)) == logits` for every architecture, which tests assert.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import timm
import torch
from torch import nn

# ImageNet RGB stats averaged to one channel; used for all inits so preprocessing is identical.
GRAY_MEAN = 0.449
GRAY_STD = 0.226

# ImageNet-1k-only weights for every family, so "ImageNet init" means the same data scale.
IMAGENET_TAGS: dict[str, str] = {
    "densenet121": "densenet121.tv_in1k",
    "resnet50": "resnet50.tv_in1k",
    "vit_small_patch16_224": "vit_small_patch16_224.augreg_in1k",
    "vit_base_patch16_224": "vit_base_patch16_224.augreg_in1k",
}


def is_vit(arch: str) -> bool:
    return arch.startswith("vit_")


@dataclass
class ModelConfig:
    arch: str = "densenet121"
    init: str = "imagenet"  # random | imagenet | ssl
    ssl_checkpoint: str | None = None  # path to encoder weights when init == "ssl"
    num_classes: int = 14
    img_size: int = 224
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1  # ViT only


class CXRClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, vit: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.vit = vit
        self.register_buffer("mean", torch.tensor(GRAY_MEAN), persistent=False)
        self.register_buffer("std", torch.tensor(GRAY_STD), persistent=False)

    @property
    def head(self) -> nn.Linear:
        h = self.backbone.get_classifier()
        assert isinstance(h, nn.Linear)
        return h

    def _spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, h, w] features right before global average pooling."""
        xn = (x - self.mean) / self.std
        f = self.backbone.forward_features(xn)
        if self.vit:
            npt = self.backbone.num_prefix_tokens
            patch = f[:, npt:]
            b, n, c = patch.shape
            s = round(n**0.5)
            return patch.transpose(1, 2).reshape(b, c, s, s)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._spatial_features(x)
        return self.head(f.mean(dim=(2, 3)))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled embedding [B, C] (used for OOD scoring)."""
        return self._spatial_features(x).mean(dim=(2, 3))

    def forward_with_cam(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits [B,K], cams [B,K,h,w] in logit units, pooled features [B,C])."""
        f = self._spatial_features(x)
        w, b = self.head.weight, self.head.bias
        cams = torch.einsum("kc,bchw->bkhw", w, f) + b.view(1, -1, 1, 1)
        pooled = f.mean(dim=(2, 3))
        return cams.mean(dim=(2, 3)), cams, pooled


def build_backbone(cfg: ModelConfig) -> nn.Module:
    kwargs: dict[str, object] = {
        "num_classes": cfg.num_classes,
        "in_chans": 1,
        "drop_rate": cfg.drop_rate,
    }
    if is_vit(cfg.arch):
        kwargs.update(
            img_size=cfg.img_size,
            global_pool="avg",
            fc_norm=False,
            drop_path_rate=cfg.drop_path_rate,
        )
    if cfg.init == "imagenet":
        # timm sums RGB first-conv weights to adapt to in_chans=1.
        return timm.create_model(IMAGENET_TAGS[cfg.arch], pretrained=True, **kwargs)
    model = timm.create_model(cfg.arch, pretrained=False, **kwargs)
    if cfg.init == "ssl":
        if not cfg.ssl_checkpoint:
            raise ValueError("init=ssl requires ssl_checkpoint")
        load_encoder(model, Path(cfg.ssl_checkpoint))
    elif cfg.init != "random":
        raise ValueError(f"unknown init {cfg.init!r}")
    return model


def load_encoder(model: nn.Module, path: Path) -> None:
    """Load SSL encoder weights; only the classification head may be missing."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    sd = blob.get("encoder", blob)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    bad_missing = set(missing) - allowed_missing
    if bad_missing or unexpected:
        raise RuntimeError(
            f"encoder load mismatch: missing={sorted(bad_missing)} unexpected={unexpected}"
        )


def build_classifier(cfg: ModelConfig) -> CXRClassifier:
    return CXRClassifier(build_backbone(cfg), vit=is_vit(cfg.arch))
