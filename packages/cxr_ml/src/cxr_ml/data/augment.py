"""Batched GPU augmentation for grayscale chest X-rays.

All geometric transforms are composed into one affine matrix per sample and applied with a single
`grid_sample`, so a batch costs one kernel regardless of how many transforms are enabled.

CXR-specific choices (see docs/augmentation.md):
- No color jitter (images are single-channel).
- No horizontal flip by default: chest anatomy is left-right asymmetric (heart, aortic arch,
  gastric bubble), radiographs carry L/R markers, and some findings are side-specific.
- Mild rotation/scale only; patient positioning varies, but extreme warps are not realistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AugmentConfig:
    enabled: bool = True
    crop_scale_min: float = 0.7  # min area fraction for random resized crop
    crop_scale_max: float = 1.0
    aspect_min: float = 0.85
    aspect_max: float = 1.0 / 0.85
    rotate_deg: float = 10.0
    translate: float = 0.05  # extra shift as fraction of half-size
    hflip_p: float = 0.0
    brightness: float = 0.1  # additive, in [0,1] intensity units
    contrast: float = 0.15  # multiplicative around mean, +-
    gamma: float = 0.15  # log-uniform gamma in [1/(1+g), 1+g]
    noise_std: float = 0.0


def u8_to_float(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,S,S] -> float32 [B,1,S,S] in [0,1] (matches cxr_common.preprocess.to_model_input)."""
    return x_u8.unsqueeze(1).float().div_(255.0)


def _uniform(
    b: int, lo: float, hi: float, device: torch.device, g: torch.Generator | None
) -> torch.Tensor:
    return torch.rand(b, device=device, generator=g) * (hi - lo) + lo


def augment(
    x: torch.Tensor,
    cfg: AugmentConfig,
    out_size: int | None = None,
    g: torch.Generator | None = None,
) -> torch.Tensor:
    """x: float [B,1,S,S] in [0,1] -> augmented float [B,1,out,out] in [0,1]."""
    if not cfg.enabled:
        if out_size and out_size != x.shape[-1]:
            return F.interpolate(
                x, size=(out_size, out_size), mode="bilinear", antialias=True, align_corners=False
            )
        return x
    b, dev = x.shape[0], x.device
    out = out_size or x.shape[-1]

    # Random resized crop: side fractions from area scale and aspect ratio.
    area = _uniform(b, cfg.crop_scale_min, cfg.crop_scale_max, dev, g)
    log_r = _uniform(b, math.log(cfg.aspect_min), math.log(cfg.aspect_max), dev, g)
    r = torch.exp(log_r)
    sw = torch.sqrt(area * r).clamp(max=1.0)
    sh = torch.sqrt(area / r).clamp(max=1.0)
    # Crop center: anywhere that keeps the crop inside the image, plus small translation.
    cx = (torch.rand(b, device=dev, generator=g) * 2 - 1) * (1 - sw) + _uniform(
        b, -cfg.translate, cfg.translate, dev, g
    )
    cy = (torch.rand(b, device=dev, generator=g) * 2 - 1) * (1 - sh) + _uniform(
        b, -cfg.translate, cfg.translate, dev, g
    )
    th = _uniform(b, -cfg.rotate_deg, cfg.rotate_deg, dev, g) * (math.pi / 180)
    flip = torch.where(torch.rand(b, device=dev, generator=g) < cfg.hflip_p, -1.0, 1.0)

    cos, sin = torch.cos(th), torch.sin(th)
    theta = torch.stack(
        [
            torch.stack([sw * cos * flip, -sh * sin, cx], dim=1),
            torch.stack([sw * sin * flip, sh * cos, cy], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(theta, [b, 1, out, out], align_corners=False)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Intensity: gamma, contrast, brightness (order chosen so gamma acts on [0,1] input).
    if cfg.gamma > 0:
        lg = math.log(1 + cfg.gamma)
        gam = torch.exp(_uniform(b, -lg, lg, dev, g)).view(b, 1, 1, 1)
        y = y.clamp(min=1e-6).pow(gam)
    if cfg.contrast > 0:
        c = _uniform(b, 1 - cfg.contrast, 1 + cfg.contrast, dev, g).view(b, 1, 1, 1)
        m = y.mean(dim=(2, 3), keepdim=True)
        y = (y - m) * c + m
    if cfg.brightness > 0:
        y = y + _uniform(b, -cfg.brightness, cfg.brightness, dev, g).view(b, 1, 1, 1)
    if cfg.noise_std > 0:
        y = y + torch.randn(y.shape, device=dev, generator=g) * cfg.noise_std
    return y.clamp_(0.0, 1.0)
