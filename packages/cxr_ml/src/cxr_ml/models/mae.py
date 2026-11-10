"""Masked Autoencoder (He et al., 2022) for single-channel chest X-rays.

The encoder is a stock timm `VisionTransformer` (in_chans=1, no head), so its state_dict loads
directly into `cxr_ml.models.classifier` for fine-tuning. Differences from the reference MAE:
- one input channel; intensity normalized with the same constants as the classifier;
- fixed 2-D sin-cos position embeddings in encoder and decoder (as in the reference), copied into
  the learnable embedding at fine-tuning time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import timm
import torch
from timm.models.vision_transformer import Block
from torch import nn

from cxr_ml.models.classifier import GRAY_MEAN, GRAY_STD


@dataclass
class MAEConfig:
    arch: str = "vit_base_patch16_224"
    img_size: int = 224
    mask_ratio: float = 0.75
    decoder_dim: int = 512
    decoder_depth: int = 8
    decoder_heads: int = 16
    norm_pix_loss: bool = True


def sincos_2d(dim: int, grid: int, cls_token: bool) -> torch.Tensor:
    """[1, grid*grid (+1), dim] fixed 2-D sin-cos embedding (MAE reference implementation)."""

    def one_d(d: int, pos: np.ndarray) -> np.ndarray:
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    gh, gw = np.meshgrid(np.arange(grid, dtype=np.float64), np.arange(grid, dtype=np.float64))
    emb = np.concatenate([one_d(dim // 2, gw), one_d(dim // 2, gh)], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros((1, dim)), emb], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)


class MAE(nn.Module):
    def __init__(self, cfg: MAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(
            cfg.arch,
            pretrained=False,
            img_size=cfg.img_size,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
            fc_norm=False,
        )
        enc = self.encoder
        d: int = enc.embed_dim  # type: ignore[assignment]
        self.patch: int = int(enc.patch_embed.patch_size[0])
        self.grid: int = cfg.img_size // self.patch
        with torch.no_grad():
            enc.pos_embed.copy_(sincos_2d(d, self.grid, cls_token=True))
        enc.pos_embed.requires_grad_(False)

        self.decoder_embed = nn.Linear(d, cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
        self.register_buffer(
            "decoder_pos_embed", sincos_2d(cfg.decoder_dim, self.grid, cls_token=True)
        )
        self.decoder_blocks = nn.ModuleList(
            Block(cfg.decoder_dim, cfg.decoder_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(cfg.decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(cfg.decoder_dim)
        self.decoder_pred = nn.Linear(cfg.decoder_dim, self.patch * self.patch)
        self.register_buffer("mean", torch.tensor(GRAY_MEAN), persistent=False)
        self.register_buffer("std", torch.tensor(GRAY_STD), persistent=False)
        self._init_decoder()

    def _init_decoder(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        for m in [self.decoder_embed, self.decoder_pred, *self.decoder_blocks.modules()]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # MAE reference: patch-embed projection initialized like a linear layer.
        w = self.encoder.patch_embed.proj.weight
        nn.init.xavier_uniform_(w.view(int(w.shape[0]), -1))

    # ------------------------------------------------------------------ helpers
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B,1,H,W] -> [B, N, p*p]"""
        b, p, g = x.shape[0], self.patch, self.grid
        x = x.reshape(b, 1, g, p, g, p)
        return torch.einsum("bchpwq->bhwpqc", x).reshape(b, g * g, p * p)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, p, g = x.shape[0], self.patch, self.grid
        x = x.reshape(b, g, g, p, p, 1)
        return torch.einsum("bhwpqc->bchpwq", x).reshape(b, 1, g * p, g * p)

    def random_masking(
        self, x: torch.Tensor, ratio: float, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, d = x.shape
        keep = int(n * (1 - ratio))
        noise = torch.rand(b, n, device=x.device, generator=generator)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        mask = torch.ones(b, n, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)  # 1 = masked
        return x_kept, mask, ids_restore

    # ------------------------------------------------------------------ forward
    def forward_encoder(
        self, x: torch.Tensor, ratio: float, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder
        x = enc.patch_embed((x - self.mean) / self.std)
        x = x + enc.pos_embed[:, 1:]
        x, mask, ids_restore = self.random_masking(x, ratio, generator)
        cls = (enc.cls_token + enc.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in enc.blocks:
            x = blk(x)
        return enc.norm(x), mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        b, n = ids_restore.shape
        masks = self.mask_token.expand(b, n + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:], masks], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1], x_], dim=1) + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:]

    def loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(x)
        if self.cfg.norm_pix_loss:
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5
        per_patch = ((pred - target) ** 2).mean(dim=-1)
        return (per_patch * mask).sum() / mask.sum()

    def forward(
        self, x: torch.Tensor, ratio: float | None = None, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: float [B,1,H,W] in [0,1]. Returns (loss, pred [B,N,p*p], mask [B,N])."""
        latent, mask, ids_restore = self.forward_encoder(
            x, self.cfg.mask_ratio if ratio is None else ratio, generator
        )
        pred = self.forward_decoder(latent, ids_restore)
        return self.loss(x, pred.float(), mask), pred, mask

    @torch.no_grad()
    def reconstruct(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(masked input, reconstruction pasted into visible pixels), both [B,1,H,W] in [0,1]."""
        _, pred, mask = self.forward(x, generator=generator)
        pred = pred.float()
        target = self.patchify(x)
        if self.cfg.norm_pix_loss:  # undo per-patch normalization using true patch stats
            mu = target.mean(dim=-1, keepdim=True)
            sd = (target.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
            pred = pred * sd + mu
        m = mask.unsqueeze(-1)
        masked = self.unpatchify(target * (1 - m))
        recon = self.unpatchify(target * (1 - m) + pred * m)
        return masked.clamp(0, 1), recon.clamp(0, 1)

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}
