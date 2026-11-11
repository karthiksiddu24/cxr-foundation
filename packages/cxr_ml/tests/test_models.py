from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cxr_ml.models.classifier import ModelConfig, build_classifier
from cxr_ml.models.mae import MAE, MAEConfig

TINY_MAE = MAEConfig(
    arch="vit_tiny_patch16_224", img_size=64, decoder_dim=32, decoder_depth=1, decoder_heads=2
)


@pytest.mark.parametrize("arch", ["resnet18", "vit_tiny_patch16_224"])
def test_cam_mean_equals_logits(arch: str) -> None:
    torch.manual_seed(0)
    m = build_classifier(ModelConfig(arch=arch, init="random", img_size=64)).eval()
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        logits, cams, pooled = m.forward_with_cam(x)
        ref = m(x)
    assert cams.shape[:2] == (2, 14)
    assert torch.allclose(cams.mean(dim=(2, 3)), ref, atol=1e-5)
    assert torch.allclose(logits, ref, atol=1e-5)
    assert pooled.ndim == 2


def test_patchify_roundtrip() -> None:
    mae = MAE(TINY_MAE)
    x = torch.rand(2, 1, 64, 64)
    assert torch.equal(mae.unpatchify(mae.patchify(x)), x)


def test_masking_ratio_and_restore() -> None:
    mae = MAE(TINY_MAE)
    tokens = torch.arange(16).float().view(1, 16, 1).expand(3, -1, 4).contiguous()
    kept, mask, _ids = mae.random_masking(tokens, 0.75)
    assert kept.shape == (3, 4, 4)
    assert (mask.sum(1) == 12).all()
    # visible tokens are exactly those with mask == 0
    for b in range(3):
        assert set(kept[b, :, 0].tolist()) == set(
            torch.nonzero(mask[b] == 0).flatten().float().tolist()
        )


def test_mae_loss_only_on_masked_patches() -> None:
    torch.manual_seed(0)
    mae = MAE(TINY_MAE)
    x = torch.rand(2, 1, 64, 64)
    loss, pred, _mask = mae(x)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert pred.shape == (2, 16, 256)
    loss.backward()
    assert mae.encoder.blocks[0].attn.qkv.weight.grad is not None
    assert not mae.encoder.pos_embed.requires_grad


def test_encoder_loads_into_classifier(tmp_path: Path) -> None:
    mae = MAE(TINY_MAE)
    p = tmp_path / "enc.pt"
    # Same layout as train/mae.py snapshots: provenance includes torch.__version__ (TorchVersion),
    # which the weights-only loader rejects unless allowlisted.
    torch.save({"encoder": mae.encoder_state_dict(), "epoch": 1, "torch": torch.__version__}, p)
    clf = build_classifier(
        ModelConfig(arch="vit_tiny_patch16_224", init="ssl", ssl_checkpoint=str(p), img_size=64)
    )
    w_src = mae.encoder.blocks[0].attn.qkv.weight
    w_dst = clf.backbone.blocks[0].attn.qkv.weight
    assert torch.equal(w_src, w_dst)
    assert torch.equal(clf.backbone.pos_embed, mae.encoder.pos_embed)
