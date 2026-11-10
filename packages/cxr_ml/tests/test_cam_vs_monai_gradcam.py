"""Cross-check: our gradient-free in-graph CAMs equal MONAI's Grad-CAM for the same model.

For a head of the form `global average pool -> linear`, Grad-CAM's channel weights are the
pooled gradients d(logit_k)/dA_c = W_kc / (h*w), so

    GradCAM_k = ReLU(sum_c W_kc * A_c) / (h*w)            (bias does not appear)

MONAI then upsamples bilinearly to the input size and min-max normalizes each map. Note that
MONAI's *default* normalizer maps max -> 0 and min -> 1 (an inverted scale, documented in
monai.visualize.class_activation_maps.default_normalizer), and maps constant inputs to 0; the
reference below reproduces that convention exactly. Our CAM is
`W_k . A + b_k`; removing the bias, applying ReLU and the same post-processing must reproduce
MONAI's output. This validates the heatmaps served by the app against an independent,
widely used implementation (monai.visualize.GradCAM), not just against our own math.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from monai.visualize import GradCAM

from cxr_ml.models.classifier import CXRClassifier, ModelConfig, build_classifier

# (timm arch, module path of the last spatial feature map, input size)
CASES = [
    ("resnet18", "backbone.layer4", 128),
    ("densenet121", "backbone.features.norm5", 128),
]


def gradcam_from_our_cam(model: CXRClassifier, x: torch.Tensor, k: int) -> torch.Tensor:
    with torch.no_grad():
        _, cams, _ = model.forward_with_cam(x)
    cam = F.relu(cams[:, k : k + 1] - model.head.bias[k])  # drop bias, Grad-CAM ReLU
    up = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    flat = up.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    span = hi - lo
    # MONAI convention: max -> 0, min -> 1; constant map -> 0.
    return torch.where(span > 0, (hi - up) / span.clamp_min(1e-12), torch.zeros_like(up))


@pytest.mark.parametrize(("arch", "layer", "size"), CASES)
def test_in_graph_cam_matches_monai_gradcam(arch: str, layer: str, size: int) -> None:
    torch.manual_seed(0)
    model = build_classifier(ModelConfig(arch=arch, init="random")).eval()
    # Random heads give near-zero logits; scale up so maps have clear structure.
    with torch.no_grad():
        model.head.weight.mul_(20)
    x = torch.rand(2, 1, size, size)
    monai_cam = GradCAM(nn_module=model, target_layers=layer)
    informative = 0
    for k in range(14):
        ref = monai_cam(x, class_idx=k)
        ours = gradcam_from_our_cam(model, x, k)
        assert ref.shape == ours.shape == (2, 1, size, size)
        torch.testing.assert_close(ours, ref.float(), atol=1e-4, rtol=1e-4)
        informative += int((ref.flatten(1).std(dim=1) > 0.05).sum())
    # Guard against a vacuous pass where every map is degenerate (all zeros on both sides).
    assert informative >= 10, f"only {informative} non-degenerate maps compared"


def test_monai_gradcam_is_class_specific() -> None:
    """Sanity check that the comparison above is not trivially satisfied by identical maps."""
    torch.manual_seed(0)
    model = build_classifier(ModelConfig(arch="resnet18", init="random")).eval()
    with torch.no_grad():
        model.head.weight.mul_(20)
    x = torch.rand(1, 1, 128, 128)
    cam = GradCAM(nn_module=model, target_layers="backbone.layer4")
    assert not torch.allclose(cam(x, class_idx=0), cam(x, class_idx=7), atol=1e-3)
