"""Classification metrics with patient-clustered bootstrap confidence intervals.

Cluster bootstrap: resample *patients* with replacement; each image gets weight = number of times
its patient was drawn. AUROC and AUPRC are computed exactly under those weights (vectorized over
resamples), which is equivalent to materializing the resampled datasets but ~100x faster.

Tied scores are handled exactly (midrank for AUROC, threshold-at-unique-score for AP, matching
scikit-learn's `average_precision_score`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

F64 = npt.NDArray[np.float64]


def _grouped(scores: F64, y: npt.NDArray[np.integer], w: F64) -> tuple[F64, F64]:
    """Aggregate weights by unique score (ascending). w: [R, n] -> pos/neg weights [R, U]."""
    uniq, inv = np.unique(scores, return_inverse=True)
    u = len(uniq)
    pos = y.astype(bool)
    r = w.shape[0]
    pw = np.zeros((r, u))
    nw = np.zeros((r, u))
    np.add.at(pw.T, inv[pos], w[:, pos].T)
    np.add.at(nw.T, inv[~pos], w[:, ~pos].T)
    return pw, nw


def weighted_auroc(scores: F64, y: npt.NDArray[np.integer], w: F64) -> F64:
    """AUROC for each row of weights w [R, n]. NaN where a class is absent."""
    pw, nw = _grouped(scores, y, w)
    neg_below = np.cumsum(nw, axis=1) - nw
    num = (pw * (neg_below + 0.5 * nw)).sum(axis=1)
    den = pw.sum(axis=1) * nw.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def weighted_auprc(scores: F64, y: npt.NDArray[np.integer], w: F64) -> F64:
    """Average precision (sklearn definition) for each row of weights."""
    pw, nw = _grouped(scores, y, w)
    pw, nw = pw[:, ::-1], nw[:, ::-1]  # descending score
    tp = np.cumsum(pw, axis=1)
    fp = np.cumsum(nw, axis=1)
    wpos = tp[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        ap = (pw / wpos * prec).sum(axis=1)
    return np.where(wpos[:, 0] > 0, ap, np.nan)


def patient_bootstrap_weights(patient_ids: npt.NDArray[np.integer], n_boot: int, seed: int) -> F64:
    """[n_boot, n_images] weights from resampling patients with replacement."""
    uniq, inv = np.unique(patient_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    counts = np.zeros((n_boot, len(uniq)))
    np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1.0)
    return counts[:, inv]


@dataclass
class BootResult:
    point: F64  # [K]
    samples: F64  # [R, K]

    def ci(self, alpha: float = 0.05) -> tuple[F64, F64]:
        lo = np.nanpercentile(self.samples, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(self.samples, 100 * (1 - alpha / 2), axis=0)
        return lo, hi


def bootstrap_metric(
    metric: str,
    y: npt.NDArray[np.integer],  # [n, K]
    scores: F64,  # [n, K]
    patient_ids: npt.NDArray[np.integer],
    n_boot: int = 2000,
    seed: int = 0,
    chunk: int = 250,
    weights: F64 | None = None,
) -> BootResult:
    """Per-class metric with point estimate and bootstrap samples. `weights` reuses resamples
    (pass the same weights to two models for a paired comparison)."""
    fn = {"auroc": weighted_auroc, "auprc": weighted_auprc}[metric]
    n, k = y.shape
    ones = np.ones((1, n))
    point = np.array([fn(scores[:, j], y[:, j], ones)[0] for j in range(k)])
    if weights is None:
        weights = patient_bootstrap_weights(patient_ids, n_boot, seed)
    samples = np.empty((weights.shape[0], k))
    for s in range(0, weights.shape[0], chunk):
        w = weights[s : s + chunk]
        for j in range(k):
            samples[s : s + chunk, j] = fn(scores[:, j], y[:, j], w)
    return BootResult(point, samples)


# ---------------------------------------------------------------- calibration


def sigmoid(x: F64) -> F64:
    return 1.0 / (1.0 + np.exp(-x))


def ece_equal_mass(p: F64, y: npt.NDArray[np.integer], n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass (quantile) bins, for one binary label."""
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    n = len(p)
    return float(sum(len(b) / n * abs(p[b].mean() - y[b].mean()) for b in bins if len(b)))


def reliability_curve(p: F64, y: npt.NDArray[np.integer], n_bins: int = 15) -> tuple[F64, F64, F64]:
    """(mean predicted, observed frequency, count) per equal-mass bin."""
    order = np.argsort(p, kind="stable")
    bins = [b for b in np.array_split(order, n_bins) if len(b)]
    return (
        np.array([p[b].mean() for b in bins]),
        np.array([y[b].mean() for b in bins]),
        np.array([len(b) for b in bins], dtype=float),
    )


def fit_temperature(logits: F64, y: npt.NDArray[np.integer]) -> float:
    """Per-class temperature T minimizing NLL of sigmoid(logits / T)."""

    def nll(log_t: float) -> float:
        z = logits / np.exp(log_t)
        # numerically stable BCE-with-logits
        return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def youden_threshold(p: F64, y: npt.NDArray[np.integer]) -> tuple[float, float, float]:
    """Threshold maximizing sensitivity + specificity - 1. Returns (thr, sens, spec)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    j = int(np.argmax(tpr - fpr))
    t = float(min(thr[j], 1.0))
    return t, float(tpr[j]), float(1 - fpr[j])


def sens_spec(p: F64, y: npt.NDArray[np.integer], thr: float) -> tuple[float, float]:
    pred = p >= thr
    pos, neg = y == 1, y == 0
    sens = float(pred[pos].mean()) if pos.any() else float("nan")
    spec = float((~pred[neg]).mean()) if neg.any() else float("nan")
    return sens, spec
