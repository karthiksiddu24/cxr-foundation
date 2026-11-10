from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from cxr_ml.eval.metrics import (
    bootstrap_metric,
    ece_equal_mass,
    fit_temperature,
    patient_bootstrap_weights,
    sigmoid,
    weighted_auprc,
    weighted_auroc,
    youden_threshold,
)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = y * 0.8 + rng.normal(0, 1, 500)
    s[:50] = np.round(s[:50], 1)  # inject ties
    return y, s


def test_auroc_matches_sklearn(data: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = data
    assert weighted_auroc(s, y, np.ones((1, len(y))))[0] == pytest.approx(roc_auc_score(y, s))


def test_auprc_matches_sklearn(data: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = data
    assert weighted_auprc(s, y, np.ones((1, len(y))))[0] == pytest.approx(
        average_precision_score(y, s)
    )


def test_weighted_equals_materialized_resample(data: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = data
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(y), len(y))
    w = np.bincount(idx, minlength=len(y)).astype(float)[None]
    assert weighted_auroc(s, y, w)[0] == pytest.approx(roc_auc_score(y[idx], s[idx]))
    assert weighted_auprc(s, y, w)[0] == pytest.approx(average_precision_score(y[idx], s[idx]))


def test_patient_weights_keep_patients_together() -> None:
    pids = np.array([1, 1, 1, 2, 3, 3])
    w = patient_bootstrap_weights(pids, 50, seed=0)
    assert (w[:, 0] == w[:, 1]).all() and (w[:, 1] == w[:, 2]).all() and (w[:, 4] == w[:, 5]).all()
    assert (w[:, [0, 3, 4]].sum(axis=1) == 3).all()  # 3 patients drawn per resample


def test_bootstrap_ci_contains_point_and_absent_class_is_nan(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    y, s = data
    Y = np.stack([y, np.zeros_like(y)], 1)
    S = np.stack([s, s], 1)
    res = bootstrap_metric("auroc", Y, S, np.arange(len(y)) // 3, n_boot=300, seed=0)
    lo, hi = res.ci()
    assert lo[0] < res.point[0] < hi[0]
    assert np.isnan(res.point[1])


def test_temperature_recovers_scaling() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < sigmoid(z)).astype(int)
    assert fit_temperature(z * 3.0, y) == pytest.approx(3.0, rel=0.1)
    assert ece_equal_mass(sigmoid(z), y) < 0.02
    assert ece_equal_mass(sigmoid(z * 3), y) > ece_equal_mass(sigmoid(z), y)


def test_youden_perfect_separation() -> None:
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    t, sens, spec = youden_threshold(p, y)
    assert 0.2 < t <= 0.8 and sens == 1.0 and spec == 1.0
