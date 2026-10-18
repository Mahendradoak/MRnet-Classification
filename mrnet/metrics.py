"""Discrimination, threshold and calibration metrics with bootstrap uncertainty.

The official MRNet validation set has 120 exams, so a point AUC alone is not
interpretable: differences of +/-0.01 between models (as in the original report)
are well inside sampling noise. Every AUC here carries a 95% bootstrap CI, and
model comparisons use a paired bootstrap on the same resampled exams.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    y, p = np.asarray(y), np.asarray(p)
    point = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_auc_difference(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> dict[str, float]:
    """AUC(a) - AUC(b) with a paired bootstrap CI and a two-sided bootstrap p-value."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diff = float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    d = np.asarray(d)
    p_two = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"auc_diff": diff, "ci_low": float(lo), "ci_high": float(hi), "p_bootstrap": p_two}


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    return {"threshold": thr, "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "accuracy": (tp + tn) / len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def summarise(y_test: np.ndarray, p_test: np.ndarray, y_val: np.ndarray, p_val: np.ndarray) -> dict[str, float]:
    """Test-set AUC (95% CI), Brier score, and sensitivity/specificity at the
    Youden threshold chosen on the internal validation set (never on test)."""
    auc, lo, hi = auc_ci(y_test, p_test)
    thr = youden_threshold(y_val, p_val)
    return {"auc": auc, "auc_ci_low": lo, "auc_ci_high": hi, "brier": float(brier_score_loss(y_test, p_test)),
            **threshold_metrics(y_test, p_test, thr), "n": int(len(y_test)), "positives": int(np.sum(y_test))}
