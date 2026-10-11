"""
metrics.py — Scores for probabilistic forecasts of target A (PLAN.md, section 4.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

def clip_proba(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Keep probabilities away from 0/1 so log loss stays finite for 0/1 forecasts."""
    return np.clip(p, eps, 1 - eps)


def paired(y: pd.DataFrame, p: pd.DataFrame, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Flatten label/forecast matrices, keeping cells where both are defined."""
    yv, pv = y.to_numpy(), p.reindex_like(y).to_numpy()
    if mask is not None:
        yv, pv = yv[mask], pv[mask]
    keep = ~np.isnan(yv) & ~np.isnan(pv)
    return yv[keep], pv[keep]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def score(y: np.ndarray, p: np.ndarray, reference_brier: float | None = None) -> dict[str, float]:
    """
    Brier, Brier skill vs a reference, log loss, ROC-AUC, PR-AUC, base rate, n.
    `reference_brier` is the Brier of the reference forecast on the same cells.
    """
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "brier": brier(y, p),
        "log_loss": float(log_loss(y, clip_proba(p), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if y.any() else np.nan,
    }
    if reference_brier is not None:
        out["bss"] = 1 - out["brier"] / reference_brier
    return out


def within_unit_auc(y: pd.DataFrame, p: pd.DataFrame, cells: np.ndarray) -> float:
    """
    Mean of per-unit ROC-AUCs over units with both classes in `cells`.

    The pooled AUC is dominated by which units are "hot" (frontline vs west); this
    one measures how well a forecast ranks hours *within* a unit, i.e. timing skill.
    """
    yv, pv = y.to_numpy(), p.reindex_like(y).to_numpy()
    aucs = []
    for j in range(yv.shape[1]):
        keep = cells[:, j] & ~np.isnan(yv[:, j]) & ~np.isnan(pv[:, j])
        yy, pp = yv[keep, j], pv[keep, j]
        if 0 < yy.sum() < len(yy):
            aucs.append(roc_auc_score(yy, pp))
    return float(np.mean(aucs)) if aucs else np.nan


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean forecast vs observed frequency per probability bin (reliability diagram)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y})
    return df.groupby("bin").agg(mean_forecast=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
