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


# ---------------------------------------------------------------------------
# Target B: counts / durations (PLAN.md, section 4.3)
# ---------------------------------------------------------------------------

def poisson_deviance_cells(y: np.ndarray, mu: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-cell Poisson deviance; works for non-integer y >= 0 (durations) too."""
    mu = np.maximum(mu, eps)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (term - (y - mu))


def poisson_deviance(y: np.ndarray, mu: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Poisson deviance."""
    return float(np.mean(poisson_deviance_cells(y, mu, eps)))


def regression_scores(y: np.ndarray, p: np.ndarray, reference_mae: float | None = None) -> dict[str, float]:
    """
    MAE, RMSE, bias, Poisson deviance; `mase` is MAE relative to the seasonal naive
    forecast's MAE on the same cells (pass it as `reference_mae`).
    """
    out = {
        "n": int(len(y)),
        "mean_y": float(np.mean(y)),
        "mae": float(np.mean(np.abs(p - y))),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "bias": float(np.mean(p - y)),
        "poisson_dev": poisson_deviance(y, p),
    }
    if reference_mae is not None:
        out["mase"] = out["mae"] / reference_mae
    return out


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def interval_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    """Empirical coverage and mean width of a central interval."""
    return {"coverage": float(np.mean((y >= lower) & (y <= upper))), "width": float(np.mean(upper - lower))}


# ---------------------------------------------------------------------------
# Significance: block bootstrap over days (PLAN.md, section 4.3)
# ---------------------------------------------------------------------------

def block_sums(loss: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-block sums of a per-cell loss and the block labels (e.g. Kyiv days)."""
    labels, inverse = np.unique(blocks, return_inverse=True)
    return np.bincount(inverse, weights=loss, minlength=len(labels)), labels


def bootstrap_skill(loss: np.ndarray, loss_ref: np.ndarray, n_boot: int = 2000,
                    level: float = 0.95, seed: int = 0) -> dict[str, float]:
    """
    Skill 1 - sum(loss) / sum(loss_ref) with a block-bootstrap interval.

    `loss` and `loss_ref` are per-block sums (block_sums) over the same cells, so
    blocks (days) are resampled whole: hours within a day are strongly dependent.
    `share_not_better` is the share of resamples with skill <= 0 (not a p-value).
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(loss), size=(n_boot, len(loss)))
    boot = 1 - loss[idx].sum(axis=1) / loss_ref[idx].sum(axis=1)
    a = (1 - level) / 2
    return {
        "skill": float(1 - loss.sum() / loss_ref.sum()),
        "ci_low": float(np.quantile(boot, a)),
        "ci_high": float(np.quantile(boot, 1 - a)),
        "share_not_better": float(np.mean(boot <= 0)),
        "n_blocks": int(len(loss)),
    }
