"""
tabular.py — Models on the long (origin × unit) design matrix (PLAN.md, section 5.2).
======================================================================================
Every model is fit per split on origins t < split.train_end(h) (so every training
label window ends at least h hours before the split starts) and within the last
`window_days` days before that bound: attack tactics change, so old history can hurt.
Imputation and scaling are fit on those training rows only (PLAN.md, section 3.1.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.splits import Split

SEED = 0


def train_rows(hours: pd.DatetimeIndex, split: Split, h: int, window_days: int) -> np.ndarray:
    """Origin rows a model for `split` may be trained on."""
    end = split.train_end(h)
    return (hours < end) & (hours >= end - pd.Timedelta(days=window_days))


def make_logit(C: float = 1.0):
    """Logistic regression: median imputation + missing flags, standardised features."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=1000),
    )


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_poisson_glm(alpha: float = 1e-3):
    """Poisson GLM (log link) for target B; counts/durations are log-compressed first."""
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import FunctionTransformer

    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        FunctionTransformer(_signed_log1p),
        StandardScaler(),
        PoissonRegressor(alpha=alpha, max_iter=1000),
    )


@dataclass
class LGBMSpec:
    """
    LightGBM settings; n_estimators is chosen by early stopping on validation.
    Small, heavily regularised trees: on validation they beat num_leaves=63 on both
    Brier and within-unit AUC (bigger trees memorise the units' base rates).
    """
    params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 3000,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    })
    n_estimators: int = 1000

    def make(self, n_estimators: int | None = None, **overrides):
        """A classifier for the binary objective, a regressor otherwise (poisson, quantile, ...)."""
        from lightgbm import LGBMClassifier, LGBMRegressor

        params = self.params | overrides
        cls = LGBMClassifier if params["objective"] == "binary" else LGBMRegressor
        return cls(n_estimators=n_estimators or self.n_estimators, **params)


def fit_lgbm_early_stopping(spec: LGBMSpec, X: pd.DataFrame, y: np.ndarray,
                            X_val: pd.DataFrame, y_val: np.ndarray, patience: int = 50, **overrides):
    """Fit with early stopping on a validation set; returns (model, best_iteration)."""
    import lightgbm

    model = spec.make(**overrides)
    metric = "binary_logloss" if (spec.params | overrides)["objective"] == "binary" else "poisson"
    model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric=metric,
              callbacks=[lightgbm.early_stopping(patience, verbose=False)])
    return model, int(model.best_iteration_ or spec.n_estimators)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class (classifiers) or the predicted value (regressors)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def to_wide(p: np.ndarray, index: pd.MultiIndex, like: pd.DataFrame) -> pd.DataFrame:
    """Put long predictions back on the (origin × unit) grid of `like`, NaN elsewhere."""
    return pd.Series(p, index=index).unstack("unit").reindex(index=like.index, columns=like.columns)


class Calibrator:
    """Platt scaling on the logit of the forecast, or isotonic regression."""

    def __init__(self, method: str = "platt"):
        self.method = method

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))[:, None]

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "platt":
            self.model = LogisticRegression(C=1e6).fit(self._logit(p), y)
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        else:
            raise ValueError(self.method)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            return self.model.predict_proba(self._logit(p))[:, 1]
        return self.model.predict(p)
