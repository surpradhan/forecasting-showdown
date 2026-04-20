"""EnsembleForecaster — averages predictions from multiple tabular sub-models.

Supports two strategies:
  mean:     simple unweighted average of all member predictions
  weighted: explicit per-model weights (normalised internally, must sum to > 0)

Config keys
-----------
members  : list[str]   model names; each must be a registered tabular model
strategy : str         "mean" | "weighted"  (default: "mean")
weights  : list[float] one weight per member (required when strategy="weighted")

Example config (configs/ensemble.yaml)
---------------------------------------
model: ensemble
members: [lgbm, xgboost, random_forest]
strategy: mean
"""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.models.base import ForecasterBase


# ---------------------------------------------------------------------------
# Registry of supported member models (lazy imports avoid loading all libs
# at module import time, which could trigger libomp conflicts on macOS ARM)
# ---------------------------------------------------------------------------

_SUPPORTED: dict[str, str] = {
    "linear":        "src.models.linear.LinearForecaster",
    "random_forest": "src.models.random_forest.RandomForestForecaster",
    "xgboost":       "src.models.xgboost_model.XGBoostForecaster",
    "lgbm":          "src.models.lgbm_model.LGBMForecaster",
}


def _get_class(name: str) -> type[ForecasterBase]:
    """Return the ForecasterBase subclass for *name*, importing lazily."""
    if name not in _SUPPORTED:
        raise ValueError(
            f"Unknown ensemble member {name!r}. "
            f"Supported: {sorted(_SUPPORTED)}"
        )
    module_path, class_name = _SUPPORTED[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# EnsembleForecaster
# ---------------------------------------------------------------------------

class EnsembleForecaster(ForecasterBase):
    """Combines predictions from multiple tabular ForecasterBase models.

    Each member is independently fit on the same (X_train, y_train) and its
    own config is loaded via load_config(name) so hyper-parameters are
    consistent with solo runs.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        members = self.config.get("members", [])
        if not members:
            raise ValueError(
                "EnsembleForecaster requires at least one member in config['members']"
            )

        strategy = self.config.get("strategy", "mean")
        if strategy not in ("mean", "weighted"):
            raise ValueError(f"Unknown ensemble strategy: {strategy!r}")

        if strategy == "weighted":
            weights = self.config.get("weights")
            if weights is None or len(weights) != len(members):
                raise ValueError(
                    f"strategy='weighted' requires 'weights' with exactly "
                    f"{len(members)} value(s), one per member"
                )
            if any(w < 0 for w in weights):
                raise ValueError("ensemble weights must all be non-negative")
            if sum(weights) == 0:
                raise ValueError("ensemble weights must not all be zero")

        self._members: list[str] = list(members)
        self._models: list[ForecasterBase] = []
        for name in self._members:
            cfg = load_config(name)
            model = _get_class(name)(cfg)
            model.fit(X, y)
            self._models.append(model)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not getattr(self, "_models", None):
            raise RuntimeError("EnsembleForecaster.predict() called before fit()")

        preds = pd.concat(
            [m.predict(X).rename(i) for i, m in enumerate(self._models)],
            axis=1,
        )

        strategy = self.config.get("strategy", "mean")
        if strategy == "mean":
            result = preds.mean(axis=1)
        else:  # weighted
            weights = self.config["weights"]
            w = pd.Series(weights, dtype=float)
            w = w / w.sum()          # normalise so they sum to 1
            result = preds.mul(w.values).sum(axis=1)

        result.name = "ensemble"
        return result
