import numpy as np
import pandas as pd
import pytest

from src.models.linear import LinearForecaster
from src.models.random_forest import RandomForestForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.lgbm_model import LGBMForecaster


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 400, n_features: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame(rng.uniform(0, 1, (n, n_features)), index=idx,
                     columns=[f"f{i}" for i in range(n_features)])
    # y is a noisy linear combination so all models can fit it
    y = pd.Series(X.sum(axis=1) + rng.normal(0, 0.1, n), index=idx, name="demand")
    return X, y


def _split(X, y, train_frac=0.8):
    k = int(len(X) * train_frac)
    return X.iloc[:k], y.iloc[:k], X.iloc[k:], y.iloc[k:]


def _assert_valid_predictions(pred: pd.Series, expected_index: pd.DatetimeIndex):
    assert isinstance(pred, pd.Series)
    assert len(pred) == len(expected_index)
    assert pred.index.equals(expected_index)
    assert not pred.isnull().any(), "predictions contain NaN"
    assert np.isfinite(pred.values).all(), "predictions contain inf"


# ---------------------------------------------------------------------------
# LinearForecaster (Ridge)
# ---------------------------------------------------------------------------

class TestLinearForecaster:
    _cfg = {"model": "linear", "alpha": 1.0}

    def _fit(self):
        X, y = _make_dataset()
        X_tr, y_tr, X_te, y_te = _split(X, y)
        m = LinearForecaster(self._cfg)
        m.fit(X_tr, y_tr)
        return m, X_te

    def test_predict_length(self):
        m, X_te = self._fit()
        assert len(m.predict(X_te)) == len(X_te)

    def test_predict_index_matches(self):
        m, X_te = self._fit()
        pred = m.predict(X_te)
        assert pred.index.equals(X_te.index)

    def test_no_nan_no_inf(self):
        m, X_te = self._fit()
        _assert_valid_predictions(m.predict(X_te), X_te.index)

    def test_get_params_returns_config(self):
        m = LinearForecaster(self._cfg)
        assert m.get_params() == self._cfg

    def test_alpha_zero_equals_ols(self):
        """alpha=0 Ridge ≈ OLS — predictions should still be finite."""
        X, y = _make_dataset()
        X_tr, y_tr, X_te, _ = _split(X, y)
        m = LinearForecaster({"model": "linear", "alpha": 0.0})
        m.fit(X_tr, y_tr)
        _assert_valid_predictions(m.predict(X_te), X_te.index)

    def test_predictions_improve_over_naive_mean(self):
        """Ridge should beat predicting the training mean (a weak sanity check)."""
        X, y = _make_dataset()
        X_tr, y_tr, X_te, y_te = _split(X, y)
        m = LinearForecaster(self._cfg)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        mae_ridge = (pred - y_te).abs().mean()
        mae_mean = (y_tr.mean() - y_te).abs().mean()
        assert mae_ridge < mae_mean


# ---------------------------------------------------------------------------
# RandomForestForecaster
# ---------------------------------------------------------------------------

class TestRandomForestForecaster:
    _cfg = {"model": "random_forest", "n_estimators": 20, "max_depth": 5,
            "min_samples_leaf": 2, "n_jobs": 1}

    def _fit(self):
        X, y = _make_dataset()
        X_tr, y_tr, X_te, _ = _split(X, y)
        m = RandomForestForecaster(self._cfg)
        m.fit(X_tr, y_tr)
        return m, X_te

    def test_predict_length(self):
        m, X_te = self._fit()
        assert len(m.predict(X_te)) == len(X_te)

    def test_predict_index_matches(self):
        m, X_te = self._fit()
        assert m.predict(X_te).index.equals(X_te.index)

    def test_no_nan_no_inf(self):
        m, X_te = self._fit()
        _assert_valid_predictions(m.predict(X_te), X_te.index)

    def test_predictions_in_training_value_range(self):
        """RF predictions are interpolations — should stay within [min, max] of y_train."""
        X, y = _make_dataset()
        X_tr, y_tr, X_te, _ = _split(X, y)
        m = RandomForestForecaster(self._cfg)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        assert pred.min() >= y_tr.min() - 1e-6
        assert pred.max() <= y_tr.max() + 1e-6

    def test_deterministic_with_fixed_seed(self):
        X, y = _make_dataset()
        X_tr, y_tr, X_te, _ = _split(X, y)
        m1 = RandomForestForecaster(self._cfg)
        m1.fit(X_tr, y_tr)
        m2 = RandomForestForecaster(self._cfg)
        m2.fit(X_tr, y_tr)
        np.testing.assert_array_equal(m1.predict(X_te).values, m2.predict(X_te).values)


# ---------------------------------------------------------------------------
# XGBoostForecaster
# ---------------------------------------------------------------------------

class TestXGBoostForecaster:
    # Small n_estimators so tests are fast; no early stopping
    _cfg_plain = {"model": "xgboost", "n_estimators": 20, "max_depth": 3,
                  "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}
    _cfg_es = {**_cfg_plain, "early_stopping_rounds": 5}

    def _fit(self, cfg=None):
        cfg = cfg or self._cfg_plain
        X, y = _make_dataset()
        X_tr, y_tr, X_te, _ = _split(X, y)
        m = XGBoostForecaster(cfg)
        m.fit(X_tr, y_tr)
        return m, X_te

    def test_predict_length(self):
        m, X_te = self._fit()
        assert len(m.predict(X_te)) == len(X_te)

    def test_predict_index_matches(self):
        m, X_te = self._fit()
        assert m.predict(X_te).index.equals(X_te.index)

    def test_no_nan_no_inf(self):
        m, X_te = self._fit()
        _assert_valid_predictions(m.predict(X_te), X_te.index)

    def test_early_stopping_path(self):
        """early_stopping_rounds triggers the internal val-split branch."""
        m, X_te = self._fit(self._cfg_es)
        _assert_valid_predictions(m.predict(X_te), X_te.index)

    def test_get_params(self):
        m = XGBoostForecaster(self._cfg_plain)
        assert m.get_params()["n_estimators"] == 20

    def test_early_stopping_zero_treated_as_disabled(self):
        """early_stopping_rounds=0 is falsy — should use the plain fit path."""
        cfg = {**self._cfg_plain, "early_stopping_rounds": 0}
        m, X_te = self._fit(cfg)
        _assert_valid_predictions(m.predict(X_te), X_te.index)


# ---------------------------------------------------------------------------
# LGBMForecaster
# ---------------------------------------------------------------------------

class TestLGBMForecaster:
    _cfg_plain = {"model": "lgbm", "n_estimators": 20, "max_depth": 4,
                  "learning_rate": 0.1, "num_leaves": 15,
                  "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1}
    _cfg_es = {**_cfg_plain, "early_stopping_rounds": 5}

    def _fit(self, cfg=None):
        cfg = cfg or self._cfg_plain
        X, y = _make_dataset()
        X_tr, y_tr, X_te, _ = _split(X, y)
        m = LGBMForecaster(cfg)
        m.fit(X_tr, y_tr)
        return m, X_te

    def test_predict_length(self):
        m, X_te = self._fit()
        assert len(m.predict(X_te)) == len(X_te)

    def test_predict_index_matches(self):
        m, X_te = self._fit()
        assert m.predict(X_te).index.equals(X_te.index)

    def test_no_nan_no_inf(self):
        m, X_te = self._fit()
        _assert_valid_predictions(m.predict(X_te), X_te.index)

    def test_early_stopping_path(self):
        m, X_te = self._fit(self._cfg_es)
        _assert_valid_predictions(m.predict(X_te), X_te.index)

    def test_get_params(self):
        m = LGBMForecaster(self._cfg_plain)
        assert m.get_params()["num_leaves"] == 15

    def test_early_stopping_zero_treated_as_disabled(self):
        """early_stopping_rounds=0 is falsy — should use the plain fit path."""
        cfg = {**self._cfg_plain, "early_stopping_rounds": 0}
        m, X_te = self._fit(cfg)
        _assert_valid_predictions(m.predict(X_te), X_te.index)
