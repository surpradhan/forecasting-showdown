"""Tests for EnsembleForecaster (Week 5).

Runs in the default pytest suite — uses only tabular (sklearn/xgboost/lgbm)
sub-models, no PyTorch or Prophet, so no libomp conflict.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import EnsembleForecaster, _get_class


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 200, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame(
        rng.uniform(0, 1, (n, n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(X.sum(axis=1) + rng.normal(0, 0.05, n), index=idx, name="demand")
    return X, y


def _split(X, y, train_frac: float = 0.8):
    k = int(len(X) * train_frac)
    return X.iloc[:k], y.iloc[:k], X.iloc[k:], y.iloc[k:]


def _assert_valid_predictions(pred: pd.Series, expected_index: pd.DatetimeIndex):
    assert isinstance(pred, pd.Series)
    assert len(pred) == len(expected_index)
    assert pred.index.equals(expected_index)
    assert not pred.isnull().any(), "predictions contain NaN"
    assert np.isfinite(pred.values).all(), "predictions contain inf"


# ---------------------------------------------------------------------------
# _get_class registry
# ---------------------------------------------------------------------------

class TestGetClass:
    def test_known_names_resolve(self):
        from src.models.linear import LinearForecaster
        from src.models.random_forest import RandomForestForecaster
        from src.models.xgboost_model import XGBoostForecaster
        from src.models.lgbm_model import LGBMForecaster

        assert _get_class("linear") is LinearForecaster
        assert _get_class("random_forest") is RandomForestForecaster
        assert _get_class("xgboost") is XGBoostForecaster
        assert _get_class("lgbm") is LGBMForecaster

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown ensemble member"):
            _get_class("prophet")

    def test_unknown_name_error_lists_supported(self):
        with pytest.raises(ValueError, match="Supported:"):
            _get_class("gru")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_empty_members_raises(self):
        m = EnsembleForecaster({"members": [], "strategy": "mean"})
        X, y = _make_dataset()
        with pytest.raises(ValueError, match="at least one member"):
            m.fit(X, y)

    def test_unknown_strategy_raises(self):
        m = EnsembleForecaster({"members": ["linear"], "strategy": "median"})
        X, y = _make_dataset()
        with pytest.raises(ValueError, match="Unknown ensemble strategy"):
            m.fit(X, y)

    def test_weighted_missing_weights_raises(self):
        m = EnsembleForecaster({"members": ["linear", "random_forest"], "strategy": "weighted"})
        X, y = _make_dataset()
        with pytest.raises(ValueError, match="requires 'weights'"):
            m.fit(X, y)

    def test_weighted_wrong_length_raises(self):
        m = EnsembleForecaster({
            "members": ["linear", "random_forest"],
            "strategy": "weighted",
            "weights": [0.5],   # one weight for two members
        })
        X, y = _make_dataset()
        with pytest.raises(ValueError, match="requires 'weights' with exactly 2"):
            m.fit(X, y)

    def test_weighted_all_zero_weights_raises(self):
        m = EnsembleForecaster({
            "members": ["linear"],
            "strategy": "weighted",
            "weights": [0.0],
        })
        X, y = _make_dataset()
        with pytest.raises(ValueError, match="must not all be zero"):
            m.fit(X, y)

    def test_predict_before_fit_raises(self):
        m = EnsembleForecaster({"members": ["linear"], "strategy": "mean"})
        X, _ = _make_dataset()
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict(X)


# ---------------------------------------------------------------------------
# Mean strategy
# ---------------------------------------------------------------------------

class TestMeanEnsemble:
    def setup_method(self):
        X, y = _make_dataset()
        self.X_tr, self.y_tr, self.X_te, self.y_te = _split(X, y)

    def test_single_member_matches_solo_model(self):
        """Ensemble of one model must produce identical output to that model alone."""
        from src.models.linear import LinearForecaster
        from src.config import load_config

        ens = EnsembleForecaster({"members": ["linear"], "strategy": "mean"})
        ens.fit(self.X_tr, self.y_tr)
        ens_pred = ens.predict(self.X_te)

        solo = LinearForecaster(load_config("linear"))
        solo.fit(self.X_tr, self.y_tr)
        solo_pred = solo.predict(self.X_te)

        pd.testing.assert_series_equal(ens_pred, solo_pred.rename("ensemble"),
                                       check_names=False)

    def test_two_member_mean_is_arithmetic_average(self):
        """Predictions should be exact average of two members."""
        from src.models.linear import LinearForecaster
        from src.models.random_forest import RandomForestForecaster
        from src.config import load_config

        ens = EnsembleForecaster({"members": ["linear", "random_forest"], "strategy": "mean"})
        ens.fit(self.X_tr, self.y_tr)
        ens_pred = ens.predict(self.X_te)

        p_lin = LinearForecaster(load_config("linear"))
        p_lin.fit(self.X_tr, self.y_tr)
        p_rf = RandomForestForecaster(load_config("random_forest"))
        p_rf.fit(self.X_tr, self.y_tr)

        expected = (p_lin.predict(self.X_te) + p_rf.predict(self.X_te)) / 2
        np.testing.assert_allclose(ens_pred.values, expected.values, rtol=1e-5)

    def test_valid_predictions_shape_and_index(self):
        ens = EnsembleForecaster({"members": ["linear", "random_forest"], "strategy": "mean"})
        ens.fit(self.X_tr, self.y_tr)
        pred = ens.predict(self.X_te)
        _assert_valid_predictions(pred, self.X_te.index)

    def test_three_member_output_is_finite(self):
        ens = EnsembleForecaster({"members": ["linear", "random_forest", "lgbm"], "strategy": "mean"})
        ens.fit(self.X_tr, self.y_tr)
        pred = ens.predict(self.X_te)
        _assert_valid_predictions(pred, self.X_te.index)


# ---------------------------------------------------------------------------
# Weighted strategy
# ---------------------------------------------------------------------------

class TestWeightedEnsemble:
    def setup_method(self):
        X, y = _make_dataset()
        self.X_tr, self.y_tr, self.X_te, self.y_te = _split(X, y)

    def test_equal_weights_matches_mean(self):
        """Weights of [1, 1] should give same result as strategy='mean'."""
        cfg_mean = {"members": ["linear", "random_forest"], "strategy": "mean"}
        cfg_w    = {"members": ["linear", "random_forest"], "strategy": "weighted",
                    "weights": [1.0, 1.0]}

        m_mean = EnsembleForecaster(cfg_mean)
        m_mean.fit(self.X_tr, self.y_tr)

        m_w = EnsembleForecaster(cfg_w)
        m_w.fit(self.X_tr, self.y_tr)

        np.testing.assert_allclose(
            m_mean.predict(self.X_te).values,
            m_w.predict(self.X_te).values,
            rtol=1e-5,
        )

    def test_weights_are_normalised(self):
        """Weights [2, 1] and [4, 2] must produce identical output."""
        def _pred(weights):
            m = EnsembleForecaster({
                "members": ["linear", "random_forest"],
                "strategy": "weighted",
                "weights": weights,
            })
            m.fit(self.X_tr, self.y_tr)
            return m.predict(self.X_te).values

        np.testing.assert_allclose(_pred([2.0, 1.0]), _pred([4.0, 2.0]), rtol=1e-5)

    def test_weight_zero_on_one_member_ignores_it(self):
        """Weight=0 on a member should give the same result as the other member alone."""
        from src.models.linear import LinearForecaster
        from src.config import load_config

        m_ens = EnsembleForecaster({
            "members": ["linear", "random_forest"],
            "strategy": "weighted",
            "weights": [1.0, 0.0],
        })
        m_ens.fit(self.X_tr, self.y_tr)

        m_solo = LinearForecaster(load_config("linear"))
        m_solo.fit(self.X_tr, self.y_tr)

        np.testing.assert_allclose(
            m_ens.predict(self.X_te).values,
            m_solo.predict(self.X_te).values,
            rtol=1e-5,
        )

    def test_valid_predictions(self):
        m = EnsembleForecaster({
            "members": ["lgbm", "xgboost", "random_forest"],
            "strategy": "weighted",
            "weights": [0.5, 0.3, 0.2],
        })
        m.fit(self.X_tr, self.y_tr)
        pred = m.predict(self.X_te)
        _assert_valid_predictions(pred, self.X_te.index)

    def test_weighted_arithmetic_correctness(self):
        """Verify the weighted sum exactly equals manually computed expected value."""
        from src.models.linear import LinearForecaster
        from src.models.random_forest import RandomForestForecaster
        from src.config import load_config

        weights = [3.0, 1.0]
        m_ens = EnsembleForecaster({
            "members": ["linear", "random_forest"],
            "strategy": "weighted",
            "weights": weights,
        })
        m_ens.fit(self.X_tr, self.y_tr)

        # Reproduce the same models independently to get raw predictions
        m_lin = LinearForecaster(load_config("linear"))
        m_lin.fit(self.X_tr, self.y_tr)
        m_rf = RandomForestForecaster(load_config("random_forest"))
        m_rf.fit(self.X_tr, self.y_tr)

        w = np.array(weights, dtype=float)
        w /= w.sum()   # normalise: [0.75, 0.25]
        expected = w[0] * m_lin.predict(self.X_te).values + w[1] * m_rf.predict(self.X_te).values

        np.testing.assert_allclose(m_ens.predict(self.X_te).values, expected, rtol=1e-5)

    def test_negative_weights_raises(self):
        m = EnsembleForecaster({
            "members": ["linear", "random_forest"],
            "strategy": "weighted",
            "weights": [-2.0, 1.0],
        })
        with pytest.raises(ValueError, match="non-negative"):
            m.fit(self.X_tr, self.y_tr)


# ---------------------------------------------------------------------------
# Production config (configs/ensemble.yaml)
# ---------------------------------------------------------------------------

class TestProductionConfig:
    def test_production_config_loads_and_fits(self):
        from src.config import load_config

        cfg = load_config("ensemble")
        assert "members" in cfg
        assert isinstance(cfg["members"], list)
        assert len(cfg["members"]) >= 1

        X, y = _make_dataset()
        X_tr, y_tr, X_te, _ = _split(X, y)
        m = EnsembleForecaster(cfg)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        _assert_valid_predictions(pred, X_te.index)
