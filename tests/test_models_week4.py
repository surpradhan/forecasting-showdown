import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.models.prophet import ProphetForecaster
from src.utils import empty_x as _empty_x


def _hourly_series(n: int = 500) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=float)
    values = 2.0 + np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(n)
    return pd.Series(values, index=idx, name="demand")


class TestProphetForecaster:
    # Disable yearly seasonality for speed; daily+weekly exercise core code path
    _cfg = {
        "model": "prophet",
        "yearly_seasonality": False,
        "weekly_seasonality": True,
        "daily_seasonality": True,
        "changepoint_prior_scale": 0.05,
    }

    def _fit(self, n: int = 500):
        y = _hourly_series(n)
        m = ProphetForecaster(self._cfg)
        m.fit(_empty_x(y.index), y)
        return m, y

    def test_predict_length(self):
        m, _ = self._fit()
        pred = m.predict(_empty_x(pd.date_range("2020-09-01", periods=24, freq="h")))
        assert len(pred) == 24

    def test_predict_index_matches(self):
        m, _ = self._fit()
        test_idx = pd.date_range("2020-09-01", periods=24, freq="h")
        pred = m.predict(_empty_x(test_idx))
        assert pred.index.equals(test_idx)

    def test_no_nan_in_predictions(self):
        m, _ = self._fit()
        pred = m.predict(_empty_x(pd.date_range("2020-09-01", periods=24, freq="h")))
        assert not pred.isnull().any()

    def test_predictions_are_finite(self):
        m, _ = self._fit()
        pred = m.predict(_empty_x(pd.date_range("2020-09-01", periods=12, freq="h")))
        assert np.isfinite(pred.values).all()

    def test_get_params_returns_config(self):
        m = ProphetForecaster(self._cfg)
        assert m.get_params() == self._cfg

    def test_predict_length_multi_day(self):
        m, _ = self._fit()
        pred = m.predict(_empty_x(pd.date_range("2020-09-01", periods=168, freq="h")))
        assert len(pred) == 168
        assert not pred.isnull().any()

    def test_production_config_keys_accepted(self):
        """Verify configs/prophet.yaml keys are all recognised by ProphetForecaster."""
        cfg = load_config("prophet")
        y = _hourly_series(500)
        m = ProphetForecaster(cfg)
        m.fit(_empty_x(y.index), y)
        pred = m.predict(_empty_x(pd.date_range("2020-09-01", periods=24, freq="h")))
        assert len(pred) == 24
        assert not pred.isnull().any()
