import numpy as np
import pandas as pd
import pytest

from src.models.arima import ARIMAForecaster
from src.models.naive import NaiveForecaster
from src.utils import empty_x as _empty_x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hourly_series(n: int = 500) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.Series(rng.uniform(1, 5, n), index=idx, name="demand")


# ---------------------------------------------------------------------------
# NaiveForecaster
# ---------------------------------------------------------------------------

class TestNaiveForecaster:
    def _model(self, strategy="seasonal", period=24):
        return NaiveForecaster({"strategy": strategy, "seasonal_period": period})

    def test_predict_length(self):
        y = _hourly_series(200)
        m = self._model()
        m.fit(_empty_x(y.index), y)
        X_test = _empty_x(pd.date_range("2020-09-01", periods=48, freq="h"))
        pred = m.predict(X_test)
        assert len(pred) == 48

    def test_predict_index_matches(self):
        y = _hourly_series(200)
        m = self._model()
        m.fit(_empty_x(y.index), y)
        test_idx = pd.date_range("2020-09-01", periods=24, freq="h")
        pred = m.predict(_empty_x(test_idx))
        assert pred.index.equals(test_idx)

    def test_seasonal_immediate_followup(self):
        """Test starts right after training — standard case."""
        idx = pd.date_range("2020-01-01", periods=100, freq="h")
        y = pd.Series(np.arange(100, dtype=float), index=idx)
        m = self._model(strategy="seasonal", period=24)
        m.fit(_empty_x(idx), y)

        # Test starts at T+1 (hour 100); last 24 training values are 76..99
        test_idx = pd.date_range("2020-01-05 04:00", periods=24, freq="h")
        pred = m.predict(_empty_x(test_idx))
        expected = np.arange(76, 100, dtype=float)
        np.testing.assert_array_equal(pred.values, expected)

    def test_seasonal_phase_aligned_with_gap(self):
        """Test starts after a gap (val set in between) — phase must remain correct."""
        idx = pd.date_range("2020-01-01", periods=100, freq="h")
        y = pd.Series(np.arange(100, dtype=float), index=idx)
        m = self._model(strategy="seasonal", period=24)
        m.fit(_empty_x(idx), y)

        # Gap: test starts 48 h after training ends (= 2 full periods later).
        # Phase offset is zero so predictions should still cycle through the same tail.
        test_idx = pd.date_range("2020-01-05 04:00", periods=24, freq="h") + pd.Timedelta("48h")
        pred = m.predict(_empty_x(test_idx))
        # h for first test ts = 100 - 99 + 48 = 49 steps from train_end
        # pos = (49-1) % 24 = 0 → tail[0] = 76
        assert pred.iloc[0] == 76.0
        # h=50 → pos=1 → tail[1]=77
        assert pred.iloc[1] == 77.0

    def test_last_strategy(self):
        y = _hourly_series(100)
        m = self._model(strategy="last")
        m.fit(_empty_x(y.index), y)
        X_test = _empty_x(pd.date_range("2020-09-01", periods=10, freq="h"))
        pred = m.predict(X_test)
        assert (pred == y.iloc[-1]).all()

    def test_predict_longer_than_period(self):
        y = _hourly_series(200)
        m = self._model(period=24)
        m.fit(_empty_x(y.index), y)
        # Test starts immediately after training
        test_idx = y.index[-1:] + pd.tseries.frequencies.to_offset("h")
        test_idx = pd.date_range(test_idx[0], periods=72, freq="h")
        pred = m.predict(_empty_x(test_idx))
        assert len(pred) == 72
        np.testing.assert_array_equal(pred.values[:24], pred.values[24:48])

    def test_no_nan_in_predictions(self):
        y = _hourly_series(200)
        m = self._model()
        m.fit(_empty_x(y.index), y)
        X_test = _empty_x(pd.date_range("2020-09-01", periods=48, freq="h"))
        pred = m.predict(X_test)
        assert not pred.isnull().any()


# ---------------------------------------------------------------------------
# ARIMAForecaster
# ---------------------------------------------------------------------------

class TestARIMAForecaster:
    # Plain ARIMA(1,0,0) — fast, exercises core fit/predict path
    _cfg_plain = {"model": "arima", "order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0]}
    # Small SARIMA with s=4 — tests the seasonal SARIMAX code path without being slow
    _cfg_sarima = {"model": "arima", "order": [1, 0, 0], "seasonal_order": [1, 0, 0, 4]}

    def _fit(self, cfg, n=200):
        y = _hourly_series(n)
        m = ARIMAForecaster(cfg)
        m.fit(_empty_x(y.index), y)
        return m, y

    def test_predict_length(self):
        m, _ = self._fit(self._cfg_plain)
        pred = m.predict(_empty_x(pd.date_range("2020-09-01", periods=24, freq="h")))
        assert len(pred) == 24

    def test_predict_index_matches(self):
        m, _ = self._fit(self._cfg_plain)
        test_idx = pd.date_range("2020-09-01", periods=24, freq="h")
        pred = m.predict(_empty_x(test_idx))
        assert pred.index.equals(test_idx)

    def test_no_nan_in_predictions(self):
        m, _ = self._fit(self._cfg_plain)
        pred = m.predict(_empty_x(pd.date_range("2020-09-01", periods=24, freq="h")))
        assert not pred.isnull().any()

    def test_predictions_are_finite(self):
        m, _ = self._fit(self._cfg_plain)
        pred = m.predict(_empty_x(pd.date_range("2020-09-01", periods=12, freq="h")))
        assert np.isfinite(pred.values).all()

    def test_sarima_seasonal_path(self):
        """SARIMA with s=4 exercises the seasonal SARIMAX code path."""
        m, _ = self._fit(self._cfg_sarima)
        pred = m.predict(_empty_x(pd.date_range("2020-09-01", periods=8, freq="h")))
        assert len(pred) == 8
        assert np.isfinite(pred.values).all()
