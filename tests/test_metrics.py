import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import mae, rmse, mape, smape, all_metrics


Y_TRUE = pd.Series([100.0, 200.0, 300.0, 400.0])
Y_PRED = pd.Series([110.0, 190.0, 310.0, 390.0])  # all off by 10


def test_mae():
    assert mae(Y_TRUE, Y_PRED) == pytest.approx(10.0)


def test_rmse():
    assert rmse(Y_TRUE, Y_PRED) == pytest.approx(10.0)


def test_mape():
    expected = np.mean([10 / 100, 10 / 200, 10 / 300, 10 / 400]) * 100
    assert mape(Y_TRUE, Y_PRED) == pytest.approx(expected)


def test_smape():
    nums = np.abs(Y_TRUE - Y_PRED)
    denoms = (np.abs(Y_TRUE) + np.abs(Y_PRED)) / 2
    expected = float(np.mean(nums / denoms) * 100)
    assert smape(Y_TRUE, Y_PRED) == pytest.approx(expected)


def test_mape_skips_zeros():
    y_true = pd.Series([0.0, 100.0, 200.0])
    y_pred = pd.Series([10.0, 110.0, 190.0])
    result = mape(y_true, y_pred)
    assert np.isfinite(result)


def test_mape_all_zeros_returns_nan():
    y_true = pd.Series([0.0, 0.0, 0.0])
    y_pred = pd.Series([1.0, 2.0, 3.0])
    with pytest.warns(RuntimeWarning, match="all y_true values are zero"):
        result = mape(y_true, y_pred)
    assert np.isnan(result)


def test_all_metrics_keys():
    result = all_metrics(Y_TRUE, Y_PRED)
    assert set(result.keys()) == {"mae", "rmse", "mape", "smape"}


def test_perfect_prediction():
    y = pd.Series([50.0, 100.0, 150.0])
    result = all_metrics(y, y)
    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["mape"] == pytest.approx(0.0)
    assert result["smape"] == pytest.approx(0.0)
