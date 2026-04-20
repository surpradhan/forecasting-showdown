import numpy as np
import pandas as pd
import pytest

from src.evaluation.runner import evaluate_model
from src.models.base import ForecasterBase


class ConstantForecaster(ForecasterBase):
    """Returns the training mean for every prediction — useful as a test double."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._mean = float(y_train.mean())

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return pd.Series([self._mean] * len(X_test), index=X_test.index)


class NumpyForecaster(ForecasterBase):
    """Returns a numpy array from predict() to verify the runner handles both types."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._mean = float(y_train.mean())

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return pd.Series(np.full(len(X_test), self._mean), index=X_test.index)


N = 100
X_TRAIN = pd.DataFrame({"f": range(N)})
Y_TRAIN = pd.Series(np.linspace(10, 50, N))
X_TEST = pd.DataFrame({"f": range(N, N + 20)})
Y_TEST = pd.Series(np.linspace(10, 50, 20))


def test_returns_expected_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = evaluate_model(
        ConstantForecaster({"model": "constant"}),
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
        run_name="test_constant",
        experiment_name="test",
    )
    assert set(result.keys()) == {"model", "mae", "rmse", "mape", "smape", "latency_s", "train_time_s"}


def test_model_name_in_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = evaluate_model(
        ConstantForecaster({"model": "constant"}),
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
        run_name="my_model",
        experiment_name="test",
    )
    assert result["model"] == "my_model"


def test_timing_fields_are_non_negative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = evaluate_model(
        ConstantForecaster({"model": "constant"}),
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
        run_name="timing_test",
        experiment_name="test",
    )
    assert result["train_time_s"] >= 0
    assert result["latency_s"] >= 0


def test_numpy_predict_output_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = evaluate_model(
        NumpyForecaster({"model": "numpy"}),
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
        run_name="numpy_test",
        experiment_name="test",
    )
    assert np.isfinite(result["mae"])


def test_preprocess_fn_transforms_data(tmp_path, monkeypatch):
    """preprocess_fn should receive and transform all four data args before fit/predict."""
    monkeypatch.chdir(tmp_path)

    calls = {}

    class RecordingForecaster(ForecasterBase):
        def fit(self, X_train, y_train):
            calls["X_train"] = X_train
            calls["y_train"] = y_train
            self._mean = float(y_train.mean())

        def predict(self, X_test):
            calls["X_test"] = X_test
            return pd.Series([self._mean] * len(X_test))

    def _double(X_tr, y_tr, X_te, y_te):
        return X_tr * 2, y_tr * 2, X_te * 2, y_te * 2

    evaluate_model(
        RecordingForecaster({"model": "rec"}),
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
        run_name="preprocess_test",
        experiment_name="test",
        preprocess_fn=_double,
    )

    # The model should have seen doubled data
    np.testing.assert_array_equal(calls["X_train"].values, (X_TRAIN * 2).values)
    np.testing.assert_array_equal(calls["y_train"].values, (Y_TRAIN * 2).values)
    np.testing.assert_array_equal(calls["X_test"].values, (X_TEST * 2).values)


def test_preprocess_fn_none_is_noop(tmp_path, monkeypatch):
    """preprocess_fn=None must leave data unchanged — backward-compatible default."""
    monkeypatch.chdir(tmp_path)
    result = evaluate_model(
        ConstantForecaster({"model": "constant"}),
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
        run_name="noop_test",
        experiment_name="test",
        preprocess_fn=None,
    )
    assert set(result.keys()) == {"model", "mae", "rmse", "mape", "smape", "latency_s", "train_time_s"}


def test_perfect_model_scores_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class PerfectForecaster(ForecasterBase):
        def fit(self, X_train, y_train): pass
        def predict(self, X_test): return Y_TEST

    result = evaluate_model(
        PerfectForecaster({"model": "perfect"}),
        X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
        run_name="perfect",
        experiment_name="test",
    )
    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
