"""Tests for scripts/run_all.py — helpers, error handling, group runners, orchestrator.

Must be run in its own process because importing run_all loads torch + prophet +
xgboost simultaneously, which conflicts with macOS ARM libomp:

    uv run pytest tests/test_run_all.py --override-ini="addopts="
"""

import numpy as np
import pandas as pd
import pytest

from src.models.base import ForecasterBase


# Deferred import: load the module once per session to avoid repeated conflicts.
@pytest.fixture(scope="module")
def run_all():
    import scripts.run_all as m
    return m


# Tiny dataset large enough to survive build_features (needs >=168 rows for hourly lags)
@pytest.fixture(scope="module")
def small_dataset():
    idx = pd.date_range("2020-01-01", periods=500, freq="h")
    y = np.sin(np.arange(500) * 2 * np.pi / 24) + 2.0
    X = pd.DataFrame({"demand": y}, index=idx)
    return X


class ConstantModel(ForecasterBase):
    def fit(self, X, y):
        self._val = float(y.mean())

    def predict(self, X):
        return pd.Series([self._val] * len(X), index=X.index)


class BrokenModel(ForecasterBase):
    def fit(self, X, y):
        raise RuntimeError("intentional failure")

    def predict(self, X):
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# _seq_preprocess
# ---------------------------------------------------------------------------

def test_seq_preprocess_output_shape(run_all):
    X = pd.DataFrame(
        {"f1": range(100), "f2": range(100)},
        index=pd.date_range("2020-01-01", periods=100, freq="h"),
    )
    y = pd.Series(np.arange(100, dtype=float), index=X.index)
    fn = run_all._seq_preprocess(seq_len=10)
    X_seq, y_seq, Xte_seq, yte_seq = fn(X, y, X, y)
    assert X_seq.shape == (90, 10, 2)
    assert len(y_seq) == 90


def test_seq_preprocess_seq_len_binding(run_all):
    """Each closure captures its own seq_len independently."""
    X = pd.DataFrame({"f": range(50)}, index=pd.date_range("2020", periods=50, freq="h"))
    y = pd.Series(np.arange(50, dtype=float), index=X.index)
    fn5  = run_all._seq_preprocess(seq_len=5)
    fn10 = run_all._seq_preprocess(seq_len=10)
    X5,  *_ = fn5(X, y, X, y)
    X10, *_ = fn10(X, y, X, y)
    assert X5.shape[0]  == 45  # 50 - 5
    assert X10.shape[0] == 40  # 50 - 10


# ---------------------------------------------------------------------------
# _print_table
# ---------------------------------------------------------------------------

def test_print_table_contains_all_models(run_all, capsys):
    results = [
        {"model": "naive",   "mae": 2.0, "rmse": 3.0, "mape": 0.2,  "smape": 0.18, "latency_s": 0.01, "train_time_s": 0.0},
        {"model": "xgboost", "mae": 0.5, "rmse": 0.8, "mape": 0.05, "smape": 0.04, "latency_s": 0.1,  "train_time_s": 5.0},
    ]
    run_all._print_table(results)
    out = capsys.readouterr().out
    assert "naive" in out
    assert "xgboost" in out
    assert "mae" in out.lower()


def test_print_table_nan_does_not_crash(run_all, capsys):
    results = [
        {"model": "bad", "mae": float("nan"), "rmse": float("nan"),
         "mape": float("nan"), "smape": float("nan"), "latency_s": 0.0, "train_time_s": 0.0},
    ]
    run_all._print_table(results)  # must not raise


# ---------------------------------------------------------------------------
# _run — error handling
# ---------------------------------------------------------------------------

def test_run_returns_none_on_failure(run_all, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X = pd.DataFrame({"f": range(10)}, index=pd.date_range("2020", periods=10, freq="h"))
    y = pd.Series(np.ones(10), index=X.index)
    result = run_all._run("broken", BrokenModel({}), X, y, X, y,
                          experiment_name="test_run_all")
    assert result is None


def test_run_returns_dict_on_success(run_all, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X = pd.DataFrame({"f": range(20)}, index=pd.date_range("2020", periods=20, freq="h"))
    y = pd.Series(np.arange(20, dtype=float), index=X.index)
    result = run_all._run("constant", ConstantModel({"model": "constant"}),
                          X, y, X, y, experiment_name="test_run_all")
    assert result is not None
    assert result["model"] == "constant"
    assert result["mae"] >= 0


# ---------------------------------------------------------------------------
# group runners — monkeypatch _run and _load_data to avoid real I/O
# ---------------------------------------------------------------------------

def _fake_load_data(small_dataset, run_all, monkeypatch):
    """Patch _load_data to return tiny in-memory splits."""
    from src.data.features import build_features
    from src.data.splits import chronological_split

    df = build_features(small_dataset.copy(), freq="h")
    train, _val, test = chronological_split(df)
    feature_cols = [c for c in df.columns if c != "demand"]

    monkeypatch.setattr(
        run_all, "_load_data",
        lambda: (train[feature_cols], train["demand"],
                 test[feature_cols],  test["demand"])
    )


def _fake_load_data_tuple(small_dataset):
    """Return a real data tuple (no monkeypatching) for tests that need to pass it as a lambda."""
    from src.data.features import build_features
    from src.data.splits import chronological_split

    df = build_features(small_dataset.copy(), freq="h")
    train, _val, test = chronological_split(df)
    feature_cols = [c for c in df.columns if c != "demand"]
    return (train[feature_cols], train["demand"], test[feature_cols], test["demand"])


def _fake_run(captured: list):
    """Return a _run replacement that records calls and returns a stub result."""
    def fn(name, model, *args, **kwargs):
        captured.append(name)
        return {"model": name, "mae": 0.5, "rmse": 0.7,
                "mape": 0.05, "smape": 0.04,
                "latency_s": 0.01, "train_time_s": 0.1}
    return fn


def test_run_univariate_calls_all_three_models(run_all, small_dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _fake_load_data(small_dataset, run_all, monkeypatch)
    called = []
    monkeypatch.setattr(run_all, "_run", _fake_run(called))
    run_all.run_univariate(only=None)
    assert called == ["naive", "arima", "prophet"]


def test_run_univariate_only_filter(run_all, small_dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _fake_load_data(small_dataset, run_all, monkeypatch)
    called = []
    monkeypatch.setattr(run_all, "_run", _fake_run(called))
    run_all.run_univariate(only="prophet")
    assert called == ["prophet"]


def test_run_tabular_calls_all_five_models(run_all, small_dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _fake_load_data(small_dataset, run_all, monkeypatch)
    called = []
    monkeypatch.setattr(run_all, "_run", _fake_run(called))
    run_all.run_tabular(only=None)
    assert called == ["linear", "random_forest", "xgboost", "lgbm", "ensemble"]


def test_run_tabular_only_filter(run_all, small_dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _fake_load_data(small_dataset, run_all, monkeypatch)
    called = []
    monkeypatch.setattr(run_all, "_run", _fake_run(called))
    run_all.run_tabular(only="lgbm")
    assert called == ["lgbm"]


def test_run_tabular_only_filter_ensemble(run_all, small_dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _fake_load_data(small_dataset, run_all, monkeypatch)
    called = []
    monkeypatch.setattr(run_all, "_run", _fake_run(called))
    run_all.run_tabular(only="ensemble")
    assert called == ["ensemble"]


def test_run_sequential_calls_all_three_models(run_all, small_dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _fake_load_data(small_dataset, run_all, monkeypatch)
    called = []
    monkeypatch.setattr(run_all, "_run", _fake_run(called))
    run_all.run_sequential(only=None)
    assert called == ["lstm", "gru", "transformer"]


# ---------------------------------------------------------------------------
# orchestrator — monkeypatch subprocess dispatch
# ---------------------------------------------------------------------------

def test_main_orchestrator_runs_all_groups(run_all, small_dataset, monkeypatch, tmp_path):
    """main() with no group/model should invoke _run_group_subprocess for each group."""
    monkeypatch.chdir(tmp_path)
    # main() calls _load_data() for the header print; patch it
    monkeypatch.setattr(run_all, "_load_data", lambda: _fake_load_data_tuple(small_dataset))

    dispatched = []

    def fake_subprocess(group, model=None):
        dispatched.append((group, model))
        return 0

    monkeypatch.setattr(run_all, "_run_group_subprocess", fake_subprocess)
    monkeypatch.setattr(run_all, "_read_mlflow_results", lambda: [])

    run_all.main(group=None, model=None)
    assert [g for g, _ in dispatched] == run_all.GROUPS
    assert all(m is None for _, m in dispatched)


def test_main_model_flag_routes_to_correct_group(run_all, small_dataset, monkeypatch, tmp_path):
    """`main(model='lgbm')` should dispatch only the tabular group subprocess."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_all, "_load_data", lambda: _fake_load_data_tuple(small_dataset))

    dispatched = []

    def fake_subprocess(group, model=None):
        dispatched.append((group, model))
        return 0

    monkeypatch.setattr(run_all, "_run_group_subprocess", fake_subprocess)
    monkeypatch.setattr(run_all, "_read_mlflow_results", lambda: [])

    run_all.main(group=None, model="lgbm")
    assert dispatched == [("tabular", "lgbm")]


def test_main_group_univariate_does_not_subprocess(run_all, small_dataset, monkeypatch, tmp_path):
    """`main(group='univariate')` must call run_univariate(), not spawn a subprocess."""
    monkeypatch.chdir(tmp_path)
    called = []
    monkeypatch.setattr(run_all, "run_univariate", lambda only=None: called.append("univariate"))
    run_all.main(group="univariate")
    assert called == ["univariate"]
