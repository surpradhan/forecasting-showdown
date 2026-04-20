import numpy as np
import pandas as pd
import pytest

from src.data.windowing import window_sequences
from src.evaluation.runner import evaluate_model
from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster

pytestmark = pytest.mark.deep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N, _SEQ_LEN, _N_FEATURES = 80, 12, 6


def _make_seq(seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=_N, freq="h")
    X_seq = rng.uniform(0, 1, (_N, _SEQ_LEN, _N_FEATURES)).astype(np.float32)
    y = pd.Series(rng.uniform(1, 5, _N), index=idx, name="demand")
    return X_seq, y


_LSTM_CFG = {"hidden_size": 16, "num_layers": 1, "dropout": 0.0, "batch_size": 16, "epochs": 2, "lr": 1e-3}
_GRU_CFG = {"hidden_size": 16, "num_layers": 1, "dropout": 0.0, "batch_size": 16, "epochs": 2, "lr": 1e-3}
_TRANSFORMER_CFG = {
    "d_model": 16, "nhead": 2, "num_encoder_layers": 1,
    "dim_feedforward": 32, "dropout": 0.0, "batch_size": 16, "epochs": 2, "lr": 5e-4,
}


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class TestLSTMForecaster:
    def test_fit_predict_roundtrip(self):
        X_seq, y = _make_seq()
        m = LSTMForecaster(_LSTM_CFG)
        m.fit(X_seq, y)
        pred = m.predict(X_seq)
        assert isinstance(pred, pd.Series)
        assert len(pred) == _N

    def test_output_is_finite(self):
        X_seq, y = _make_seq()
        m = LSTMForecaster(_LSTM_CFG)
        m.fit(X_seq, y)
        assert np.isfinite(m.predict(X_seq).values).all()

    def test_no_nan(self):
        X_seq, y = _make_seq()
        m = LSTMForecaster(_LSTM_CFG)
        m.fit(X_seq, y)
        assert not m.predict(X_seq).isnull().any()

    def test_get_params(self):
        m = LSTMForecaster(_LSTM_CFG)
        assert m.get_params() is _LSTM_CFG

    def test_multilayer_dropout(self):
        cfg = {**_LSTM_CFG, "num_layers": 2, "dropout": 0.2}
        X_seq, y = _make_seq()
        m = LSTMForecaster(cfg)
        m.fit(X_seq, y)
        assert len(m.predict(X_seq)) == _N


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class TestGRUForecaster:
    def test_fit_predict_roundtrip(self):
        X_seq, y = _make_seq()
        m = GRUForecaster(_GRU_CFG)
        m.fit(X_seq, y)
        pred = m.predict(X_seq)
        assert isinstance(pred, pd.Series)
        assert len(pred) == _N

    def test_output_is_finite(self):
        X_seq, y = _make_seq()
        m = GRUForecaster(_GRU_CFG)
        m.fit(X_seq, y)
        assert np.isfinite(m.predict(X_seq).values).all()

    def test_no_nan(self):
        X_seq, y = _make_seq()
        m = GRUForecaster(_GRU_CFG)
        m.fit(X_seq, y)
        assert not m.predict(X_seq).isnull().any()

    def test_get_params(self):
        m = GRUForecaster(_GRU_CFG)
        assert m.get_params() is _GRU_CFG

    def test_multilayer_dropout(self):
        cfg = {**_GRU_CFG, "num_layers": 2, "dropout": 0.2}
        X_seq, y = _make_seq()
        m = GRUForecaster(cfg)
        m.fit(X_seq, y)
        assert len(m.predict(X_seq)) == _N


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class TestTransformerForecaster:
    def test_fit_predict_roundtrip(self):
        X_seq, y = _make_seq()
        m = TransformerForecaster(_TRANSFORMER_CFG)
        m.fit(X_seq, y)
        pred = m.predict(X_seq)
        assert isinstance(pred, pd.Series)
        assert len(pred) == _N

    def test_output_is_finite(self):
        X_seq, y = _make_seq()
        m = TransformerForecaster(_TRANSFORMER_CFG)
        m.fit(X_seq, y)
        assert np.isfinite(m.predict(X_seq).values).all()

    def test_no_nan(self):
        X_seq, y = _make_seq()
        m = TransformerForecaster(_TRANSFORMER_CFG)
        m.fit(X_seq, y)
        assert not m.predict(X_seq).isnull().any()

    def test_get_params(self):
        m = TransformerForecaster(_TRANSFORMER_CFG)
        assert m.get_params() is _TRANSFORMER_CFG

    def test_nhead_not_dividing_d_model_raises(self):
        cfg = {**_TRANSFORMER_CFG, "d_model": 16, "nhead": 3}
        X_seq, y = _make_seq()
        with pytest.raises(ValueError, match="d_model"):
            TransformerForecaster(cfg).fit(X_seq, y)

    def test_nhead_divides_d_model(self):
        cfg = {**_TRANSFORMER_CFG, "nhead": 4, "d_model": 16}
        X_seq, y = _make_seq()
        m = TransformerForecaster(cfg)
        m.fit(X_seq, y)
        assert len(m.predict(X_seq)) == _N


# ---------------------------------------------------------------------------
# Runner integration — preprocess_fn wiring and DatetimeIndex restoration
# ---------------------------------------------------------------------------

def _make_flat(n: int = 200, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame(rng.uniform(0, 1, (n, n_features)), index=idx,
                     columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.uniform(1, 5, n), index=idx, name="demand")
    return X, y


_SEQ_LEN = 10


def _make_preprocess_fn(seq_len: int):
    return lambda Xtr, ytr, Xte, yte: (
        *window_sequences(Xtr, ytr, seq_len),
        *window_sequences(Xte, yte, seq_len),
    )


class TestRunnerIntegration:
    def _run(self, model, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        X, y = _make_flat()
        split = 160
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]
        return evaluate_model(
            model, X_train, y_train, X_test, y_test,
            run_name="test_seq",
            experiment_name="test",
            preprocess_fn=_make_preprocess_fn(_SEQ_LEN),
        )

    def test_lstm_runner_returns_expected_keys(self, tmp_path, monkeypatch):
        result = self._run(LSTMForecaster({**_LSTM_CFG, "epochs": 1}), tmp_path, monkeypatch)
        assert set(result.keys()) == {"model", "mae", "rmse", "mape", "smape", "latency_s", "train_time_s"}

    def test_gru_runner_returns_expected_keys(self, tmp_path, monkeypatch):
        result = self._run(GRUForecaster({**_GRU_CFG, "epochs": 1}), tmp_path, monkeypatch)
        assert set(result.keys()) == {"model", "mae", "rmse", "mape", "smape", "latency_s", "train_time_s"}

    def test_transformer_runner_returns_expected_keys(self, tmp_path, monkeypatch):
        result = self._run(TransformerForecaster({**_TRANSFORMER_CFG, "epochs": 1}), tmp_path, monkeypatch)
        assert set(result.keys()) == {"model", "mae", "rmse", "mape", "smape", "latency_s", "train_time_s"}

    def test_scores_are_finite(self, tmp_path, monkeypatch):
        result = self._run(LSTMForecaster({**_LSTM_CFG, "epochs": 1}), tmp_path, monkeypatch)
        for key in ("mae", "rmse", "mape", "smape"):
            assert np.isfinite(result[key]), f"{key} is not finite"

    def test_y_pred_gets_datetimeindex(self, tmp_path, monkeypatch):
        """Runner must re-attach DatetimeIndex to sequence model predictions."""
        monkeypatch.chdir(tmp_path)
        X, y = _make_flat()
        split = 160
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]

        captured = {}
        original_predict = LSTMForecaster.predict

        def _patched_predict(self, X_seq):
            pred = original_predict(self, X_seq)
            captured["raw_index_type"] = type(pred.index).__name__
            return pred

        model = LSTMForecaster({**_LSTM_CFG, "epochs": 1})
        model.predict = lambda X_seq: _patched_predict(model, X_seq)

        evaluate_model(
            model, X_train, y_train, X_test, y_test,
            run_name="index_test", experiment_name="test",
            preprocess_fn=_make_preprocess_fn(_SEQ_LEN),
        )
        assert captured["raw_index_type"] == "RangeIndex"
