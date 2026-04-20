import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.models.base_seq import SeqForecasterBase

pytestmark = pytest.mark.deep


# ---------------------------------------------------------------------------
# Minimal concrete implementation for testing the base class
# ---------------------------------------------------------------------------

class _LinearSeqForecaster(SeqForecasterBase):
    """Wraps a single linear layer — trivial but exercises all base helpers."""

    def fit(self, X_seq: np.ndarray, y: pd.Series) -> None:
        device = self._default_device()
        X_t, y_t = self._to_tensors(X_seq, y)
        _, seq_len, n_features = X_t.shape

        self._net = nn.Linear(seq_len * n_features, 1).to(device)
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=float(self.config.get("lr", 1e-3))
        )
        criterion = nn.MSELoss()
        loader = self._make_loader(X_t, y_t, batch_size=int(self.config.get("batch_size", 32)))

        # Flatten seq dim before the linear layer
        class _FlatWrapper(nn.Module):
            def __init__(self, net): super().__init__(); self.net = net
            def forward(self, x): return self.net(x.flatten(1))

        self._wrapped = _FlatWrapper(self._net).to(device)
        self._run_epochs(
            loader, self._wrapped, optimizer, criterion,
            epochs=int(self.config.get("epochs", 2)), device=device,
        )
        self._device = device

    def predict(self, X_seq: np.ndarray) -> pd.Series:
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self._device)
        self._wrapped.eval()
        with torch.no_grad():
            preds = self._wrapped(X_t).squeeze(-1).cpu().numpy()
        return pd.Series(preds, dtype=float)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seq(n_windows: int = 50, seq_len: int = 6, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_windows, freq="h")
    X_seq = rng.uniform(0, 1, (n_windows, seq_len, n_features)).astype(np.float64)
    y = pd.Series(rng.uniform(1, 5, n_windows), index=idx, name="demand")
    return X_seq, y


_CFG = {"lr": 1e-3, "epochs": 2, "batch_size": 16}


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class TestSeqForecasterBaseInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            SeqForecasterBase({"model": "seq"})  # type: ignore[abstract]

    def test_get_params_returns_config(self):
        m = _LinearSeqForecaster(_CFG)
        assert m.get_params() is _CFG

    def test_fit_predict_roundtrip(self):
        X_seq, y = _make_seq()
        m = _LinearSeqForecaster(_CFG)
        m.fit(X_seq, y)
        pred = m.predict(X_seq)
        assert isinstance(pred, pd.Series)
        assert len(pred) == len(y)

    def test_predict_output_is_finite(self):
        X_seq, y = _make_seq()
        m = _LinearSeqForecaster(_CFG)
        m.fit(X_seq, y)
        pred = m.predict(X_seq)
        assert np.isfinite(pred.values).all()

    def test_predict_no_nan(self):
        X_seq, y = _make_seq()
        m = _LinearSeqForecaster(_CFG)
        m.fit(X_seq, y)
        assert not m.predict(X_seq).isnull().any()


# ---------------------------------------------------------------------------
# _to_tensors
# ---------------------------------------------------------------------------

class TestToTensors:
    def test_shapes_preserved(self):
        X_seq, y = _make_seq(n_windows=20, seq_len=5, n_features=3)
        X_t, y_t = SeqForecasterBase._to_tensors(X_seq, y)
        assert X_t.shape == (20, 5, 3)
        assert y_t.shape == (20,)

    def test_dtype_float32(self):
        X_seq, y = _make_seq()
        X_t, y_t = SeqForecasterBase._to_tensors(X_seq, y)
        assert X_t.dtype == torch.float32
        assert y_t.dtype == torch.float32

    def test_values_match(self):
        X_seq, y = _make_seq(n_windows=10, seq_len=3, n_features=2)
        X_t, y_t = SeqForecasterBase._to_tensors(X_seq, y)
        np.testing.assert_allclose(X_t.numpy(), X_seq, rtol=1e-5)
        np.testing.assert_allclose(y_t.numpy(), y.values, rtol=1e-5)


# ---------------------------------------------------------------------------
# _make_loader
# ---------------------------------------------------------------------------

class TestMakeLoader:
    def test_loader_length(self):
        X_seq, y = _make_seq(n_windows=40)
        X_t, y_t = SeqForecasterBase._to_tensors(X_seq, y)
        loader = SeqForecasterBase._make_loader(X_t, y_t, batch_size=10)
        assert len(loader) == 4  # 40 / 10

    def test_batch_shapes(self):
        X_seq, y = _make_seq(n_windows=30, seq_len=6, n_features=4)
        X_t, y_t = SeqForecasterBase._to_tensors(X_seq, y)
        loader = SeqForecasterBase._make_loader(X_t, y_t, batch_size=10)
        X_b, y_b = next(iter(loader))
        assert X_b.shape == (10, 6, 4)
        assert y_b.shape == (10,)

    def test_no_shuffle_preserves_order(self):
        """First batch must contain the first batch_size samples."""
        X_seq, y = _make_seq(n_windows=20, seq_len=3, n_features=2)
        X_t, y_t = SeqForecasterBase._to_tensors(X_seq, y)
        loader = SeqForecasterBase._make_loader(X_t, y_t, batch_size=5)
        X_b, y_b = next(iter(loader))
        torch.testing.assert_close(X_b, X_t[:5])
        torch.testing.assert_close(y_b, y_t[:5])


# ---------------------------------------------------------------------------
# _default_device
# ---------------------------------------------------------------------------

class TestDefaultDevice:
    def test_returns_torch_device(self):
        device = SeqForecasterBase._default_device()
        assert isinstance(device, torch.device)

    def test_device_is_valid(self):
        device = SeqForecasterBase._default_device()
        # Should be one of cpu / cuda / mps
        assert device.type in {"cpu", "cuda", "mps"}
