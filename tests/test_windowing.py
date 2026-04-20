import numpy as np
import pandas as pd
import pytest

from src.data.windowing import window_sequences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int, n_features: int = 3, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame(rng.uniform(0, 1, (n, n_features)), index=idx,
                     columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(np.arange(n, dtype=float), index=idx, name="demand")
    return X, y


# ---------------------------------------------------------------------------
# Shape correctness
# ---------------------------------------------------------------------------

class TestWindowSequencesShape:
    def test_horizon1_shape(self):
        X, y = _make_df(100, n_features=3)
        X_seq, y_seq = window_sequences(X, y, seq_len=10, horizon=1)
        assert X_seq.shape == (90, 10, 3)   # 100 - 10 windows
        assert len(y_seq) == 90

    def test_horizon_gt1_shape(self):
        X, y = _make_df(100, n_features=3)
        X_seq, y_seq = window_sequences(X, y, seq_len=10, horizon=3)
        # n_windows = 100 - 10 - 3 + 1 = 88
        assert X_seq.shape == (88, 10, 3)
        assert len(y_seq) == 88

    def test_seq_len_equals_n_minus_1(self):
        X, y = _make_df(50, n_features=2)
        X_seq, y_seq = window_sequences(X, y, seq_len=49, horizon=1)
        assert X_seq.shape == (1, 49, 2)
        assert len(y_seq) == 1

    def test_single_feature(self):
        X, y = _make_df(20, n_features=1)
        X_seq, y_seq = window_sequences(X, y, seq_len=5)
        assert X_seq.shape == (15, 5, 1)

    def test_dtype_is_float(self):
        X, y = _make_df(30)
        X_seq, _ = window_sequences(X, y, seq_len=5)
        assert X_seq.dtype == np.float64


# ---------------------------------------------------------------------------
# Target alignment
# ---------------------------------------------------------------------------

class TestWindowSequencesAlignment:
    def test_first_window_content(self):
        """X_seq[0] must equal X.iloc[0:seq_len].values exactly."""
        X, y = _make_df(50)
        seq_len = 7
        X_seq, _ = window_sequences(X, y, seq_len=seq_len)
        np.testing.assert_array_equal(X_seq[0], X.iloc[:seq_len].values)

    def test_last_window_content(self):
        """X_seq[-1] must equal X.iloc[n-seq_len:n].values (horizon=1)."""
        X, y = _make_df(50)
        seq_len = 7
        X_seq, _ = window_sequences(X, y, seq_len=seq_len)
        np.testing.assert_array_equal(X_seq[-1], X.iloc[50 - seq_len - 1 : 50 - 1].values)

    def test_target_is_step_after_window(self):
        """y_seq[i] == y.iloc[i + seq_len] for horizon=1."""
        X, y = _make_df(30)
        seq_len = 5
        X_seq, y_seq = window_sequences(X, y, seq_len=seq_len, horizon=1)
        for i in range(len(y_seq)):
            assert y_seq.iloc[i] == y.iloc[i + seq_len]

    def test_target_horizon3(self):
        """y_seq[i] == y.iloc[i + seq_len + horizon - 1] for horizon=3."""
        X, y = _make_df(30)
        seq_len, horizon = 4, 3
        _, y_seq = window_sequences(X, y, seq_len=seq_len, horizon=horizon)
        for i in range(len(y_seq)):
            assert y_seq.iloc[i] == y.iloc[i + seq_len + horizon - 1]

    def test_y_seq_index_is_datetimeindex(self):
        X, y = _make_df(40)
        _, y_seq = window_sequences(X, y, seq_len=5)
        assert isinstance(y_seq.index, pd.DatetimeIndex)

    def test_y_seq_index_matches_prediction_timestamps(self):
        """y_seq.index[i] == y.index[i + seq_len] for horizon=1."""
        X, y = _make_df(40)
        seq_len = 6
        _, y_seq = window_sequences(X, y, seq_len=seq_len, horizon=1)
        expected_index = y.index[seq_len : seq_len + len(y_seq)]
        assert y_seq.index.equals(expected_index)

    def test_consecutive_windows_overlap_correctly(self):
        """Window i and window i+1 share seq_len-1 rows."""
        X, y = _make_df(20)
        seq_len = 5
        X_seq, _ = window_sequences(X, y, seq_len=seq_len)
        # X_seq[1][0] == X_seq[0][1], X_seq[1][1] == X_seq[0][2], ...
        np.testing.assert_array_equal(X_seq[1][:-1], X_seq[0][1:])


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------

class TestWindowSequencesErrors:
    def test_seq_len_zero_raises(self):
        X, y = _make_df(20)
        with pytest.raises(ValueError, match="seq_len"):
            window_sequences(X, y, seq_len=0)

    def test_horizon_zero_raises(self):
        X, y = _make_df(20)
        with pytest.raises(ValueError, match="horizon"):
            window_sequences(X, y, seq_len=5, horizon=0)

    def test_too_short_raises(self):
        X, y = _make_df(10)
        with pytest.raises(ValueError, match="Not enough rows"):
            window_sequences(X, y, seq_len=10, horizon=1)

    def test_exactly_at_limit_raises(self):
        """n_windows = 0 should raise; n_windows = 1 should not."""
        X, y = _make_df(10)
        # seq_len=9, horizon=1 → n_windows = 10-9-1+1 = 1 → valid, no raise
        X_seq, y_seq = window_sequences(X, y, seq_len=9, horizon=1)
        assert len(y_seq) == 1
        # seq_len=10, horizon=1 → n_windows = 10-10-1+1 = 0 → raises
        with pytest.raises(ValueError, match="Not enough rows"):
            window_sequences(X, y, seq_len=10, horizon=1)

    def test_no_data_mutation(self):
        """window_sequences must not modify the input arrays."""
        X, y = _make_df(30)
        X_copy = X.copy()
        y_copy = y.copy()
        window_sequences(X, y, seq_len=5)
        pd.testing.assert_frame_equal(X, X_copy)
        pd.testing.assert_series_equal(y, y_copy)
