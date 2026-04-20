import numpy as np
import pandas as pd


def window_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    seq_len: int,
    horizon: int = 1,
) -> tuple[np.ndarray, pd.Series]:
    """Slide a window over (X, y) to produce fixed-length sequence samples.

    For each window i:
      X_seq[i]  = X.iloc[i : i + seq_len].values       shape (seq_len, n_features)
      y_seq[i]  = y.iloc[i + seq_len + horizon - 1]    one-step or multi-step target

    The returned y_seq carries the prediction timestamps as its DatetimeIndex so
    callers can align predictions back to the original time axis.

    Args:
        X:        Feature DataFrame with DatetimeIndex, shape (N, F).
        y:        Target Series with DatetimeIndex, shape (N,).
        seq_len:  Number of time steps per input window (>= 1).
        horizon:  Steps ahead to predict (>= 1). Default 1 = one-step-ahead.

    Returns:
        X_seq:  ndarray of shape (N_windows, seq_len, F).
        y_seq:  Series of shape (N_windows,) indexed by prediction timestamps.

    Raises:
        ValueError: if seq_len or horizon < 1, or if data is too short to form
                    even one window.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    n = len(X)
    n_windows = n - seq_len - horizon + 1
    if n_windows <= 0:
        raise ValueError(
            f"Not enough rows to form a window: need > {seq_len + horizon - 1}, got {n}"
        )

    X_arr = X.values  # avoid repeated DataFrame overhead inside the loop
    X_seq = np.stack([X_arr[i : i + seq_len] for i in range(n_windows)])

    target_start = seq_len + horizon - 1
    y_seq = y.iloc[target_start : target_start + n_windows].copy()

    return X_seq, y_seq
