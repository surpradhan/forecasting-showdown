import pandas as pd


# Lags and rolling window keyed by canonical pandas freq alias (lowercase).
# Covers the frequencies most likely to appear in energy benchmarks.
_LAG_MAP: dict[str, list[int]] = {
    "h":     [1, 24, 168],   # hourly: 1h, 1d, 1w
    "d":     [1, 7, 30],     # daily:  1d, 1w, ~1mo
    "15min": [4, 96, 672],   # 15-min: 1h, 1d, 1w
    "30min": [2, 48, 336],   # 30-min: 1h, 1d, 1w
}
_ROLL_MAP: dict[str, int] = {
    "h":     24,
    "d":     7,
    "15min": 96,
    "30min": 48,
}
_DEFAULT_LAGS = [1, 24, 168]
_DEFAULT_ROLL = 24


def build_features(
    df: pd.DataFrame,
    freq: str = "h",
    target_col: str = "demand",
) -> pd.DataFrame:
    """Add lag, rolling, and calendar features to a time-indexed DataFrame."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in df")

    freq_key = freq.lower()
    lags = _LAG_MAP.get(freq_key, _DEFAULT_LAGS)
    roll_window = _ROLL_MAP.get(freq_key, _DEFAULT_ROLL)

    out = df.copy()

    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    shifted = out[target_col].shift(1)
    out[f"roll_mean_{roll_window}"] = shifted.rolling(roll_window).mean()
    out[f"roll_std_{roll_window}"]  = shifted.rolling(roll_window).std()
    out[f"roll_min_{roll_window}"]  = shifted.rolling(roll_window).min()
    out[f"roll_max_{roll_window}"]  = shifted.rolling(roll_window).max()

    idx = out.index
    out["hour"]        = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["month"]       = idx.month
    out["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    out["season"]      = (idx.month % 12 // 3)  # 0=winter,1=spring,2=summer,3=autumn

    return out.dropna()
