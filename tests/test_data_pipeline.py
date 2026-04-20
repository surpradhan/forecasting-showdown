import pandas as pd
import pytest

from src.data.features import build_features
from src.data.splits import chronological_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hourly_df(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({"demand": range(n)}, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------

def test_load_raw_roundtrip(tmp_path):
    from src.data.loader import load_raw

    df = _make_hourly_df(48)
    csv_path = tmp_path / "energy.csv"
    df.reset_index().rename(columns={"index": "timestamp"}).to_csv(csv_path, index=False)

    loaded = load_raw(csv_path)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert "demand" in loaded.columns
    assert len(loaded) == 48


def test_load_raw_sorts_by_timestamp(tmp_path):
    from src.data.loader import load_raw

    df = _make_hourly_df(10).reset_index().rename(columns={"index": "timestamp"})
    df = df.iloc[::-1]  # reverse order
    csv_path = tmp_path / "energy.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_raw(csv_path)
    assert loaded.index.is_monotonic_increasing


def test_load_raw_missing_column_raises(tmp_path):
    from src.data.loader import load_raw

    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=5, freq="h"),
                        "not_demand": range(5)})
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_raw(csv_path)


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

def test_build_features_adds_expected_columns():
    df = _make_hourly_df(300)
    out = build_features(df)

    expected = {"lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_std_24",
                "roll_min_24", "roll_max_24", "hour", "day_of_week", "month",
                "is_weekend", "season"}
    assert expected.issubset(out.columns)


def test_build_features_no_nan():
    df = _make_hourly_df(300)
    out = build_features(df)
    assert not out.isnull().any().any()


def test_build_features_shorter_than_input():
    df = _make_hourly_df(300)
    out = build_features(df)
    # lag_168 dominates: first 168 rows become NaN and are dropped
    assert len(out) == len(df) - 168


def test_build_features_requires_datetimeindex():
    df = pd.DataFrame({"demand": range(50)})
    with pytest.raises(ValueError):
        build_features(df)


def test_build_features_missing_target_col_raises():
    df = _make_hourly_df(300)
    with pytest.raises(ValueError, match="target column"):
        build_features(df, target_col="load")


def test_build_features_lag_values_correct():
    df = _make_hourly_df(300)
    out = build_features(df)
    # lag_1 at row i should equal demand at row i-1 in the original df
    # After dropna the first surviving row has original index 168 (0-based).
    # Check a handful of rows.
    for orig_idx in [168, 169, 200]:
        ts = df.index[orig_idx]
        assert out.loc[ts, "lag_1"]   == df["demand"].iloc[orig_idx - 1]
        assert out.loc[ts, "lag_24"]  == df["demand"].iloc[orig_idx - 24]
        assert out.loc[ts, "lag_168"] == df["demand"].iloc[orig_idx - 168]


def test_build_features_daily_freq_uses_correct_lags():
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({"demand": range(200)}, index=idx, dtype=float)
    out = build_features(df, freq="D")
    assert "lag_7" in out.columns
    assert "lag_24" not in out.columns


def test_build_features_hour_range():
    df = _make_hourly_df(300)
    out = build_features(df)
    assert out["hour"].between(0, 23).all()


def test_build_features_season_range():
    df = _make_hourly_df(300)
    out = build_features(df)
    assert out["season"].between(0, 3).all()


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------

def test_chronological_split_sizes():
    df = _make_hourly_df(1000)
    train, val, test = chronological_split(df, val_frac=0.1, test_frac=0.1)
    assert len(train) + len(val) + len(test) == 1000
    assert abs(len(val) - 100) <= 1
    assert abs(len(test) - 100) <= 1


def test_chronological_split_no_overlap():
    df = _make_hourly_df(1000)
    train, val, test = chronological_split(df, val_frac=0.1, test_frac=0.1)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_chronological_split_order_preserved():
    df = _make_hourly_df(1000)
    train, val, test = chronological_split(df)
    for part in (train, val, test):
        assert part.index.is_monotonic_increasing


def test_chronological_split_rejects_bad_fracs():
    df = _make_hourly_df(100)
    with pytest.raises(ValueError):
        chronological_split(df, val_frac=0.5, test_frac=0.6)
    with pytest.raises(ValueError):
        chronological_split(df, val_frac=0.0, test_frac=0.1)
