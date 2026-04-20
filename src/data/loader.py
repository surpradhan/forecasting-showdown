from pathlib import Path

import pandas as pd

_REQUIRED_COLUMNS = {"timestamp", "demand"}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load raw CSV dataset. Expected columns: timestamp, demand, plus optional exogenous."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df.sort_values("timestamp").set_index("timestamp")
    return df
