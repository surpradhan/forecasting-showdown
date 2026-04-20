import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train / val / test without shuffling."""
    if val_frac <= 0 or test_frac <= 0:
        raise ValueError("val_frac and test_frac must both be positive")
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac + test_frac = {val_frac + test_frac:.3f} leaves no training data"
        )

    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - val_frac - test_frac))

    train = df.iloc[:val_start]
    val = df.iloc[val_start:test_start]
    test = df.iloc[test_start:]
    return train, val, test
