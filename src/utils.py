"""Shared helpers used across models, scripts, and tests."""

import pandas as pd


def empty_x(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return an empty DataFrame with the given DatetimeIndex.

    Passed to univariate models (Naive, ARIMA, Prophet) that derive all signal
    from ``y_train`` and never inspect the feature matrix.
    """
    return pd.DataFrame(index=index)
