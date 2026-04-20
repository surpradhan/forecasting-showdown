import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base import ForecasterBase


class ARIMAForecaster(ForecasterBase):
    """SARIMA model via statsmodels SARIMAX."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        order = tuple(self.config.get("order", [1, 1, 1]))
        seasonal_order = tuple(self.config.get("seasonal_order", [1, 1, 1, 24]))
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp=False)
        if not result.mle_retvals.get("converged", True):
            raise RuntimeError(
                f"SARIMAX failed to converge (order={order}, seasonal_order={seasonal_order}). "
                "Try adjusting model orders or differencing."
            )
        self._result = result

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        n = len(X_test)
        forecasts = self._result.forecast(steps=n)
        return pd.Series(forecasts.values, index=X_test.index, dtype=float)
