import logging

import pandas as pd
from prophet import Prophet

from src.models.base import ForecasterBase


class ProphetForecaster(ForecasterBase):
    """Facebook Prophet — additive seasonality decomposition, univariate."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        # Suppress Stan/Prophet noise. setLevel is process-global but called
        # here (not at import) so it only fires when a model is actually trained.
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)

        df = pd.DataFrame({"ds": y_train.index, "y": y_train.values})
        self._model = Prophet(
            yearly_seasonality=self.config.get("yearly_seasonality", True),
            weekly_seasonality=self.config.get("weekly_seasonality", True),
            daily_seasonality=self.config.get("daily_seasonality", True),
            changepoint_prior_scale=float(
                self.config.get("changepoint_prior_scale", 0.05)
            ),
        )
        self._model.fit(df)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        future = pd.DataFrame({"ds": X_test.index})
        forecast = self._model.predict(future)
        return pd.Series(forecast["yhat"].values, index=X_test.index, dtype=float)
