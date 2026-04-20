import pandas as pd
from sklearn.linear_model import Ridge

from src.models.base import ForecasterBase


class LinearForecaster(ForecasterBase):
    """Ridge regression forecaster."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        alpha = float(self.config.get("alpha", 1.0))
        self._model = Ridge(alpha=alpha)
        self._model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        preds = self._model.predict(X_test)
        return pd.Series(preds, index=X_test.index, dtype=float)
