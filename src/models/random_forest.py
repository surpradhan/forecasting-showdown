import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.base import ForecasterBase


class RandomForestForecaster(ForecasterBase):
    """Random Forest regression forecaster."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._model = RandomForestRegressor(
            n_estimators=int(self.config.get("n_estimators", 200)),
            max_depth=int(self.config.get("max_depth", 10)),
            min_samples_leaf=int(self.config.get("min_samples_leaf", 5)),
            n_jobs=int(self.config.get("n_jobs", -1)),
            random_state=int(self.config.get("random_state", 42)),
        )
        self._model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        preds = self._model.predict(X_test)
        return pd.Series(preds, index=X_test.index, dtype=float)
