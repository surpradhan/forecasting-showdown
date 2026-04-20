import pandas as pd
import xgboost as xgb

from src.models.base import ForecasterBase


class XGBoostForecaster(ForecasterBase):
    """XGBoost regression forecaster."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        early_stopping_rounds = self.config.get("early_stopping_rounds")
        params = {
            "n_estimators": int(self.config.get("n_estimators", 500)),
            "max_depth": int(self.config.get("max_depth", 6)),
            "learning_rate": float(self.config.get("learning_rate", 0.05)),
            "subsample": float(self.config.get("subsample", 0.8)),
            "colsample_bytree": float(self.config.get("colsample_bytree", 0.8)),
            "random_state": int(self.config.get("random_state", 42)),
        }

        if early_stopping_rounds:
            # Use last 10% of training data as internal validation set.
            split = int(len(X_train) * 0.9)
            X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
            y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]
            self._model = xgb.XGBRegressor(
                **params, early_stopping_rounds=int(early_stopping_rounds)
            )
            self._model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self._model = xgb.XGBRegressor(**params)
            self._model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        preds = self._model.predict(X_test)
        return pd.Series(preds, index=X_test.index, dtype=float)
