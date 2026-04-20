import pandas as pd
import lightgbm as lgb

from src.models.base import ForecasterBase


class LGBMForecaster(ForecasterBase):
    """LightGBM regression forecaster."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        early_stopping_rounds = self.config.get("early_stopping_rounds")
        params = {
            "n_estimators": int(self.config.get("n_estimators", 500)),
            "max_depth": int(self.config.get("max_depth", -1)),
            "learning_rate": float(self.config.get("learning_rate", 0.05)),
            "num_leaves": int(self.config.get("num_leaves", 31)),
            "subsample": float(self.config.get("subsample", 0.8)),
            "subsample_freq": int(self.config.get("subsample_freq", 1)),
            "colsample_bytree": float(self.config.get("colsample_bytree", 0.8)),
            "n_jobs": int(self.config.get("n_jobs", -1)),
            "random_state": int(self.config.get("random_state", 42)),
            "verbose": -1,
        }

        callbacks = [lgb.log_evaluation(period=-1)]

        if early_stopping_rounds:
            split = int(len(X_train) * 0.9)
            X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
            y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]
            self._model = lgb.LGBMRegressor(**params)
            callbacks.append(lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False))
            self._model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            self._model = lgb.LGBMRegressor(**params)
            self._model.fit(X_train, y_train, callbacks=callbacks)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        preds = self._model.predict(X_test)
        return pd.Series(preds, index=X_test.index, dtype=float)
