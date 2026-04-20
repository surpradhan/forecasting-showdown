import numpy as np
import pandas as pd

from src.models.base import ForecasterBase


class NaiveForecaster(ForecasterBase):
    """Seasonal naive: repeat the last full seasonal cycle, phase-aligned to timestamps."""

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._strategy = self.config.get("strategy", "seasonal")
        self._period = int(self.config.get("seasonal_period", 24))

        if self._strategy == "last":
            self._last_value = float(y_train.iloc[-1])
            return

        # Store tail and step size for phase-correct lookup at predict time.
        # Step = median successive difference, robust to occasional gaps.
        self._tail = y_train.iloc[-self._period:].values.copy()
        self._train_end = y_train.index[-1]
        diffs = pd.Series(y_train.index[1:] - y_train.index[:-1])
        self._step: pd.Timedelta = diffs.mode().iloc[0]

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        n = len(X_test)
        if self._strategy == "last":
            return pd.Series(self._last_value, index=X_test.index, dtype=float)

        # For each test timestamp t, the seasonal naive forecast is the value
        # from the last training cycle at the same phase:
        #   h  = steps from training end to t
        #   pos = (h - 1) % period  →  index into tail
        # This correctly handles any gap between training end and test start.
        values = np.empty(n, dtype=float)
        for i, ts in enumerate(X_test.index):
            h = round((ts - self._train_end) / self._step)
            values[i] = self._tail[(h - 1) % self._period]
        return pd.Series(values, index=X_test.index, dtype=float)
