from abc import ABC, abstractmethod

import pandas as pd


class ForecasterBase(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series: ...

    def get_params(self) -> dict:
        return self.config
