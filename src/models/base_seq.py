from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SeqForecasterBase(ABC):
    """Abstract base for sequence forecasters (LSTM, GRU, Transformer).

    Accepts 3D numpy arrays produced by window_sequences() rather than the
    flat 2D DataFrames used by ForecasterBase. The runner receives these via
    the preprocess_fn hook in evaluate_model().

    Subclass responsibilities:
      - Implement fit(X_seq, y) — build and train a torch.nn.Module.
      - Implement predict(X_seq) — run inference, return pd.Series.
      - Call _run_epochs(), _make_loader(), _to_tensors() as needed.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fit(self, X_seq: np.ndarray, y: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, X_seq: np.ndarray) -> pd.Series: ...

    def get_params(self) -> dict:
        return self.config

    # ------------------------------------------------------------------
    # Protected helpers — use these inside fit() / predict()
    # ------------------------------------------------------------------

    @staticmethod
    def _default_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _to_tensors(
        X_seq: np.ndarray, y: pd.Series
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32)
        return X_t, y_t

    @staticmethod
    def _make_loader(
        X_t: torch.Tensor, y_t: torch.Tensor, batch_size: int
    ) -> DataLoader:
        ds = TensorDataset(X_t, y_t)
        # shuffle=False — time series order must be preserved during training
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    @staticmethod
    def _run_epochs(
        loader: DataLoader,
        torch_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        device: torch.device,
    ) -> None:
        torch_model.train()
        for _ in range(epochs):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                pred = torch_model(X_batch).squeeze(-1)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
