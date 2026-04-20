import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.base_seq import SeqForecasterBase


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMForecaster(SeqForecasterBase):
    def fit(self, X_seq: np.ndarray, y: pd.Series) -> None:
        device = self._default_device()
        X_t, y_t = self._to_tensors(X_seq, y)
        _, _, n_features = X_t.shape

        self._net = _LSTMNet(
            n_features=n_features,
            hidden_size=int(self.config.get("hidden_size", 64)),
            num_layers=int(self.config.get("num_layers", 2)),
            dropout=float(self.config.get("dropout", 0.2)),
        ).to(device)

        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=float(self.config.get("lr", 1e-3))
        )
        loader = self._make_loader(X_t, y_t, batch_size=int(self.config.get("batch_size", 64)))
        self._run_epochs(
            loader, self._net, optimizer, nn.MSELoss(),
            int(self.config.get("epochs", 50)), device,
        )
        self._device = device

    def predict(self, X_seq: np.ndarray) -> pd.Series:
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self._device)
        self._net.eval()
        with torch.no_grad():
            preds = self._net(X_t).squeeze(-1).cpu().numpy()
        return pd.Series(preds, dtype=float)
