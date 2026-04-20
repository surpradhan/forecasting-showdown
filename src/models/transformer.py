import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.base_seq import SeqForecasterBase


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _TransformerNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])


class TransformerForecaster(SeqForecasterBase):
    def fit(self, X_seq: np.ndarray, y: pd.Series) -> None:
        d_model = int(self.config.get("d_model", 64))
        nhead = int(self.config.get("nhead", 4))
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        device = self._default_device()
        X_t, y_t = self._to_tensors(X_seq, y)
        _, _, n_features = X_t.shape

        self._net = _TransformerNet(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=int(self.config.get("num_encoder_layers", 2)),
            dim_feedforward=int(self.config.get("dim_feedforward", 256)),
            dropout=float(self.config.get("dropout", 0.1)),
        ).to(device)

        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=float(self.config.get("lr", 5e-4))
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
