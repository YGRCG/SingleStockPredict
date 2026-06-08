"""
LSTM 模型（PyTorch）。可选依赖，需安装 torch。
"""

import numpy as np
import pandas as pd
from src.models.base import BaseModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))


class LSTMModel(BaseModel):
    name = "lstm"

    def __init__(self, params: dict):
        if not _TORCH_AVAILABLE:
            raise ImportError("请先安装 torch：uv add torch")
        self.params = params
        self.net: _LSTMNet | None = None
        self.seq_len = params.get("seq_len", 30)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        seqs = []
        for i in range(self.seq_len, len(X) + 1):
            seqs.append(X[i - self.seq_len: i])
        return np.array(seqs)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        p = self.params
        X_arr = X_train.values.astype(np.float32)
        y_arr = y_train.values.astype(np.float32)

        # 序列化：X_seq.shape = (N - seq_len, seq_len, features)
        X_seq = self._make_sequences(X_arr)
        y_seq = y_arr[self.seq_len - 1:]

        self.net = _LSTMNet(
            input_size  = X_arr.shape[1],
            hidden_size = p.get("hidden_size", 64),
            num_layers  = p.get("num_layers", 2),
            dropout     = p.get("dropout", 0.2),
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=p.get("learning_rate", 0.001))
        criterion = nn.BCELoss()

        dataset = TensorDataset(
            torch.tensor(X_seq).to(self.device),
            torch.tensor(y_seq).unsqueeze(1).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=p.get("batch_size", 64), shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(p.get("epochs", 100)):
            self.net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()

            if X_val is not None:
                val_loss = self._eval_loss(X_val, y_val, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= p.get("patience", 10):
                        break

    def _eval_loss(self, X_val, y_val, criterion):
        self.net.eval()
        with torch.no_grad():
            X_arr = X_val.values.astype(np.float32)
            X_seq = self._make_sequences(X_arr)
            y_arr = y_val.values.astype(np.float32)[self.seq_len - 1:]
            xb = torch.tensor(X_seq).to(self.device)
            yb = torch.tensor(y_arr).unsqueeze(1).to(self.device)
            return criterion(self.net(xb), yb).item()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            X_arr = X.values.astype(np.float32)
            X_seq = self._make_sequences(X_arr)
            xb = torch.tensor(X_seq).to(self.device)
            proba = self.net(xb).cpu().numpy().flatten()
        # 前 seq_len-1 行无预测，填 NaN
        pad = np.full(self.seq_len - 1, np.nan)
        return np.concatenate([pad, proba])
