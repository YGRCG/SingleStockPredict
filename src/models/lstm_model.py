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

    class _LSTMNet(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                     dropout: float, is_regression: bool = False):
            super().__init__()
            self.is_regression = is_regression
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            logit = self.fc(out[:, -1, :])
            if self.is_regression:
                return logit.squeeze(-1)
            return torch.sigmoid(logit)

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class LSTMModel(BaseModel):
    name = "lstm"

    def __init__(self, params: dict, label_type: str = "binary"):
        if not _TORCH_AVAILABLE:
            raise ImportError("请先安装 torch：uv add torch")
        self.params = params
        self.label_type = label_type
        self._is_regression = label_type in ("return", "triple_barrier")
        self._is_multiclass = False
        self.net: _LSTMNet | None = None
        self.seq_len = params.get("seq_len", 30)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        seqs, labels = [], []
        for i in range(self.seq_len, len(X) + 1):
            seqs.append(X[i - self.seq_len: i])
            if y is not None:
                labels.append(y[i - 1])
        if not seqs:
            return np.empty((0, self.seq_len, X.shape[1])), np.empty(0)
        if y is not None:
            return np.array(seqs), np.array(labels)
        return np.array(seqs), None

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

        X_seq, y_seq = self._make_sequences(X_arr, y_arr)
        if len(X_seq) == 0:
            return

        self.net = _LSTMNet(
            input_size  = X_arr.shape[1],
            hidden_size = p.get("hidden_size", 64),
            num_layers  = p.get("num_layers", 2),
            dropout     = p.get("dropout", 0.2),
            is_regression = self._is_regression,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=p.get("learning_rate", 0.001))
        criterion = nn.MSELoss() if self._is_regression else nn.BCELoss()

        dataset = TensorDataset(
            torch.tensor(X_seq, device=self.device),
            torch.tensor(y_seq, device=self.device).float(),
        )
        loader = DataLoader(dataset, batch_size=p.get("batch_size", 64), shuffle=True)
        clip_norm = p.get("clip_grad_norm", 1.0)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(p.get("epochs", 100)):
            self.net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.net(xb)
                if not self._is_regression:
                    pred = pred.squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), clip_norm)
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
            y_arr = y_val.values.astype(np.float32)
            X_seq, y_seq = self._make_sequences(X_arr, y_arr)
            if len(X_seq) == 0:
                return float("inf")
            xb = torch.tensor(X_seq, device=self.device)
            yb = torch.tensor(y_seq, device=self.device).float()
            pred = self.net(xb)
            if not self._is_regression:
                pred = pred.squeeze(-1)
            return criterion(pred, yb).item()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            X_arr = X.values.astype(np.float32)
            n_rows = len(X_arr)

            if n_rows >= self.seq_len:
                X_seq, _ = self._make_sequences(X_arr)
                xb = torch.tensor(X_seq, device=self.device)
                proba = self.net(xb).cpu().numpy().flatten()
                pad = np.full(self.seq_len - 1, np.nan)
                return np.concatenate([pad, proba])
            else:
                # 不足 seq_len 行时，前面补零凑成一个完整序列
                pad_rows = np.zeros((self.seq_len - n_rows, X_arr.shape[1]), dtype=np.float32)
                padded = np.concatenate([pad_rows, X_arr], axis=0)
                xb = torch.tensor(padded[np.newaxis, :, :], device=self.device)
                proba = self.net(xb).cpu().numpy().flatten()
                result = np.full(n_rows, np.nan)
                result[-1] = proba[0]
                return result
