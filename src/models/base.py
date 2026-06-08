"""统一模型接口，所有模型继承此基类。"""

from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回正类概率，shape (n,)。"""
        ...

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        return joblib.load(path)
