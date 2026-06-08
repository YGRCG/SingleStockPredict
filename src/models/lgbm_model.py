import numpy as np
import pandas as pd
import lightgbm as lgb
from src.models.base import BaseModel


class LGBMModel(BaseModel):
    name = "lgbm"

    def __init__(self, params: dict):
        self.params = params
        self.model: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        callbacks = [lgb.log_evaluation(period=100)]
        if X_val is not None:
            callbacks.append(lgb.early_stopping(self.params.get("early_stopping_rounds", 50), verbose=False))

        self.model = lgb.LGBMClassifier(**{k: v for k, v in self.params.items() if k != "early_stopping_rounds"})
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_name_,
        ).sort_values(ascending=False)
