import numpy as np
import pandas as pd
import lightgbm as lgb
from src.models.base import BaseModel


class LGBMModel(BaseModel):
    name = "lgbm"

    def __init__(self, params: dict):
        self.params = params
        self.model = None
        self._is_multiclass = False
        self._is_regression = False

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

        params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}

        # 判断是回归还是分类
        n_classes = y_train.nunique()
        if n_classes > 20 or y_train.dtype == float:
            # 连续值标签 → 回归
            params["objective"] = "regression"
            params["metric"] = "mae"
            self._is_regression = True
            self._is_multiclass = False
            self.model = lgb.LGBMRegressor(**params)
        elif n_classes > 2:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
            params.pop("metric", None)
            params.setdefault("metric", "multi_logloss")
            self._is_multiclass = True
            self._is_regression = False
            self.model = lgb.LGBMClassifier(**params)
        else:
            self._is_multiclass = False
            self._is_regression = False
            self.model = lgb.LGBMClassifier(**params)

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """分类：返回正类概率；回归：返回预测值。"""
        if self._is_regression:
            return self.model.predict(X)
        if self._is_multiclass:
            proba = self.model.predict_proba(X)
            return proba[:, -1]
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_name_,
        ).sort_values(ascending=False)
