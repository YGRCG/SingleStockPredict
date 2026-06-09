import numpy as np
import pandas as pd
import xgboost as xgb
from src.models.base import BaseModel


class XGBModel(BaseModel):
    name = "xgb"

    def __init__(self, params: dict, label_type: str = "binary"):
        self.params = params
        self.label_type = label_type
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
        params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
        params["early_stopping_rounds"] = self.params.get("early_stopping_rounds", 50)

        if self.label_type in ("return", "triple_barrier"):
            params["objective"] = "reg:squarederror"
            params.pop("eval_metric", None)
            params["eval_metric"] = "mae"
            self._is_regression = True
            self._is_multiclass = False
            self.model = xgb.XGBRegressor(**params)
        elif self.label_type == "ternary":
            n_classes = y_train.nunique()
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes
            params.pop("eval_metric", None)
            params["eval_metric"] = "mlogloss"
            self._is_multiclass = True
            self._is_regression = False
            self.model = xgb.XGBClassifier(**params)
        else:
            self._is_multiclass = False
            self._is_regression = False
            self.model = xgb.XGBClassifier(**params)

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._is_regression:
            return self.model.predict(X)
        if self._is_multiclass:
            return self.model.predict_proba(X)[:, -1]
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.get_booster().feature_names,
        ).sort_values(ascending=False)
