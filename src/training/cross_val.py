"""
时序交叉验证：用于超参搜索和模型评估。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def time_series_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_cls: type[BaseModel],
    model_params: dict,
    n_splits: int = 5,
    gap: int = 3,               # 训练集末尾与验证集开头的间隔（防标签泄露）
) -> dict:
    """
    Args:
        gap: 跳过训练集最后 gap 行，避免标签与未来特征重叠

    Returns:
        含各折指标的字典
    """
    df = df.dropna(subset=["label"]).copy()
    X = df[feature_cols]
    y = df["label"]

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    auc_scores, acc_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

        if y_train.nunique() < 2:
            continue

        model = model_cls(model_params)
        model.fit(X_train, y_train, X_val, y_val)

        proba = model.predict_proba(X_val)
        pred  = model.predict(X_val)

        auc = roc_auc_score(y_val, proba)
        acc = accuracy_score(y_val, pred)
        auc_scores.append(auc)
        acc_scores.append(acc)
        logger.info(f"  Fold {fold+1} | AUC={auc:.4f} | ACC={acc:.4f}")

    result = {
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std":  float(np.std(auc_scores)),
        "acc_mean": float(np.mean(acc_scores)),
        "acc_std":  float(np.std(acc_scores)),
        "n_folds":  len(auc_scores),
    }
    logger.info(f"CV 结果: AUC {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
    return result
