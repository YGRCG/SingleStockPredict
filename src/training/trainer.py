"""
滚动训练调度器。
支持两种模式：
  rolling   — 固定训练窗口，每步向前滑动 step 个交易日
  expanding — 扩展窗口，训练集逐步累积
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainResult:
    date: pd.Timestamp
    model: BaseModel
    val_score: float
    feature_cols: list[str]


def rolling_train(
    df: pd.DataFrame,           # 含 label 列的特征宽表
    feature_cols: list[str],
    model_cls: type[BaseModel],
    model_params: dict,
    mode: str = "rolling",      # rolling / expanding
    train_window: int = 500,
    val_ratio: float = 0.1,
    step: int = 1,
    backtest_start: str | None = None,
    save_dir: str | None = None,
) -> list[TrainResult]:
    """
    Walk-forward 训练，返回每个预测时间点对应的模型列表。

    Returns:
        results: list of TrainResult，按日期排序
    """
    from sklearn.metrics import roc_auc_score

    df = df.dropna(subset=["label"]).copy()
    dates = df.index

    if backtest_start:
        start_idx = dates.searchsorted(pd.Timestamp(backtest_start))
    else:
        start_idx = train_window

    results: list[TrainResult] = []
    predict_indices = range(start_idx, len(dates), step)

    logger.info(f"开始 {mode} 训练 | 窗口={train_window} | 预测点数={len(predict_indices)}")

    for i, pred_idx in enumerate(predict_indices):
        if mode == "rolling":
            train_start = max(0, pred_idx - train_window)
        else:
            train_start = 0

        train_end   = pred_idx
        val_size    = max(1, int((train_end - train_start) * val_ratio))
        val_start   = train_end - val_size

        X_train = df.iloc[train_start:val_start][feature_cols]
        y_train = df.iloc[train_start:val_start]["label"]
        X_val   = df.iloc[val_start:train_end][feature_cols]
        y_val   = df.iloc[val_start:train_end]["label"]

        if len(X_train) < 50 or y_train.nunique() < 2:
            continue

        model = model_cls(model_params)
        model.fit(X_train, y_train, X_val, y_val)

        val_proba = model.predict_proba(X_val)
        try:
            score = roc_auc_score(y_val, val_proba)
        except Exception:
            score = float("nan")

        pred_date = dates[pred_idx]
        results.append(TrainResult(
            date=pred_date,
            model=model,
            val_score=score,
            feature_cols=feature_cols,
        ))

        if (i + 1) % 20 == 0:
            logger.info(f"  [{i+1}/{len(predict_indices)}] {pred_date.date()} | val_auc={score:.4f}")

        if save_dir:
            path = Path(save_dir) / f"{model.name}_{pred_date.strftime('%Y%m%d')}.pkl"
            model.save(str(path))

    logger.info(f"训练完成，共 {len(results)} 个模型")
    return results
