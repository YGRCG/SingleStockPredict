"""
特征流水线：整合多周期特征、形态特征，输出最终特征矩阵。
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from src.data.loader import load_all_periods
from src.features.technical import build_technical_features
from src.features.multi_period import (
    build_weekly_features, build_monthly_features,
    build_minute_features, build_cross_period_features, merge_multi_period,
)
from src.features.price_pattern import build_pattern_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 原始 OHLCV 列（不作为模型特征）
_OHLCV_COLS = ["open", "high", "low", "close", "volume", "amount",
               "adjustflag", "tradestatus", "isST"]


def _drop_highly_correlated(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """移除相关系数绝对值 > threshold 的冗余列，保留先出现的列。"""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    if to_drop:
        logger.info(f"移除 {len(to_drop)} 列高相关特征 (>{threshold}): {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")
        df = df.drop(columns=to_drop)
    return df


def build_feature_matrix(
    symbol: str,
    cfg: dict,
    data_dir: str = "data/raw",
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Args:
        symbol:    股票代码
        cfg:       config.yaml 解析后的字典
        data_dir:  原始数据目录
        save_path: 若指定则保存 parquet
    Returns:
        含所有特征的宽表 DataFrame（索引 date）
    """
    fcfg = cfg["features"]
    data = load_all_periods(symbol, data_dir)

    # 日线技术指标 + 形态特征
    daily_feat = build_technical_features(
        data["daily"],
        ma_windows  = fcfg["ma_windows"],
        rsi_period  = fcfg["rsi_period"],
        macd_fast   = fcfg["macd_fast"],
        macd_slow   = fcfg["macd_slow"],
        macd_signal = fcfg["macd_signal"],
        boll_window = fcfg["boll_window"],
        atr_period  = fcfg["atr_period"],
        volume_ma   = fcfg["volume_ma"],
    )
    daily_feat = build_pattern_features(daily_feat)

    # 周线 / 月线对齐（独立参数）
    w_kwargs = fcfg.get("weekly_kwargs", {})
    m_kwargs = fcfg.get("monthly_kwargs", {})
    weekly_feat  = build_weekly_features(daily_feat.index, data["weekly"], **w_kwargs)
    monthly_feat = build_monthly_features(daily_feat.index, data["monthly"], **m_kwargs)

    # 分钟线特征
    minute_feats = {}
    minute_cfg = fcfg.get("minute", {})
    for freq in ("min5", "min15", "min30", "min60"):
        if freq in data:
            mc = minute_cfg.get(freq, {})
            minute_feats[freq] = build_minute_features(
                daily_feat.index, data[freq],
                freq_label=freq,
                agg_bars=mc.get("agg_bars", 8),
            )

    # 先合并日线+周线+月线，再计算跨周期交互特征
    merged_for_cross = daily_feat.join(
        weekly_feat[[c for c in weekly_feat.columns if c not in _OHLCV_COLS]], how="left"
    ).join(
        monthly_feat[[c for c in monthly_feat.columns if c not in _OHLCV_COLS]], how="left"
    )
    cross_feat = build_cross_period_features(merged_for_cross)

    # 合并
    feat = merge_multi_period(
        daily_feat, weekly_feat, monthly_feat,
        minute_feats=minute_feats or None,
        cross_feat=cross_feat,
    )

    # 去掉原始 OHLCV（保留 close 供标签使用，后续可按需剔除）
    drop_cols = [c for c in _OHLCV_COLS if c in feat.columns and c != "close"]
    feat = feat.drop(columns=drop_cols, errors="ignore")

    # 去掉全 NaN 列、dropna 首尾
    feat = feat.dropna(axis=1, how="all")
    feat = feat.dropna()

    # 高相关性过滤：移除相关系数 > threshold 的冗余列
    corr_thresh = fcfg.get("corr_threshold", 0.95)
    if corr_thresh < 1.0:
        feat = _drop_highly_correlated(feat, corr_thresh)

    logger.info(f"特征矩阵: {feat.shape[0]} 行 × {feat.shape[1]} 列")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(save_path)
        logger.info(f"已保存特征矩阵: {save_path}")

    return feat


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
