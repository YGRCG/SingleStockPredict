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
from src.features.alpha.builder import build_alpha_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 原始 OHLCV 列（不作为模型特征）
_OHLCV_COLS = ["open", "high", "low", "close", "volume", "amount",
               "adjustflag", "tradestatus", "isST"]


def _relativize_features(df: pd.DataFrame) -> pd.DataFrame:
    """将绝对价格/量级特征转为相对值，提升跨时期泛化能力。
    须在跨周期特征计算之后、去除 OHLCV 列之前调用。"""
    close = df["close"]
    to_drop = []

    for col in list(df.columns):
        # MA / EMA：转为偏离度 (close / ma - 1)
        if any(col.startswith(p) for p in
               ("ma_", "ema_", "w_ma_", "w_ema_", "m_ma_", "m_ema_")):
            df[col] = close / df[col].replace(0, np.nan) - 1
            continue

        # 布林带绝对值：已有 boll_width/boll_pos，直接丢弃
        if col in ("boll_mid", "boll_upper", "boll_lower",
                    "w_boll_mid", "w_boll_upper", "w_boll_lower",
                    "m_boll_mid", "m_boll_upper", "m_boll_lower"):
            to_drop.append(col)
            continue

        # MACD：除以 close 归一化
        if any(col.startswith(p) for p in ("macd_", "w_macd_", "m_macd_")):
            df[col] = df[col] / close
            continue

        # ATR：除以 close 转为百分比
        if any(col.startswith(p) for p in ("atr_", "w_atr_", "m_atr_")):
            df[col] = df[col] / close
            continue

        # vol_ma_X：已有 vol_ratio，直接丢弃
        if any(col.startswith(p) for p in ("vol_ma_", "w_vol_ma_", "m_vol_ma_")):
            to_drop.append(col)
            continue

        # OBV：取滚动均值的变化率，避免累积值过零导致 pct_change 爆炸
        if col in ("obv", "w_obv", "m_obv"):
            obv_ma = df[col].rolling(20, min_periods=1).mean()
            df[col] = (df[col] - obv_ma) / obv_ma.replace(0, np.nan)
            continue

        # 分钟线绝对价格：high/low 转为相对 close
        if any(col.endswith(s) for s in ("_intraday_high", "_intraday_low")):
            df[col] = df[col] / close - 1
            continue

        # 分钟线冗余绝对值：丢弃
        if any(col.endswith(s) for s in
               ("_intraday_open", "_intraday_close", "_intraday_vol")) or \
           (col.endswith("_vwap") and not col.endswith("_vwap_premium")):
            to_drop.append(col)
            continue

    return df.drop(columns=to_drop, errors="ignore")


def _rolling_zscore(df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """滚动 z-score 标准化，使特征更平稳。
    每个特征用过去 window 日的均值/标准差做标准化，不引入未来信息。"""
    exclude = {"close", "label", "future_ret"}
    cols = [c for c in df.columns if c not in exclude]
    result = df.copy()
    for col in cols:
        rolling_mean = df[col].rolling(window, min_periods=20).mean()
        rolling_std  = df[col].rolling(window, min_periods=20).std()
        result[col]  = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)
    return result


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
    data = load_all_periods(symbol, data_dir, periods=cfg.get("periods"))

    # 日线技术指标：过滤掉非指标的结构性 key，其余全部透传
    _NON_TECH_KEYS = {"corr_threshold", "zscore", "weekly_kwargs", "monthly_kwargs", "minute"}
    tech_kwargs = {k: v for k, v in fcfg.items() if k not in _NON_TECH_KEYS}
    daily_feat = build_technical_features(data["daily"], **tech_kwargs)
    daily_feat = build_pattern_features(daily_feat)

    # Alpha101 因子（可选）
    alpha_cfg = fcfg.get("alphas", {})
    if alpha_cfg.get("enabled"):
        alpha_feat = build_alpha_features(data["daily"], alpha_cfg)
        daily_feat = daily_feat.join(alpha_feat, how="left")

    # 周线 / 月线对齐（仅在对应数据已加载时计算）
    weekly_feat  = None
    monthly_feat = None
    if "weekly" in data:
        weekly_feat  = build_weekly_features(daily_feat.index, data["weekly"],  **fcfg.get("weekly_kwargs",  {}))
    if "monthly" in data:
        monthly_feat = build_monthly_features(daily_feat.index, data["monthly"], **fcfg.get("monthly_kwargs", {}))

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
    merged_for_cross = daily_feat.copy()
    for pf in (weekly_feat, monthly_feat):
        if pf is not None:
            merged_for_cross = merged_for_cross.join(
                pf[[c for c in pf.columns if c not in _OHLCV_COLS]], how="left"
            )
    cross_feat = build_cross_period_features(merged_for_cross)

    # 合并
    feat = merge_multi_period(
        daily_feat, weekly_feat, monthly_feat,
        minute_feats=minute_feats or None,
        cross_feat=cross_feat,
    )

    # 绝对特征相对化（跨周期特征已算完，绝对值无用途）
    feat = _relativize_features(feat)

    # 去掉原始 OHLCV（保留 close 供标签使用，后续可按需剔除）
    drop_cols = [c for c in _OHLCV_COLS if c in feat.columns and c != "close"]
    feat = feat.drop(columns=drop_cols, errors="ignore")

    # 去掉所有非数值列（如 adjustflag/tradestatus/isST 的前缀版本）
    num_cols = feat.select_dtypes(include=[np.number]).columns.tolist()
    # 保留 close（供标签使用）
    if "close" in feat.columns and "close" not in num_cols:
        num_cols.append("close")
    feat = feat[num_cols]

    # 去掉全 NaN 列
    feat = feat.dropna(axis=1, how="all")

    # 分析 NaN 分布
    nan_ratio = feat.isna().mean()
    high_nan_cols = nan_ratio[nan_ratio > 0.3].sort_values(ascending=False)
    if len(high_nan_cols) > 0:
        logger.info(
            f"NaN 比例 >30% 的列（共 {len(high_nan_cols)} 列）: "
            f"{dict(list(high_nan_cols.items())[:10])}"
        )

    # 中性值填充：指标预热期不足时，用中性值代替 NaN
    # 语义："数据不足时假设指标处于中性/均衡状态"
    _FILL_RULES = [
        ("ma_", 0.0), ("ema_", 0.0),
        ("w_ma_", 0.0), ("w_ema_", 0.0),
        ("m_ma_", 0.0), ("m_ema_", 0.0),
        ("macd_", 0.0), ("w_macd_", 0.0), ("m_macd_", 0.0),
        ("atr_", 0.0), ("w_atr_", 0.0), ("m_atr_", 0.0),
        ("rsi_", 50.0), ("w_rsi_", 50.0), ("m_rsi_", 50.0),
        ("kdj_k", 50.0), ("kdj_d", 50.0), ("kdj_j", 50.0),
        ("w_kdj_", 50.0), ("m_kdj_", 50.0),
        ("boll_width", 0.0), ("boll_pos", 0.5),
        ("w_boll_", 0.0), ("m_boll_", 0.0),
        ("vol_ratio", 1.0), ("w_vol_ratio", 1.0), ("m_vol_ratio", 1.0),
        ("obv", 0.0), ("w_obv", 0.0), ("m_obv", 0.0),
        ("ret_", 0.0), ("w_ret_", 0.0), ("m_ret_", 0.0),
        ("cross_", 0.0),
        ("turn", 0.0), ("pctChg", 0.0),
        ("min", 0.0),
        ("alpha", 0.0),
    ]
    for col in feat.columns:
        if feat[col].isna().any():
            for prefix, fill_val in _FILL_RULES:
                if col.startswith(prefix) or col.endswith(prefix):
                    feat[col] = feat[col].fillna(fill_val)
                    break

    # 剩余无法匹配规则的 NaN 列，用 0 填充
    remaining_nan = feat.isna().sum()
    remaining_nan_cols = remaining_nan[remaining_nan > 0]
    if len(remaining_nan_cols) > 0:
        logger.info(f"未匹配填充规则的 NaN 列，用 0 填充: {dict(remaining_nan_cols)}")
        feat = feat.fillna(0.0)

    # 高相关性过滤：移除相关系数 > threshold 的冗余列
    corr_thresh = fcfg.get("corr_threshold", 0.95)
    if corr_thresh < 1.0:
        feat = _drop_highly_correlated(feat, corr_thresh)

    # 滚动 z-score 标准化（可选，默认关闭）
    zscore_cfg = fcfg.get("zscore", {})
    if zscore_cfg.get("enabled", False):
        zscore_window = zscore_cfg.get("window", 120)
        feat = _rolling_zscore(feat, zscore_window)
        feat = feat.dropna()
        logger.info(f"滚动 z-score 标准化完成 | window={zscore_window}")

    logger.info(f"特征矩阵: {feat.shape[0]} 行 × {feat.shape[1]} 列")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(save_path)
        logger.info(f"已保存特征矩阵: {save_path}")

    return feat


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
