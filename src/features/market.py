"""
市场环境特征：从大盘/板块指数中提取特征，捕捉系统性行情信息。
指数特征与个股日频对齐（reindex + ffill），不引入未来信息。
"""

import numpy as np
import pandas as pd
from src.features.technical import build_technical_features


def build_index_features(
    daily_index: pd.DatetimeIndex,
    index_df: pd.DataFrame,
    prefix: str = "idx",
) -> pd.DataFrame:
    """
    在指数日线上计算技术指标 + 收益率 + 波动率，加前缀后对齐到个股日频。

    Args:
        daily_index: 个股的日期索引
        index_df:    指数日线 DataFrame（需含 close, volume）
        prefix:      列名前缀（如 "idx" / "cyb"）
    """
    tech = build_technical_features(
        index_df,
        ma_windows=[5, 10, 20, 60],
        rsi_period=14,
        macd_fast=12, macd_slow=26, macd_signal=9,
        boll_window=20,
        atr_periods=[14],
        volume_ma=[5, 10, 20],
    )

    close = tech["close"]

    # MA/EMA 相对化（用指数自身 close，不是个股 close）
    for col in list(tech.columns):
        if col.startswith(("ma_", "ema_")):
            tech[col] = close / tech[col].replace(0, np.nan) - 1
        elif col.startswith("macd_") or col.startswith("atr_"):
            tech[col] = tech[col] / close
    # 丢弃绝对价格列
    drop_abs = [c for c in tech.columns
                if c in ("boll_mid", "boll_upper", "boll_lower")
                or c.startswith("vol_ma_")]
    tech = tech.drop(columns=drop_abs, errors="ignore")

    # N 日 log return
    for n in (1, 5, 10, 20):
        tech[f"ret_{n}d"] = np.log(close / close.shift(n))

    # 20 日已实现波动率（年化）
    tech["volatility_20"] = tech["ret_1d"].rolling(20, min_periods=5).std() * np.sqrt(252)

    # 去掉原始 OHLCV
    ohlcv = {"open", "high", "low", "close", "volume", "amount",
             "adjustflag", "tradestatus", "isST", "turn", "pctChg",
             "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"}
    keep_cols = [c for c in tech.columns if c not in ohlcv]
    feat = tech[keep_cols]

    # 加前缀
    feat = feat.rename(columns={c: f"{prefix}_{c}" for c in feat.columns})

    # 对齐到个股日期
    feat = feat.reindex(daily_index, method="ffill")
    return feat


def build_stock_index_features(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    prefix: str = "mkt",
) -> pd.DataFrame:
    """
    股票-指数交互特征：相对强弱、滚动相关性、大盘状态。

    Args:
        stock_df:  个股日线（需含 close, volume）
        index_df:  指数日线（需含 close）
        prefix:    列名前缀
    """
    cross = pd.DataFrame(index=stock_df.index)

    s_close = stock_df["close"]
    i_close = index_df["close"].reindex(stock_df.index, method="ffill")

    s_ret_1d = np.log(s_close / s_close.shift(1))
    i_ret_1d = np.log(i_close / i_close.shift(1))

    # 相对强弱（超额收益）
    for n in (1, 5, 20):
        s_ret = np.log(s_close / s_close.shift(n))
        i_ret = np.log(i_close / i_close.shift(n))
        cross[f"{prefix}_rel_ret_{n}d"] = s_ret - i_ret

    # 滚动相关性
    cross[f"{prefix}_corr_20"] = s_ret_1d.rolling(20, min_periods=10).corr(i_ret_1d)

    # 大盘 RSI 状态
    delta = i_close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    idx_rsi = 100 - 100 / (1 + rs)
    cross[f"{prefix}_idx_rsi_overbought"] = (idx_rsi > 70).astype(int)
    cross[f"{prefix}_idx_rsi_oversold"] = (idx_rsi < 30).astype(int)

    # 大盘趋势方向
    idx_ma5 = i_close.rolling(5, min_periods=1).mean()
    idx_ma20 = i_close.rolling(20, min_periods=1).mean()
    cross[f"{prefix}_idx_trend_up"] = (idx_ma5 > idx_ma20).astype(int)

    return cross
