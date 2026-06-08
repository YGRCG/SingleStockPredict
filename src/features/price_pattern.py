"""
形态特征：涨跌停、跳空缺口、量价背离、ST 状态等。
"""

import pandas as pd
import numpy as np


def add_limit_flag(df: pd.DataFrame, limit_pct: float = 0.099) -> pd.DataFrame:
    """涨停/跌停标记（A 股默认 ±10%，ST 股 ±5%）。"""
    pct = df["close"].pct_change()
    df["is_limit_up"]   = (pct >= limit_pct).astype(int)
    df["is_limit_down"] = (pct <= -limit_pct).astype(int)
    return df


def add_gap(df: pd.DataFrame) -> pd.DataFrame:
    """跳空缺口：今开盘 vs 昨收盘。"""
    df["gap_up"]   = ((df["open"] > df["close"].shift()) & (df["low"] > df["close"].shift())).astype(int)
    df["gap_down"] = ((df["open"] < df["close"].shift()) & (df["high"] < df["close"].shift())).astype(int)
    df["gap_pct"]  = df["open"] / df["close"].shift() - 1
    return df


def add_candlestick(df: pd.DataFrame) -> pd.DataFrame:
    """K 线实体、上下影线比率。"""
    body  = (df["close"] - df["open"]).abs()
    total = df["high"] - df["low"]
    df["body_ratio"]       = body / total.replace(0, np.nan)
    df["upper_shadow_ratio"] = (df["high"] - df[["close", "open"]].max(axis=1)) / total.replace(0, np.nan)
    df["lower_shadow_ratio"] = (df[["close", "open"]].min(axis=1) - df["low"])  / total.replace(0, np.nan)
    df["is_bullish"]       = (df["close"] > df["open"]).astype(int)
    return df


def add_consecutive(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """连涨/连跌天数（截至当日）。"""
    up   = (df["close"] > df["close"].shift()).astype(int)
    down = (df["close"] < df["close"].shift()).astype(int)

    def streak(s: pd.Series) -> pd.Series:
        result = s.copy().astype(float)
        for i in range(1, len(s)):
            if s.iloc[i] == 1:
                result.iloc[i] = result.iloc[i - 1] + 1
            else:
                result.iloc[i] = 0
        return result

    df["consec_up"]   = streak(up)
    df["consec_down"] = streak(down)
    return df


def build_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_limit_flag(df)
    df = add_gap(df)
    df = add_candlestick(df)
    df = add_consecutive(df)
    return df
