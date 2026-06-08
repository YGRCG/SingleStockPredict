"""
单周期技术指标计算。
输入：标准 OHLCV DataFrame（索引为 date）
输出：新增指标列的 DataFrame（不修改原始列）
"""

import pandas as pd
import numpy as np


# ── 趋势类 ─────────────────────────────────────────────────────────────────────

def add_ma(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
    return df


def add_ema(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd_dif"] = ema_fast - ema_slow
    df["macd_dea"] = df["macd_dif"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2
    return df


# ── 震荡类 ─────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - 100 / (1 + rs)
    return df


def add_kdj(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    low_min  = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    rsv = (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
    return df


def add_boll(df: pd.DataFrame, window: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    mid = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    df["boll_mid"]   = mid
    df["boll_upper"] = mid + std_mult * std
    df["boll_lower"] = mid - std_mult * std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / mid
    df["boll_pos"]   = (df["close"] - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"]).replace(0, np.nan)
    return df


# ── 波动 / 量 类 ────────────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.rolling(period).mean()
    return df


def add_volume_ma(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"vol_ma_{w}"] = df["volume"].rolling(w).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma_5"].replace(0, np.nan)
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["obv"] = (direction * df["volume"]).cumsum()
    return df


# ── 价格衍生特征 ────────────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ret_1d"]  = df["close"].pct_change(1)
    df["ret_5d"]  = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)
    df["ret_20d"] = df["close"].pct_change(20)
    df["high_low_ratio"] = df["high"] / df["low"].replace(0, np.nan) - 1
    df["close_open_ratio"] = df["close"] / df["open"].replace(0, np.nan) - 1
    return df


# ── 汇总入口 ───────────────────────────────────────────────────────────────────

def build_technical_features(
    df: pd.DataFrame,
    ma_windows:   list[int] = None,
    rsi_period:   int = 14,
    macd_fast:    int = 12,
    macd_slow:    int = 26,
    macd_signal:  int = 9,
    boll_window:  int = 20,
    atr_period:   int = 14,
    volume_ma:    list[int] = None,
) -> pd.DataFrame:
    df = df.copy()
    ma_windows = ma_windows or [5, 10, 20, 60, 120]
    volume_ma  = volume_ma  or [5, 10, 20]

    df = add_ma(df, ma_windows)
    df = add_ema(df, [12, 26])
    df = add_macd(df, macd_fast, macd_slow, macd_signal)
    df = add_rsi(df, rsi_period)
    df = add_kdj(df)
    df = add_boll(df, boll_window)
    df = add_atr(df, atr_period)
    df = add_volume_ma(df, volume_ma)
    df = add_obv(df)
    df = add_price_features(df)
    return df
