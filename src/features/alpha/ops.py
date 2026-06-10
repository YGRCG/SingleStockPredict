"""
Alpha101 原子操作。
所有函数接受 pd.Series，返回 pd.Series，不引入未来数据。

单股场景说明：
  原论文的 rank() 是横截面排名。这里改为滚动时序百分位排名，
  窗口由 rank_window 参数控制（默认 252 交易日）。
"""

import numpy as np
import pandas as pd


def delay(x: pd.Series, d: int) -> pd.Series:
    return x.shift(d)


def delta(x: pd.Series, d: int) -> pd.Series:
    return x - x.shift(d)


def sign(x: pd.Series) -> pd.Series:
    return np.sign(x)


def log(x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan))


def signed_power(x: pd.Series, e: float) -> pd.Series:
    return np.sign(x) * (x.abs() ** e)


def rank(x: pd.Series, window: int = 252) -> pd.Series:
    """滚动时序百分位排名，替代原论文的横截面 rank。"""
    return x.rolling(window, min_periods=window // 4).rank(pct=True)


def ts_rank(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=max(1, d // 2)).rank(pct=True)


def ts_max(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).max()


def ts_min(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=1).min()


def ts_argmax(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d).apply(np.argmax, raw=True) + 1


def ts_argmin(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d).apply(np.argmin, raw=True) + 1


def ts_mean(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=max(1, d // 2)).mean()


def ts_sum(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=max(1, d // 2)).sum()


def ts_std(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=max(2, d // 2)).std()


def ts_corr(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=max(2, d // 2)).corr(y)


def ts_cov(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=max(2, d // 2)).cov(y)


def decay_linear(x: pd.Series, d: int) -> pd.Series:
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    return x.rolling(d, min_periods=d).apply(lambda v: np.dot(v, w), raw=True)


def scale(x: pd.Series, window: int = 252) -> pd.Series:
    total = x.abs().rolling(window, min_periods=1).sum()
    return x / total.replace(0, np.nan)


def adv(volume: pd.Series, d: int) -> pd.Series:
    return volume.rolling(d, min_periods=max(1, d // 2)).mean()


def vwap(amount: pd.Series, volume: pd.Series) -> pd.Series:
    return amount / volume.replace(0, np.nan)
