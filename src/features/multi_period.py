"""
多周期特征对齐：将周线、月线、分钟线指标 reindex 到日频，
用 forward-fill 填充，保证不引入未来信息。
同时生成跨周期交互特征，捕捉多周期共振/背离信号。
"""

import numpy as np
import pandas as pd
from src.features.technical import build_technical_features


def _add_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """给所有非 OHLCV 列加前缀，避免列名冲突。"""
    ohlcv = {"open", "high", "low", "close", "volume", "amount"}
    rename = {c: f"{prefix}_{c}" for c in df.columns if c not in ohlcv}
    return df.rename(columns=rename)


_WEEKLY_DEFAULTS = dict(
    ma_windows=[5, 10, 20, 60],
    rsi_period=14,
    macd_fast=12, macd_slow=26, macd_signal=9,
    boll_window=20,
    atr_period=14,
    volume_ma=[5, 10, 20],
)

_MONTHLY_DEFAULTS = dict(
    ma_windows=[3, 6, 12, 24],
    rsi_period=9,
    macd_fast=12, macd_slow=26, macd_signal=9,
    boll_window=12,
    atr_period=14,
    volume_ma=[3, 6, 12],
)


def _merge_kwargs(defaults: dict, overrides: dict) -> dict:
    merged = defaults.copy()
    merged.update(overrides)
    return merged


def build_weekly_features(
    daily_index: pd.DatetimeIndex,
    weekly_df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    在周线上计算技术指标，然后对齐到 daily_index。
    周线的每根 bar 对应其所在周的最后一个交易日，
    forward-fill 到该周的所有交易日。
    """
    params = _merge_kwargs(_WEEKLY_DEFAULTS, kwargs)
    feat = build_technical_features(weekly_df, **params)
    feat = _add_prefix(feat, "w")
    feat = feat.reindex(daily_index, method="ffill")
    return feat


def build_monthly_features(
    daily_index: pd.DatetimeIndex,
    monthly_df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """同 build_weekly_features，但基于月线，使用月线独立默认参数。"""
    params = _merge_kwargs(_MONTHLY_DEFAULTS, kwargs)
    feat = build_technical_features(monthly_df, **params)
    feat = _add_prefix(feat, "m")
    feat = feat.reindex(daily_index, method="ffill")
    return feat


def build_minute_features(
    target_index: pd.DatetimeIndex,
    minute_df: pd.DataFrame,
    freq_label: str,
    agg_bars: int = 8,
) -> pd.DataFrame:
    """
    分钟 K 线 → 日频特征。
    每个交易日取截至收盘前所有分钟 bar，提取：
      - 日内振幅、VWAP、量比、价格位置
      - 日内趋势（上午/下午走势差异）
      - 已实现波动率
      - 尾盘动量（最后 N 根 bar）
    然后 forward-fill 对齐到 target_index（日频）。
    """
    p = freq_label

    minute_df = minute_df.copy()
    minute_df.index = pd.to_datetime(minute_df.index)
    day_key = minute_df.index.normalize()

    agg = minute_df.groupby(day_key).agg(
        **{
            f"{p}_intraday_high":  ("high",   "max"),
            f"{p}_intraday_low":   ("low",    "min"),
            f"{p}_intraday_open":  ("open",   "first"),
            f"{p}_intraday_close": ("close",  "last"),
            f"{p}_intraday_vol":   ("volume", "sum"),
            f"{p}_bar_count":      ("close",  "count"),
        }
    )

    agg[f"{p}_amplitude"] = (
        agg[f"{p}_intraday_high"] - agg[f"{p}_intraday_low"]
    ) / agg[f"{p}_intraday_open"].replace(0, float("nan"))

    if "amount" in minute_df.columns:
        vwap_agg = minute_df.groupby(day_key).apply(
            lambda g: g["amount"].sum() / g["volume"].sum() if g["volume"].sum() > 0 else float("nan"),
            include_groups=False,
        )
        agg[f"{p}_vwap"] = vwap_agg.values
        agg[f"{p}_vwap_premium"] = (
            agg[f"{p}_intraday_close"] - agg[f"{p}_vwap"]
        ) / agg[f"{p}_vwap"].replace(0, float("nan"))

    def _intraday_trend(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        if n < 2:
            return pd.Series({f"{p}_am_trend": 0.0, f"{p}_pm_trend": 0.0, f"{p}_am_pm_div": 0.0})
        mid = n // 2
        am_ret = g["close"].iloc[mid - 1] / g["open"].iloc[0] - 1 if g["open"].iloc[0] != 0 else 0.0
        pm_ret = g["close"].iloc[-1] / g["open"].iloc[mid] - 1 if g["open"].iloc[mid] != 0 else 0.0
        return pd.Series({
            f"{p}_am_trend": am_ret,
            f"{p}_pm_trend": pm_ret,
            f"{p}_am_pm_div": am_ret - pm_ret,
        })

    trend_feat = minute_df.groupby(day_key).apply(_intraday_trend, include_groups=False)
    agg = agg.join(trend_feat, how="left")

    def _realized_vol(g: pd.DataFrame) -> float:
        ret = g["close"].pct_change().dropna()
        return ret.std() * np.sqrt(len(ret)) if len(ret) > 1 else float("nan")

    agg[f"{p}_realized_vol"] = minute_df.groupby(day_key).apply(
        _realized_vol, include_groups=False,
    ).values

    def _tail_momentum(g: pd.DataFrame, n: int) -> pd.Series:
        tail = g.tail(n)
        if len(tail) < 2 or tail["open"].iloc[0] == 0:
            return pd.Series({f"{p}_tail_ret": 0.0, f"{p}_tail_vol_ratio": 0.0})
        tail_ret = tail["close"].iloc[-1] / tail["open"].iloc[0] - 1
        total_vol = g["volume"].sum()
        tail_vol_ratio = tail["volume"].sum() / total_vol if total_vol > 0 else 0.0
        return pd.Series({f"{p}_tail_ret": tail_ret, f"{p}_tail_vol_ratio": tail_vol_ratio})

    tail_feat = minute_df.groupby(day_key).apply(
        lambda g: _tail_momentum(g, agg_bars), include_groups=False,
    )
    agg = agg.join(tail_feat, how="left")

    agg[f"{p}_close_position"] = (
        agg[f"{p}_intraday_close"] - agg[f"{p}_intraday_low"]
    ) / (agg[f"{p}_intraday_high"] - agg[f"{p}_intraday_low"]).replace(0, float("nan"))

    agg.index = pd.to_datetime(agg.index)
    feat = agg.reindex(target_index, method="ffill")
    return feat


def build_cross_period_features(daily_feat: pd.DataFrame) -> pd.DataFrame:
    """
    跨周期交互特征：比较日线与周线/月线指标，捕捉共振/背离。
    要求 daily_feat 已包含 w_ / m_ 前缀的周线/月线列。
    """
    cross = pd.DataFrame(index=daily_feat.index)

    ma_pairs = [(5, 5), (10, 5), (20, 10), (60, 20)]
    for d_ma, w_ma in ma_pairs:
        d_col = f"ma_{d_ma}"
        w_col = f"w_ma_{w_ma}"
        if d_col in daily_feat.columns and w_col in daily_feat.columns:
            cross[f"cross_d{d_ma}_w{w_ma}_bias"] = (
                daily_feat[d_col] - daily_feat[w_col]
            ) / daily_feat[w_col].replace(0, float("nan"))

    m_ma_pairs = [(5, 3), (10, 6), (20, 12), (60, 24)]
    for d_ma, m_ma in m_ma_pairs:
        d_col = f"ma_{d_ma}"
        m_col = f"m_ma_{m_ma}"
        if d_col in daily_feat.columns and m_col in daily_feat.columns:
            cross[f"cross_d{d_ma}_m{m_ma}_bias"] = (
                daily_feat[d_col] - daily_feat[m_col]
            ) / daily_feat[m_col].replace(0, float("nan"))

    for pfx in ("", "w_", "m_"):
        col = f"{pfx}rsi_14" if pfx == "" else f"{pfx}rsi_14"
        if pfx == "w_":
            col = "w_rsi_14"
        elif pfx == "m_":
            col = "m_rsi_9"
        if col not in daily_feat.columns:
            continue
        cross[f"cross_{pfx}rsi_overbought"] = (daily_feat[col] > 70).astype(int)
        cross[f"cross_{pfx}rsi_oversold"] = (daily_feat[col] < 30).astype(int)

    if "rsi_14" in daily_feat.columns and "w_rsi_14" in daily_feat.columns:
        cross["cross_d_rsi_w_rsi_diff"] = daily_feat["rsi_14"] - daily_feat["w_rsi_14"]
    if "rsi_14" in daily_feat.columns and "m_rsi_9" in daily_feat.columns:
        cross["cross_d_rsi_m_rsi_diff"] = daily_feat["rsi_14"] - daily_feat["m_rsi_9"]

    trend_cols = []
    for pfx, ma_col in [("d", "ma_5"), ("w", "w_ma_5"), ("m", "m_ma_3")]:
        full_col = ma_col if pfx == "d" else ma_col
        if full_col in daily_feat.columns and f"{pfx[0] if pfx != 'd' else ''}ma_20" if pfx == "d" else f"{pfx}_ma_20" if pfx == "w" else f"m_ma_12" in daily_feat.columns:
            short = full_col
            long = "ma_20" if pfx == "d" else ("w_ma_20" if pfx == "w" else "m_ma_12")
            if short in daily_feat.columns and long in daily_feat.columns:
                trend_cols.append((pfx, short, long))

    bullish_count = pd.Series(0, index=daily_feat.index, dtype=float)
    for pfx, short, long in trend_cols:
        bullish_count += (daily_feat[short] > daily_feat[long]).astype(float)
    cross["cross_trend_alignment"] = bullish_count / max(len(trend_cols), 1)

    for pfx, col in [("d", "boll_pos"), ("w", "w_boll_pos"), ("m", "m_boll_pos")]:
        if col in daily_feat.columns:
            cross[f"cross_{pfx}_boll_extreme_high"] = (daily_feat[col] > 0.9).astype(int)
            cross[f"cross_{pfx}_boll_extreme_low"] = (daily_feat[col] < 0.1).astype(int)

    return cross


def merge_multi_period(
    daily_feat: pd.DataFrame,
    weekly_feat: pd.DataFrame | None = None,
    monthly_feat: pd.DataFrame | None = None,
    minute_feats: dict[str, pd.DataFrame] | None = None,
    cross_feat: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """横向拼接所有周期特征（含跨周期交互特征），索引对齐到日频。"""
    ohlcv = ["open", "high", "low", "close", "volume", "amount"]

    merged = daily_feat.copy()
    if weekly_feat is not None:
        w_cols = [c for c in weekly_feat.columns if c not in ohlcv]
        merged = merged.join(weekly_feat[w_cols], how="left")
    if monthly_feat is not None:
        m_cols = [c for c in monthly_feat.columns if c not in ohlcv]
        merged = merged.join(monthly_feat[m_cols], how="left")

    if minute_feats:
        for freq, mfeat in minute_feats.items():
            min_cols = [c for c in mfeat.columns if c not in ohlcv]
            merged = merged.join(mfeat[min_cols], how="left")

    if cross_feat is not None:
        merged = merged.join(cross_feat, how="left")

    return merged
