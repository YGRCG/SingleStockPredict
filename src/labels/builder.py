"""
标签构建，支持四种模式：
  binary         — 未来 horizon 日收益 > 0 → 1，否则 0
  ternary        — 涨幅 > +threshold → 2，跌幅 > threshold → 0，中间 → 1
  return         — 未来 horizon 日对数收益（回归目标）
  triple_barrier — Marcos Lopez de Prado《AFML》第 3 章三重障碍法
                   上轨（止盈）先触及 → 1
                   下轨（止损）先触及 → -1
                   时间障碍先到       → sign(未来收益) 或 0

严禁未来泄露：所有标签仅依赖 t 时刻之后的价格数据。
"""

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── 基础标签 ──────────────────────────────────────────────────────────────────

def _binary(close: pd.Series, horizon: int) -> pd.Series:
    ret = close.shift(-horizon) / close - 1
    label = (ret > 0).astype(float)
    label[ret.isna()] = np.nan
    return label


def _ternary(close: pd.Series, horizon: int, threshold: float) -> pd.Series:
    ret = close.shift(-horizon) / close - 1
    label = pd.Series(np.nan, index=close.index)
    label[ret >  threshold] = 2.0
    label[ret < -threshold] = 0.0
    label[(ret >= -threshold) & (ret <= threshold)] = 1.0
    return label


def _return(close: pd.Series, horizon: int) -> pd.Series:
    return np.log(close.shift(-horizon) / close)


# ── Triple Barrier ────────────────────────────────────────────────────────────

def _triple_barrier_single(
    prices: np.ndarray,
    t: int,
    horizon: int,
    upper_pct: float,
    lower_pct: float,
) -> float:
    """
    对单个时间点 t 计算三重障碍标签。

    Returns:
        1  : 上轨（止盈）先触及
       -1  : 下轨（止损）先触及
        0  : 时间障碍先到，且区间内收益绝对值 < (upper_pct + lower_pct) / 2
        返回值可选 {-1, 0, 1}，也可改为 {0, 1, 2} 用于多分类
    """
    p0    = prices[t]
    upper = p0 * (1 + upper_pct)
    lower = p0 * (1 - lower_pct)
    end   = min(t + horizon, len(prices) - 1)

    for i in range(t + 1, end + 1):
        p = prices[i]
        if p >= upper:
            return 1.0
        if p <= lower:
            return -1.0

    # 时间障碍：用区间终点收益的符号作为标签（可改为 0）
    final_ret = prices[end] / p0 - 1
    if final_ret > 0:
        return 1.0
    elif final_ret < 0:
        return -1.0
    return 0.0


def _triple_barrier(
    close: pd.Series,
    horizon: int,
    upper_pct: float,
    lower_pct: float,
) -> pd.Series:
    """向量化外层循环，对每个时间点调用单点计算。"""
    prices = close.values
    n      = len(prices)
    labels = np.full(n, np.nan)

    for t in range(n - 1):
        if t + horizon >= n:
            break
        labels[t] = _triple_barrier_single(prices, t, horizon, upper_pct, lower_pct)

    return pd.Series(labels, index=close.index)


# ── 公共入口 ──────────────────────────────────────────────────────────────────

def build_labels(
    df: pd.DataFrame,
    horizon:    int   = 3,
    label_type: str   = "binary",
    threshold:  float = 0.02,
    upper_pct:  float = 0.03,
    lower_pct:  float = 0.02,
    close_col:  str   = "close",
) -> pd.DataFrame:
    """
    Args:
        df:          含 close 列的 DataFrame，索引为 date/datetime
        horizon:     预测未来第 N 根 bar
        label_type:  binary / ternary / return / triple_barrier
        threshold:   ternary 模式阈值
        upper_pct:   triple_barrier 止盈线（如 0.03 = 3%）
        lower_pct:   triple_barrier 止损线（如 0.02 = 2%）
        close_col:   收盘价列名

    Returns:
        原 df + label、future_ret 两列
    """
    close = df[close_col]

    if label_type == "binary":
        label = _binary(close, horizon)

    elif label_type == "ternary":
        label = _ternary(close, horizon, threshold)

    elif label_type == "return":
        label = _return(close, horizon)

    elif label_type == "triple_barrier":
        label = _triple_barrier(close, horizon, upper_pct, lower_pct)

    else:
        raise ValueError(f"未知 label_type: {label_type}，"
                         f"可选: binary / ternary / return / triple_barrier")

    result = df.copy()
    result["label"]      = label
    result["future_ret"] = close.shift(-horizon) / close - 1

    valid = result["label"].notna().sum()
    dist  = result["label"].value_counts(dropna=True).to_dict()
    logger.info(
        f"标签构建 | type={label_type} | horizon={horizon} | "
        f"有效样本={valid} | 分布={dist}"
    )
    return result


def drop_label_na(df: pd.DataFrame) -> pd.DataFrame:
    """删除末尾无标签的行（用于训练前）。"""
    return df.dropna(subset=["label"])


def get_feature_cols(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """返回特征列名（排除标签和原始价格列）。"""
    default_exclude = {
        "label", "future_ret",
        "close", "open", "high", "low", "volume", "amount",
        "adjustflag", "tradestatus", "isST",
    }
    exclude_set = default_exclude | set(exclude or [])
    return [c for c in df.columns if c not in exclude_set]
