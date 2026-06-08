"""回测绩效指标计算。"""

import numpy as np
import pandas as pd


def calc_metrics(result_df: pd.DataFrame, annual_factor: int = 252) -> dict:
    """
    Args:
        result_df:     run_backtest 返回的 DataFrame
        annual_factor: 年化因子（日频=252）
    """
    nav  = result_df["nav"]
    ret  = result_df["ret"].dropna()
    bm   = result_df["benchmark_nav"]

    total_ret    = nav.iloc[-1] - 1
    annual_ret   = (1 + total_ret) ** (annual_factor / len(nav)) - 1
    annual_vol   = ret.std() * np.sqrt(annual_factor)
    sharpe       = annual_ret / annual_vol if annual_vol > 0 else np.nan

    # 最大回撤
    roll_max     = nav.cummax()
    drawdown     = (nav - roll_max) / roll_max
    max_drawdown = drawdown.min()

    # 胜率（信号为 1 且次日上涨）
    if "signal" in result_df.columns and "ret" in result_df.columns:
        long_ret = result_df.loc[result_df["signal"] == 1, "ret"].dropna()
        win_rate = (long_ret > 0).mean() if len(long_ret) > 0 else np.nan
    else:
        win_rate = np.nan

    # 超额收益
    excess_ret   = total_ret - (bm.iloc[-1] - 1)

    return {
        "total_return":    round(total_ret,    4),
        "annual_return":   round(annual_ret,   4),
        "annual_vol":      round(annual_vol,   4),
        "sharpe":          round(sharpe,       4),
        "max_drawdown":    round(max_drawdown, 4),
        "win_rate":        round(win_rate,     4) if not np.isnan(win_rate) else np.nan,
        "excess_return":   round(excess_ret,   4),
        "benchmark_return": round(bm.iloc[-1] - 1, 4),
        "n_days":          len(nav),
    }


def print_metrics(metrics: dict) -> None:
    print("\n===== 回测绩效 =====")
    for k, v in metrics.items():
        pct_keys = {"total_return", "annual_return", "annual_vol", "max_drawdown",
                    "win_rate", "excess_return", "benchmark_return"}
        val_str = f"{v*100:.2f}%" if k in pct_keys else str(v)
        print(f"  {k:<22}: {val_str}")
    print("====================\n")
