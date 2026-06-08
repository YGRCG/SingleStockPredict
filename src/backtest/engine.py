"""
Walk-forward 回测引擎。
每个交易日用当日收盘前已训练好的模型生成信号，
次日开盘买入/卖出（T+1 执行），计算净值曲线。
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.training.trainer import TrainResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    commission:  float = 0.0003   # 单边佣金率
    slippage:    float = 0.001    # 滑点率（按成交额）
    position:    float = 1.0      # 仓位（1.0 = 全仓）
    threshold:   float = 0.5      # 预测概率阈值，> threshold 做多
    init_cash:   float = 1_000_000.0


def run_backtest(
    df: pd.DataFrame,              # 含 close、label、feature 列的完整表
    feature_cols: list[str],
    train_results: list[TrainResult],
    cfg: BacktestConfig | None = None,
) -> pd.DataFrame:
    """
    Returns:
        每日持仓、收益、净值的 DataFrame
    """
    if cfg is None:
        cfg = BacktestConfig()

    if not train_results:
        raise ValueError("train_results 为空，无法回测。请检查滚动训练步骤是否正常完成。")

    # 构建 date → model 映射（每个预测日用对应模型）
    model_map: dict[pd.Timestamp, TrainResult] = {r.date: r for r in train_results}
    sorted_dates = sorted(model_map.keys())
    logger.info(f"共 {len(sorted_dates)} 个预测日，df 行数={len(df)}")

    records = []
    cash    = cfg.init_cash
    holding = 0.0       # 持股市值
    position = 0        # 当前持仓方向：1 多 / 0 空

    close_series = df["close"]

    skipped_missing = 0
    skipped_no_future = 0
    for i, date in enumerate(sorted_dates):
        if date not in df.index:
            skipped_missing += 1
            continue

        result = model_map[date]
        row    = df.loc[date, feature_cols]
        proba  = result.model.predict_proba(pd.DataFrame([row]))[0]
        signal = int(proba >= cfg.threshold)  # 1=买 0=不持有

        # 取次日开盘执行（简化：用次日收盘价代替）
        future_dates = df.index[df.index > date]
        if len(future_dates) == 0:
            skipped_no_future += 1
            continue
        exec_date  = future_dates[0]
        exec_price = close_series.loc[exec_date]

        # 换仓逻辑
        cost = 0.0
        if signal == 1 and position == 0:
            # 买入
            shares = (cash * cfg.position) / exec_price
            cost   = shares * exec_price * (cfg.commission + cfg.slippage)
            cash  -= shares * exec_price + cost
            holding = shares
            position = 1

        elif signal == 0 and position == 1:
            # 卖出
            proceeds = holding * exec_price
            cost     = proceeds * (cfg.commission + cfg.slippage)
            cash    += proceeds - cost
            holding  = 0.0
            position = 0

        portfolio_value = cash + holding * exec_price
        records.append({
            "date":        exec_date,
            "signal":      signal,
            "proba":       round(proba, 4),
            "price":       exec_price,
            "position":    position,
            "cash":        round(cash, 2),
            "holding_val": round(holding * exec_price, 2),
            "portfolio":   round(portfolio_value, 2),
            "trade_cost":  round(cost, 2),
        })

    if skipped_missing:
        logger.warning(f"  {skipped_missing} 个预测日不在 df.index 中，已跳过")
    if skipped_no_future:
        logger.warning(f"  {skipped_no_future} 个预测日无未来交易日（已是最后一日），已跳过")

    if not records:
        raise ValueError(
            f"回测记录为空。预测日总数={len(sorted_dates)}，"
            f"不在df中={skipped_missing}，无未来日={skipped_no_future}。"
            "请检查数据日期范围与 backtest.start_date 是否匹配。"
        )

    result_df = pd.DataFrame(records).set_index("date").sort_index()
    result_df["nav"] = result_df["portfolio"] / cfg.init_cash
    result_df["ret"] = result_df["nav"].pct_change()

    # 基准：买入持有
    first_price = close_series.loc[result_df.index[0]]
    result_df["benchmark_nav"] = close_series.reindex(result_df.index) / first_price

    logger.info(
        f"回测完成 | {result_df.index[0].date()} ~ {result_df.index[-1].date()} | "
        f"最终净值={result_df['nav'].iloc[-1]:.4f} | 基准={result_df['benchmark_nav'].iloc[-1]:.4f}"
    )
    return result_df
