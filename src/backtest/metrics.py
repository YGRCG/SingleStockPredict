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


def calc_accuracy(result_df: pd.DataFrame, rolling_window: int = 20) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    valid = result_df.dropna(subset=["actual", "hit"])
    y_true = valid["actual"].astype(int)
    y_pred = valid["pred_dir"].astype(int)

    accuracy = valid["hit"].mean()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    rolling_acc = valid["hit"].rolling(rolling_window, min_periods=1).mean()
    monthly_acc = valid.groupby(valid.index.to_period("M"))["hit"].mean()

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_predictions": len(valid),
        "rolling_accuracy": rolling_acc,
        "monthly_accuracy": monthly_acc,
    }


def print_accuracy(metrics: dict) -> None:
    print("\n===== 预测准确率 =====")
    pct_keys = {"accuracy", "precision", "recall", "f1"}
    for k in ["accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn", "n_predictions"]:
        v = metrics[k]
        val_str = f"{v*100:.2f}%" if k in pct_keys else str(v)
        print(f"  {k:<22}: {val_str}")
    print("====================\n")


def optimize_threshold(result_df: pd.DataFrame, lo: float = 0.30, hi: float = 0.70, step: float = 0.01) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score

    valid = result_df.dropna(subset=["actual", "pred"])
    y_true = valid["actual"].astype(int)
    preds = valid["pred"]
    n_total = len(valid)

    best_acc_t, best_acc = 0.5, 0.0
    best_f1_t, best_f1 = 0.5, 0.0
    rows = []
    for t_int in range(int(lo * 100), int(hi * 100) + 1, int(step * 100)):
        t = t_int / 100
        y_pred = (preds >= t).astype(int)
        n_long = int(y_pred.sum())
        long_ratio = n_long / n_total
        acc = (y_pred == y_true).mean()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if acc > best_acc:
            best_acc, best_acc_t = acc, t
        if f1 > best_f1 and 0.2 < long_ratio < 0.8:
            best_f1, best_f1_t = f1, t
        rows.append({"threshold": t, "accuracy": acc, "precision": prec,
                      "recall": rec, "f1": f1, "n_long": n_long})

    return {"best_acc_threshold": best_acc_t, "best_accuracy": round(best_acc, 4),
            "best_f1_threshold": best_f1_t, "best_f1": round(best_f1, 4),
            "n_total": n_total, "scan": rows}


def print_threshold_scan(result: dict, top_n: int = 8) -> None:
    print(f"\n===== 阈值优化 =====")
    print(f"  最优 accuracy: threshold={result['best_acc_threshold']:.2f}  accuracy={result['best_accuracy']*100:.2f}%")
    print(f"  最优 F1 (均衡): threshold={result['best_f1_threshold']:.2f}  F1={result['best_f1']*100:.2f}%")
    print(f"\n  按 accuracy 排序 Top-{top_n}:")
    rows = sorted(result["scan"], key=lambda r: r["accuracy"], reverse=True)[:top_n]
    print(f"  {'threshold':>9}  {'accuracy':>9}  {'precision':>10}  {'recall':>7}  {'f1':>7}  {'n_long':>7}")
    for r in rows:
        print(f"  {r['threshold']:9.2f}  {r['accuracy']*100:8.2f}%  {r['precision']*100:9.2f}%  {r['recall']*100:6.2f}%  {r['f1']*100:6.2f}%  {r['n_long']:7d}")
    print("====================\n")


def print_metrics(metrics: dict) -> None:
    print("\n===== 回测绩效 =====")
    for k, v in metrics.items():
        pct_keys = {"total_return", "annual_return", "annual_vol", "max_drawdown",
                    "win_rate", "excess_return", "benchmark_return"}
        val_str = f"{v*100:.2f}%" if k in pct_keys else str(v)
        print(f"  {k:<22}: {val_str}")
    print("====================\n")
