"""
加载已保存的模型，执行回测并输出报告。
用法：python scripts/run_backtest.py --model lgbm
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import joblib
import pandas as pd
from src.labels.builder import build_labels, drop_label_na, get_feature_cols
from src.training.trainer import TrainResult
from src.backtest.engine import run_backtest, BacktestConfig
from src.backtest.metrics import calc_metrics, print_metrics
from src.utils.plot import plot_nav, plot_feature_importance
from src.utils.logger import get_logger

logger = get_logger("backtest")


def load_saved_models(model_dir: str) -> list[TrainResult]:
    """加载目录下所有 .pkl 模型文件，按日期排序。"""
    results = []
    for p in sorted(Path(model_dir).glob("*.pkl")):
        date_str = p.stem.split("_")[-1]
        date = pd.Timestamp(date_str)
        model = joblib.load(p)
        results.append(TrainResult(date=date, model=model, val_score=float("nan"),
                                   feature_cols=getattr(model, "feature_cols", [])))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm")
    args = parser.parse_args()

    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["stock"]["symbol"]
    feat_df  = pd.read_parquet(f"data/processed/{symbol}_features.parquet")
    lcfg     = cfg["label"]
    full_df  = build_labels(feat_df, lcfg["horizon"], lcfg["type"], lcfg["threshold"])
    full_df  = drop_label_na(full_df)
    feature_cols = get_feature_cols(full_df)

    train_results = load_saved_models(f"output/models/{args.model}")
    if not train_results:
        logger.error(f"未找到模型文件，请先运行 run_train.py")
        return

    bt_cfg = BacktestConfig(
        commission = cfg["backtest"]["commission"],
        slippage   = cfg["backtest"]["slippage"],
        threshold  = cfg["backtest"].get("threshold", 0.5),
    )
    result_df = run_backtest(full_df, feature_cols, train_results, bt_cfg)
    result_df.to_csv(f"output/predictions/{symbol}_{args.model}_backtest.csv")

    metrics = calc_metrics(result_df)
    print_metrics(metrics)
    plot_nav(result_df, f"{symbol} {args.model} 回测",
             save_path=f"output/reports/{symbol}_{args.model}_nav.png")


if __name__ == "__main__":
    main()
