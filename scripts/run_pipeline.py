"""
完整流水线：下载 → 特征 → 标签 → 训练 → 回测
用法：python scripts/run_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data.downloader import download_all
from src.features.pipeline import build_feature_matrix
from src.labels.builder import build_labels, drop_label_na, get_feature_cols
from src.models.lgbm_model import LGBMModel
from src.training.trainer import rolling_train
from src.backtest.engine import run_backtest, BacktestConfig
from src.backtest.metrics import calc_metrics, print_metrics
from src.utils.plot import plot_nav
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def load_config(path="config/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model_config(path="config/model_config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg       = load_config()
    model_cfg = load_model_config()
    symbol    = cfg["stock"]["symbol"]

    # 1. 下载数据（跳过已存在且完整的数据）
    if cfg["data"].get("skip_download", False):
        logger.info("=== 步骤 1：数据下载（已跳过） ===")
    else:
        logger.info("=== 步骤 1：数据下载 ===")
        download_all(
            symbol, cfg["data"]["start_date"], cfg["data"]["end_date"],
            skip_existing=True,  # 已下载过的不再重复下载
            include_minutes=cfg.get("include_minutes", True),
        )

    # 2. 特征工程
    logger.info("=== 步骤 2：特征构建 ===")
    feat_df = build_feature_matrix(
        symbol, cfg,
        save_path=f"data/processed/{symbol}_features.parquet",
    )

    # 3. 标签构建
    logger.info("=== 步骤 3：标签构建 ===")
    lcfg   = cfg["label"]
    full_df = build_labels(
        feat_df,
        horizon    = lcfg["horizon"],
        label_type = lcfg["type"],
        threshold  = lcfg["threshold"],
        upper_pct  = lcfg.get("upper_pct", 0.03),
        lower_pct  = lcfg.get("lower_pct", 0.02),
    )
    full_df = drop_label_na(full_df)
    feature_cols = get_feature_cols(full_df)

    # 4. 滚动训练
    logger.info("=== 步骤 4：滚动训练 ===")
    tcfg = cfg["training"]
    train_results = rolling_train(
        full_df,
        feature_cols  = feature_cols,
        model_cls     = LGBMModel,
        model_params  = model_cfg["lgbm"],
        mode          = tcfg["mode"],
        train_window  = tcfg["train_window"],
        val_ratio     = tcfg["val_ratio"],
        step          = tcfg["step"],
        backtest_start = cfg["backtest"]["start_date"],
        save_dir      = "output/models",
        label_type    = lcfg["type"],
    )

    if not train_results:
        raise RuntimeError(
            "滚动训练未产生任何模型。请检查：\n"
            "  1. backtest.start_date 是否在数据范围内且留有足够训练窗口\n"
            "  2. training.train_window 是否小于 backtest_start 前的数据量\n"
            f"  当前 backtest_start={cfg['backtest']['start_date']}, "
            f"train_window={tcfg['train_window']}, full_df 行数={len(full_df)}"
        )
    logger.info(f"训练完成，共 {len(train_results)} 个模型")

    # 5. 回测
    logger.info("=== 步骤 5：回测 ===")
    bt_cfg = BacktestConfig(
        commission = cfg["backtest"]["commission"],
        slippage   = cfg["backtest"]["slippage"],
        threshold  = cfg["backtest"].get("threshold", 0.5),
    )
    result_df = run_backtest(full_df, feature_cols, train_results, bt_cfg)
    
    # 创建输出目录
    import os
    os.makedirs("output/predictions", exist_ok=True)
    result_df.to_csv(f"output/predictions/{symbol}_backtest.csv")

    metrics = calc_metrics(result_df)
    print_metrics(metrics)
    plot_nav(result_df, title=f"{symbol} 回测净值", save_path=f"output/reports/{symbol}_nav.png")


if __name__ == "__main__":
    main()
