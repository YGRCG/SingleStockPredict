"""
单独训练并保存模型。
用法：python scripts/run_train.py [--model lgbm|xgb|lstm]
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from src.labels.builder import build_labels, drop_label_na, get_feature_cols
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel
from src.training.trainer import rolling_train
from src.training.cross_val import time_series_cv
from src.utils.logger import get_logger

logger = get_logger("train")

MODEL_MAP = {"lgbm": LGBMModel, "xgb": XGBModel}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm", choices=["lgbm", "xgb", "lstm"])
    parser.add_argument("--cv", action="store_true", help="先做时序 CV 评估")
    args = parser.parse_args()

    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open("config/model_config.yaml", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    symbol = cfg["stock"]["symbol"]
    feat_path = f"data/processed/{symbol}_features.parquet"

    feat_df  = pd.read_parquet(feat_path)
    lcfg     = cfg["label"]
    full_df  = build_labels(feat_df, lcfg["horizon"], lcfg["type"], lcfg["threshold"])
    full_df  = drop_label_na(full_df)
    feature_cols = get_feature_cols(full_df)

    model_cls    = MODEL_MAP[args.model]
    model_params = model_cfg[args.model]

    if args.cv:
        logger.info("执行时序 CV…")
        cv_res = time_series_cv(full_df, feature_cols, model_cls, model_params)
        logger.info(f"CV: {cv_res}")

    tcfg = cfg["training"]
    rolling_train(
        full_df, feature_cols, model_cls, model_params,
        mode          = tcfg["mode"],
        train_window  = tcfg["train_window"],
        val_ratio     = tcfg["val_ratio"],
        step          = tcfg["step"],
        backtest_start = cfg["backtest"]["start_date"],
        save_dir      = f"output/models/{args.model}",
    )


if __name__ == "__main__":
    main()
