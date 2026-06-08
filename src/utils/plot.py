"""可视化工具：净值曲线、特征重要性、混淆矩阵。"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


def plot_nav(result_df: pd.DataFrame, title: str = "净值曲线", save_path: str | None = None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(result_df.index, result_df["nav"],           label="策略净值", linewidth=1.5)
    axes[0].plot(result_df.index, result_df["benchmark_nav"], label="基准净值",  linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("净值")
    axes[0].legend()
    axes[0].set_title(title)

    # 回撤
    roll_max  = result_df["nav"].cummax()
    drawdown  = (result_df["nav"] - roll_max) / roll_max
    axes[1].fill_between(result_df.index, drawdown, 0, alpha=0.4, color="red")
    axes[1].set_ylabel("回撤")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 30,
    save_path: str | None = None,
) -> None:
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(8, top_n * 0.3 + 1))
    top[::-1].plot(kind="barh", ax=ax)
    ax.set_title(f"特征重要性 Top {top_n}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion(y_true: pd.Series, y_pred: pd.Series, save_path: str | None = None) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["跌", "涨"]).plot(ax=ax, colorbar=False)
    ax.set_title("混淆矩阵")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
