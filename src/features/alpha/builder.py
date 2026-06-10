import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from .registry import REGISTRY

logger = get_logger(__name__)


def build_alpha_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    根据 config 计算 Alpha101 因子。

    cfg 示例：
        enabled: [1, 2, 6, 12]   # 因子编号列表，或 "all"
        rank_window: 252          # 滚动 rank 窗口
    """
    enabled = cfg.get("enabled", [])
    if enabled == "all":
        enabled = sorted(REGISTRY.keys())
    if not enabled:
        return pd.DataFrame(index=df.index)

    rank_window = int(cfg.get("rank_window", 252))

    data = dict(
        close=df["close"],
        open_=df["open"],
        high=df["high"],
        low=df["low"],
        volume=df["volume"],
        amount=df.get("amount", df["volume"]),  # 无 amount 列时退化为 volume
        rank_window=rank_window,
    )

    result = pd.DataFrame(index=df.index)
    for num in enabled:
        if num not in REGISTRY:
            logger.warning(f"alpha{num:03d} 未在 registry 中注册，跳过")
            continue
        try:
            col = f"alpha{num:03d}"
            result[col] = REGISTRY[num](**data)
        except Exception as e:
            logger.warning(f"alpha{num:03d} 计算失败: {e}")

    implemented = [n for n in enabled if n in REGISTRY]
    logger.info(f"Alpha 因子: 计算 {len(result.columns)} 个 "
                f"({implemented})")
    return result
