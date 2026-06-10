"""
从本地 parquet 加载数据，提供统一接口。
若文件不存在则自动触发下载。
"""

from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_kline(symbol: str, period: str, data_dir: str = "data/raw") -> pd.DataFrame:
    """
    加载指定周期的 K 线数据。

    Args:
        symbol:  股票代码，如 000001.SZ
        period:  daily / weekly / monthly
        data_dir: 数据根目录

    Returns:
        以 date 为索引的 DataFrame
    """
    path = Path(data_dir) / f"{symbol}_{period}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，请先运行 downloader.download_{period}()"
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


def load_all_periods(
    symbol: str,
    data_dir: str = "data/raw",
    periods: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """加载指定周期数据，返回字典。
    periods 为 None 时加载全部；daily 始终必须存在。
    """
    all_periods = ["daily", "weekly", "monthly", "min5", "min15", "min30", "min60"]
    requested = set(periods) if periods else set(all_periods)
    requested.add("daily")  # daily 始终加载

    result = {}
    for period in all_periods:
        if period not in requested:
            continue
        try:
            result[period] = load_kline(symbol, period, data_dir)
        except FileNotFoundError:
            if period == "daily":
                raise
            logger.info(f"跳过 {period} 数据（未启用或文件不存在）")
    return result
