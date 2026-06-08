"""
从 baostock 下载日/周/月 K 线数据，保存为 parquet。

baostock 股票代码规则：
  上证：sh.600000
  深证：sz.000001
"""

import baostock as bs
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# baostock 日线字段
DAILY_FIELDS = (
    "date,open,high,low,close,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)

WEEKLY_FIELDS = "date,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
MONTHLY_FIELDS = WEEKLY_FIELDS


def _bs_code(symbol: str) -> str:
    """将 000001.SZ → sz.000001，600000.SH → sh.600000"""
    code, market = symbol.split(".")
    return f"{market.lower()}.{code}"


def _to_numeric(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    exclude = exclude or ["date", "adjustflag", "tradestatus", "isST"]
    cols = [c for c in df.columns if c not in exclude]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df


def download_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "3",          # 1=后复权 2=前复权 3=不复权
    save_dir: str = "data/raw",
) -> pd.DataFrame:
    """下载日线数据并保存。"""
    bs_code = _bs_code(symbol)
    save_path = Path(save_dir) / f"{symbol}_daily.parquet"

    logger.info(f"下载日线 {symbol}  {start_date} ~ {end_date}")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

    rs = bs.query_history_k_data_plus(
        bs_code,
        DAILY_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjust,
    )
    bs.logout()

    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        raise ValueError(f"未获取到数据: {symbol}")

    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = _to_numeric(df)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path)
    logger.info(f"已保存 {save_path}，共 {len(df)} 条")
    return df


def download_weekly(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "3",
    save_dir: str = "data/raw",
) -> pd.DataFrame:
    """下载周线数据并保存。"""
    bs_code = _bs_code(symbol)
    save_path = Path(save_dir) / f"{symbol}_weekly.parquet"

    logger.info(f"下载周线 {symbol}  {start_date} ~ {end_date}")
    bs.login()
    rs = bs.query_history_k_data_plus(
        bs_code,
        WEEKLY_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="w",
        adjustflag=adjust,
    )
    bs.logout()

    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = _to_numeric(df)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path)
    logger.info(f"已保存 {save_path}，共 {len(df)} 条")
    return df


def download_monthly(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "3",
    save_dir: str = "data/raw",
) -> pd.DataFrame:
    """下载月线数据并保存。"""
    bs_code = _bs_code(symbol)
    save_path = Path(save_dir) / f"{symbol}_monthly.parquet"

    logger.info(f"下载月线 {symbol}  {start_date} ~ {end_date}")
    bs.login()
    rs = bs.query_history_k_data_plus(
        bs_code,
        MONTHLY_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="m",
        adjustflag=adjust,
    )
    bs.logout()

    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = _to_numeric(df)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path)
    logger.info(f"已保存 {save_path}，共 {len(df)} 条")
    return df


# baostock 分钟线字段（不含基本面字段）
MINUTE_FIELDS = "date,time,open,high,low,close,volume,amount,adjustflag"

# baostock frequency 代码映射
_MINUTE_FREQ_MAP = {"min5": "5", "min15": "15", "min30": "30", "min60": "60"}


def download_minute(
    symbol: str,
    start_date: str,
    end_date: str,
    freq: str = "min5",           # min5 / min15 / min30 / min60
    adjust: str = "3",
    save_dir: str = "data/raw",
    chunk_months: int = 3,        # 每段最多查询的月数
) -> pd.DataFrame:
    """
    下载分钟级 K 线。baostock 分钟数据单次最多返回约 500 条，
    需按 chunk_months 个月分段下载再拼接。
    """
    if freq not in _MINUTE_FREQ_MAP:
        raise ValueError(f"freq 必须为 {list(_MINUTE_FREQ_MAP.keys())}，当前: {freq}")

    bs_code  = _bs_code(symbol)
    bs_freq  = _MINUTE_FREQ_MAP[freq]
    save_path = Path(save_dir) / f"{symbol}_{freq}.parquet"

    # 生成分段日期列表
    date_ranges = _split_date_range(start_date, end_date, chunk_months)

    logger.info(f"下载{freq}线 {symbol}  {start_date} ~ {end_date}，共 {len(date_ranges)} 段")

    all_dfs = []
    for i, (s, e) in enumerate(date_ranges):
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

        rs = bs.query_history_k_data_plus(
            bs_code,
            MINUTE_FIELDS,
            start_date=s,
            end_date=e,
            frequency=bs_freq,
            adjustflag=adjust,
        )

        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        bs.logout()

        if rows:
            chunk_df = pd.DataFrame(rows, columns=rs.fields)
            all_dfs.append(chunk_df)
            logger.info(f"  段 {i+1}/{len(date_ranges)}: {s} ~ {e}，获取 {len(rows)} 条")
        else:
            logger.info(f"  段 {i+1}/{len(date_ranges)}: {s} ~ {e}，无数据")

    if not all_dfs:
        raise ValueError(f"未获取到分钟数据: {symbol} {freq}")

    df = pd.concat(all_dfs, ignore_index=True)
    # baostock 分钟线 time 格式：093000000 → 合并 datetime
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"].str[:6], format="%Y-%m-%d %H%M%S")
    df = df.drop(columns=["date", "time"]).set_index("datetime").sort_index()
    # 去重（分段边界可能重叠）
    df = df[~df.index.duplicated(keep="first")]
    df = _to_numeric(df, exclude=["adjustflag"])

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path)
    logger.info(f"已保存 {save_path}，共 {len(df)} 条")
    return df


def _split_date_range(
    start_date: str, end_date: str, chunk_months: int
) -> list[tuple[str, str]]:
    """将日期范围按 chunk_months 个月分段，返回 [(start, end), ...] 列表。"""
    from dateutil.relativedelta import relativedelta

    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    ranges = []
    cur = s
    while cur < e:
        chunk_end = cur + relativedelta(months=chunk_months) - pd.Timedelta(days=1)
        chunk_end = min(chunk_end, e)
        ranges.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + pd.Timedelta(days=1)
    return ranges


def download_all(
    symbol: str,
    start_date: str,
    end_date: str,
    save_dir: str = "data/raw",
    include_minutes: bool = True,
) -> dict[str, pd.DataFrame]:
    """一次性下载日/周/月以及分钟级别数据。"""
    result = {
        "daily":   download_daily(symbol, start_date, end_date, save_dir=save_dir),
        "weekly":  download_weekly(symbol, start_date, end_date, save_dir=save_dir),
        "monthly": download_monthly(symbol, start_date, end_date, save_dir=save_dir),
    }
    if include_minutes:
        for freq in ("min5", "min15", "min30", "min60"):
            try:
                result[freq] = download_minute(symbol, start_date, end_date, freq=freq, save_dir=save_dir)
            except Exception as e:
                logger.warning(f"分钟数据下载失败 {freq}: {e}")
    return result
