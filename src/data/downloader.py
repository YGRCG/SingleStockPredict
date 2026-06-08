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


def _bs_query_with_retry(
    bs_code: str,
    fields: str,
    start_date: str,
    end_date: str,
    frequency: str,
    adjust: str = "3",
    max_retries: int = 3,
) -> tuple[list[list], list[str]]:
    """带重试的 baostock 查询，解决连接不稳定问题。返回 (rows, fields)。"""
    import time as _time
    for attempt in range(max_retries):
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

        rs = bs.query_history_k_data_plus(
            bs_code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjust,
        )

        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        col_names = rs.fields
        bs.logout()

        if rows:
            return rows, col_names
        logger.warning(f"  第 {attempt+1} 次查询无数据 ({bs_code} {frequency} {start_date}~{end_date})，重试...")
        _time.sleep(2)

    return [], []


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
    rows, col_names = _bs_query_with_retry(bs_code, DAILY_FIELDS, start_date, end_date, "d", adjust)

    if not rows:
        raise ValueError(f"未获取到数据: {symbol}")

    df = pd.DataFrame(rows, columns=col_names)
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
    rows, col_names = _bs_query_with_retry(bs_code, WEEKLY_FIELDS, start_date, end_date, "w", adjust)

    if not rows:
        raise ValueError(f"未获取到周线数据: {symbol}")

    df = pd.DataFrame(rows, columns=col_names)
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
    rows, col_names = _bs_query_with_retry(bs_code, MONTHLY_FIELDS, start_date, end_date, "m", adjust)

    if not rows:
        raise ValueError(f"未获取到月线数据: {symbol}")

    df = pd.DataFrame(rows, columns=col_names)
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
    注意：baostock 分钟数据仅保留 2020-01-03 至今的数据。
    """
    if freq not in _MINUTE_FREQ_MAP:
        raise ValueError(f"freq 必须为 {list(_MINUTE_FREQ_MAP.keys())}，当前: {freq}")
    
    # 日期范围校验：baostock 分钟数据仅从2020年开始
    earliest_available = pd.Timestamp("2020-01-03")
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    if s < earliest_available:
        logger.info(f"baostock 分钟数据仅从2020-01-03开始，自动调整起始日期")
        s = earliest_available
        start_date = s.strftime("%Y-%m-%d")
    if s >= e:
        raise ValueError(f"有效日期范围为空（已调整为 {start_date} ~ {end_date}")

    bs_code  = _bs_code(symbol)
    bs_freq  = _MINUTE_FREQ_MAP[freq]
    save_path = Path(save_dir) / f"{symbol}_{freq}.parquet"

    # 生成分段日期列表
    date_ranges = _split_date_range(start_date, end_date, chunk_months)

    logger.info(f"下载{freq}线 {symbol}  {start_date} ~ {end_date}，共 {len(date_ranges)} 段")

    all_dfs = []
    for i, (s, e) in enumerate(date_ranges):
        # 每段最多重试 3 次（baostock 连接不稳定时可能返回空数据）
        rows = []
        for attempt in range(3):
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
                break
            import time as _time
            logger.warning(f"  段 {i+1}/{len(date_ranges)}: {s} ~ {e}，第 {attempt+1} 次尝试无数据，重试...")
            _time.sleep(2)

        if rows:
            chunk_df = pd.DataFrame(rows, columns=rs.fields)
            all_dfs.append(chunk_df)
            logger.info(f"  段 {i+1}/{len(date_ranges)}: {s} ~ {e}，获取 {len(rows)} 条")
        else:
            logger.warning(f"  段 {i+1}/{len(date_ranges)}: {s} ~ {e}，3 次尝试均无数据，跳过")

    if not all_dfs:
        raise ValueError(f"未获取到分钟数据: {symbol} {freq}")

    df = pd.concat(all_dfs, ignore_index=True)
    # baostock 分钟线 time 格式：YYYYMMDDHHMMSS000（17位），如 20250603093500000
    # 直接用 time 字段解析为完整 datetime
    df["datetime"] = pd.to_datetime(df["time"].str[:14], format="%Y%m%d%H%M%S")
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


def _check_data_integrity(
    symbol: str, period: str, start_date: str, end_date: str, save_dir: str
) -> bool:
    """检查本地数据是否完整（覆盖起始日期、行数合理、数据较新）。"""
    path = Path(save_dir) / f"{symbol}_{period}.parquet"
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if len(df) == 0:
            return False
        # 起始日期必须覆盖
        if df.index.min() > pd.Timestamp(start_date) + pd.Timedelta(days=5):
            return False
        # 结束日期：取 end_date 和今天中较早的那个，数据不应差太多
        effective_end = min(pd.Timestamp(end_date), pd.Timestamp.now())
        if period in ("min5", "min15", "min30", "min60"):
            # 分钟线允许差 3 天（baostock 分钟数据更新有延迟）
            max_gap = pd.Timedelta(days=3)
        else:
            # 日线/周线/月线允许差 2 天
            max_gap = pd.Timedelta(days=2)
        if df.index.max() < effective_end - max_gap:
            return False
        # 检查行数是否合理
        actual_days = (df.index.max() - df.index.min()).days
        if actual_days <= 0:
            return False
        if period == "daily":
            expected_rows = actual_days * 250 // 365
        elif period == "weekly":
            expected_rows = actual_days * 52 // 365
        elif period == "monthly":
            expected_rows = actual_days * 12 // 365
        else:  # 分钟线
            bars_per_day = {"min5": 48, "min15": 16, "min30": 8, "min60": 4}
            expected_rows = actual_days * bars_per_day.get(period, 48) * 250 // 365
        # 允许30%误差（停牌、节假日等）
        if len(df) < expected_rows * 0.7:
            return False
        return True
    except Exception as e:
        logger.warning(f"数据完整性检查失败 {period}: {e}")
        return False


def download_all(
    symbol: str,
    start_date: str,
    end_date: str,
    save_dir: str = "data/raw",
    include_minutes: bool = True,
    skip_existing: bool = True,  # 跳过已存在且完整的数据
) -> dict[str, pd.DataFrame]:
    """一次性下载日/周/月以及分钟级别数据。"""
    result = {}
    # 日线/周线/月线
    for period, download_func in [
        ("daily", download_daily),
        ("weekly", download_weekly),
        ("monthly", download_monthly),
    ]:
        if skip_existing and _check_data_integrity(symbol, period, start_date, end_date, save_dir):
            logger.info(f"跳过 {period} 数据（已存在且完整）")
            result[period] = pd.read_parquet(Path(save_dir) / f"{symbol}_{period}.parquet")
            continue
        result[period] = download_func(symbol, start_date, end_date, save_dir=save_dir)
    
    # 分钟线
    if include_minutes:
        for freq in ("min5", "min15", "min30", "min60"):
            try:
                if skip_existing and _check_data_integrity(symbol, freq, start_date, end_date, save_dir):
                    logger.info(f"跳过 {freq} 数据（已存在且完整）")
                    result[freq] = pd.read_parquet(Path(save_dir) / f"{symbol}_{freq}.parquet")
                    continue
                result[freq] = download_minute(symbol, start_date, end_date, freq=freq, save_dir=save_dir)
            except Exception as e:
                logger.warning(f"分钟数据下载失败 {freq}: {e}")
    return result
