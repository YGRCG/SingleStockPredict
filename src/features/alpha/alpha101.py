"""
Alpha101 因子实现（部分）。
参考：Kakushadze 2016, "101 Formulaic Alphas"

每个函数签名统一为 (close, open_, high, low, volume, amount, rank_window, **_)，
用 **_ 忽略不需要的字段，方便 builder 统一调用。
"""

import numpy as np
import pandas as pd

from .ops import (
    delay, delta, sign, log, signed_power,
    rank, ts_rank, ts_max, ts_min, ts_argmax, ts_argmin,
    ts_mean, ts_sum, ts_std, ts_corr, ts_cov,
    decay_linear, scale, adv, vwap,
)


def alpha001(close, rank_window=252, **_):
    """动量反转：下跌时用波动率替代价格，捕捉超跌后的反弹信号。
    rank(Ts_ArgMax(SignedPower((returns<0 ? stddev(returns,20) : close), 2), 5)) - 0.5"""
    r = close.pct_change()
    base = ts_std(r, 20).where(r < 0, close)
    return rank(ts_argmax(signed_power(base, 2), 5), rank_window) - 0.5


def alpha002(open_, close, volume, rank_window=252, **_):
    """量价背离：成交量加速与日内涨幅的相关性为负时看多。
    -1 * corr(rank(delta(log(volume), 2)), rank((close-open)/open), 6)"""
    x = rank(delta(log(volume), 2), rank_window)
    y = rank((close - open_) / open_.replace(0, np.nan), rank_window)
    return -1 * ts_corr(x, y, 6)


def alpha003(open_, volume, rank_window=252, **_):
    """开盘价与成交量负相关：量增价跌的做空逻辑。
    -1 * corr(rank(open), rank(volume), 10)"""
    return -1 * ts_corr(rank(open_, rank_window), rank(volume, rank_window), 10)


def alpha006(open_, volume, **_):
    """开盘量价负相关：高开放量视为短期利空信号。
    -1 * corr(open, volume, 10)"""
    return -1 * ts_corr(open_, volume, 10)


def alpha007(close, volume, **_):
    """成交量放大时，近期价格波动越大、方向越强，信号越反向。
    (adv20<volume) ? (-1*ts_rank(|delta(close,7)|,60)*sign(delta(close,7))) : -1"""
    adv20 = adv(volume, 20)
    d7 = delta(close, 7)
    a = (-1 * ts_rank(d7.abs(), 60)) * sign(d7)
    return a.where(adv20 < volume, -1.0)


def alpha008(open_, close, rank_window=252, **_):
    """开盘价与收益率乘积的短期动量衰减：近期共振弱于历史则看多。
    -1 * rank(sum(open,5)*sum(returns,5) - delay(sum(open,5)*sum(returns,5), 10))"""
    r = close.pct_change()
    x = ts_sum(open_, 5) * ts_sum(r, 5)
    return -1 * rank(x - delay(x, 10), rank_window)


def alpha009(close, **_):
    """趋势跟随：连续上涨则顺势，连续下跌则顺势，震荡时反转。
    ts_min(delta,5)>0 ? delta : ts_max(delta,5)<0 ? delta : -delta"""
    d = delta(close, 1)
    return d.where(ts_min(d, 5) > 0,
           d.where(ts_max(d, 5) < 0, -1 * d))


def alpha010(close, rank_window=252, **_):
    """alpha009 的排名版，趋势/反转信号的相对强弱。
    rank(ts_min(delta,4)>0 ? delta : ts_max(delta,4)<0 ? delta : -delta)"""
    d = delta(close, 1)
    inner = d.where(ts_min(d, 4) > 0,
             d.where(ts_max(d, 4) < 0, -1 * d))
    return rank(inner, rank_window)


def alpha011(close, volume, amount, rank_window=252, **_):
    """VWAP 偏离度结合成交量变化：量增且偏离扩大时信号增强。
    (rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3))"""
    vw = vwap(amount, volume)
    diff = vw - close
    return (rank(ts_max(diff, 3), rank_window) + rank(ts_min(diff, 3), rank_window)) \
           * rank(delta(volume, 3), rank_window)


def alpha012(close, volume, **_):
    """成交量方向与价格变动反向：放量下跌或缩量上涨时看多。
    sign(delta(volume,1)) * (-1 * delta(close,1))"""
    return sign(delta(volume, 1)) * (-1 * delta(close, 1))


def alpha014(open_, close, volume, rank_window=252, **_):
    """收益率动量衰减叠加量价相关性：两者共振时信号更强。
    -1 * rank(delta(returns,3)) * corr(open, volume, 10)"""
    r = close.pct_change()
    return (-1 * rank(delta(r, 3), rank_window)) * ts_corr(open_, volume, 10)


def alpha015(high, volume, rank_window=252, **_):
    """高价与成交量相关性的累积排名：持续量价背离看空。
    -1 * sum(rank(corr(rank(high), rank(volume), 3)), 3)"""
    c = ts_corr(rank(high, rank_window), rank(volume, rank_window), 3)
    return -1 * ts_sum(rank(c, rank_window), 3)


def alpha016(high, volume, rank_window=252, **_):
    """高价与成交量协方差的排名：量价协同上涨反而是短期顶部信号。
    -1 * rank(cov(rank(high), rank(volume), 5))"""
    return -1 * rank(ts_cov(rank(high, rank_window), rank(volume, rank_window), 5), rank_window)


def alpha019(close, rank_window=252, **_):
    """中期趋势反转叠加长期收益排名：趋势越强、长期涨幅越高时反转信号越强。
    (-1*sign(delta(close,7))) * (1 + rank(1 + sum(returns,250)))"""
    r = close.pct_change()
    return (-1 * sign(delta(close, 7))) * (1 + rank(1 + ts_sum(r, 250), rank_window))


def alpha025(close, high, volume, amount, rank_window=252, **_):
    """多因子复合：日收益反向 × 成交量 × VWAP × 上影线，捕捉滞涨特征。
    rank((-1*returns) * adv20 * vwap * (high-close))"""
    r = close.pct_change()
    vw = vwap(amount, volume)
    return rank((-1 * r) * adv(volume, 20) * vw * (high - close), rank_window)


def alpha026(high, volume, **_):
    """高价与成交量时序排名的滚动相关性峰值取反：量价共振后反转。
    -1 * ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)"""
    return -1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)


def alpha004(low, rank_window=252, **_):
    """近期低价排名的时序排名取反：低价越强势越看空。
    -1 * Ts_Rank(rank(low), 9)"""
    return -1 * ts_rank(rank(low, rank_window), 9)


def alpha005(open_, close, volume, amount, rank_window=252, **_):
    """开盘偏离VWAP均值与收盘偏离VWAP的组合：两者方向一致时信号增强。
    rank(open - mean(vwap,10)) * (-1 * |rank(close - vwap)|)"""
    vw = vwap(amount, volume)
    return rank(open_ - ts_mean(vw, 10), rank_window) * (-1 * rank(close - vw, rank_window).abs())


def alpha013(close, volume, rank_window=252, **_):
    """收盘价与成交量排名的协方差取反：量价协同的反转信号。
    -1 * rank(cov(rank(close), rank(volume), 5))"""
    return -1 * rank(ts_cov(rank(close, rank_window), rank(volume, rank_window), 5), rank_window)


def alpha017(close, volume, rank_window=252, **_):
    """价格时序排名 × 价格加速度 × 成交量比的复合反转信号。
    (-1*rank(ts_rank(close,10))) * rank(delta(delta(close,1),1)) * rank(ts_rank(vol/adv20,5))"""
    adv20_ = adv(volume, 20)
    return (-1 * rank(ts_rank(close, 10), rank_window)) \
           * rank(delta(delta(close, 1), 1), rank_window) \
           * rank(ts_rank(volume / adv20_.replace(0, np.nan), 5), rank_window)


def alpha018(open_, close, rank_window=252, **_):
    """日内振幅、收盘偏离与量价相关的综合排名：波动大方向不利时看空。
    -1 * rank(stddev(|close-open|,5) + (close-open) + corr(close,open,10))"""
    x = ts_std((close - open_).abs(), 5) + (close - open_) + ts_corr(close, open_, 10)
    return -1 * rank(x, rank_window)


def alpha020(open_, high, low, close, rank_window=252, **_):
    """开盘价相对于昨日高低收的三向偏离乘积：低开强势信号。
    (-1*rank(open-delay(high,1))) * rank(open-delay(close,1)) * rank(open-delay(low,1))"""
    return (-1 * rank(open_ - delay(high, 1), rank_window)) \
           * rank(open_ - delay(close, 1), rank_window) \
           * rank(open_ - delay(low, 1), rank_window)


def alpha021(close, volume, **_):
    """均价与波动带的位置结合成交量：突破上轨放量做多，跌破下轨做空。
    (sma8+std8 < sma2) ? -1 : (sma2 < sma8-std8) ? 1 : (vol/adv20>=1) ? 1 : -1"""
    sma8 = ts_mean(close, 8)
    std8 = ts_std(close, 8)
    sma2 = ts_mean(close, 2)
    vol_ratio = volume / adv(volume, 20).replace(0, np.nan)
    return pd.Series(
        np.where(sma8 + std8 < sma2, -1.0,
        np.where(sma2 < sma8 - std8,  1.0,
        np.where(vol_ratio >= 1,       1.0, -1.0))),
        index=close.index, dtype=float,
    )


def alpha022(high, close, volume, rank_window=252, **_):
    """高价量相关性变化叠加波动率排名：量价关系恶化时看空。
    -1 * (delta(corr(high,volume,5), 5) * rank(stddev(close,20)))"""
    return -1 * (delta(ts_corr(high, volume, 5), 5) * rank(ts_std(close, 20), rank_window))


def alpha023(high, **_):
    """高价突破20日均值后的短期反转：突破均值时做空近期涨幅。
    (mean(high,20) < high) ? -1*delta(high,2) : 0"""
    return (-1 * delta(high, 2)).where(ts_mean(high, 20) < high, 0.0)


def alpha024(close, **_):
    """长期均线斜率极低时做空反弹，否则跟随3日动量反转。
    (delta(sma100,100)/delay(close,100) <= 0.05) ? -1*(close-ts_min(close,100)) : -delta(close,3)"""
    sma100 = ts_mean(close, 100)
    ratio = delta(sma100, 100) / delay(close, 100).replace(0, np.nan)
    return (-1 * (close - ts_min(close, 100))).where(ratio <= 0.05, -1 * delta(close, 3))


def alpha027(close, volume, amount, rank_window=252, **_):
    """量VWAP相关性均值的排名：持续量价共振时做空。
    (rank(mean(corr(rank(vol),rank(vwap),6),2)) > 0.5) ? -1 : 1"""
    vw = vwap(amount, volume)
    x = ts_mean(ts_corr(rank(volume, rank_window), rank(vw, rank_window), 6), 2)
    return pd.Series(
        np.where(rank(x, rank_window) > 0.5, -1.0, 1.0),
        index=close.index, dtype=float,
    )


def alpha028(close, high, low, volume, **_):
    """成交量均值与低价相关性加均价减收盘的归一化：捕捉价格与量能的偏离。
    scale(corr(adv20,low,5) + (high+low)/2 - close)"""
    return scale(ts_corr(adv(volume, 20), low, 5) + (high + low) / 2 - close)


def alpha029(close, rank_window=252, **_):
    """多层嵌套排名与动量的组合：提取价格趋势的深层非线性特征。
    ts_min(rank(rank(scale(log(ts_min(rank(rank(-rank(delta(close-1,5)))),2))))),5) + ts_rank(delay(-ret,6),5)"""
    r = close.pct_change()
    x = -1 * rank(delta(close - 1, 5), rank_window)
    x = rank(rank(x, rank_window), rank_window)
    x = ts_min(x, 2)
    x = np.log(x.clip(lower=1e-8))
    x = scale(x)
    x = rank(rank(x, rank_window), rank_window)
    return ts_min(x, 5) + ts_rank(delay(-1 * r, 6), 5)


def alpha030(close, volume, rank_window=252, **_):
    """近3日涨跌符号累积排名叠加短期成交量比：逆势放量信号。
    (1 - rank(sign3)) * sum_vol5 / sum_vol20"""
    s = (sign(delta(close, 1))
         + sign(delay(close, 1) - delay(close, 2))
         + sign(delay(close, 2) - delay(close, 3)))
    return (1.0 - rank(s, rank_window)) \
           * ts_sum(volume, 5) / ts_sum(volume, 20).replace(0, np.nan)


def alpha031(close, low, volume, rank_window=252, **_):
    """多层衰减排名的趋势反转叠加短期动量与量价相关符号。
    rank(rank(rank(decay_linear(-rank(rank(delta(close,10))),10)))) + rank(-delta(close,3)) + sign(scale(corr(adv20,low,12)))"""
    x = decay_linear(-1 * rank(rank(delta(close, 10), rank_window), rank_window), 10)
    part1 = rank(rank(rank(x, rank_window), rank_window), rank_window)
    part2 = rank(-1 * delta(close, 3), rank_window)
    part3 = sign(scale(ts_corr(adv(volume, 20), low, 12)))
    return part1 + part2 + part3


def alpha032(close, volume, amount, **_):
    """VWAP与滞后收盘价的长期相关性加收盘偏离均价：捕捉价格回归均值信号。
    scale(mean(close,7) - close) + 20*scale(corr(vwap, delay(close,5), 230))"""
    vw = vwap(amount, volume)
    return scale(ts_mean(close, 7) - close) + 20 * scale(ts_corr(vw, delay(close, 5), 230))


def alpha033(open_, close, rank_window=252, **_):
    """开收比排名：开盘相对收盘越低越看多。
    rank(-(1 - open/close))"""
    return rank(-1 * (1 - open_ / close.replace(0, np.nan)), rank_window)


def alpha034(close, rank_window=252, **_):
    """短期波动率比值与价格动量的反向排名组合。
    rank((1-rank(std(ret,2)/std(ret,5))) + (1-rank(delta(close,1))))"""
    r = close.pct_change()
    std_ratio = ts_std(r, 2) / ts_std(r, 5).replace(0, np.nan)
    return rank((1 - rank(std_ratio, rank_window)) + (1 - rank(delta(close, 1), rank_window)), rank_window)


def alpha035(close, high, low, volume, **_):
    """成交量时序排名 × (1-价格区间排名) × (1-收益率排名)：放量低波动反转。
    ts_rank(volume,32) * (1-ts_rank(close+high-low,16)) * (1-ts_rank(returns,32))"""
    r = close.pct_change()
    return ts_rank(volume, 32) \
           * (1 - ts_rank(close + high - low, 16)) \
           * (1 - ts_rank(r, 32))


def alpha036(open_, close, volume, amount, rank_window=252, **_):
    """五项加权组合：日内量价、方向、反转动量、VWAP相关、均值回归。
    2.21*rank(corr(close-open,delay(vol,1),15)) + 0.7*rank(open-close) + 0.73*rank(ts_rank(delay(-ret,6),5)) + rank(|corr(vwap,adv20,6)|) + 0.45*rank(zscore(ret,20))"""
    vw = vwap(amount, volume)
    r = close.pct_change()
    part1 = 2.21 * rank(ts_corr(close - open_, delay(volume, 1), 15), rank_window)
    part2 = 0.7  * rank(open_ - close, rank_window)
    part3 = 0.73 * rank(ts_rank(delay(-1 * r, 6), 5), rank_window)
    part4 = rank(ts_corr(vw, adv(volume, 20), 6).abs(), rank_window)
    part5 = 0.45 * rank((ts_mean(r, 20) - r) / ts_std(r, 20).replace(0, np.nan), rank_window)
    return part1 + part2 + part3 + part4 + part5


def alpha037(open_, close, rank_window=252, **_):
    """开收差的滞后相关性加日内方向排名：趋势延续信号。
    rank(corr(delay(open-close,1), close, 200)) + rank(open-close)"""
    return rank(ts_corr(delay(open_ - close, 1), close, 200), rank_window) \
           + rank(open_ - close, rank_window)


def alpha038(open_, close, rank_window=252, **_):
    """收盘时序排名反向 × 收开比排名：趋势末端做空。
    -1 * rank(ts_rank(close,10)) * rank(close/open)"""
    return (-1 * rank(ts_rank(close, 10), rank_window)) \
           * rank(close / open_.replace(0, np.nan), rank_window)


def alpha039(close, volume, rank_window=252, **_):
    """7日价格变动叠加量比衰减权重与长期收益排名的反转信号。
    -1*rank(delta(close,7)*(1-rank(decay_linear(vol/adv20,9)))) * (1+rank(sum(ret,250)))"""
    r = close.pct_change()
    weight = 1 - rank(decay_linear(volume / adv(volume, 20).replace(0, np.nan), 9), rank_window)
    return (-1 * rank(delta(close, 7) * weight, rank_window)) \
           * (1 + rank(ts_sum(r, 250), rank_window))


def alpha040(high, volume, rank_window=252, **_):
    """高价波动率排名与量价相关的乘积取反：高波动量价共振看空。
    -1 * rank(stddev(high,10)) * corr(high,volume,10)"""
    return (-1 * rank(ts_std(high, 10), rank_window)) * ts_corr(high, volume, 10)


def alpha041(high, low, volume, amount, **_):
    """高低价几何均值与VWAP之差：VWAP偏离公允价值的度量。
    sqrt(high*low) - vwap"""
    return (high * low) ** 0.5 - vwap(amount, volume)


def alpha042(close, volume, amount, rank_window=252, **_):
    """VWAP超越收盘的排名比率：资金定价能力相对强弱。
    rank(vwap-close) / rank(vwap+close)"""
    vw = vwap(amount, volume)
    return rank(vw - close, rank_window) / rank(vw + close, rank_window).replace(0, np.nan)


def alpha043(close, volume, **_):
    """量比时序排名 × 反向7日价格动量时序排名：放量反转信号。
    ts_rank(vol/adv20, 20) * ts_rank(-delta(close,7), 8)"""
    return ts_rank(volume / adv(volume, 20).replace(0, np.nan), 20) \
           * ts_rank(-1 * delta(close, 7), 8)


def alpha044(high, volume, rank_window=252, **_):
    """高价与成交量排名的相关性取反：量价正相关是短期顶部信号。
    -1 * corr(high, rank(volume), 5)"""
    return -1 * ts_corr(high, rank(volume, rank_window), 5)


def alpha045(close, volume, rank_window=252, **_):
    """滞后收盘均值排名 × 量价短期相关 × 收盘和的相关排名的乘积取反。
    -1 * rank(mean(delay(close,5),20)) * corr(close,vol,2) * rank(corr(sum(close,5),sum(close,20),2))"""
    return -1 * (rank(ts_sum(delay(close, 5), 20) / 20, rank_window)
                 * ts_corr(close, volume, 2)
                 * rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2), rank_window))


def alpha046(close, **_):
    """价格加速度判断趋势状态：加速过猛做空，加速向上做多，震荡时跟随日动量反向。
    accel>0.25 ? -1 : accel<0 ? 1 : -delta(close,1)"""
    accel = ((delay(close, 20) - delay(close, 10)) / 10) \
          - ((delay(close, 10) - close) / 10)
    return pd.Series(
        np.where(accel > 0.25, -1.0,
        np.where(accel < 0,    1.0,
                 -1.0 * delta(close, 1))),
        index=close.index, dtype=float,
    )


def alpha047(close, high, volume, amount, rank_window=252, **_):
    """价格倒数量比 × 高价偏离加权，减VWAP动量排名：量价综合强弱信号。
    (rank(1/close)*vol/adv20) * (high*rank(high-close)/mean(high,5)) - rank(vwap-delay(vwap,5))"""
    vw = vwap(amount, volume)
    adv20_ = adv(volume, 20)
    part1 = (rank(1 / close.replace(0, np.nan), rank_window) * volume
             / adv20_.replace(0, np.nan)) \
            * (high * rank(high - close, rank_window)
               / ts_mean(high, 5).replace(0, np.nan))
    return part1 - rank(vw - delay(vw, 5), rank_window)


def alpha049(close, **_):
    """价格加速度深度向上时做多，否则跟随日动量反向。
    accel < -0.1 ? 1 : -delta(close,1)"""
    accel = ((delay(close, 20) - delay(close, 10)) / 10) \
          - ((delay(close, 10) - close) / 10)
    return pd.Series(
        np.where(accel < -0.1, 1.0, -1.0 * delta(close, 1)),
        index=close.index, dtype=float,
    )


def alpha050(close, volume, amount, rank_window=252, **_):
    """量VWAP相关排名的滚动峰值取反：量价共振极端时反转。
    -1 * ts_max(rank(corr(rank(vol),rank(vwap),5)), 5)"""
    vw = vwap(amount, volume)
    c = ts_corr(rank(volume, rank_window), rank(vw, rank_window), 5)
    return -1 * ts_max(rank(c, rank_window), 5)


def alpha051(close, **_):
    """价格加速度适度向上时做多，否则跟随日动量反向（阈值较046宽松）。
    accel < -0.05 ? 1 : -delta(close,1)"""
    accel = ((delay(close, 20) - delay(close, 10)) / 10) \
          - ((delay(close, 10) - close) / 10)
    return pd.Series(
        np.where(accel < -0.05, 1.0, -1.0 * delta(close, 1)),
        index=close.index, dtype=float,
    )


def alpha052(close, low, volume, rank_window=252, **_):
    """低价5日极值反弹叠加长短期收益差排名与量排名：超跌反弹信号。
    (-ts_min(low,5)+delay(ts_min(low,5),5)) * rank((sum(ret,240)-sum(ret,20))/220) * ts_rank(vol,5)"""
    r = close.pct_change()
    low_min = ts_min(low, 5)
    long_ret = (ts_sum(r, 240) - ts_sum(r, 20)) / 220
    return (-1 * low_min + delay(low_min, 5)) \
           * rank(long_ret, rank_window) \
           * ts_rank(volume, 5)


def alpha053(close, high, low, **_):
    """收盘位置动量变化：收盘越靠近高价且趋势加速时信号越强。
    -1 * delta(((close-low)-(high-close)) / (close-low), 9)"""
    x = ((close - low) - (high - close)) / (close - low).replace(0, np.nan)
    return -1 * delta(x, 9)


def alpha054(open_, close, high, low, **_):
    """低收差与开盘价高次幂的比值：捕捉收盘相对低价的强度异常。
    (-1*(low-close)*open^5) / ((low-high)*close^5)"""
    numer = -1 * (low - close) * (open_ ** 5)
    denom = (low - high).replace(0, np.nan) * (close ** 5).replace(0, np.nan)
    return numer / denom


def alpha055(close, high, low, volume, rank_window=252, **_):
    """12日价格区间内收盘位置排名与成交量排名的相关性取反：强势区间放量看空。
    -1 * corr(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))), rank(vol), 6)"""
    low12  = ts_min(low, 12)
    high12 = ts_max(high, 12)
    x = (close - low12) / (high12 - low12).replace(0, np.nan)
    return -1 * ts_corr(rank(x, rank_window), rank(volume, rank_window), 6)


def alpha057(close, volume, amount, rank_window=252, **_):
    """收盘偏离VWAP除以最高价时序位置的衰减：偏离越大且处于高位时信号越强。
    -(close-vwap) / decay_linear(rank(ts_argmax(close,30)), 2)"""
    vw = vwap(amount, volume)
    denom = decay_linear(rank(ts_argmax(close, 30), rank_window), 2).replace(0, np.nan)
    return -1 * (close - vw) / denom


def alpha060(close, high, low, volume, rank_window=252, **_):
    """收盘强度加权成交量的排名与价格新高排名的对比：量价共振背离信号。
    -(2*scale(rank(((close-low-(high-close))/(high-low))*vol)) - scale(rank(ts_argmax(close,10))))"""
    denom = (high - low).replace(0, np.nan)
    x = ((close - low) - (high - close)) / denom * volume
    return -1 * (2 * scale(rank(x, rank_window)) - scale(rank(ts_argmax(close, 10), rank_window)))


def alpha061(close, volume, amount, rank_window=252, **_):
    """VWAP突破近期低点的排名 vs 长期量相关排名：量能支撑突破时做多。
    (rank(vwap-ts_min(vwap,16)) < rank(corr(vwap,adv180,18))) ? 1 : 0"""
    vw = vwap(amount, volume)
    cond = rank(vw - ts_min(vw, 16), rank_window) < rank(ts_corr(vw, adv(volume, 180), 18), rank_window)
    return pd.Series(np.where(cond, 1.0, 0.0), index=close.index, dtype=float)


def alpha062(open_, high, low, close, volume, amount, rank_window=252, **_):
    """VWAP与量均相关排名 vs 开盘相对中间价强弱：量价背离时做空。
    (rank(corr(vwap,sum(adv20,22),10)) < rank((rank(open)*2) < (rank((high+low)/2)+rank(high)))) * -1"""
    vw = vwap(amount, volume)
    x = rank(ts_corr(vw, ts_sum(adv(volume, 20), 22), 10), rank_window)
    open_rank2 = rank(open_, rank_window) * 2
    mid_rank = rank((high + low) / 2, rank_window) + rank(high, rank_window)
    y = rank((open_rank2 < mid_rank).astype(float), rank_window)
    return (x < y).astype(float) * -1


def alpha064(open_, high, low, close, volume, amount, rank_window=252, **_):
    """加权开低价与长期量均相关排名 vs 加权均价变化排名：量价关系转折信号。
    (rank(corr(sum(0.178*open+0.822*low,13),sum(adv120,13),17)) < rank(delta(0.178*(high+low)/2+0.822*vwap,4))) * -1"""
    vw = vwap(amount, volume)
    ol = open_ * 0.178404 + low * (1 - 0.178404)
    hl_vw = (high + low) / 2 * 0.178404 + vw * (1 - 0.178404)
    x = rank(ts_corr(ts_sum(ol, 13), ts_sum(adv(volume, 120), 13), 17), rank_window)
    y = rank(delta(hl_vw, 4), rank_window)
    return (x < y).astype(float) * -1


def alpha065(open_, close, volume, amount, rank_window=252, **_):
    """加权开VWAP与中期量均相关排名 vs 开盘相对低点排名：量能不支撑开盘时做空。
    (rank(corr(0.008*open+0.992*vwap,sum(adv60,9),6)) < rank(open-ts_min(open,14))) * -1"""
    vw = vwap(amount, volume)
    wt = open_ * 0.00817522 + vw * (1 - 0.00817522)
    x = rank(ts_corr(wt, ts_sum(adv(volume, 60), 9), 6), rank_window)
    y = rank(open_ - ts_min(open_, 14), rank_window)
    return (x < y).astype(float) * -1


def alpha066(open_, high, low, close, volume, amount, rank_window=252, **_):
    """VWAP动量衰减排名与低价偏离比率时序排名之和取反。
    (rank(decay_linear(delta(vwap,4),7)) + ts_rank(decay_linear((low-vwap)/(open-(high+low)/2),11),7)) * -1"""
    vw = vwap(amount, volume)
    denom = (open_ - (high + low) / 2).replace(0, np.nan)
    part1 = rank(decay_linear(delta(vw, 4), 7), rank_window)
    part2 = ts_rank(decay_linear((low - vw) / denom, 11), 7)
    return (part1 + part2) * -1


def alpha068(close, high, low, volume, rank_window=252, **_):
    """高价与中期量均相关的时序排名 vs 加权收低价变化排名：量价共振末端反转。
    (ts_rank(corr(rank(high),rank(adv15),9),14) < rank(delta(0.518*close+0.482*low,1))) * -1"""
    hl = close * 0.518371 + low * (1 - 0.518371)
    x = ts_rank(ts_corr(rank(high, rank_window), rank(adv(volume, 15), rank_window), 9), 14)
    y = rank(delta(hl, 1), rank_window)
    return (x < y).astype(float) * -1


def alpha071(open_, close, low, volume, amount, rank_window=252, **_):
    """收盘与长期量均相关衰减时序排名 vs 开低相对双倍VWAP偏离平方衰减时序排名，取大值。
    max(ts_rank(decay_linear(corr(ts_rank(close,3),ts_rank(adv180,12),18),4),16),
        ts_rank(decay_linear(rank((low+open-2*vwap)^2),16),4))"""
    vw = vwap(amount, volume)
    adv180_ = adv(volume, 180)
    part1 = ts_rank(decay_linear(ts_corr(ts_rank(close, 3), ts_rank(adv180_, 12), 18), 4), 16)
    part2 = ts_rank(decay_linear(rank((low + open_ - 2 * vw) ** 2, rank_window), 16), 4)
    return pd.concat([part1, part2], axis=1).max(axis=1)


def alpha072(close, high, low, volume, amount, rank_window=252, **_):
    """中间价与量均相关衰减排名 / VWAP与成交量时序排名相关衰减排名：两类量价比率。
    rank(decay_linear(corr((high+low)/2,adv40,9),10)) / rank(decay_linear(corr(ts_rank(vwap,4),ts_rank(vol,19),7),3))"""
    vw = vwap(amount, volume)
    numer = rank(decay_linear(ts_corr((high + low) / 2, adv(volume, 40), 9), 10), rank_window)
    denom = rank(decay_linear(ts_corr(ts_rank(vw, 4), ts_rank(volume, 19), 7), 3), rank_window)
    return numer / denom.replace(0, np.nan)


def alpha073(open_, close, low, volume, amount, rank_window=252, **_):
    """VWAP变化衰减排名与加权开低价收益率衰减时序排名，取大值后取反。
    max(rank(decay_linear(delta(vwap,5),3)), ts_rank(decay_linear(-delta(0.147*open+0.853*low,2)/(0.147*open+0.853*low),3),17)) * -1"""
    vw = vwap(amount, volume)
    ol = open_ * 0.147155 + low * (1 - 0.147155)
    part1 = rank(decay_linear(delta(vw, 5), 3), rank_window)
    part2 = ts_rank(decay_linear(-1 * delta(ol, 2) / ol.replace(0, np.nan), 3), 17)
    return pd.concat([part1, part2], axis=1).max(axis=1) * -1


def alpha074(close, high, volume, amount, rank_window=252, **_):
    """收盘与长期量均相关排名 vs 加权高VWAP的量排名相关排名：量价关系转折信号。
    (rank(corr(close,sum(adv30,37),15)) < rank(corr(rank(0.026*high+0.974*vwap),rank(vol),11))) * -1"""
    vw = vwap(amount, volume)
    x = rank(ts_corr(close, ts_sum(adv(volume, 30), 37), 15), rank_window)
    y = rank(ts_corr(rank(high * 0.0261661 + vw * (1 - 0.0261661), rank_window),
                     rank(volume, rank_window), 11), rank_window)
    return (x < y).astype(float) * -1


def alpha075(close, low, volume, amount, rank_window=252, **_):
    """VWAP与成交量短期相关排名 vs 低价与长期量均相关排名：量价结构差异信号。
    rank(corr(vwap,vol,4)) < rank(corr(rank(low),rank(adv50),12))"""
    vw = vwap(amount, volume)
    x = rank(ts_corr(vw, volume, 4), rank_window)
    y = rank(ts_corr(rank(low, rank_window), rank(adv(volume, 50), rank_window), 12), rank_window)
    return (x < y).astype(float)


def alpha077(close, high, low, volume, amount, rank_window=252, **_):
    """中间价偏离VWAP的衰减排名与量均相关衰减排名，取较小值：均值回归双重信号的弱者。
    min(rank(decay_linear((high+low)/2-vwap,20)), rank(decay_linear(corr((high+low)/2,adv40,3),6)))"""
    vw = vwap(amount, volume)
    part1 = rank(decay_linear((high + low) / 2 - vw, 20), rank_window)
    part2 = rank(decay_linear(ts_corr((high + low) / 2, adv(volume, 40), 3), 6), rank_window)
    return pd.concat([part1, part2], axis=1).min(axis=1)


def alpha078(close, low, volume, amount, rank_window=252, **_):
    """加权低价VWAP均值与量均相关排名的幂次：两类量价关系的乘法叠加。
    rank(corr(sum(0.352*low+0.648*vwap,20),sum(adv40,20),7)) ^ rank(corr(rank(vwap),rank(vol),6))"""
    vw = vwap(amount, volume)
    lv = low * 0.352233 + vw * (1 - 0.352233)
    adv40_ = adv(volume, 40)
    x = rank(ts_corr(ts_sum(lv, 20), ts_sum(adv40_, 20), 7), rank_window)
    y = rank(ts_corr(rank(vw, rank_window), rank(volume, rank_window), 6), rank_window)
    return x ** y


def alpha081(close, volume, amount, rank_window=252, **_):
    """VWAP与量均相关的高次幂排名滚动乘积对数排名 vs VWAP量相关排名：非线性量价共振信号。
    (rank(log(product(rank((rank(corr(vwap,sum(adv10,50),8))^4)),15))) < rank(corr(rank(vwap),rank(vol),5))) * -1"""
    vw = vwap(amount, volume)
    adv10_ = adv(volume, 10)
    inner_rank = rank(rank(ts_corr(vw, ts_sum(adv10_, 50), 8), rank_window) ** 4, rank_window)
    prod15 = inner_rank.rolling(15, min_periods=5).apply(np.prod, raw=True)
    x = rank(np.log(prod15.clip(lower=1e-8)), rank_window)
    y = rank(ts_corr(rank(vw, rank_window), rank(volume, rank_window), 5), rank_window)
    return (x < y).astype(float) * -1


def alpha083(close, high, low, volume, amount, rank_window=252, **_):
    """高低价比均值的滞后排名与成交量二次排名的乘积，除以高低比与VWAP偏离之比。
    rank(delay(hl_ratio,2))*rank(rank(vol)) / (hl_ratio/(vwap-close))"""
    vw = vwap(amount, volume)
    hl_ratio = (high - low) / ts_mean(close, 5).replace(0, np.nan)
    numer = rank(delay(hl_ratio, 2), rank_window) * rank(rank(volume, rank_window), rank_window)
    denom = hl_ratio / (vw - close).replace(0, np.nan)
    return numer / denom.replace(0, np.nan)


def alpha084(close, volume, amount, **_):
    """VWAP偏离近期高点的时序排名，以价格5日变化为指数：趋势强度的幂次放大。
    SignedPower(ts_rank(vwap-ts_max(vwap,15), 21), delta(close,5))"""
    vw = vwap(amount, volume)
    base = ts_rank(vw - ts_max(vw, 15), 21)
    exp = delta(close, 5)
    return np.sign(base) * (base.abs() ** exp)


def alpha085(close, high, low, volume, rank_window=252, **_):
    """加权高收价与量均相关排名的幂次：两类量价共振程度的幂次叠加。
    rank(corr(0.877*high+0.123*close,adv30,10)) ^ rank(corr(ts_rank((high+low)/2,4),ts_rank(vol,10),7))"""
    adv30_ = adv(volume, 30)
    hc = high * 0.876703 + close * (1 - 0.876703)
    x = rank(ts_corr(hc, adv30_, 10), rank_window)
    y = rank(ts_corr(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7), rank_window)
    return x ** y


def alpha086(open_, close, volume, amount, rank_window=252, **_):
    """收盘与量均相关时序排名 vs 收盘偏离VWAP排名：量价背离时做空。
    (ts_rank(corr(close,sum(adv20,15),6),20) < rank(close-vwap)) * -1"""
    vw = vwap(amount, volume)
    x = ts_rank(ts_corr(close, ts_sum(adv(volume, 20), 15), 6), 20)
    y = rank(close - vw, rank_window)
    return (x < y).astype(float) * -1


def alpha088(open_, close, high, low, volume, rank_window=252, **_):
    """OHLC四价排名差的衰减排名与收盘量均相关衰减时序排名，取较小值。
    min(rank(decay_linear(rank(open)+rank(low)-rank(high)-rank(close),8)),
        ts_rank(decay_linear(corr(ts_rank(close,8),ts_rank(adv60,21),8),7),3))"""
    adv60_ = adv(volume, 60)
    rl = (rank(open_, rank_window) + rank(low, rank_window)
          - rank(high, rank_window) - rank(close, rank_window))
    part1 = rank(decay_linear(rl, 8), rank_window)
    part2 = ts_rank(decay_linear(ts_corr(ts_rank(close, 8), ts_rank(adv60_, 21), 8), 7), 3)
    return pd.concat([part1, part2], axis=1).min(axis=1)


def alpha092(open_, close, high, low, volume, rank_window=252, **_):
    """价格结构与低价量均相关的双重衰减时序排名，取较小值。
    min(ts_rank(decay_linear((hl_mid+close)<(low+open),15),19),
        ts_rank(decay_linear(corr(rank(low),rank(adv30),8),7),7))"""
    adv30_ = adv(volume, 30)
    cond = (((high + low) / 2 + close) < (low + open_)).astype(float)
    part1 = ts_rank(decay_linear(cond, 15), 19)
    part2 = ts_rank(decay_linear(ts_corr(rank(low, rank_window), rank(adv30_, rank_window), 8), 7), 7)
    return pd.concat([part1, part2], axis=1).min(axis=1)


def alpha094(close, volume, amount, rank_window=252, **_):
    """VWAP突破近期低点排名的幂次，以量均时序排名相关为指数：量价动量的幂次放大。
    (rank(vwap-ts_min(vwap,12)) ^ ts_rank(corr(ts_rank(vwap,20),ts_rank(adv60,4),18),3)) * -1"""
    vw = vwap(amount, volume)
    adv60_ = adv(volume, 60)
    base = rank(vw - ts_min(vw, 12), rank_window)
    exp = ts_rank(ts_corr(ts_rank(vw, 20), ts_rank(adv60_, 4), 18), 3)
    return (base ** exp) * -1


def alpha095(open_, high, low, volume, rank_window=252, **_):
    """开盘突破近期低点排名 vs 中间价与量均相关排名高次幂时序排名：量价共振强度信号。
    rank(open-ts_min(open,12)) < ts_rank(rank(corr(sum((high+low)/2,19),sum(adv40,19),13))^5,12)"""
    adv40_ = adv(volume, 40)
    x = rank(open_ - ts_min(open_, 12), rank_window)
    corr_rank = rank(ts_corr(ts_sum((high + low) / 2, 19), ts_sum(adv40_, 19), 13), rank_window)
    y = ts_rank(corr_rank ** 5, 12)
    return (x < y).astype(float)


def alpha096(close, volume, amount, rank_window=252, **_):
    """VWAP量相关衰减时序排名与收盘量均相关极值位置衰减时序排名，取大值后取反。
    max(ts_rank(decay_linear(corr(rank(vwap),rank(vol),4),4),8),
        ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7),ts_rank(adv60,4),4),13),14),13)) * -1"""
    vw = vwap(amount, volume)
    adv60_ = adv(volume, 60)
    part1 = ts_rank(decay_linear(ts_corr(rank(vw, rank_window), rank(volume, rank_window), 4), 4), 8)
    part2 = ts_rank(decay_linear(ts_argmax(ts_corr(ts_rank(close, 7), ts_rank(adv60_, 4), 4), 13), 14), 13)
    return pd.concat([part1, part2], axis=1).max(axis=1) * -1


def alpha098(open_, close, volume, amount, rank_window=252, **_):
    """VWAP与短期量均相关衰减排名 减 开盘量均相关极小值位置的衰减时序排名。
    rank(decay_linear(corr(vwap,sum(adv5,26),5),7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open),rank(adv15),21),9),7),2))"""
    vw = vwap(amount, volume)
    adv5_  = adv(volume, 5)
    adv15_ = adv(volume, 15)
    part1 = rank(decay_linear(ts_corr(vw, ts_sum(adv5_, 26), 5), 7), rank_window)
    part2 = rank(decay_linear(
        ts_rank(ts_argmin(ts_corr(rank(open_, rank_window), rank(adv15_, rank_window), 21), 9), 7),
        2), rank_window)
    return part1 - part2


def alpha099(close, high, low, volume, rank_window=252, **_):
    """中间价与量均相关排名 vs 低价量相关排名：量价关系强弱比较。
    (rank(corr(sum((high+low)/2,20),sum(adv60,20),9)) < rank(corr(low,vol,6))) * -1"""
    adv60_ = adv(volume, 60)
    x = rank(ts_corr(ts_sum((high + low) / 2, 20), ts_sum(adv60_, 20), 9), rank_window)
    y = rank(ts_corr(low, volume, 6), rank_window)
    return (x < y).astype(float) * -1


def alpha101(open_, close, high, low, **_):
    """日内收盘偏离开盘相对价格区间的比值：量化当日收盘强弱位置。
    (close - open) / (high - low + 0.001)"""
    return (close - open_) / (high - low + 0.001)
