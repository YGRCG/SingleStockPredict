import pytest
import pandas as pd
import numpy as np
from src.features.technical import build_technical_features
from src.features.price_pattern import build_pattern_features


def make_dummy_ohlcv(n=200) -> pd.DataFrame:
    np.random.seed(42)
    close = 10 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame({
        "open":   close * (1 + np.random.randn(n) * 0.005),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.01),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.01),
        "close":  close,
        "volume": np.abs(np.random.randn(n)) * 1e6 + 1e6,
        "amount": np.abs(np.random.randn(n)) * 1e7 + 1e7,
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    return df


def test_technical_features_no_future_leak():
    df = make_dummy_ohlcv()
    feat = build_technical_features(df)
    # 特征不含未来信息：第 i 行的 ma_5 只依赖 i 及之前
    assert "ma_5" in feat.columns
    assert feat["ma_5"].iloc[4] == pytest.approx(df["close"].iloc[:5].mean(), rel=1e-5)


def test_no_negative_atr():
    df = make_dummy_ohlcv()
    feat = build_technical_features(df)
    assert (feat["atr_14"].dropna() >= 0).all()


def test_pattern_features():
    df = make_dummy_ohlcv()
    feat = build_pattern_features(df)
    assert "is_bullish" in feat.columns
    assert feat["body_ratio"].between(0, 1).all()
