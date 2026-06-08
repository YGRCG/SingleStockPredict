import pytest
import pandas as pd
import numpy as np
from src.labels.builder import build_labels, drop_label_na


def make_dummy_df(n=100) -> pd.DataFrame:
    np.random.seed(0)
    close = 10 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({"close": close, "feat_a": np.random.randn(n)},
                        index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_binary_label_no_future_data():
    df = make_dummy_df()
    out = build_labels(df, horizon=3, label_type="binary")
    # 最后 3 行应为 NaN
    assert out["label"].iloc[-1] is np.nan or pd.isna(out["label"].iloc[-1])
    assert out["label"].iloc[-3] is np.nan or pd.isna(out["label"].iloc[-3])


def test_binary_label_values():
    df = make_dummy_df()
    out = drop_label_na(build_labels(df, horizon=3, label_type="binary"))
    assert set(out["label"].unique()).issubset({0.0, 1.0})


def test_ternary_label_values():
    df = make_dummy_df()
    out = drop_label_na(build_labels(df, horizon=3, label_type="ternary", threshold=0.01))
    assert set(out["label"].unique()).issubset({0.0, 1.0, 2.0})


def test_return_label():
    df = make_dummy_df()
    out = build_labels(df, horizon=1, label_type="return")
    # 第 0 行的 label 应等于 close[1]/close[0] - 1
    expected = df["close"].iloc[1] / df["close"].iloc[0] - 1
    assert out["label"].iloc[0] == pytest.approx(expected, rel=1e-6)
