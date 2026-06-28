"""기술적 지표 및 특성 엔지니어링 모듈."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표를 추가합니다."""
    data = df.copy()

    data["return_1d"] = data["close"].pct_change()
    data["return_5d"] = data["close"].pct_change(5)
    data["return_10d"] = data["close"].pct_change(10)

    data["sma_5"] = data["close"].rolling(5).mean()
    data["sma_10"] = data["close"].rolling(10).mean()
    data["sma_20"] = data["close"].rolling(20).mean()
    data["sma_50"] = data["close"].rolling(50).mean()

    data["ema_12"] = data["close"].ewm(span=12, adjust=False).mean()
    data["ema_26"] = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    delta = data["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))

    data["bb_mid"] = data["close"].rolling(20).mean()
    bb_std = data["close"].rolling(20).std()
    data["bb_upper"] = data["bb_mid"] + 2 * bb_std
    data["bb_lower"] = data["bb_mid"] - 2 * bb_std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_mid"]

    data["volume_sma"] = data["volume"].rolling(20).mean()
    data["volume_ratio"] = data["volume"] / data["volume_sma"]

    data["high_low_ratio"] = (data["high"] - data["low"]) / data["close"]
    data["close_sma20_ratio"] = data["close"] / data["sma_20"]

    return data


def prepare_features(
    df: pd.DataFrame,
    forecast_days: int = 5,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    학습용 특성과 타겟을 준비합니다.

    Args:
        df: OHLCV 데이터프레임
        forecast_days: 예측할 미래 일수

    Returns:
        (특성 데이터프레임, 타겟 시리즈, 특성 컬럼 목록)
    """
    data = add_technical_indicators(df)

    data["target"] = data["close"].shift(-forecast_days)
    data["target_return"] = data["target"] / data["close"] - 1

    feature_cols = [
        "return_1d",
        "return_5d",
        "return_10d",
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "macd",
        "macd_signal",
        "rsi",
        "bb_width",
        "volume_ratio",
        "high_low_ratio",
        "close_sma20_ratio",
    ]

    data = data.dropna(subset=feature_cols + ["target", "target_return"])
    X = data[feature_cols]
    y = data["target"]
    return X, y, feature_cols


def get_latest_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """최신 데이터의 특성을 반환합니다."""
    data = add_technical_indicators(df)
    latest = data[feature_cols].iloc[[-1]].copy()
    if latest.isna().any().any():
        raise ValueError("최신 데이터에 충분한 기술적 지표가 없습니다. 더 긴 기간의 데이터가 필요합니다.")
    return latest