"""시장 국면(레짐) 판별 — 상승장/하락장 이중 전략."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from src.backtest.position_plan import (
    BULL_PLAN,
    CASH_PLAN,
    CapitalAllocation,
    REGIME_PLANS,
    apply_drawdown_scaling,
)
from src.features import add_technical_indicators


class MarketRegime(str, Enum):
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    CRISIS = "crisis"


def build_index_features(index_close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"close": index_close, "open": index_close, "high": index_close,
                       "low": index_close, "volume": 1.0})
    return add_technical_indicators(df)


def detect_market_regime(index_close: pd.Series, as_of: pd.Timestamp) -> MarketRegime:
    """KOSPI 지수만으로 상승/하락/횡보 판별 (포트폴리오 DD 제외)."""
    hist = index_close[index_close.index <= as_of]
    if len(hist) < 60:
        return MarketRegime.NEUTRAL

    row = build_index_features(hist).iloc[-1]
    close = float(row["close"])
    sma20 = float(row["sma_20"])
    sma50 = float(row["sma_50"])
    sma200 = float(row["sma_50"])  # fallback if no 200
    if len(hist) >= 200:
        sma200 = float(hist.tail(200).mean())

    ret60 = float(row.get("return_10d", 0) or 0)
    rsi = float(row.get("rsi", 50) or 50)

    bull_signals = sum([
        close > sma50,
        sma20 > sma50,
        close > sma200 if len(hist) >= 200 else close > sma50,
        ret60 > 0,
        rsi >= 48,
    ])
    bear_signals = sum([
        close < sma50,
        sma20 < sma50,
        ret60 < 0,
        rsi < 45,
    ])

    if bull_signals >= 4:
        return MarketRegime.BULL
    if bear_signals >= 3:
        return MarketRegime.BEAR
    return MarketRegime.NEUTRAL


def get_allocation_plan(
    index_close: pd.Series,
    as_of: pd.Timestamp,
    portfolio_drawdown: float = 0.0,
    max_dd_limit: float = 0.15,
) -> CapitalAllocation:
    """시장 국면 + DD 스케일링을 반영한 자본 배분 계획."""
    if portfolio_drawdown <= -max_dd_limit + 0.03:
        return CASH_PLAN

    regime = detect_market_regime(index_close, as_of)
    if portfolio_drawdown <= -(max_dd_limit - 0.03):
        return CASH_PLAN

    base = REGIME_PLANS[regime.value]
    return apply_drawdown_scaling(base, portfolio_drawdown)