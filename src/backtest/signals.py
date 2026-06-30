"""과거 시점(point-in-time) 매매 신호 — 국면별 전략."""

from __future__ import annotations

import pandas as pd

from src.features import add_technical_indicators
from src.invest_analysis import _score_investor_flow
from src.screener import (
    _score_momentum,
    _score_short_term,
    _score_trend,
    _score_volume,
)


def _score_relative_strength(df: pd.DataFrame) -> float:
    """60일 상대 모멘텀 — EW 벤치마크 대비 초과수익 후보 가점."""
    if len(df) < 60:
        return 0.0
    ret60 = float(df["close"].iloc[-1] / df["close"].iloc[-60] - 1)
    bonus = 0.0
    if ret60 > 0.15:
        bonus += 15
    elif ret60 > 0.08:
        bonus += 10
    elif ret60 > 0.03:
        bonus += 5
    elif ret60 < -0.10:
        bonus -= 10
    return bonus


def _score_momentum_mode(row: pd.Series, df: pd.DataFrame, use_flow: bool) -> float:
    total = 0.0
    total += _score_trend(row)[0]
    total += _score_momentum(row)[0]
    total += _score_volume(row)[0]
    total += _score_short_term(row)[0]
    total += _score_relative_strength(df)
    if use_flow:
        total += _score_investor_flow(df)[0]
    return total


def _score_selective_mode(row: pd.Series, df: pd.DataFrame, use_flow: bool) -> float:
    """횡보장: 추세+수급 모두 양호할 때만 고득점."""
    total = _score_momentum_mode(row, df, use_flow)
    if row["close"] <= row["sma_20"]:
        total -= 15
    if use_flow and "foreigner_net" in df.columns:
        f5 = df["foreigner_net"].tail(5).sum()
        if f5 <= 0:
            total -= 10
    if row["rsi"] > 68:
        total -= 10
    return max(0, total)


def _score_defensive_mode(row: pd.Series, df: pd.DataFrame, use_flow: bool) -> float:
    """하락장: 과매도 반등 + 외국인 순매수."""
    total = 0.0
    rsi = float(row["rsi"])
    if 28 <= rsi <= 42:
        total += 25
    elif rsi < 28:
        total += 15
    else:
        total -= 10

    if use_flow and "foreigner_net" in df.columns:
        f3 = df["foreigner_net"].tail(3).sum()
        f10 = df["foreigner_net"].tail(10).sum()
        if f3 > 0:
            total += 20
        if f10 > 0:
            total += 10

    if row["close"] > row["sma_5"]:
        total += 15
    if row.get("return_1d", 0) > 0:
        total += 10

    if row["volume_ratio"] > 1.3:
        total += 10

    return max(0, min(total, 90))


def score_at_date(
    df: pd.DataFrame,
    use_flow: bool = True,
    mode: str = "momentum",
) -> float | None:
    """주어진 과거 데이터만으로 당시 매수 점수 (lookahead 없음)."""
    if len(df) < 60:
        return None

    data = add_technical_indicators(df)
    row = data.iloc[-1]
    if row[["sma_50", "rsi", "macd"]].isna().any():
        return None

    if mode == "cash":
        return 0.0
    if mode == "defensive":
        return round(_score_defensive_mode(row, data, use_flow), 1)
    if mode == "selective":
        return round(_score_selective_mode(row, data, use_flow), 1)
    raw = _score_momentum_mode(row, data, use_flow)
    return round(min(raw, 100.0), 1)


def rank_universe(
    data_map: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    use_flow: bool = True,
    min_history: int = 60,
    mode: str = "momentum",
    min_score: float = 0,
    market_caps: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """특정 일자 기준 유니버스 종목 점수 순위."""
    scores: list[tuple[str, float]] = []
    for code, df in data_map.items():
        hist = df[df.index <= as_of]
        if len(hist) < min_history:
            continue
        s = score_at_date(hist, use_flow=use_flow, mode=mode)
        if s is None:
            continue
        if market_caps and mode == "momentum":
            cap = market_caps.get(code, 0)
            s += min(15, cap / 1e13 * 10) if cap else 0
        if s >= min_score or mode == "momentum":
            scores.append((code, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores