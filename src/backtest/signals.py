"""과거 시점(point-in-time) 매매 신호 점수."""

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


def score_at_date(df: pd.DataFrame, use_flow: bool = True) -> float | None:
    """
    주어진 과거 데이터만으로 당시 매수 점수를 계산합니다.
    미래 데이터 누수(lookahead) 없음.
    """
    if len(df) < 60:
        return None

    data = add_technical_indicators(df)
    row = data.iloc[-1]
    if row[["sma_50", "rsi", "macd"]].isna().any():
        return None

    total = 0.0
    total += _score_trend(row)[0]
    total += _score_momentum(row)[0]
    total += _score_volume(row)[0]
    total += _score_short_term(row)[0]

    if use_flow and "foreigner_net" in df.columns:
        total += _score_investor_flow(df)[0]

    return round(total, 1)


def rank_universe(
    data_map: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    use_flow: bool = True,
    min_history: int = 60,
) -> list[tuple[str, float]]:
    """특정 일자 기준 유니버스 종목 점수 순위."""
    scores: list[tuple[str, float]] = []
    for code, df in data_map.items():
        hist = df[df.index <= as_of]
        if len(hist) < min_history:
            continue
        s = score_at_date(hist, use_flow=use_flow)
        if s is not None:
            scores.append((code, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores