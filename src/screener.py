"""한국 주식 매수 스크리너."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .data_fetcher import fetch_multiple_stock_data
from .features import add_technical_indicators
from .korean_universe import KoreanStock, get_stock_map, get_universe
from .model import quick_predict


@dataclass
class StockScore:
    """종목 분석 결과."""

    ticker: str
    name: str
    sector: str
    market: str
    current_price: float
    predicted_return: float
    score: float
    recommendation: str
    rsi: float
    return_1d: float
    return_5d: float
    volume_ratio: float
    signals: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class ScreeningResult:
    """스크리닝 전체 결과."""

    date: str
    universe_label: str
    picks: list[StockScore]
    all_picks: list[StockScore]
    analyzed_count: int
    failed_tickers: list[str]


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _recommendation(score: float) -> str:
    if score >= 75:
        return "강력 매수"
    if score >= 60:
        return "매수"
    if score >= 45:
        return "관망"
    return "회피"


def _score_ml_prediction(predicted_return: float) -> tuple[float, list[str]]:
    """ML 예측 점수 (최대 30점)."""
    signals = []
    pct = predicted_return * 100

    if pct >= 3:
        score = 30
        signals.append(f"ML 5일 예측 +{pct:.1f}% (강한 상승)")
    elif pct >= 1.5:
        score = 24
        signals.append(f"ML 5일 예측 +{pct:.1f}% (상승)")
    elif pct >= 0.5:
        score = 18
        signals.append(f"ML 5일 예측 +{pct:.1f}% (완만한 상승)")
    elif pct >= 0:
        score = 12
        signals.append(f"ML 5일 예측 {pct:+.1f}% (보합)")
    elif pct >= -1.5:
        score = 6
        signals.append(f"ML 5일 예측 {pct:+.1f}% (약한 하락)")
    else:
        score = 0
        signals.append(f"ML 5일 예측 {pct:+.1f}% (하락)")

    return score, signals


def _score_trend(row: pd.Series) -> tuple[float, list[str]]:
    """추세 점수 (최대 25점)."""
    signals = []
    score = 0.0
    close = row["close"]
    sma5, sma20, sma50 = row["sma_5"], row["sma_20"], row["sma_50"]

    if close > sma5 > sma20 > sma50:
        score += 15
        signals.append("정배열 (SMA5 > 20 > 50)")
    elif close > sma20 > sma50:
        score += 10
        signals.append("중기 상승 추세")
    elif close > sma20:
        score += 6
        signals.append("20일선 위")
    else:
        signals.append("20일선 아래 (약세)")

    if close > sma50:
        score += 5
        signals.append("50일선 위")
    if row["close_sma20_ratio"] > 1.02:
        score += 5
        signals.append("20일선 대비 강세")

    return _clamp(score, 0, 25), signals


def _score_momentum(row: pd.Series) -> tuple[float, list[str]]:
    """모멘텀 점수 (최대 20점)."""
    signals = []
    score = 0.0
    rsi = row["rsi"]

    if 45 <= rsi <= 65:
        score += 10
        signals.append(f"RSI {rsi:.0f} (적정 구간)")
    elif 35 <= rsi < 45:
        score += 8
        signals.append(f"RSI {rsi:.0f} (반등 가능)")
    elif 65 < rsi <= 70:
        score += 5
        signals.append(f"RSI {rsi:.0f} (과열 주의)")
    elif rsi < 35:
        score += 4
        signals.append(f"RSI {rsi:.0f} (과매도)")
    else:
        signals.append(f"RSI {rsi:.0f} (과매수)")

    if row["macd"] > row["macd_signal"]:
        score += 7
        signals.append("MACD 골든크로스")
    elif row["macd"] > 0:
        score += 4
        signals.append("MACD 양수")

    if row["return_5d"] > 0:
        score += 3
        signals.append(f"5일 수익률 {row['return_5d']*100:+.1f}%")

    return _clamp(score, 0, 20), signals


def _score_volume(row: pd.Series) -> tuple[float, list[str]]:
    """거래량 점수 (최대 15점)."""
    signals = []
    ratio = row["volume_ratio"]

    if ratio >= 2.0:
        score = 15
        signals.append(f"거래량 {ratio:.1f}배 급증")
    elif ratio >= 1.5:
        score = 12
        signals.append(f"거래량 {ratio:.1f}배 증가")
    elif ratio >= 1.2:
        score = 8
        signals.append(f"거래량 {ratio:.1f}배 (활발)")
    elif ratio >= 0.8:
        score = 5
        signals.append("거래량 보통")
    else:
        score = 2
        signals.append("거래량 저조")

    return score, signals


def _score_short_term(row: pd.Series) -> tuple[float, list[str]]:
    """단기 강도 점수 (최대 10점)."""
    signals = []
    score = 0.0
    r1 = row["return_1d"] * 100

    if 0 < r1 <= 3:
        score += 6
        signals.append(f"전일 +{r1:.1f}%")
    elif -1 <= r1 <= 0:
        score += 4
        signals.append(f"전일 {r1:+.1f}% (눌림목)")
    elif r1 > 3:
        score += 3
        signals.append(f"전일 +{r1:.1f}% (급등, 추격 주의)")
    else:
        signals.append(f"전일 {r1:+.1f}%")

    if row["return_10d"] > 0:
        score += 4

    return _clamp(score, 0, 10), signals


def analyze_stock(
    df: pd.DataFrame,
    stock: KoreanStock,
    forecast_days: int = 5,
) -> StockScore:
    """단일 종목을 분석하고 점수를 산출합니다."""
    data = add_technical_indicators(df)
    row = data.iloc[-1]

    _, _, predicted_return = quick_predict(df, forecast_days=forecast_days)

    breakdown: dict[str, float] = {}
    all_signals: list[str] = []

    ml_score, ml_signals = _score_ml_prediction(predicted_return)
    breakdown["ml"] = ml_score
    all_signals.extend(ml_signals)

    trend_score, trend_signals = _score_trend(row)
    breakdown["trend"] = trend_score
    all_signals.extend(trend_signals)

    mom_score, mom_signals = _score_momentum(row)
    breakdown["momentum"] = mom_score
    all_signals.extend(mom_signals)

    vol_score, vol_signals = _score_volume(row)
    breakdown["volume"] = vol_score
    all_signals.extend(vol_signals)

    st_score, st_signals = _score_short_term(row)
    breakdown["short_term"] = st_score
    all_signals.extend(st_signals)

    total = sum(breakdown.values())

    return StockScore(
        ticker=stock.ticker,
        name=stock.name,
        sector=stock.sector,
        market=stock.market,
        current_price=float(row["close"]),
        predicted_return=predicted_return,
        score=round(total, 1),
        recommendation=_recommendation(total),
        rsi=float(row["rsi"]),
        return_1d=float(row["return_1d"]),
        return_5d=float(row["return_5d"]),
        volume_ratio=float(row["volume_ratio"]),
        signals=all_signals,
        score_breakdown=breakdown,
    )


def screen_korean_stocks(
    universe: str = "kospi_large",
    period: str = "1y",
    forecast_days: int = 5,
    top_n: int | None = None,
) -> ScreeningResult:
    """한국 주식 유니버스를 스크리닝합니다."""
    label, stocks = get_universe(universe)
    stock_map = get_stock_map(universe)
    tickers = [s.ticker for s in stocks]

    data_map = fetch_multiple_stock_data(tickers, period=period)
    picks: list[StockScore] = []
    failed: list[str] = []

    for ticker in tickers:
        stock = stock_map[ticker]
        df = data_map.get(ticker)
        if df is None or len(df) < 60:
            failed.append(ticker)
            continue
        try:
            picks.append(analyze_stock(df, stock, forecast_days=forecast_days))
        except Exception:
            failed.append(ticker)

    picks.sort(key=lambda p: p.score, reverse=True)
    total_analyzed = len(picks)
    display_picks = picks[:top_n] if top_n else picks

    return ScreeningResult(
        date=datetime.now().strftime("%Y-%m-%d"),
        universe_label=label,
        picks=display_picks,
        all_picks=picks,
        analyzed_count=total_analyzed,
        failed_tickers=failed,
    )


def picks_to_dataframe(picks: list[StockScore]) -> pd.DataFrame:
    """분석 결과를 데이터프레임으로 변환합니다."""
    rows = []
    for i, p in enumerate(picks, 1):
        rows.append(
            {
                "순위": i,
                "종목": p.name,
                "티커": p.ticker,
                "섹터": p.sector,
                "현재가": p.current_price,
                "점수": p.score,
                "추천": p.recommendation,
                "ML예측(5일)": f"{p.predicted_return * 100:+.1f}%",
                "RSI": round(p.rsi, 1),
                "1일수익": f"{p.return_1d * 100:+.1f}%",
                "5일수익": f"{p.return_5d * 100:+.1f}%",
                "거래량비율": round(p.volume_ratio, 2),
            }
        )
    return pd.DataFrame(rows)