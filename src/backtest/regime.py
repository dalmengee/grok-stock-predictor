"""시장 국면(레짐) 판별."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.features import add_technical_indicators


class MarketRegime(str, Enum):
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    CRISIS = "crisis"


@dataclass
class RegimeState:
    regime: MarketRegime
    exposure_cap: float
    entry_score: float
    max_positions: int
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    strategy_mode: str
    label: str


REGIME_PROFILES: dict[MarketRegime, RegimeState] = {
    MarketRegime.BULL: RegimeState(
        regime=MarketRegime.BULL,
        exposure_cap=0.90,
        entry_score=55.0,
        max_positions=5,
        stop_loss_pct=0.09,
        take_profit_pct=0.20,
        trailing_stop_pct=0.07,
        strategy_mode="momentum",
        label="상승장 — 모멘텀·수급 적극",
    ),
    MarketRegime.NEUTRAL: RegimeState(
        regime=MarketRegime.NEUTRAL,
        exposure_cap=0.55,
        entry_score=62.0,
        max_positions=3,
        stop_loss_pct=0.06,
        take_profit_pct=0.12,
        trailing_stop_pct=0.05,
        strategy_mode="selective",
        label="횡보장 — 선별 매수",
    ),
    MarketRegime.BEAR: RegimeState(
        regime=MarketRegime.BEAR,
        exposure_cap=0.25,
        entry_score=70.0,
        max_positions=2,
        stop_loss_pct=0.05,
        take_profit_pct=0.08,
        trailing_stop_pct=0.04,
        strategy_mode="defensive",
        label="하락장 — 방어·현금 비중 확대",
    ),
    MarketRegime.CRISIS: RegimeState(
        regime=MarketRegime.CRISIS,
        exposure_cap=0.05,
        entry_score=99.0,
        max_positions=0,
        stop_loss_pct=0.04,
        take_profit_pct=0.06,
        trailing_stop_pct=0.03,
        strategy_mode="cash",
        label="위기 — 포트폴리오 DD 차단, 현금 대기",
    ),
}


def build_index_features(index_close: pd.Series) -> pd.DataFrame:
    """KOSPI 지수 기술적 지표."""
    df = pd.DataFrame({"close": index_close})
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = 1.0
    return add_technical_indicators(df)


def detect_regime(
    index_close: pd.Series,
    as_of: pd.Timestamp,
    portfolio_drawdown: float = 0.0,
    crisis_dd_threshold: float = 0.12,
) -> RegimeState:
    """
    KOSPI 지수 + 포트폴리오 DD로 시장 국면을 판별합니다.
    portfolio_drawdown: 0 ~ -1 (e.g. -0.15 = -15%)
    """
    if portfolio_drawdown <= -crisis_dd_threshold:
        return REGIME_PROFILES[MarketRegime.CRISIS]

    hist = index_close[index_close.index <= as_of]
    if len(hist) < 50:
        return REGIME_PROFILES[MarketRegime.NEUTRAL]

    feat = build_index_features(hist)
    row = feat.iloc[-1]
    close = float(row["close"])
    sma20 = float(row["sma_20"])
    sma50 = float(row["sma_50"])
    ret20 = float(row.get("return_10d", 0) or 0)
    rsi = float(row.get("rsi", 50) or 50)

    if close > sma50 and sma20 > sma50 and ret20 > 0 and rsi >= 45:
        return REGIME_PROFILES[MarketRegime.BULL]
    if close < sma50 and sma20 < sma50 and ret20 < 0:
        return REGIME_PROFILES[MarketRegime.BEAR]
    return REGIME_PROFILES[MarketRegime.NEUTRAL]


def regime_history(
    index_close: pd.Series,
    trading_dates: list[pd.Timestamp],
    equity_curve: pd.Series | None = None,
    crisis_dd_threshold: float = 0.12,
) -> pd.DataFrame:
    """일별 국면 이력."""
    peak = None
    rows = []
    for dt in trading_dates:
        dd = 0.0
        if equity_curve is not None and dt in equity_curve.index:
            val = equity_curve.loc[dt]
            peak = val if peak is None else max(peak, val)
            dd = (val - peak) / peak if peak > 0 else 0.0
        state = detect_regime(index_close, dt, dd, crisis_dd_threshold)
        rows.append({
            "date": dt,
            "regime": state.regime.value,
            "exposure_cap": state.exposure_cap,
            "strategy": state.strategy_mode,
            "label": state.label,
        })
    return pd.DataFrame(rows).set_index("date")