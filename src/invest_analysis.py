"""투자 의사결정용 종목 분석."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import add_technical_indicators
from .korean_universe import KoreanStock
from .local_db import fetch_daily_quote, get_company_info
from .model import quick_predict

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class InvestmentReport:
    """투자 분석 리포트."""

    code: str
    name: str
    sector: str
    market: str
    current_price: float
    action: str
    confidence: float
    score: float
    risk_level: str
    entry_price: float
    stop_loss: float
    target_price: float
    position_pct: float
    predicted_return: float
    volatility_20d: float
    max_drawdown_60d: float
    rsi: float
    flow_5d: dict[str, float]
    flow_20d: dict[str, float]
    market_regime: str = ""
    regime_label: str = ""
    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)


def _calc_atr(data: pd.DataFrame, period: int = 14) -> float:
    high, low, close = data["high"], data["low"], data["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _calc_max_drawdown(prices: pd.Series) -> float:
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return float(drawdown.min())


def _flow_sum(df: pd.DataFrame, col: str, days: int) -> float:
    if col not in df.columns:
        return 0.0
    return float(df[col].tail(days).sum())


def _score_investor_flow(df: pd.DataFrame) -> tuple[float, list[str], list[str]]:
    """수급 점수 (최대 20점)."""
    score = 0.0
    reasons: list[str] = []
    cautions: list[str] = []

    f5 = _flow_sum(df, "foreigner_net", 5)
    i5 = _flow_sum(df, "institution_net", 5)
    p5 = _flow_sum(df, "program_net", 5)

    if f5 > 0:
        score += 8
        reasons.append(f"외국인 5일 순매수 {f5/1e4:,.0f}만주")
    elif f5 < 0:
        cautions.append(f"외국인 5일 순매도 {abs(f5)/1e4:,.0f}만주")

    if i5 > 0:
        score += 6
        reasons.append(f"기관 5일 순매수 {i5/1e4:,.0f}만주")
    elif i5 < 0:
        cautions.append(f"기관 5일 순매도 {abs(i5)/1e4:,.0f}만주")

    if p5 > 0:
        score += 4
        reasons.append(f"프로그램 5일 순매수 {p5/1e4:,.0f}만주")

    if "short_sell_qty" in df.columns:
        short5 = _flow_sum(df, "short_sell_qty", 5)
        short20 = _flow_sum(df, "short_sell_qty", 20)
        if short5 > short20 / 4 * 1.5:
            cautions.append("공매도 최근 증가")

    return min(score, 20), reasons, cautions


def _determine_action(
    total_score: float,
    risk_level: str,
    rsi: float,
    predicted_return: float,
) -> tuple[str, float, float]:
    """투자 액션, 신뢰도, 포지션 비중을 결정합니다."""
    confidence = min(total_score, 100)

    if total_score >= 75 and risk_level != "높음":
        action = "매수"
        position = 12.0 if risk_level == "낮음" else 8.0
    elif total_score >= 65:
        action = "분할 매수"
        position = 8.0 if risk_level == "낮음" else 5.0
    elif total_score >= 50:
        action = "관망"
        position = 3.0
    elif total_score >= 35:
        action = "보유/관망"
        position = 0.0
    else:
        action = "회피"
        position = 0.0

    if rsi > 72:
        action = "관망" if action in ("매수", "분할 매수") else action
        position *= 0.5
        confidence -= 10

    if predicted_return < -0.02:
        action = "관망" if action == "매수" else action
        position *= 0.5

    return action, max(confidence, 0), round(position, 1)


def _risk_level(volatility: float, drawdown: float) -> str:
    if volatility > 0.035 or drawdown < -0.15:
        return "높음"
    if volatility > 0.02 or drawdown < -0.08:
        return "보통"
    return "낮음"


def analyze_for_investment(
    code: str,
    period: str = "1y",
    forecast_days: int = 5,
    portfolio_value: float = 10_000_000,
) -> InvestmentReport:
    """종목 투자 분석 리포트를 생성합니다."""
    from src.backtest.kospi200 import build_kospi200_equal_weight_benchmark
    from src.backtest.regime import get_allocation_plan
    from src.backtest.signals import score_at_date

    info = get_company_info(code)
    df_ext = fetch_daily_quote(code, period=period, extended=True)
    df = df_ext[_OHLCV_COLS].copy()

    data = add_technical_indicators(df)
    row = data.iloc[-1]
    current = float(row["close"])

    _, predicted_price, predicted_return = quick_predict(df, forecast_days=forecast_days)

    volatility = float(data["return_1d"].tail(20).std())
    drawdown = _calc_max_drawdown(data["close"].tail(60))
    atr = _calc_atr(data)
    risk = _risk_level(volatility, drawdown)

    from .screener import (
        _score_ml_prediction,
        _score_momentum,
        _score_short_term,
        _score_trend,
        _score_volume,
    )

    total = 0.0
    reasons: list[str] = []
    cautions: list[str] = []

    for scorer, args in [
        (_score_ml_prediction, (predicted_return,)),
        (_score_trend, (row,)),
        (_score_momentum, (row,)),
        (_score_volume, (row,)),
        (_score_short_term, (row,)),
    ]:
        s, sigs = scorer(*args)
        total += s
        reasons.extend(sigs[:2])

    flow_score, flow_reasons, flow_cautions = _score_investor_flow(df_ext)
    total += flow_score
    reasons.extend(flow_reasons)
    cautions.extend(flow_cautions)

    if risk == "높음":
        cautions.append(f"변동성 높음 (일간 σ {volatility*100:.1f}%)")
        total -= 5
    if row["rsi"] > 70:
        cautions.append(f"RSI {row['rsi']:.0f} 과열 구간")
    if current < row["sma_50"]:
        cautions.append("50일선 아래 (중기 약세)")

    action, confidence, position_pct = _determine_action(
        total, risk, float(row["rsi"]), predicted_return
    )

    from datetime import datetime, timedelta
    _end = datetime.now().strftime("%Y-%m-%d")
    _start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    index_close = build_kospi200_equal_weight_benchmark(_start, _end)
    as_of = df_ext.index[-1]
    plan = get_allocation_plan(index_close, as_of)
    mode_score = score_at_date(df_ext, use_flow=True, mode=plan.strategy_mode) or 0

    position_pct = round(min(position_pct, plan.per_position_pct), 1)
    reasons.insert(0, f"시장 국면: {plan.label}")
    reasons.insert(1, f"종목당 비중 {plan.per_position_pct}% / 전체 주식 {plan.total_stock_pct}% / 현금 {plan.cash_pct}%")

    if plan.strategy_mode == "cash":
        action, position_pct = "관망", 0.0
        cautions.append("MDD 방어 — 신규 매수 중단")
    elif mode_score < plan.entry_score:
        if action in ("매수", "분할 매수"):
            action = "관망"
            position_pct = 0.0
            cautions.append(f"국면별 점수 {mode_score:.0f} < 진입기준 {plan.entry_score:.0f}")
    elif plan.regime == "bear":
        cautions.append(f"하락장 — 최대 {plan.per_position_pct}% 소액·{plan.max_hold_days}일 단기만")

    stop_loss = min(
        current - 2 * atr,
        current * (1 - plan.stop_loss_pct),
        float(row["sma_20"]) * 0.97,
    )
    target_price = max(predicted_price, float(row.get("bb_upper", current * 1.05)))

    if action in ("매수", "분할 매수"):
        entry_price = min(current, float(row["sma_5"]))
    else:
        entry_price = current

    stock = KoreanStock(
        ticker=info["code"],
        name=info["name"],
        sector=info.get("sector", ""),
        market=info.get("market", ""),
    )

    return InvestmentReport(
        code=stock.ticker,
        name=stock.name,
        sector=stock.sector,
        market=stock.market,
        current_price=current,
        action=action,
        confidence=round(confidence, 1),
        score=round(total, 1),
        risk_level=risk,
        entry_price=round(entry_price),
        stop_loss=round(stop_loss),
        target_price=round(target_price),
        position_pct=position_pct,
        predicted_return=predicted_return,
        volatility_20d=round(volatility * 100, 2),
        max_drawdown_60d=round(drawdown * 100, 2),
        rsi=float(row["rsi"]),
        flow_5d={
            "foreigner": _flow_sum(df_ext, "foreigner_net", 5),
            "institution": _flow_sum(df_ext, "institution_net", 5),
            "individual": _flow_sum(df_ext, "individual_net", 5),
            "program": _flow_sum(df_ext, "program_net", 5),
        },
        flow_20d={
            "foreigner": _flow_sum(df_ext, "foreigner_net", 20),
            "institution": _flow_sum(df_ext, "institution_net", 20),
            "individual": _flow_sum(df_ext, "individual_net", 20),
            "program": _flow_sum(df_ext, "program_net", 20),
        },
        market_regime=plan.regime,
        regime_label=plan.label,
        reasons=reasons[:8],
        cautions=cautions[:6],
    )


def suggest_order_amount(report: InvestmentReport, portfolio_value: float) -> dict:
    """포트폴리오 규모 기준 매수 금액·수량을 제안합니다."""
    if report.position_pct <= 0 or report.action in ("회피", "관망", "보유/관망"):
        return {"amount": 0, "quantity": 0, "note": "매수 비중 0%"}

    amount = portfolio_value * (report.position_pct / 100)
    quantity = int(amount / report.entry_price) if report.entry_price > 0 else 0
    actual_amount = quantity * report.entry_price

    return {
        "amount": actual_amount,
        "quantity": quantity,
        "entry_price": report.entry_price,
        "stop_loss": report.stop_loss,
        "target_price": report.target_price,
        "max_loss": (report.entry_price - report.stop_loss) * quantity if quantity else 0,
        "note": f"포트폴리오의 {report.position_pct}% 배분",
    }