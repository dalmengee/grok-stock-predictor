"""오늘의 투자 브리핑."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .invest_analysis import InvestmentReport, analyze_for_investment, suggest_order_amount
from .korean_universe import get_universe
from .local_db import get_latest_quote_date
from .portfolio import analyze_portfolio, load_portfolio
from .screener import screen_korean_stocks


@dataclass
class MarketBrief:
    """일일 투자 브리핑."""

    date: str
    db_latest: str
    market_mood: str
    market_regime: str
    regime_label: str
    exposure_cap: float
    buy_ideas: list[InvestmentReport]
    watch_list: list[InvestmentReport]
    portfolio_summary: dict
    portfolio_actions: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _mood_from_screening(buy_count: int, total: int, avg_score: float) -> str:
    ratio = buy_count / total if total else 0
    if ratio >= 0.25 and avg_score >= 55:
        return "긍정적"
    if ratio >= 0.1 or avg_score >= 50:
        return "중립"
    return "신중"


def generate_market_brief(
    universe: str = "kospi_large",
    portfolio_value: float | None = None,
    top_ideas: int = 5,
) -> MarketBrief:
    """오늘의 투자 브리핑을 생성합니다."""
    from src.backtest.benchmark import fetch_index_close
    from src.backtest.regime import detect_regime

    screening = screen_korean_stocks(universe=universe, top_n=top_ideas)
    db_latest = get_latest_quote_date() or ""

    from datetime import datetime, timedelta
    _end = datetime.now().strftime("%Y-%m-%d")
    _start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    index_close = fetch_index_close("001", _start, _end)
    if not index_close.empty:
        regime = detect_regime(index_close, index_close.index[-1])
    else:
        from src.backtest.regime import REGIME_PROFILES, MarketRegime
        regime = REGIME_PROFILES[MarketRegime.NEUTRAL]

    buy_codes = [p.ticker for p in screening.all_picks if p.recommendation in ("강력 매수", "매수")]
    watch_codes = [p.ticker for p in screening.all_picks if p.recommendation == "관망"][:3]

    pf_data = load_portfolio()
    if portfolio_value is None:
        portfolio_value = pf_data.cash + sum(h.quantity * h.avg_price for h in pf_data.holdings)

    buy_ideas: list[InvestmentReport] = []
    for code in buy_codes[:top_ideas]:
        try:
            buy_ideas.append(analyze_for_investment(code, portfolio_value=portfolio_value))
        except Exception:
            continue

    watch_list: list[InvestmentReport] = []
    for code in watch_codes:
        try:
            watch_list.append(analyze_for_investment(code, portfolio_value=portfolio_value))
        except Exception:
            continue

    avg_score = sum(p.score for p in screening.all_picks) / len(screening.all_picks) if screening.all_picks else 0
    mood = _mood_from_screening(len(buy_codes), screening.analyzed_count, avg_score)

    pf = analyze_portfolio()
    portfolio_actions = []
    for h in pf.holdings:
        item = {"code": h.code, "name": h.name, "action": h.action, "pnl_pct": h.pnl_pct}
        if h.current_price <= h.stop_loss and h.stop_loss > 0:
            item["action"] = "손절 검토"
            item["alert"] = f"현재가 ₩{h.current_price:,.0f} ≤ 손절 ₩{h.stop_loss:,.0f}"
        elif h.target_price and h.current_price >= h.target_price:
            item["action"] = "익절 검토"
            item["alert"] = f"목표가 ₩{h.target_price:,.0f} 도달"
        portfolio_actions.append(item)

    notes = [
        f"시장 국면: {regime.label} (투자 한도 {regime.exposure_cap*100:.0f}%)",
        f"{screening.universe_label} {screening.analyzed_count}개 분석, 매수 후보 {len(buy_codes)}개",
        "수급 + 기술적 지표 + ML + 시장국면을 종합한 판단입니다.",
    ]
    if mood == "신중":
        notes.append("시장 전반 점수가 낮습니다. 분할 매수와 리스크 관리를 권장합니다.")

    return MarketBrief(
        date=datetime.now().strftime("%Y-%m-%d"),
        db_latest=db_latest,
        market_mood=mood,
        market_regime=regime.regime.value,
        regime_label=regime.label,
        exposure_cap=regime.exposure_cap,
        buy_ideas=buy_ideas,
        watch_list=watch_list,
        portfolio_summary={
            "total_value": pf.total_value,
            "cash": pf.cash,
            "stock_value": pf.stock_value,
            "total_pnl": pf.total_pnl,
            "total_pnl_pct": pf.total_pnl_pct,
            "holdings_count": len(pf.holdings),
        },
        portfolio_actions=portfolio_actions,
        notes=notes,
    )


def brief_buy_recommendations(brief: MarketBrief, portfolio_value: float) -> list[dict]:
    """브리핑 매수 아이디어에 주문 제안을 붙입니다."""
    results = []
    for report in brief.buy_ideas:
        order = suggest_order_amount(report, portfolio_value)
        results.append({
            "code": report.code,
            "name": report.name,
            "action": report.action,
            "score": report.score,
            "confidence": report.confidence,
            "order": order,
            "reasons": report.reasons[:3],
        })
    return results