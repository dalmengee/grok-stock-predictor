"""포트폴리오 관리 (프로젝트 로컬 JSON 저장)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .data_fetcher import fetch_stock_data
from .invest_analysis import analyze_for_investment
from .local_db import get_company_info, is_korean_code

PORTFOLIO_PATH = Path(__file__).resolve().parents[1] / "data" / "portfolio.json"


@dataclass
class Holding:
    code: str
    quantity: int
    avg_price: float
    added_at: str = ""


@dataclass
class Portfolio:
    cash: float = 10_000_000
    holdings: list[Holding] = field(default_factory=list)
    updated_at: str = ""


@dataclass
class HoldingStatus:
    code: str
    name: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    pnl: float
    pnl_pct: float
    weight: float
    action: str = ""
    stop_loss: float = 0
    target_price: float = 0


@dataclass
class PortfolioSummary:
    total_value: float
    cash: float
    stock_value: float
    total_pnl: float
    total_pnl_pct: float
    holdings: list[HoldingStatus]
    updated_at: str


def _ensure_data_dir() -> None:
    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_portfolio() -> Portfolio:
    """포트폴리오를 불러옵니다."""
    if not PORTFOLIO_PATH.exists():
        return Portfolio(updated_at=datetime.now().isoformat())
    data = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    holdings = [Holding(**h) for h in data.get("holdings", [])]
    return Portfolio(
        cash=float(data.get("cash", 10_000_000)),
        holdings=holdings,
        updated_at=data.get("updated_at", ""),
    )


def save_portfolio(portfolio: Portfolio) -> None:
    """포트폴리오를 저장합니다 (프로젝트 data/ 폴더만 수정)."""
    _ensure_data_dir()
    portfolio.updated_at = datetime.now().isoformat()
    payload = {
        "cash": portfolio.cash,
        "holdings": [asdict(h) for h in portfolio.holdings],
        "updated_at": portfolio.updated_at,
    }
    PORTFOLIO_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_holding(code: str, quantity: int, price: float) -> Portfolio:
    """종목을 추가하거나 수량을 늘립니다."""
    portfolio = load_portfolio()
    cost = quantity * price

    for h in portfolio.holdings:
        if h.code == code:
            total_qty = h.quantity + quantity
            h.avg_price = (h.avg_price * h.quantity + cost) / total_qty
            h.quantity = total_qty
            save_portfolio(portfolio)
            return portfolio

    portfolio.holdings.append(
        Holding(code=code, quantity=quantity, avg_price=price, added_at=datetime.now().strftime("%Y-%m-%d"))
    )
    portfolio.cash = max(0, portfolio.cash - cost)
    save_portfolio(portfolio)
    return portfolio


def remove_holding(code: str, quantity: int | None = None) -> Portfolio:
    """종목을 매도(삭제)합니다."""
    portfolio = load_portfolio()
    for i, h in enumerate(portfolio.holdings):
        if h.code != code:
            continue
        sell_qty = quantity or h.quantity
        sell_qty = min(sell_qty, h.quantity)
        try:
            price = float(fetch_stock_data(code, period="5d")["close"].iloc[-1])
        except Exception:
            price = h.avg_price
        portfolio.cash += sell_qty * price
        h.quantity -= sell_qty
        if h.quantity <= 0:
            portfolio.holdings.pop(i)
        save_portfolio(portfolio)
        return portfolio
    return portfolio


def set_cash(amount: float) -> Portfolio:
    """현금 잔고를 설정합니다."""
    portfolio = load_portfolio()
    portfolio.cash = amount
    save_portfolio(portfolio)
    return portfolio


def _get_name(code: str) -> str:
    if is_korean_code(code):
        return get_company_info(code).get("name", code)
    return code


def analyze_portfolio() -> PortfolioSummary:
    """포트폴리오 현황과 종목별 투자 판단을 반환합니다."""
    portfolio = load_portfolio()
    statuses: list[HoldingStatus] = []
    stock_value = 0.0
    total_cost = 0.0

    for h in portfolio.holdings:
        try:
            current = float(fetch_stock_data(h.code, period="1mo")["close"].iloc[-1])
        except Exception:
            current = h.avg_price

        mv = current * h.quantity
        cost = h.avg_price * h.quantity
        stock_value += mv
        total_cost += cost

        action, stop_loss, target = "", 0.0, 0.0
        if is_korean_code(h.code):
            try:
                report = analyze_for_investment(h.code)
                action = report.action
                stop_loss = report.stop_loss
                target_price = report.target_price
            except Exception:
                target_price = 0.0
        else:
            target_price = 0.0

        statuses.append(
            HoldingStatus(
                code=h.code,
                name=_get_name(h.code),
                quantity=h.quantity,
                avg_price=h.avg_price,
                current_price=current,
                market_value=mv,
                pnl=mv - cost,
                pnl_pct=(current / h.avg_price - 1) * 100 if h.avg_price else 0,
                weight=0,
                action=action,
                stop_loss=stop_loss,
                target_price=target_price,
            )
        )

    total_value = stock_value + portfolio.cash
    for s in statuses:
        s.weight = (s.market_value / total_value * 100) if total_value else 0

    return PortfolioSummary(
        total_value=total_value,
        cash=portfolio.cash,
        stock_value=stock_value,
        total_pnl=stock_value - total_cost,
        total_pnl_pct=((stock_value / total_cost - 1) * 100) if total_cost else 0,
        holdings=statuses,
        updated_at=portfolio.updated_at,
    )