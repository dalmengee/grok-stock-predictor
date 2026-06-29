"""국면별 자본 배분 계획 (포지션 사이징)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapitalAllocation:
    """자본금 대비 포지션 배분 설계."""

    regime: str
    label: str
    strategy_mode: str
    total_stock_pct: float
    cash_pct: float
    max_positions: int
    per_position_pct: float
    entry_score: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    max_hold_days: int

    @property
    def max_deployable_pct(self) -> float:
        return self.per_position_pct * self.max_positions


# ── 상승장: 모멘텀 추종, 적극 투자 ──
BULL_PLAN = CapitalAllocation(
    regime="bull",
    label="상승장 — 모멘텀·수급 추종",
    strategy_mode="momentum",
    total_stock_pct=60.0,
    cash_pct=40.0,
    max_positions=4,
    per_position_pct=15.0,
    entry_score=55.0,
    stop_loss_pct=0.08,
    take_profit_pct=0.18,
    trailing_stop_pct=0.06,
    max_hold_days=30,
)

# ── 하락장: 과매도 반등만 소액, 현금 우선 ──
BEAR_PLAN = CapitalAllocation(
    regime="bear",
    label="하락장 — 과매도 반등 스캘핑",
    strategy_mode="defensive",
    total_stock_pct=15.0,
    cash_pct=85.0,
    max_positions=1,
    per_position_pct=15.0,
    entry_score=65.0,
    stop_loss_pct=0.04,
    take_profit_pct=0.06,
    trailing_stop_pct=0.03,
    max_hold_days=10,
)

NEUTRAL_PLAN = CapitalAllocation(
    regime="neutral",
    label="횡보장 — 선별 매수",
    strategy_mode="selective",
    total_stock_pct=30.0,
    cash_pct=70.0,
    max_positions=2,
    per_position_pct=15.0,
    entry_score=60.0,
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
    trailing_stop_pct=0.04,
    max_hold_days=15,
)

CASH_PLAN = CapitalAllocation(
    regime="crisis",
    label="위기 — 전량 현금 (MDD 방어)",
    strategy_mode="cash",
    total_stock_pct=0.0,
    cash_pct=100.0,
    max_positions=0,
    per_position_pct=0.0,
    entry_score=99.0,
    stop_loss_pct=0.03,
    take_profit_pct=0.05,
    trailing_stop_pct=0.02,
    max_hold_days=5,
)

REGIME_PLANS = {
    "bull": BULL_PLAN,
    "bear": BEAR_PLAN,
    "neutral": NEUTRAL_PLAN,
    "crisis": CASH_PLAN,
}


def apply_drawdown_scaling(plan: CapitalAllocation, drawdown: float) -> CapitalAllocation:
    """
    DD에 따라 비중을 단계적으로 축소 (최대손실 15% 이내 방어).
    drawdown: 0 ~ -1
    """
    if drawdown <= -0.12 or plan.regime == "crisis":
        return CASH_PLAN
    if drawdown <= -0.08:
        factor = 0.4
    elif drawdown <= -0.05:
        factor = 0.6
    else:
        return plan

    return CapitalAllocation(
        regime=plan.regime,
        label=plan.label + f" (DD축소 ×{factor})",
        strategy_mode=plan.strategy_mode,
        total_stock_pct=plan.total_stock_pct * factor,
        cash_pct=100.0 - plan.total_stock_pct * factor,
        max_positions=max(0, int(plan.max_positions * factor)),
        per_position_pct=plan.per_position_pct * factor,
        entry_score=plan.entry_score,
        stop_loss_pct=plan.stop_loss_pct,
        take_profit_pct=plan.take_profit_pct,
        trailing_stop_pct=plan.trailing_stop_pct,
        max_hold_days=plan.max_hold_days,
    )


def allocate_by_plan(
    equity: float,
    cash: float,
    plan: CapitalAllocation,
    candidates: list[tuple[str, float]],
    prices: dict[str, float],
    current_holdings: set[str],
) -> dict[str, int]:
    """
    자본금 % 기반 수량 배분.
    종목당 plan.per_position_pct % of equity.
    """
    if plan.max_positions <= 0 or plan.per_position_pct <= 0:
        return {}

    slots = plan.max_positions - len(current_holdings)
    if slots <= 0:
        return {}

    allocations: dict[str, int] = {}
    budget_per_stock = equity * (plan.per_position_pct / 100.0)
    total_budget_cap = equity * (plan.total_stock_pct / 100.0)

    for code, _score in candidates:
        if code in current_holdings or slots <= 0:
            continue
        price = prices.get(code, 0)
        if price <= 0:
            continue
        spend = min(budget_per_stock, cash)
        if spend < price:
            continue
        qty = int(spend / price)
        if qty <= 0:
            continue
        cost = qty * price
        if cost > cash:
            qty = int(cash / price)
            if qty <= 0:
                continue
            cost = qty * price
        allocations[code] = qty
        cash -= cost
        slots -= 1
        if sum(allocations[c] * prices[c] for c in allocations) >= total_budget_cap:
            break

    return allocations