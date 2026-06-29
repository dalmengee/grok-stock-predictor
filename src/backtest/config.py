"""백테스트 설정."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

DEFAULT_DATA_START = "2016-01-04"
DEFAULT_DATA_END = "2026-06-29"
MIN_BACKTEST_YEARS = 10


@dataclass
class BacktestConfig:
    """백테스트 파라미터."""

    start_date: str = DEFAULT_DATA_START
    end_date: str = DEFAULT_DATA_END
    initial_cash: float = 10_000_000
    rebalance_days: int = 5
    commission_rate: float = 0.00015
    sell_tax_rate: float = 0.0023
    slippage_pct: float = 0.001
    min_history_days: int = 120
    use_flow_score: bool = True

    adaptive: bool = True
    dual_strategy: bool = True
    max_drawdown_limit: float = 0.15
    crisis_dd_trigger: float = 0.12
    recovery_dd_threshold: float = 0.05

    risk_per_trade: float = 0.01
    max_hold_days: int = 30

    max_positions: int = 5
    entry_score: float = 60.0
    exit_score: float = 45.0
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.15


def validate_period(start: str, end: str, min_years: int = MIN_BACKTEST_YEARS) -> None:
    """백테스트 기간이 최소 N년 이상인지 검증."""
    s, e = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
    years = (e - s).days / 365.25
    if years < min_years:
        raise ValueError(
            f"백테스트 기간이 {years:.1f}년입니다. 최소 {min_years}년 이상 필요합니다. "
            f"(DB 데이터: {DEFAULT_DATA_START} ~ {DEFAULT_DATA_END})"
        )


def max_available_period() -> tuple[str, str]:
    return DEFAULT_DATA_START, DEFAULT_DATA_END