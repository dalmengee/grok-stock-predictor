"""포트폴리오 리스크 관리."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RiskState:
    peak_equity: float
    current_drawdown: float
    in_crisis: bool


class RiskManager:
    """포트폴리오 수준 리스크 관리."""

    def __init__(
        self,
        initial_cash: float,
        crisis_dd: float = 0.12,
        recovery_dd: float = 0.06,
    ):
        self.peak_equity = initial_cash
        self.global_peak = initial_cash
        self.crisis_dd = crisis_dd
        self.recovery_dd = recovery_dd
        self.in_crisis = False
        self.cash_lock_until: pd.Timestamp | None = None

    def update(self, equity: float, dt: pd.Timestamp | None = None) -> RiskState:
        self.peak_equity = max(self.peak_equity, equity)
        self.global_peak = max(self.global_peak, equity)
        dd = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if dd <= -self.crisis_dd:
            self.in_crisis = True
        elif self.in_crisis and dd > -self.recovery_dd:
            self.in_crisis = False

        return RiskState(
            peak_equity=self.peak_equity,
            current_drawdown=dd,
            in_crisis=self.in_crisis,
        )

    def reset_peak(self, equity: float) -> None:
        """위기 청산 후 사이클 고점을 재설정해 재진입을 허용."""
        self.peak_equity = equity
        if self.cash_lock_until is None:
            self.in_crisis = False

    def locked_in_cash(self, dt: pd.Timestamp) -> bool:
        if self.cash_lock_until is not None and dt >= self.cash_lock_until:
            self.cash_lock_until = None
            self.in_crisis = False
        return self.cash_lock_until is not None and dt < self.cash_lock_until

    def global_drawdown(self, equity: float) -> float:
        if self.global_peak <= 0:
            return 0.0
        return (equity - self.global_peak) / self.global_peak

    def trigger_cash_lock(self, dt: pd.Timestamp, days: int = 10) -> None:
        self.in_crisis = True
        self.cash_lock_until = dt + pd.Timedelta(days=days)


def should_trailing_stop(
    current_price: float,
    high_watermark: float,
    trailing_pct: float,
) -> bool:
    """고점 대비 trailing stop 발동."""
    if high_watermark <= 0:
        return False
    return current_price <= high_watermark * (1 - trailing_pct)


def target_stock_value(equity: float, exposure_cap: float, cash_buffer: float = 0.05) -> float:
    """목표 주식 평가액."""
    return equity * max(0, exposure_cap - cash_buffer)