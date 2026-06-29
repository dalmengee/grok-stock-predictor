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
        self.crisis_dd = crisis_dd
        self.recovery_dd = recovery_dd
        self.in_crisis = False

    def update(self, equity: float) -> RiskState:
        self.peak_equity = max(self.peak_equity, equity)
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