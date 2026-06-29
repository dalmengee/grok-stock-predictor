"""백테스트 설정."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """백테스트 파라미터."""

    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    initial_cash: float = 10_000_000
    max_positions: int = 5
    entry_score: float = 60.0
    exit_score: float = 45.0
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.15
    max_hold_days: int = 20
    rebalance_days: int = 5
    commission_rate: float = 0.00015
    sell_tax_rate: float = 0.0023
    slippage_pct: float = 0.001
    min_history_days: int = 60
    use_flow_score: bool = True