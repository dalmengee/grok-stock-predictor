"""백테스트 성과 지표."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_return_pct: float
    cagr_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    avg_hold_days: float
    final_value: float


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[dict],
    benchmark_curve: pd.Series | None,
    initial_cash: float,
    risk_free_rate: float = 0.035,
) -> PerformanceMetrics:
    """성과 지표를 계산합니다."""
    if equity_curve.empty:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, initial_cash)

    final_value = float(equity_curve.iloc[-1])
    total_return = (final_value / initial_cash - 1) * 100

    days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
    years = days / 365.25
    cagr = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

    daily_ret = equity_curve.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = (daily_ret.mean() - risk_free_rate / 252) / daily_ret.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = float(drawdown.min()) * 100

    sells = [t for t in trades if t["side"] == "sell"]
    wins = [t for t in sells if t.get("pnl", 0) > 0]
    win_rate = len(wins) / len(sells) * 100 if sells else 0

    gross_profit = sum(t.get("pnl", 0) for t in wins)
    gross_loss = abs(sum(t.get("pnl", 0) for t in sells if t.get("pnl", 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    hold_days = [t.get("hold_days", 0) for t in sells if t.get("hold_days")]
    avg_hold = float(np.mean(hold_days)) if hold_days else 0

    bench_return = 0.0
    alpha = total_return
    if benchmark_curve is not None and not benchmark_curve.empty:
        aligned = benchmark_curve.reindex(equity_curve.index).ffill().dropna()
        if len(aligned) > 1:
            bench_return = (aligned.iloc[-1] / aligned.iloc[0] - 1) * 100
            alpha = total_return - bench_return

    return PerformanceMetrics(
        total_return_pct=round(total_return, 2),
        cagr_pct=round(cagr, 2),
        benchmark_return_pct=round(bench_return, 2),
        alpha_pct=round(alpha, 2),
        sharpe_ratio=round(float(sharpe), 2),
        max_drawdown_pct=round(max_dd, 2),
        win_rate_pct=round(win_rate, 1),
        profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
        total_trades=len(trades),
        avg_hold_days=round(avg_hold, 1),
        final_value=round(final_value),
    )