"""워크포워드(out-of-sample) 검증."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.config import BacktestConfig
from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.metrics import PerformanceMetrics


@dataclass
class WalkForwardResult:
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    split_date: str
    is_metrics: PerformanceMetrics
    oos_metrics: PerformanceMetrics
    robust: bool
    summary: str


def run_walk_forward(
    universe: str = "kospi_large",
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    train_ratio: float = 0.7,
    base_config: BacktestConfig | None = None,
) -> WalkForwardResult:
    """
    기간을 in-sample / out-of-sample로 나눠 전략 견고성을 검증합니다.
    동일 파라미터로 양 구간 성과를 비교합니다.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    split = start + (end - start) * train_ratio
    split_str = split.strftime("%Y-%m-%d")

    cfg = base_config or BacktestConfig()
    is_cfg = BacktestConfig(**{**cfg.__dict__, "start_date": start_date, "end_date": split_str})
    oos_start = (split + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    oos_cfg = BacktestConfig(**{**cfg.__dict__, "start_date": oos_start, "end_date": end_date})

    is_result = run_backtest(universe=universe, config=is_cfg)
    oos_result = run_backtest(universe=universe, config=oos_cfg)

    is_m = is_result.metrics
    oos_m = oos_result.metrics

    robust = (
        oos_m.total_return_pct > 0
        and oos_m.sharpe_ratio > 0
        and oos_m.alpha_pct > -10
    )

    if robust:
        summary = "OOS 구간에서도 양호 — 전략 견고성 있음"
    elif oos_m.total_return_pct > 0:
        summary = "OOS 수익은 있으나 샤프/알파 약함 — 보수적 접근 권장"
    else:
        summary = "OOS 구간 손실 — 실전 적용 전 파라미터 재검토 필요"

    return WalkForwardResult(
        in_sample=is_result,
        out_of_sample=oos_result,
        split_date=split_str,
        is_metrics=is_m,
        oos_metrics=oos_m,
        robust=robust,
        summary=summary,
    )