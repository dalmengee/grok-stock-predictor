"""변동성·리스크 기반 포지션 사이징."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import add_technical_indicators


def stock_volatility(df: pd.DataFrame, window: int = 20) -> float:
    """연환산 변동성 추정."""
    if len(df) < window + 5:
        return 0.25
    data = add_technical_indicators(df)
    vol = data["return_1d"].tail(window).std()
    if pd.isna(vol) or vol <= 0:
        return 0.25
    return float(vol * np.sqrt(252))


def risk_budget_size(
    equity: float,
    price: float,
    stop_loss_pct: float,
    risk_per_trade: float = 0.02,
    max_position_pct: float = 0.20,
    vol: float = 0.25,
    target_vol: float = 0.20,
) -> int:
    """
    리스크 예산 기반 수량.
    - 포트폴리오의 risk_per_trade (2%)만 손절 시 잃도록 수량 결정
    - 변동성 높을수록 수량 축소 (target_vol / vol)
    """
    if price <= 0 or stop_loss_pct <= 0:
        return 0

    risk_amount = equity * risk_per_trade
    stop_distance = price * stop_loss_pct
    base_qty = int(risk_amount / stop_distance)

    vol_scalar = min(1.5, max(0.4, target_vol / max(vol, 0.05)))
    qty = int(base_qty * vol_scalar)

    max_qty = int(equity * max_position_pct / price)
    return max(0, min(qty, max_qty))


def allocate_by_scores(
    equity: float,
    exposure_cap: float,
    candidates: list[tuple[str, float]],
    prices: dict[str, float],
    vols: dict[str, float],
    stop_loss_pct: float,
    risk_per_trade: float = 0.015,
) -> dict[str, int]:
    """
    점수 가중 + 리스크 예산으로 종목별 수량 배분.
    exposure_cap: 전체 투자 가능 비율 (0~1)
    """
    if not candidates:
        return {}

    investable = equity * exposure_cap
    scores = np.array([s for _, s in candidates], dtype=float)
    weights = scores / scores.sum() if scores.sum() > 0 else np.ones(len(candidates)) / len(candidates)

    allocations: dict[str, int] = {}
    used = 0.0

    for (code, score), w in zip(candidates, weights):
        price = prices.get(code, 0)
        if price <= 0:
            continue
        budget = investable * w
        vol = vols.get(code, 0.25)
        qty = risk_budget_size(
            equity=equity,
            price=price,
            stop_loss_pct=stop_loss_pct,
            risk_per_trade=risk_per_trade * (0.8 + 0.4 * w),
            max_position_pct=min(0.25, budget / equity) if equity > 0 else 0.2,
            vol=vol,
        )
        cost = qty * price
        if used + cost > investable:
            qty = int((investable - used) / price)
        if qty > 0:
            allocations[code] = qty
            used += qty * price

    return allocations