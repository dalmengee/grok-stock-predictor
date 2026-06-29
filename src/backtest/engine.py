"""백테스트 시뮬레이션 엔진."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.backtest.benchmark import benchmark_equity_curve, fetch_index_close
from src.backtest.config import BacktestConfig
from src.backtest.metrics import PerformanceMetrics, compute_metrics
from src.backtest.signals import rank_universe, score_at_date
from src.korean_universe import get_universe
from src.local_db import fetch_multiple_daily_quotes


@dataclass
class Position:
    code: str
    quantity: int
    entry_price: float
    entry_date: pd.Timestamp
    entry_score: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: list[dict]
    daily_scores: dict[str, list] = field(default_factory=dict)


def _apply_slippage(price: float, side: str, slippage: float) -> float:
    if side == "buy":
        return price * (1 + slippage)
    return price * (1 - slippage)


def _trade_cost(amount: float, side: str, cfg: BacktestConfig) -> float:
    fee = amount * cfg.commission_rate
    if side == "sell":
        fee += amount * cfg.sell_tax_rate
    return fee


def _load_universe_data(
    universe: str,
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    _, stocks = get_universe(universe)
    codes = [s.ticker for s in stocks]
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    return fetch_multiple_daily_quotes(codes, start=warmup_start, end=end, extended=True)


def run_backtest(
    universe: str = "kospi_large",
    config: BacktestConfig | None = None,
    benchmark_code: str = "001",
) -> BacktestResult:
    """유니버스 로테이션 전략 백테스트를 실행합니다."""
    cfg = config or BacktestConfig()
    data_map = _load_universe_data(universe, cfg.start_date, cfg.end_date)

    all_dates: set[pd.Timestamp] = set()
    for df in data_map.values():
        mask = (df.index >= cfg.start_date) & (df.index <= cfg.end_date)
        all_dates.update(df.index[mask])
    trading_dates = sorted(all_dates)
    if not trading_dates:
        raise ValueError("백테스트 기간에 거래일이 없습니다.")

    cash = cfg.initial_cash
    positions: dict[str, Position] = {}
    trades: list[dict] = []
    equity_records: list[tuple[pd.Timestamp, float]] = []
    last_rebalance: pd.Timestamp | None = None

    for dt in trading_dates:
        # --- 1. 청산 조건 확인 ---
        to_close: list[tuple[str, str]] = []
        for code, pos in positions.items():
            df = data_map.get(code)
            if df is None or dt not in df.index:
                continue
            price = float(df.loc[dt, "close"])
            hold_days = (dt - pos.entry_date).days
            ret = price / pos.entry_price - 1

            hist = df[df.index <= dt]
            current_score = score_at_date(hist, use_flow=cfg.use_flow_score) or 0

            if ret <= -cfg.stop_loss_pct:
                to_close.append((code, "stop_loss"))
            elif ret >= cfg.take_profit_pct:
                to_close.append((code, "take_profit"))
            elif hold_days >= cfg.max_hold_days:
                to_close.append((code, "max_hold"))
            elif current_score < cfg.exit_score:
                to_close.append((code, "score_exit"))

        for code, reason in to_close:
            pos = positions.pop(code)
            df = data_map[code]
            raw_price = float(df.loc[dt, "close"])
            price = _apply_slippage(raw_price, "sell", cfg.slippage_pct)
            amount = price * pos.quantity
            fee = _trade_cost(amount, "sell", cfg)
            pnl = (price - pos.entry_price) * pos.quantity - fee
            cash += amount - fee
            trades.append({
                "date": dt.strftime("%Y-%m-%d"),
                "code": code,
                "side": "sell",
                "price": price,
                "quantity": pos.quantity,
                "amount": amount,
                "fee": fee,
                "pnl": pnl,
                "hold_days": (dt - pos.entry_date).days,
                "reason": reason,
            })

        # --- 2. 리밸런싱 ---
        do_rebalance = (
            last_rebalance is None
            or (dt - last_rebalance).days >= cfg.rebalance_days
        )
        if do_rebalance:
            last_rebalance = dt
            rankings = rank_universe(
                data_map, dt, use_flow=cfg.use_flow_score, min_history=cfg.min_history_days,
            )
            candidates = [(c, s) for c, s in rankings if s >= cfg.entry_score]
            target_codes = {c for c, _ in candidates[: cfg.max_positions]}

            # 보유 중인데 후보에서 빠진 종목 매도
            for code in list(positions):
                if code not in target_codes:
                    pos = positions.pop(code)
                    raw_price = float(data_map[code].loc[dt, "close"])
                    price = _apply_slippage(raw_price, "sell", cfg.slippage_pct)
                    amount = price * pos.quantity
                    fee = _trade_cost(amount, "sell", cfg)
                    pnl = (price - pos.entry_price) * pos.quantity - fee
                    cash += amount - fee
                    trades.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "code": code,
                        "side": "sell",
                        "price": price,
                        "quantity": pos.quantity,
                        "amount": amount,
                        "fee": fee,
                        "pnl": pnl,
                        "hold_days": (dt - pos.entry_date).days,
                        "reason": "rebalance_out",
                    })

            # 신규 매수
            slots = cfg.max_positions - len(positions)
            new_buys = [c for c, _ in candidates if c not in positions][:slots]
            if new_buys and cash > 0:
                per_stock = cash / len(new_buys)
                for code in new_buys:
                    raw_price = float(data_map[code].loc[dt, "close"])
                    price = _apply_slippage(raw_price, "buy", cfg.slippage_pct)
                    qty = int(per_stock / price)
                    if qty <= 0:
                        continue
                    amount = price * qty
                    fee = _trade_cost(amount, "buy", cfg)
                    total_cost = amount + fee
                    if total_cost > cash:
                        qty = int((cash - fee) / price)
                        if qty <= 0:
                            continue
                        amount = price * qty
                        fee = _trade_cost(amount, "buy", cfg)
                        total_cost = amount + fee
                    cash -= total_cost
                    score = next(s for c, s in candidates if c == code)
                    positions[code] = Position(code, qty, price, dt, score)
                    trades.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "code": code,
                        "side": "buy",
                        "price": price,
                        "quantity": qty,
                        "amount": amount,
                        "fee": fee,
                        "pnl": 0,
                        "hold_days": 0,
                        "reason": f"entry_score_{score}",
                    })

        # --- 3. 일별 자산 평가 ---
        stock_value = 0.0
        for code, pos in positions.items():
            df = data_map.get(code)
            if df is not None and dt in df.index:
                stock_value += float(df.loc[dt, "close"]) * pos.quantity
            else:
                stock_value += pos.entry_price * pos.quantity
        equity_records.append((dt, cash + stock_value))

    equity_curve = pd.Series(
        [v for _, v in equity_records],
        index=pd.DatetimeIndex([d for d, _ in equity_records]),
    )

    index_series = fetch_index_close(benchmark_code, cfg.start_date, cfg.end_date)
    bench_curve = benchmark_equity_curve(cfg.initial_cash, index_series, equity_curve.index)

    metrics = compute_metrics(
        equity_curve, trades, index_series, cfg.initial_cash,
    )

    return BacktestResult(
        config=cfg,
        metrics=metrics,
        equity_curve=equity_curve,
        benchmark_curve=bench_curve,
        trades=trades,
    )