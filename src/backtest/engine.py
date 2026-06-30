"""이중 전략 백테스트 엔진 (상승장/하락장 + MDD 15% 제한)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.backtest.benchmark import benchmark_equity_curve, fetch_index_close
from src.backtest.config import BacktestConfig, validate_period
from src.backtest.kospi200 import (
    BENCHMARK_LABEL,
    build_kospi200_equal_weight_benchmark,
    select_kospi200_codes,
)
from src.backtest.metrics import PerformanceMetrics, compute_metrics
from src.backtest.position_plan import CASH_PLAN, allocate_by_plan
from src.backtest.regime import get_allocation_plan
from src.backtest.risk import RiskManager, should_trailing_stop
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
    high_watermark: float
    regime: str


@dataclass
class BacktestResult:
    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: list[dict]
    regime_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    exposure_log: pd.Series = field(default_factory=pd.Series)
    allocation_log: pd.DataFrame = field(default_factory=pd.DataFrame)


def _apply_slippage(price: float, side: str, slippage: float) -> float:
    return price * (1 + slippage) if side == "buy" else price * (1 - slippage)


def _trade_cost(amount: float, side: str, cfg: BacktestConfig) -> float:
    fee = amount * cfg.commission_rate
    if side == "sell":
        fee += amount * cfg.sell_tax_rate
    return fee


def _sell_position(code, pos, dt, raw_price, cfg, reason) -> tuple[float, dict]:
    price = _apply_slippage(raw_price, "sell", cfg.slippage_pct)
    amount = price * pos.quantity
    fee = _trade_cost(amount, "sell", cfg)
    pnl = (price - pos.entry_price) * pos.quantity - fee
    return amount - fee, {
        "date": dt.strftime("%Y-%m-%d"), "code": code, "side": "sell",
        "price": price, "quantity": pos.quantity, "amount": amount,
        "fee": fee, "pnl": pnl, "hold_days": (dt - pos.entry_date).days,
        "reason": reason, "regime": pos.regime,
    }


def _portfolio_value(cash, positions, data_map, dt) -> float:
    stock_val = 0.0
    for code, pos in positions.items():
        df = data_map.get(code)
        if df is not None and dt in df.index:
            stock_val += float(df.loc[dt, "close"]) * pos.quantity
        else:
            stock_val += pos.entry_price * pos.quantity
    return cash + stock_val


def _is_kospi200_universe(universe: str) -> bool:
    return universe in ("kospi200", "kospi200_ex")


def _load_universe_data(universe: str, start: str, end: str) -> dict[str, pd.DataFrame]:
    if _is_kospi200_universe(universe):
        from src.backtest.kospi200 import _static_kospi200_codes

        codes = _static_kospi200_codes(exclude_strategy=False)
    else:
        _, stocks = get_universe(universe)
        codes = [s.ticker for s in stocks]
    warmup = (pd.Timestamp(start) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
    return fetch_multiple_daily_quotes(codes, start=warmup, end=end, extended=True)


def _strategy_codes(universe: str, dt: pd.Timestamp, data_map: dict, cfg: BacktestConfig) -> set[str]:
    if _is_kospi200_universe(universe):
        return set(select_kospi200_codes(dt, data_map, for_strategy=True))
    _, stocks = get_universe(universe)
    return {s.ticker for s in stocks}


def _resolve_benchmark_series(
    cfg: BacktestConfig,
    data_map: dict,
    equity_index: pd.DatetimeIndex,
    warmup_start: str,
) -> pd.Series:
    if cfg.benchmark_type == "kospi200_ew":
        return build_kospi200_equal_weight_benchmark(
            warmup_start, cfg.end_date, data_map=data_map,
        )
    return fetch_index_close("001", warmup_start, cfg.end_date)


def _liquidate_all(positions, data_map, dt, cfg, reason) -> tuple[float, list]:
    cash_gain = 0.0
    trades = []
    for code, pos in list(positions.items()):
        if code not in data_map or dt not in data_map[code].index:
            continue
        proceeds, trade = _sell_position(code, pos, dt, float(data_map[code].loc[dt, "close"]), cfg, reason)
        cash_gain += proceeds
        trades.append(trade)
        del positions[code]
    return cash_gain, trades


def _run_dual_strategy(
    cfg: BacktestConfig,
    data_map: dict,
    trading_dates: list,
    index_close: pd.Series,
    universe: str = "kospi200_ex",
) -> BacktestResult:
    cash = cfg.initial_cash
    positions: dict[str, Position] = {}
    trades: list[dict] = []
    equity_records: list[tuple[pd.Timestamp, float]] = []
    regime_rows: list[dict] = []
    exposure_rows: list[tuple[pd.Timestamp, float]] = []
    alloc_rows: list[dict] = []
    last_rebalance = None
    last_universe_refresh = None
    strategy_pool: set[str] = _strategy_codes(universe, trading_dates[0], data_map, cfg)
    risk_mgr = RiskManager(cfg.initial_cash, cfg.crisis_dd_trigger, cfg.recovery_dd_threshold)

    for dt in trading_dates:
        if (
            last_universe_refresh is None
            or (dt - last_universe_refresh).days >= cfg.rebalance_universe_days
        ):
            strategy_pool = _strategy_codes(universe, dt, data_map, cfg)
            last_universe_refresh = dt
        equity = _portfolio_value(cash, positions, data_map, dt)
        risk = risk_mgr.update(equity, dt)
        global_dd = risk_mgr.global_drawdown(equity)
        if cfg.global_mdd_limit and global_dd <= -cfg.max_drawdown_limit and positions:
            gain, ts = _liquidate_all(positions, data_map, dt, cfg, "global_mdd_stop")
            cash += gain
            trades.extend(ts)
            risk_mgr.reset_peak(cash)
            risk_mgr.trigger_cash_lock(dt)
            equity = cash
            risk = risk_mgr.update(equity, dt)
        elif risk.current_drawdown <= -cfg.max_drawdown_limit and positions:
            gain, ts = _liquidate_all(positions, data_map, dt, cfg, "mdd_hard_stop")
            cash += gain
            trades.extend(ts)
            risk_mgr.reset_peak(cash)
            equity = cash
            risk = risk_mgr.update(equity)

        plan = get_allocation_plan(index_close, dt, risk.current_drawdown, cfg.max_drawdown_limit)

        regime_rows.append({
            "date": dt, "regime": plan.regime, "strategy": plan.strategy_mode,
            "stock_pct": plan.total_stock_pct, "cash_pct": plan.cash_pct,
            "per_position_pct": plan.per_position_pct, "max_positions": plan.max_positions,
            "drawdown": risk.current_drawdown, "label": plan.label,
        })

        if plan.regime == "crisis" and positions:
            gain, ts = _liquidate_all(positions, data_map, dt, cfg, "mdd_defense")
            cash += gain
            trades.extend(ts)
            risk_mgr.reset_peak(cash)

        if not cfg.rebalance_only_exits:
            for code, pos in list(positions.items()):
                df = data_map.get(code)
                if df is None or dt not in df.index:
                    continue
                price = float(df.loc[dt, "close"])
                pos.high_watermark = max(pos.high_watermark, price)
                ret = price / pos.entry_price - 1
                hold_days = (dt - pos.entry_date).days
                hist = df[df.index <= dt]
                sc = score_at_date(hist, cfg.use_flow_score, plan.strategy_mode) or 0

                reason = None
                if ret <= -plan.stop_loss_pct:
                    reason = "stop_loss"
                elif ret >= plan.take_profit_pct:
                    reason = "take_profit"
                elif should_trailing_stop(price, pos.high_watermark, plan.trailing_stop_pct) and ret > 0:
                    reason = "trailing_stop"
                elif hold_days >= plan.max_hold_days:
                    reason = "max_hold"
                elif sc < plan.entry_score - 20:
                    reason = "score_exit"
                elif plan.strategy_mode == "cash":
                    reason = "regime_cash"

                if reason:
                    proceeds, trade = _sell_position(code, pos, dt, price, cfg, reason)
                    del positions[code]
                    cash += proceeds
                    trades.append(trade)
        elif plan.strategy_mode == "cash" and positions:
            gain, ts = _liquidate_all(positions, data_map, dt, cfg, "regime_cash")
            cash += gain
            trades.extend(ts)

        equity = _portfolio_value(cash, positions, data_map, dt)
        stock_val = equity - cash
        exposure_rows.append((dt, stock_val / equity if equity > 0 else 0))

        do_rebalance = last_rebalance is None or (dt - last_rebalance).days >= cfg.rebalance_days
        if risk_mgr.locked_in_cash(dt):
            plan = CASH_PLAN
        if do_rebalance and plan.strategy_mode != "cash":
            last_rebalance = dt
            tradeable = {c: data_map[c] for c in strategy_pool if c in data_map}
            from src.backtest.kospi200 import market_cap_at_date, _company_meta
            meta = _company_meta()
            caps = {c: market_cap_at_date(c, dt, data_map, meta) for c in tradeable}
            rankings = rank_universe(
                tradeable, dt, cfg.use_flow_score, cfg.min_history_days,
                mode=plan.strategy_mode, min_score=plan.entry_score,
                market_caps=caps,
            )
            candidates = rankings[: plan.max_positions]
            target_codes = {c for c, _ in candidates}

            for code in list(positions):
                if code not in target_codes:
                    proceeds, trade = _sell_position(
                        code, positions.pop(code), dt,
                        float(data_map[code].loc[dt, "close"]), cfg, "rebalance_out",
                    )
                    cash += proceeds
                    trades.append(trade)

            equity = _portfolio_value(cash, positions, data_map, dt)
            prices = {c: float(data_map[c].loc[dt, "close"]) for c, _ in candidates
                      if c in data_map and dt in data_map[c].index}
            alloc = allocate_by_plan(
                equity, cash, plan, candidates, prices, set(positions.keys()),
            )
            for code, qty in alloc.items():
                if qty <= 0 or code in positions:
                    continue
                raw = prices[code]
                price = _apply_slippage(raw, "buy", cfg.slippage_pct)
                cost = price * qty + _trade_cost(price * qty, "buy", cfg)
                if cost > cash:
                    continue
                cash -= cost
                score = next(s for c, s in candidates if c == code)
                positions[code] = Position(code, qty, price, dt, score, price, plan.regime)
                trades.append({
                    "date": dt.strftime("%Y-%m-%d"), "code": code, "side": "buy",
                    "price": price, "quantity": qty, "amount": price * qty,
                    "reason": f"{plan.regime}_{plan.per_position_pct}pct",
                    "regime": plan.regime,
                })
            alloc_rows.append({
                "date": dt, "regime": plan.regime,
                "positions": len(positions), "stock_pct": plan.total_stock_pct,
            })

        equity_records.append((dt, _portfolio_value(cash, positions, data_map, dt)))

    return _finalize(cfg, equity_records, trades, index_close, regime_rows, exposure_rows, alloc_rows, data_map)


def _run_simple(cfg, data_map, trading_dates, index_close, universe: str = "kospi200_ex") -> BacktestResult:
    """레거시 단순 로테이션."""
    cash = cfg.initial_cash
    positions: dict[str, Position] = {}
    trades: list[dict] = []
    equity_records: list[tuple[pd.Timestamp, float]] = []
    last_rebalance = None

    for dt in trading_dates:
        for code, pos in list(positions.items()):
            df = data_map.get(code)
            if df is None or dt not in df.index:
                continue
            price = float(df.loc[dt, "close"])
            ret = price / pos.entry_price - 1
            sc = score_at_date(df[df.index <= dt], cfg.use_flow_score) or 0
            reason = None
            if ret <= -cfg.stop_loss_pct:
                reason = "stop_loss"
            elif ret >= cfg.take_profit_pct:
                reason = "take_profit"
            elif (dt - pos.entry_date).days >= cfg.max_hold_days:
                reason = "max_hold"
            elif sc < cfg.exit_score:
                reason = "score_exit"
            if reason:
                p, t = _sell_position(code, pos, dt, price, cfg, reason)
                del positions[code]
                cash += p
                trades.append(t)

        if last_rebalance is None or (dt - last_rebalance).days >= cfg.rebalance_days:
            last_rebalance = dt
            pool = _strategy_codes(universe, dt, data_map, cfg)
            tradeable = {c: data_map[c] for c in pool if c in data_map}
            rankings = rank_universe(tradeable, dt, cfg.use_flow_score, cfg.min_history_days)
            candidates = [(c, s) for c, s in rankings if s >= cfg.entry_score]
            targets = {c for c, _ in candidates[: cfg.max_positions]}
            for code in list(positions):
                if code not in targets:
                    p, t = _sell_position(code, positions.pop(code), dt, float(data_map[code].loc[dt, "close"]), cfg, "rebalance_out")
                    cash += p
                    trades.append(t)
            slots = cfg.max_positions - len(positions)
            for code, score in candidates:
                if code in positions or slots <= 0:
                    continue
                price = _apply_slippage(float(data_map[code].loc[dt, "close"]), "buy", cfg.slippage_pct)
                qty = int((cash / slots) / price)
                if qty <= 0:
                    continue
                cost = price * qty + _trade_cost(price * qty, "buy", cfg)
                if cost > cash:
                    continue
                cash -= cost
                positions[code] = Position(code, qty, price, dt, score, price, "simple")
                trades.append({"date": dt.strftime("%Y-%m-%d"), "code": code, "side": "buy",
                                 "price": price, "quantity": qty, "reason": f"entry_{score}", "regime": "simple"})
                slots -= 1
        equity_records.append((dt, _portfolio_value(cash, positions, data_map, dt)))

    return _finalize(cfg, equity_records, trades, index_close, [], [], [], data_map)


def _finalize(cfg, equity_records, trades, index_close, regime_rows, exposure_rows, alloc_rows, data_map=None):
    equity_curve = pd.Series([v for _, v in equity_records], index=pd.DatetimeIndex([d for d, _ in equity_records]))
    if cfg.benchmark_type == "kospi200_ew" and data_map is not None:
        index_series = build_kospi200_equal_weight_benchmark(
            cfg.start_date, cfg.end_date, data_map=data_map,
        )
    elif index_close is not None and not index_close.empty:
        index_series = index_close.loc[index_close.index >= cfg.start_date]
    else:
        index_series = fetch_index_close("001", cfg.start_date, cfg.end_date)
    bench_curve = benchmark_equity_curve(cfg.initial_cash, index_series, equity_curve.index)
    metrics = compute_metrics(equity_curve, trades, index_series, cfg.initial_cash)
    regime_df = pd.DataFrame(regime_rows).set_index("date") if regime_rows else pd.DataFrame()
    exposure_series = pd.Series([e for _, e in exposure_rows], index=pd.DatetimeIndex([d for d, _ in exposure_rows])) if exposure_rows else pd.Series(dtype=float)
    alloc_df = pd.DataFrame(alloc_rows) if alloc_rows else pd.DataFrame()
    return BacktestResult(cfg, metrics, equity_curve, bench_curve, trades, regime_df, exposure_series, alloc_df)


def run_backtest(
    universe: str | None = None,
    config: BacktestConfig | None = None,
    benchmark_code: str = "001",
) -> BacktestResult:
    cfg = config or BacktestConfig()
    universe = universe or cfg.universe
    validate_period(cfg.start_date, cfg.end_date)
    data_map = _load_universe_data(universe, cfg.start_date, cfg.end_date)

    all_dates: set[pd.Timestamp] = set()
    for df in data_map.values():
        mask = (df.index >= cfg.start_date) & (df.index <= cfg.end_date)
        all_dates.update(df.index[mask])
    trading_dates = sorted(all_dates)
    if not trading_dates:
        raise ValueError("백테스트 기간에 거래일이 없습니다.")

    warmup = (pd.Timestamp(cfg.start_date) - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    if cfg.benchmark_type == "kospi200_ew":
        index_close = _resolve_benchmark_series(cfg, data_map, pd.DatetimeIndex(trading_dates), warmup)
    else:
        index_close = fetch_index_close(benchmark_code, warmup, cfg.end_date)

    if cfg.adaptive and cfg.dual_strategy:
        return _run_dual_strategy(cfg, data_map, trading_dates, index_close, universe)
    if cfg.adaptive:
        return _run_dual_strategy(cfg, data_map, trading_dates, index_close, universe)
    return _run_simple(cfg, data_map, trading_dates, index_close, universe)