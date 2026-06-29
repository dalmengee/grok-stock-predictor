"""적응형 백테스트 시뮬레이션 엔진."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.backtest.benchmark import benchmark_equity_curve, fetch_index_close
from src.backtest.config import BacktestConfig
from src.backtest.metrics import PerformanceMetrics, compute_metrics
from src.backtest.position_sizing import allocate_by_scores, stock_volatility
from src.backtest.regime import MarketRegime, detect_regime
from src.backtest.risk import RiskManager, should_trailing_stop, target_stock_value
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


def _apply_slippage(price: float, side: str, slippage: float) -> float:
    return price * (1 + slippage) if side == "buy" else price * (1 - slippage)


def _trade_cost(amount: float, side: str, cfg: BacktestConfig) -> float:
    fee = amount * cfg.commission_rate
    if side == "sell":
        fee += amount * cfg.sell_tax_rate
    return fee


def _sell_position(
    code: str,
    pos: Position,
    dt: pd.Timestamp,
    raw_price: float,
    cfg: BacktestConfig,
    reason: str,
) -> tuple[float, dict]:
    price = _apply_slippage(raw_price, "sell", cfg.slippage_pct)
    amount = price * pos.quantity
    fee = _trade_cost(amount, "sell", cfg)
    pnl = (price - pos.entry_price) * pos.quantity - fee
    trade = {
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
        "regime": pos.regime,
    }
    return amount - fee, trade


def _portfolio_value(cash: float, positions: dict[str, Position], data_map: dict, dt: pd.Timestamp) -> float:
    stock_val = 0.0
    for code, pos in positions.items():
        df = data_map.get(code)
        if df is not None and dt in df.index:
            stock_val += float(df.loc[dt, "close"]) * pos.quantity
        else:
            stock_val += pos.entry_price * pos.quantity
    return cash + stock_val


def _load_universe_data(universe: str, start: str, end: str) -> dict[str, pd.DataFrame]:
    _, stocks = get_universe(universe)
    codes = [s.ticker for s in stocks]
    warmup = (pd.Timestamp(start) - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    return fetch_multiple_daily_quotes(codes, start=warmup, end=end, extended=True)


def _run_simple(cfg: BacktestConfig, data_map: dict, trading_dates: list, index_close: pd.Series) -> BacktestResult:
    """기존 단순 로테이션 (adaptive=False)."""
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
            hist = df[df.index <= dt]
            sc = score_at_date(hist, use_flow=cfg.use_flow_score) or 0
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
                proceeds, trade = _sell_position(code, pos, dt, price, cfg, reason)
                del positions[code]
                cash += proceeds
                trades.append(trade)

        if last_rebalance is None or (dt - last_rebalance).days >= cfg.rebalance_days:
            last_rebalance = dt
            rankings = rank_universe(data_map, dt, cfg.use_flow_score, cfg.min_history_days)
            candidates = [(c, s) for c, s in rankings if s >= cfg.entry_score]
            targets = {c for c, _ in candidates[: cfg.max_positions]}
            for code in list(positions):
                if code not in targets:
                    proceeds, trade = _sell_position(
                        code, positions.pop(code), dt,
                        float(data_map[code].loc[dt, "close"]), cfg, "rebalance_out",
                    )
                    cash += proceeds
                    trades.append(trade)
            slots = cfg.max_positions - len(positions)
            for code, score in candidates:
                if code in positions or slots <= 0:
                    continue
                price = float(data_map[code].loc[dt, "close"])
                price = _apply_slippage(price, "buy", cfg.slippage_pct)
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

    return _finalize(cfg, equity_records, trades, index_close, pd.DataFrame(), pd.Series())


def _run_adaptive(
    cfg: BacktestConfig,
    data_map: dict,
    trading_dates: list,
    index_close: pd.Series,
) -> BacktestResult:
    """국면 전환 + 리스크 관리 + 포지션 사이징."""
    cash = cfg.initial_cash
    positions: dict[str, Position] = {}
    trades: list[dict] = []
    equity_records: list[tuple[pd.Timestamp, float]] = []
    regime_rows: list[dict] = []
    exposure_rows: list[tuple[pd.Timestamp, float]] = []
    last_rebalance = None
    risk_mgr = RiskManager(cfg.initial_cash, cfg.crisis_dd_threshold, cfg.recovery_dd_threshold)

    for dt in trading_dates:
        equity = _portfolio_value(cash, positions, data_map, dt)
        risk = risk_mgr.update(equity)
        regime_state = detect_regime(
            index_close, dt, risk.current_drawdown, cfg.crisis_dd_threshold,
        )
        if risk.in_crisis:
            from src.backtest.regime import REGIME_PROFILES
            regime_state = REGIME_PROFILES[MarketRegime.CRISIS]

        regime_rows.append({
            "date": dt,
            "regime": regime_state.regime.value,
            "exposure_cap": regime_state.exposure_cap,
            "strategy": regime_state.strategy_mode,
            "drawdown": risk.current_drawdown,
        })

        # --- 1. 청산: 손절/익절/trailing/점수/국면전환 ---
        for code, pos in list(positions.items()):
            df = data_map.get(code)
            if df is None or dt not in df.index:
                continue
            price = float(df.loc[dt, "close"])
            pos.high_watermark = max(pos.high_watermark, price)
            ret = price / pos.entry_price - 1
            hist = df[df.index <= dt]
            sc = score_at_date(hist, cfg.use_flow_score, regime_state.strategy_mode) or 0

            reason = None
            if ret <= -regime_state.stop_loss_pct:
                reason = "stop_loss"
            elif ret >= regime_state.take_profit_pct:
                reason = "take_profit"
            elif should_trailing_stop(price, pos.high_watermark, regime_state.trailing_stop_pct) and ret > 0:
                reason = "trailing_stop"
            elif (dt - pos.entry_date).days >= cfg.max_hold_days:
                reason = "max_hold"
            elif sc < regime_state.entry_score - 15:
                reason = "score_exit"
            elif regime_state.strategy_mode == "cash":
                reason = "crisis_exit"

            if reason:
                proceeds, trade = _sell_position(code, pos, dt, price, cfg, reason)
                del positions[code]
                cash += proceeds
                trades.append(trade)

        equity = _portfolio_value(cash, positions, data_map, dt)
        stock_val = equity - cash
        exposure = stock_val / equity if equity > 0 else 0
        exposure_rows.append((dt, exposure))

        # --- 2. 국면 전환 시 과잉 비중 축소 ---
        tgt_val = target_stock_value(equity, regime_state.exposure_cap)
        if stock_val > tgt_val * 1.05 and positions:
            sorted_pos = sorted(
                positions.items(),
                key=lambda x: score_at_date(
                    data_map[x[0]][data_map[x[0]].index <= dt],
                    cfg.use_flow_score,
                    regime_state.strategy_mode,
                ) or 0,
            )
            for code, pos in sorted_pos:
                if stock_val <= tgt_val:
                    break
                if dt not in data_map[code].index:
                    continue
                price = float(data_map[code].loc[dt, "close"])
                proceeds, trade = _sell_position(code, pos, dt, price, cfg, "exposure_reduce")
                del positions[code]
                cash += proceeds
                trades.append(trade)
                stock_val -= pos.quantity * price

        # --- 3. 리밸런싱 & 신규 매수 ---
        do_rebalance = last_rebalance is None or (dt - last_rebalance).days >= cfg.rebalance_days
        if do_rebalance and regime_state.strategy_mode != "cash":
            last_rebalance = dt
            rankings = rank_universe(
                data_map, dt, cfg.use_flow_score, cfg.min_history_days,
                mode=regime_state.strategy_mode,
                min_score=regime_state.entry_score,
            )
            candidates = rankings[: regime_state.max_positions]
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
            stock_val = equity - cash
            remaining_budget = max(0, target_stock_value(equity, regime_state.exposure_cap) - stock_val)

            new_candidates = [(c, s) for c, s in candidates if c not in positions]
            if new_candidates and remaining_budget > 0:
                prices = {
                    c: float(data_map[c].loc[dt, "close"]) for c, _ in new_candidates if dt in data_map[c].index
                }
                vols = {
                    c: stock_volatility(data_map[c][data_map[c].index <= dt])
                    for c, _ in new_candidates if c in prices
                }
                alloc = allocate_by_scores(
                    equity=equity,
                    exposure_cap=remaining_budget / equity if equity > 0 else 0,
                    candidates=[(c, s) for c, s in new_candidates if c in prices],
                    prices=prices,
                    vols=vols,
                    stop_loss_pct=regime_state.stop_loss_pct,
                    risk_per_trade=cfg.risk_per_trade,
                )
                for code, qty in alloc.items():
                    if qty <= 0:
                        continue
                    raw = prices[code]
                    price = _apply_slippage(raw, "buy", cfg.slippage_pct)
                    cost = price * qty + _trade_cost(price * qty, "buy", cfg)
                    if cost > cash:
                        qty = int((cash - _trade_cost(price, "buy", cfg)) / price)
                        if qty <= 0:
                            continue
                        cost = price * qty + _trade_cost(price * qty, "buy", cfg)
                    cash -= cost
                    score = next(s for c, s in new_candidates if c == code)
                    positions[code] = Position(
                        code, qty, price, dt, score, price, regime_state.regime.value,
                    )
                    trades.append({
                        "date": dt.strftime("%Y-%m-%d"), "code": code, "side": "buy",
                        "price": price, "quantity": qty, "amount": price * qty,
                        "reason": f"{regime_state.regime.value}_entry_{score}",
                        "regime": regime_state.regime.value,
                    })

        equity_records.append((dt, _portfolio_value(cash, positions, data_map, dt)))

    regime_df = pd.DataFrame(regime_rows).set_index("date") if regime_rows else pd.DataFrame()
    exposure_series = pd.Series(
        [e for _, e in exposure_rows],
        index=pd.DatetimeIndex([d for d, _ in exposure_rows]),
    ) if exposure_rows else pd.Series(dtype=float)

    return _finalize(cfg, equity_records, trades, index_close, regime_df, exposure_series)


def _finalize(
    cfg: BacktestConfig,
    equity_records: list,
    trades: list,
    index_close: pd.Series,
    regime_df: pd.DataFrame,
    exposure_series: pd.Series,
) -> BacktestResult:
    equity_curve = pd.Series(
        [v for _, v in equity_records],
        index=pd.DatetimeIndex([d for d, _ in equity_records]),
    )
    index_series = fetch_index_close("001", cfg.start_date, cfg.end_date)
    bench_curve = benchmark_equity_curve(cfg.initial_cash, index_series, equity_curve.index)
    metrics = compute_metrics(equity_curve, trades, index_series, cfg.initial_cash)
    return BacktestResult(
        config=cfg, metrics=metrics, equity_curve=equity_curve,
        benchmark_curve=bench_curve, trades=trades,
        regime_log=regime_df, exposure_log=exposure_series,
    )


def run_backtest(
    universe: str = "kospi_large",
    config: BacktestConfig | None = None,
    benchmark_code: str = "001",
) -> BacktestResult:
    """백테스트 실행 (adaptive=True 기본)."""
    cfg = config or BacktestConfig()
    data_map = _load_universe_data(universe, cfg.start_date, cfg.end_date)

    all_dates: set[pd.Timestamp] = set()
    for df in data_map.values():
        mask = (df.index >= cfg.start_date) & (df.index <= cfg.end_date)
        all_dates.update(df.index[mask])
    trading_dates = sorted(all_dates)
    if not trading_dates:
        raise ValueError("백테스트 기간에 거래일이 없습니다.")

    warmup = (pd.Timestamp(cfg.start_date) - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    index_close = fetch_index_close(benchmark_code, warmup, cfg.end_date)

    if cfg.adaptive:
        return _run_adaptive(cfg, data_map, trading_dates, index_close)
    return _run_simple(cfg, data_map, trading_dates, index_close)