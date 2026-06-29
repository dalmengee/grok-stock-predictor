#!/usr/bin/env python3
"""전략 백테스트 및 검증 CLI."""

from __future__ import annotations

import argparse
import sys

from src.backtest.config import BacktestConfig, DEFAULT_DATA_END, DEFAULT_DATA_START
from src.backtest.position_plan import REGIME_PLANS
from src.backtest.engine import run_backtest
from src.backtest.validation import run_walk_forward


def _print_metrics(label: str, m) -> None:
    print(f"\n  [{label}]")
    print(f"    총수익률:   {m.total_return_pct:+.2f}%")
    print(f"    CAGR:       {m.cagr_pct:+.2f}%")
    print(f"    KOSPI 대비: {m.alpha_pct:+.2f}%p (벤치마크 {m.benchmark_return_pct:+.2f}%)")
    print(f"    샤프비율:   {m.sharpe_ratio:.2f}")
    print(f"    최대낙폭:   {m.max_drawdown_pct:.2f}%")
    print(f"    승률:       {m.win_rate_pct:.1f}%")
    print(f"    손익비:     {m.profit_factor:.2f}")
    print(f"    거래횟수:   {m.total_trades}회")
    print(f"    최종자산:   ₩{m.final_value:,.0f}")


def _verdict(m) -> str:
    if m.alpha_pct > 0 and m.sharpe_ratio > 0.3:
        return "✅ KOSPI 대비 초과수익"
    if m.total_return_pct > m.benchmark_return_pct:
        return "✅ 벤치마크 수익률 상회"
    if m.total_return_pct > 0:
        return "⚠️  절대수익은 있으나 벤치마크 열세"
    return "❌ 손실 구간"


def cmd_run(args: argparse.Namespace) -> int:
    cfg = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash,
        adaptive=not args.simple,
        rebalance_days=args.rebalance,
        crisis_dd_trigger=args.crisis_dd / 100,
    )
    mode = "이중 전략 (상승/하락 국면 + MDD 15%)" if cfg.adaptive and cfg.dual_strategy else (
        "적응형" if cfg.adaptive else "단순 로테이션"
    )
    print(f"\n🔬 백테스트 — {mode}")
    print(f"   {args.universe}, {args.start} ~ {args.end}")
    if cfg.adaptive and cfg.dual_strategy:
        print(f"   MDD 한도: {cfg.max_drawdown_limit*100:.0f}% | 위기 DD: {cfg.crisis_dd_trigger*100:.0f}%\n")
    else:
        print()

    result = run_backtest(universe=args.universe, config=cfg)
    m = result.metrics

    print("=" * 60)
    _print_metrics("성과", m)

    if not result.regime_log.empty and cfg.adaptive:
        counts = result.regime_log["regime"].value_counts()
        print("\n  [시장 국면 분포]")
        for regime, cnt in counts.items():
            pct = cnt / len(result.regime_log) * 100
            plan = REGIME_PLANS.get(regime)
            sizing = f"주식 {plan.total_stock_pct:.0f}% / 종목당 {plan.per_position_pct:.0f}%" if plan else ""
            print(f"    {regime}: {pct:.0f}%  ({sizing})")
        if not result.exposure_log.empty:
            print(f"    평균 주식비중: {result.exposure_log.mean()*100:.1f}%")

    mdd_ok = abs(m.max_drawdown_pct) <= cfg.max_drawdown_limit * 100
    print(f"\n  MDD 제한: {'✅' if mdd_ok else '❌'} {abs(m.max_drawdown_pct):.1f}% / {cfg.max_drawdown_limit*100:.0f}%")
    print(f"  판정: {_verdict(m)}")
    print("=" * 60 + "\n")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """단순 vs 적응형 전략 비교."""
    base = dict(
        start_date=args.start, end_date=args.end,
        initial_cash=args.cash, rebalance_days=args.rebalance,
    )
    print(f"\n🔬 전략 비교 ({args.start} ~ {args.end})\n")
    print("=" * 60)

    for label, adaptive in [("단순 로테이션", False), ("적응형 전략", True)]:
        cfg = BacktestConfig(**base, adaptive=adaptive)
        r = run_backtest(args.universe, cfg)
        print(f"\n  ▶ {label}")
        print(f"    수익률 {r.metrics.total_return_pct:+.1f}% | 알파 {r.metrics.alpha_pct:+.1f}%p | MDD {r.metrics.max_drawdown_pct:.1f}% | 샤프 {r.metrics.sharpe_ratio:.2f}")

    print("\n" + "=" * 60 + "\n")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    print(f"\n🔬 워크포워드 검증 (적응형, {args.start} ~ {args.end})\n")
    wf = run_walk_forward(
        universe=args.universe,
        start_date=args.start,
        end_date=args.end,
        train_ratio=args.train_ratio,
        base_config=BacktestConfig(adaptive=True),
    )
    print("=" * 60)
    print(f"  분할일: {wf.split_date}")
    _print_metrics("In-Sample", wf.is_metrics)
    _print_metrics("Out-of-Sample", wf.oos_metrics)
    print(f"\n  판정: {wf.summary}")
    print("=" * 60 + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="전략 백테스트")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("run", "백테스트 실행"),
        ("compare", "단순 vs 적응형 비교"),
        ("validate", "워크포워드 검증"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--universe", default="kospi_large", choices=["kospi_large", "kosdaq", "all"])
        p.add_argument("--start", default=DEFAULT_DATA_START)
        p.add_argument("--end", default=DEFAULT_DATA_END)
        p.add_argument("--cash", type=float, default=10_000_000)
        p.add_argument("--rebalance", type=int, default=5)
        if name == "run":
            p.add_argument("--simple", action="store_true", help="단순 전략 (구버전)")
            p.add_argument("--crisis-dd", type=float, default=12, help="위기 DD 차단 %")
        if name == "validate":
            p.add_argument("--train-ratio", type=float, default=0.7)
        p.set_defaults(func={"run": cmd_run, "compare": cmd_compare, "validate": cmd_validate}[name])

    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as e:
        print(f"❌ 오류: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())