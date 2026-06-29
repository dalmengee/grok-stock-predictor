#!/usr/bin/env python3
"""전략 백테스트 및 검증 CLI."""

from __future__ import annotations

import argparse
import sys

from src.backtest.config import BacktestConfig
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
    print(f"    평균보유:   {m.avg_hold_days:.1f}일")
    print(f"    최종자산:   ₩{m.final_value:,.0f}")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash,
        max_positions=args.positions,
        entry_score=args.entry,
        exit_score=args.exit,
        stop_loss_pct=args.stop / 100,
        take_profit_pct=args.profit / 100,
        rebalance_days=args.rebalance,
    )
    print(f"\n🔬 백테스트 실행 ({args.universe}, {args.start} ~ {args.end})\n")
    result = run_backtest(universe=args.universe, config=cfg)
    m = result.metrics

    print("=" * 60)
    print("  전략: 점수 기반 유니버스 로테이션")
    print(f"  진입 {args.entry}점 / 청산 {args.exit}점 / 최대 {args.positions}종목")
    print("=" * 60)
    _print_metrics("성과", m)

    sells = [t for t in result.trades if t["side"] == "sell"]
    if sells:
        print("\n  최근 매도 5건:")
        for t in sells[-5:]:
            print(
                f"    {t['date']} {t['code']} {t['reason']} "
                f"₩{t['pnl']:+,.0f} ({t['hold_days']}일)"
            )
    print("\n" + "=" * 60)
    if m.alpha_pct > 0 and m.sharpe_ratio > 0.5:
        print("  ✅ KOSPI 대비 초과수익 — 전략 유효성 있음 (과거 기준)")
    elif m.total_return_pct > 0:
        print("  ⚠️  수익은 있으나 벤치마크 대비 열세 — 개선 필요")
    else:
        print("  ❌ 손실 구간 — 파라미터 조정 또는 전략 변경 필요")
    print("=" * 60)
    print("\n⚠️  과거 성과가 미래 수익을 보장하지 않습니다.\n")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    print(f"\n🔬 워크포워드 검증 ({args.start} ~ {args.end}, train {args.train_ratio:.0%})\n")
    wf = run_walk_forward(
        universe=args.universe,
        start_date=args.start,
        end_date=args.end,
        train_ratio=args.train_ratio,
    )
    print("=" * 60)
    print(f"  분할일: {wf.split_date}")
    _print_metrics("In-Sample (학습 구간)", wf.is_metrics)
    _print_metrics("Out-of-Sample (검증 구간)", wf.oos_metrics)
    print(f"\n  판정: {wf.summary}")
    print("=" * 60 + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="전략 백테스트 및 워크포워드 검증")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="백테스트 실행")
    p_run.add_argument("--universe", default="kospi_large", choices=["kospi_large", "kosdaq", "all"])
    p_run.add_argument("--start", default="2024-01-01")
    p_run.add_argument("--end", default="2025-06-30")
    p_run.add_argument("--cash", type=float, default=10_000_000)
    p_run.add_argument("--positions", type=int, default=5)
    p_run.add_argument("--entry", type=float, default=60, help="진입 점수")
    p_run.add_argument("--exit", type=float, default=45, help="청산 점수")
    p_run.add_argument("--stop", type=float, default=8, help="손절 %")
    p_run.add_argument("--profit", type=float, default=15, help="익절 %")
    p_run.add_argument("--rebalance", type=int, default=5, help="리밸런싱 주기(일)")
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("validate", help="워크포워드 검증")
    p_val.add_argument("--universe", default="kospi_large", choices=["kospi_large", "kosdaq", "all"])
    p_val.add_argument("--start", default="2023-01-01")
    p_val.add_argument("--end", default="2025-06-30")
    p_val.add_argument("--train-ratio", type=float, default=0.7)
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as e:
        print(f"❌ 오류: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())