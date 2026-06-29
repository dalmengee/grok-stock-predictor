#!/usr/bin/env python3
"""투자 의사결정 CLI."""

from __future__ import annotations

import argparse
import sys

from src.invest_analysis import analyze_for_investment, suggest_order_amount
from src.market_brief import brief_buy_recommendations, generate_market_brief
from src.portfolio import add_holding, analyze_portfolio, load_portfolio, remove_holding, set_cash


def _fmt_won(v: float) -> str:
    return f"₩{v:,.0f}"


def cmd_analyze(args: argparse.Namespace) -> int:
    report = analyze_for_investment(args.code, portfolio_value=args.portfolio)
    order = suggest_order_amount(report, args.portfolio)

    print(f"\n📊 {report.name} ({report.code}) 투자 분석\n")
    print("=" * 60)
    print(f"  판단:     {report.action}  (신뢰도 {report.confidence}%, 점수 {report.score})")
    print(f"  리스크:   {report.risk_level}")
    print(f"  현재가:   {_fmt_won(report.current_price)}")
    print(f"  진입가:   {_fmt_won(report.entry_price)}")
    print(f"  손절가:   {_fmt_won(report.stop_loss)}")
    print(f"  목표가:   {_fmt_won(report.target_price)}")
    print(f"  ML 5일:   {report.predicted_return*100:+.1f}%")
    print(f"  비중제안: 포트폴리오의 {report.position_pct}%")
    print("-" * 60)
    print("  리스크 지표:")
    print(f"    변동성(20일): {report.volatility_20d}%")
    print(f"    최대낙폭(60일): {report.max_drawdown_60d}%")
    print(f"    RSI: {report.rsi:.0f}")
    print("-" * 60)
    print("  수급 (5일 순매수, 주):")
    for k, v in report.flow_5d.items():
        print(f"    {k}: {v:+,}")
    print("-" * 60)
    if report.reasons:
        print("  근거:")
        for r in report.reasons:
            print(f"    ✓ {r}")
    if report.cautions:
        print("  주의:")
        for c in report.cautions:
            print(f"    ⚠ {c}")
    if order["quantity"] > 0:
        print("-" * 60)
        print(f"  매수 제안: {order['quantity']}주 ({_fmt_won(order['amount'])})")
        print(f"  최대 손실(손절 시): {_fmt_won(order['max_loss'])}")
    print("=" * 60)
    print("\n⚠️  참고용 분석이며, 투자 결정은 본인 책임입니다.\n")
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    brief = generate_market_brief(universe=args.universe, portfolio_value=args.portfolio)
    recs = brief_buy_recommendations(brief, args.portfolio)

    print(f"\n📋 오늘의 투자 브리핑 ({brief.date})\n")
    print("=" * 60)
    print(f"  DB 최신: {brief.db_latest}")
    print(f"  시장 분위기: {brief.market_mood}")
    ps = brief.portfolio_summary
    print(f"  포트폴리오: {_fmt_won(ps['total_value'])} (주식 {_fmt_won(ps['stock_value'])}, 현금 {_fmt_won(ps['cash'])})")
    if ps["holdings_count"]:
        print(f"  평가손익: {_fmt_won(ps['total_pnl'])} ({ps['total_pnl_pct']:+.1f}%)")
    print("=" * 60)

    if recs:
        print("\n💡 매수 아이디어\n")
        for r in recs:
            o = r["order"]
            print(f"  [{r['action']}] {r['name']} ({r['code']}) — 점수 {r['score']}")
            if o["quantity"]:
                print(f"     {o['quantity']}주 / {_fmt_won(o['amount'])} / 손절 {_fmt_won(o['stop_loss'])}")
            for reason in r["reasons"]:
                print(f"     · {reason}")
            print()
    else:
        print("\n⚠️  현재 적극 매수 아이디어가 없습니다.\n")

    if brief.portfolio_actions:
        print("📁 보유 종목 액션\n")
        for a in brief.portfolio_actions:
            line = f"  {a['name']} ({a['code']}): {a['action']} (수익률 {a['pnl_pct']:+.1f}%)"
            if a.get("alert"):
                line += f"\n     ⚠ {a['alert']}"
            print(line)
        print()

    for note in brief.notes:
        print(f"  · {note}")
    print("\n⚠️  참고용 브리핑이며, 투자 결정은 본인 책임입니다.\n")
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    if args.action == "show":
        pf = analyze_portfolio()
        print(f"\n💼 포트폴리오 (총 {_fmt_won(pf.total_value)})\n")
        print(f"  현금: {_fmt_won(pf.cash)}  |  주식: {_fmt_won(pf.stock_value)}  |  손익: {_fmt_won(pf.total_pnl)} ({pf.total_pnl_pct:+.1f}%)")
        if not pf.holdings:
            print("\n  보유 종목 없음\n")
            return 0
        print()
        for h in pf.holdings:
            print(f"  {h.name} ({h.code})")
            print(f"    {h.quantity}주 × {_fmt_won(h.avg_price)} → {_fmt_won(h.current_price)}")
            print(f"    평가: {_fmt_won(h.market_value)}  손익: {_fmt_won(h.pnl)} ({h.pnl_pct:+.1f}%)  비중: {h.weight:.1f}%")
            if h.action:
                print(f"    판단: {h.action}  손절: {_fmt_won(h.stop_loss)}  목표: {_fmt_won(h.target_price)}")
            print()
        return 0

    if args.action == "add":
        add_holding(args.code, args.qty, args.price)
        print(f"✅ {args.code} {args.qty}주 @ {_fmt_won(args.price)} 추가")
        return 0

    if args.action == "sell":
        remove_holding(args.code, args.qty)
        qty_msg = f"{args.qty}주" if args.qty else "전량"
        print(f"✅ {args.code} {qty_msg} 매도 반영")
        return 0

    if args.action == "cash":
        set_cash(args.amount)
        print(f"✅ 현금 잔고 → {_fmt_won(args.amount)}")
        return 0

    return 1


def main() -> int:
    pf = load_portfolio()
    default_portfolio = pf.cash + sum(h.quantity * h.avg_price for h in pf.holdings)

    parser = argparse.ArgumentParser(description="주식 투자 의사결정 도구")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="종목 투자 분석")
    p_analyze.add_argument("code", help="종목코드 (예: 005930)")
    p_analyze.add_argument("--portfolio", type=float, default=default_portfolio, help="포트폴리오 총액")
    p_analyze.set_defaults(func=cmd_analyze)

    p_brief = sub.add_parser("brief", help="오늘의 투자 브리핑")
    p_brief.add_argument("--universe", default="kospi_large", choices=["kospi_large", "kosdaq", "all"])
    p_brief.add_argument("--portfolio", type=float, default=default_portfolio)
    p_brief.set_defaults(func=cmd_brief)

    p_pf = sub.add_parser("portfolio", help="포트폴리오 관리")
    p_pf_sub = p_pf.add_subparsers(dest="action", required=True)

    p_show = p_pf_sub.add_parser("show", help="포트폴리오 조회")
    p_show.set_defaults(func=cmd_portfolio)

    p_add = p_pf_sub.add_parser("add", help="종목 추가")
    p_add.add_argument("code")
    p_add.add_argument("qty", type=int)
    p_add.add_argument("price", type=float)
    p_add.set_defaults(func=cmd_portfolio)

    p_sell = p_pf_sub.add_parser("sell", help="종목 매도")
    p_sell.add_argument("code")
    p_sell.add_argument("--qty", type=int, default=None)
    p_sell.set_defaults(func=cmd_portfolio)

    p_cash = p_pf_sub.add_parser("cash", help="현금 설정")
    p_cash.add_argument("amount", type=float)
    p_cash.set_defaults(func=cmd_portfolio)

    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as e:
        print(f"❌ 오류: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())