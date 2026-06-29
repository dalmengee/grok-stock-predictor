#!/usr/bin/env python3
"""오늘의 한국 주식 매수 스크리너 CLI."""

from __future__ import annotations

import argparse
import sys

from src.korean_universe import UNIVERSES
from src.local_db import get_latest_quote_date
from src.screener import picks_to_dataframe, screen_korean_stocks

RECOMMENDATION_ICON = {
    "강력 매수": "🟢",
    "매수": "🔵",
    "관망": "🟡",
    "회피": "🔴",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="오늘 살 만한 한국 주식을 스크리닝합니다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python kr_pick.py
  python kr_pick.py --universe all --top 10
  python kr_pick.py --universe kosdaq --top 5
        """,
    )
    parser.add_argument(
        "--universe",
        choices=list(UNIVERSES),
        default="kospi_large",
        help="분석 대상 종목군 (기본: kospi_large)",
    )
    parser.add_argument("--top", type=int, default=10, help="상위 N개 표시 (기본: 10)")
    parser.add_argument("--period", default="1y", help="데이터 기간 (기본: 1y)")
    parser.add_argument("--days", type=int, default=5, help="ML 예측 기간 (기본: 5)")
    args = parser.parse_args()

    print(f"\n🇰🇷 한국 주식 매수 스크리너 ({args.universe})\n")
    print("종목 데이터를 불러오고 분석 중... (1~2분 소요)\n")

    try:
        result = screen_korean_stocks(
            universe=args.universe,
            period=args.period,
            forecast_days=args.days,
            top_n=args.top,
        )
    except Exception as e:
        print(f"❌ 오류: {e}", file=sys.stderr)
        return 1

    latest_dt = get_latest_quote_date()
    print("=" * 72)
    print(f"  분석일: {result.date}")
    if latest_dt:
        print(f"  DB 최신 일봉: {latest_dt}")
    print(f"  대상: {result.universe_label} ({result.analyzed_count}개 분석)")
    if result.failed_tickers:
        print(f"  분석 실패: {len(result.failed_tickers)}개")
    print("=" * 72)

    buy_candidates = [p for p in result.all_picks if p.recommendation in ("강력 매수", "매수")]

    if buy_candidates:
        print(f"\n✅ 오늘 매수 후보 ({len(buy_candidates)}개)\n")
        for p in buy_candidates:
            icon = RECOMMENDATION_ICON[p.recommendation]
            print(f"  {icon} [{p.recommendation}] {p.name} ({p.ticker})")
            print(f"     현재가 ₩{p.current_price:,.0f}  |  점수 {p.score}/100  |  ML 5일 예측 {p.predicted_return*100:+.1f}%")
            print(f"     RSI {p.rsi:.0f}  |  1일 {p.return_1d*100:+.1f}%  |  5일 {p.return_5d*100:+.1f}%  |  거래량 {p.volume_ratio:.1f}배")
            key_signals = p.signals[:3]
            print(f"     근거: {', '.join(key_signals)}")
            print()
    else:
        print("\n⚠️  현재 매수 추천 종목이 없습니다. 관망을 권장합니다.\n")

    print("-" * 72)
    print(f"  전체 순위 (상위 {args.top}개)\n")

    df = picks_to_dataframe(result.picks)
    print(df.to_string(index=False))
    print()
    print("=" * 72)
    print("\n⚠️  본 분석은 참고용이며, 투자 결정의 유일한 근거로 사용하지 마세요.")
    print("    반드시 본인의 판단과 리스크 관리 후 투자하세요.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())