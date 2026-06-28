#!/usr/bin/env python3
"""주가 예측 CLI 프로그램."""

from __future__ import annotations

import argparse
import sys

from src.data_fetcher import fetch_stock_data, get_ticker_info
from src.model import train_and_predict
from src.visualizer import save_matplotlib_chart


def format_price(price: float, currency: str = "USD") -> str:
    if currency == "KRW":
        return f"₩{price:,.0f}"
    return f"${price:,.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="머신러닝 기반 주가 예측 프로그램",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python predict.py AAPL
  python predict.py 005930.KS --days 10
  python predict.py TSLA --model gradient_boosting --period 3y --chart output.png
        """,
    )
    parser.add_argument("ticker", help="종목 심볼 (예: AAPL, 005930.KS, TSLA)")
    parser.add_argument("--days", type=int, default=5, help="예측 기간 (일, 기본: 5)")
    parser.add_argument(
        "--model",
        choices=["random_forest", "gradient_boosting"],
        default="random_forest",
        help="사용할 모델 (기본: random_forest)",
    )
    parser.add_argument("--period", default="2y", help="데이터 조회 기간 (기본: 2y)")
    parser.add_argument("--chart", metavar="FILE", help="차트 저장 경로 (예: chart.png)")
    args = parser.parse_args()

    print(f"\n📈 {args.ticker} 주가 예측을 시작합니다...\n")

    try:
        info = get_ticker_info(args.ticker)
        df = fetch_stock_data(args.ticker, period=args.period)
        result = train_and_predict(
            df,
            ticker=args.ticker,
            forecast_days=args.days,
            model_name=args.model,
        )
    except Exception as e:
        print(f"❌ 오류: {e}", file=sys.stderr)
        return 1

    currency = info.get("currency", "USD")
    return_pct = result.predicted_return * 100

    print("=" * 50)
    print(f"  종목: {info['name']} ({result.ticker})")
    print(f"  모델: {args.model}")
    print("=" * 50)
    print(f"  현재가:     {format_price(result.current_price, currency)}")
    print(f"  예측가 ({result.forecast_days}일 후): {format_price(result.predicted_price, currency)}")
    print(f"  예상 수익률: {return_pct:+.2f}%")
    print(f"  예측 방향:   {result.direction}")
    print("-" * 50)
    print("  모델 성능 (교차 검증):")
    print(f"    MAE:  {format_price(result.metrics['mae'], currency)}")
    print(f"    RMSE: {format_price(result.metrics['rmse'], currency)}")
    print(f"    R²:   {result.metrics['r2']:.4f}")
    print("-" * 50)
    print("  주요 특성 (상위 5개):")
    for _, row in result.feature_importance.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    print("=" * 50)
    print("\n⚠️  본 예측은 참고용이며, 투자 결정의 유일한 근거로 사용하지 마세요.\n")

    if args.chart:
        save_matplotlib_chart(result, args.chart)
        print(f"📊 차트가 저장되었습니다: {args.chart}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())