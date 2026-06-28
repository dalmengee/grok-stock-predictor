"""주가 데이터 수집 모듈."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Yahoo Finance에서 주가 데이터를 가져옵니다.

    Args:
        ticker: 종목 심볼 (예: AAPL, 005930.KS)
        period: 조회 기간 (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        start: 시작일 (YYYY-MM-DD)
        end: 종료일 (YYYY-MM-DD)

    Returns:
        OHLCV 데이터프레임
    """
    ticker = ticker.strip().upper()

    if start and end:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    else:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if data.empty:
        raise ValueError(f"'{ticker}' 종목의 데이터를 가져올 수 없습니다. 심볼을 확인해주세요.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    data = data[["open", "high", "low", "close", "volume"]].dropna()
    data.index = pd.to_datetime(data.index)
    return data


def get_ticker_info(ticker: str) -> dict:
    """종목 기본 정보를 반환합니다."""
    stock = yf.Ticker(ticker.strip().upper())
    info = stock.info
    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "currency": info.get("currency", "USD"),
        "exchange": info.get("exchange", ""),
        "sector": info.get("sector", ""),
    }


def default_date_range(years: int = 2) -> tuple[str, str]:
    """기본 날짜 범위를 반환합니다."""
    end = datetime.now()
    start = end - timedelta(days=365 * years)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")