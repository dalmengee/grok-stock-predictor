"""주가 데이터 수집 모듈."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from .local_db import (
    fetch_daily_quote,
    fetch_multiple_daily_quotes,
    get_company_info,
    is_korean_code,
    normalize_korean_code,
)


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    주가 데이터를 가져옵니다.
    한국 종목은 로컬 DB, 해외 종목은 Yahoo Finance를 사용합니다.
    """
    if is_korean_code(ticker):
        return fetch_daily_quote(ticker, period=period, start=start, end=end)

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
    if is_korean_code(ticker):
        return get_company_info(ticker)

    stock = yf.Ticker(ticker.strip().upper())
    info = stock.info
    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "currency": info.get("currency", "USD"),
        "exchange": info.get("exchange", ""),
        "sector": info.get("sector", ""),
    }


def fetch_multiple_stock_data(
    tickers: list[str],
    period: str = "1y",
) -> dict[str, pd.DataFrame]:
    """여러 종목의 주가 데이터를 한 번에 가져옵니다."""
    if not tickers:
        return {}

    korean_codes = []
    foreign_tickers = []
    for ticker in tickers:
        code = normalize_korean_code(ticker)
        if code:
            korean_codes.append(code)
        else:
            foreign_tickers.append(ticker.strip().upper())

    result = fetch_multiple_daily_quotes(korean_codes, period=period)

    if foreign_tickers:
        raw = yf.download(
            foreign_tickers,
            period=period,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
        )
        if not raw.empty:
            if len(foreign_tickers) == 1:
                ticker = foreign_tickers[0]
                data = raw.copy()
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
                cleaned = data[["open", "high", "low", "close", "volume"]].dropna()
                if not cleaned.empty:
                    cleaned.index = pd.to_datetime(cleaned.index)
                    result[ticker] = cleaned
            else:
                for ticker in foreign_tickers:
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    data = raw[ticker].copy()
                    data = data.rename(
                        columns={
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        }
                    )
                    cleaned = data[["open", "high", "low", "close", "volume"]].dropna()
                    if cleaned.empty:
                        continue
                    cleaned.index = pd.to_datetime(cleaned.index)
                    result[ticker] = cleaned

    return result


def default_date_range(years: int = 2) -> tuple[str, str]:
    """기본 날짜 범위를 반환합니다."""
    end = datetime.now()
    start = end - timedelta(days=365 * years)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")