"""벤치마크 지수 데이터."""

from __future__ import annotations

import pandas as pd

from src.local_db import _connect_readonly


def fetch_index_close(
    inds_cd: str = "001",
    start: str = "2024-01-01",
    end: str = "2025-12-31",
) -> pd.Series:
    """KOSPI/KOSDAQ 종합지수 종가 시계열 (정규화: 시작=100)."""
    conn = _connect_readonly("index_quote.sqlite3")
    try:
        rows = conn.execute(
            """
            SELECT dt, close FROM index_quote
            WHERE inds_cd = ? AND dt >= ? AND dt <= ?
            ORDER BY dt
            """,
            (inds_cd, start, end),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=["dt", "close"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt")
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty:
        return pd.Series(dtype=float)

    normalized = close / close.iloc[0] * 100
    return normalized


def benchmark_equity_curve(
    initial_cash: float,
    index_series: pd.Series,
    trading_dates: pd.DatetimeIndex,
) -> pd.Series:
    """지수 추종 buy&hold 자산 곡선."""
    if index_series.empty:
        return pd.Series(initial_cash, index=trading_dates)

    aligned = index_series.reindex(trading_dates).ffill()
    if aligned.isna().all():
        return pd.Series(initial_cash, index=trading_dates)
    aligned = aligned.bfill().fillna(100)
    return initial_cash * (aligned / aligned.iloc[0])