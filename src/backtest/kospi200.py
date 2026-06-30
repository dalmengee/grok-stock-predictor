"""KOSPI200 유니버스 및 동일가중 벤치마크."""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from src.local_db import _connect_readonly, fetch_multiple_daily_quotes, list_active_companies

KOSPI200_SIZE = 200
DEFAULT_EXCLUDE_CODES = frozenset({"005930", "000660"})
PREFERRED_NAME = re.compile(r"\d*우[A-C]?$")
BENCHMARK_LABEL = "KOSPI200 동일가중"


@lru_cache(maxsize=1)
def _company_meta() -> dict[str, dict]:
    conn = _connect_readonly("master.sqlite3")
    try:
        rows = conn.execute(
            """
            SELECT code, name, market_code, up_name, list_count
            FROM companies
            WHERE is_active = 1 AND market_code = '0'
            """
        ).fetchall()
    finally:
        conn.close()
    return {
        code: {
            "name": name,
            "market_code": market_code,
            "sector": up_name or "",
            "list_count": float(list_count or 0),
        }
        for code, name, market_code, up_name, list_count in rows
    }


def is_preferred_stock(name: str) -> bool:
    return bool(PREFERRED_NAME.search(name or ""))


def market_cap_at_date(
    code: str,
    as_of: pd.Timestamp,
    data_map: dict[str, pd.DataFrame],
    meta: dict[str, dict] | None = None,
) -> float:
    """종가 × 상장주식수로 시가총액 추정."""
    meta = meta or _company_meta()
    df = data_map.get(code)
    if df is None or as_of not in df.index:
        hist = df[df.index <= as_of] if df is not None else None
        if hist is None or hist.empty:
            return 0.0
        close = float(hist.iloc[-1]["close"])
    else:
        close = float(df.loc[as_of, "close"])
    shares = meta.get(code, {}).get("list_count", 0)
    return close * shares if shares > 0 else 0.0


def select_kospi200_codes(
    as_of: pd.Timestamp,
    data_map: dict[str, pd.DataFrame],
    exclude: frozenset[str] | set[str] | None = None,
    for_strategy: bool = False,
) -> list[str]:
    """
    시가총액 상위 200개 코스피 종목 (우선주 제외).
    for_strategy=True 이면 삼성전자·SK하이닉스 제외.
    """
    exclude = set(exclude or set())
    if for_strategy:
        exclude |= set(DEFAULT_EXCLUDE_CODES)

    meta = _company_meta()
    ranked: list[tuple[str, float]] = []
    for code, info in meta.items():
        if code in exclude or is_preferred_stock(info["name"]):
            continue
        cap = market_cap_at_date(code, as_of, data_map, meta)
        if cap > 0:
            ranked.append((code, cap))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [code for code, _ in ranked[:KOSPI200_SIZE]]


def _codes_to_stocks(codes: list[str]) -> list:
    from src.korean_universe import KoreanStock

    companies = {c["code"]: c for c in list_active_companies(markets=("0",))}
    stocks = []
    for code in codes:
        info = companies.get(code)
        if info:
            stocks.append(KoreanStock(code, info["name"], info.get("sector") or "기타", info.get("market") or "KOSPI"))
        else:
            stocks.append(KoreanStock(code, code, "기타", "KOSPI"))
    return stocks


def get_kospi200_universe(
    exclude_strategy: bool = True,
    as_of: pd.Timestamp | None = None,
    data_map: dict[str, pd.DataFrame] | None = None,
) -> tuple[str, list]:
    """정적(최신 시총) 또는 시점 기준 KOSPI200 유니버스."""
    if as_of is not None and data_map is not None:
        codes = select_kospi200_codes(as_of, data_map, for_strategy=exclude_strategy)
    else:
        codes = _static_kospi200_codes(exclude_strategy=exclude_strategy)
    label = "KOSPI200 (삼성·SK하이닉스 제외)" if exclude_strategy else "KOSPI200"
    return label, _codes_to_stocks(codes)


def _static_kospi200_codes(exclude_strategy: bool = True) -> list[str]:
    """최신 거래일 기준 정적 멤버십 (초기 로딩용)."""
    conn_daily = _connect_readonly("daily_quote.sqlite3")
    try:
        closes = dict(
            conn_daily.execute(
                """
                SELECT d.code, d.close
                FROM daily_quote d
                JOIN (SELECT code, MAX(dt) AS mx FROM daily_quote GROUP BY code) m
                  ON d.code = m.code AND d.dt = m.mx
                WHERE d.close > 0
                """
            ).fetchall()
        )
    finally:
        conn_daily.close()

    meta = _company_meta()
    exclude = set(DEFAULT_EXCLUDE_CODES) if exclude_strategy else set()
    ranked: list[tuple[str, float]] = []
    for code, info in meta.items():
        if code in exclude or is_preferred_stock(info["name"]):
            continue
        close = closes.get(code)
        shares = info.get("list_count", 0)
        if close and shares:
            ranked.append((code, close * shares))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [code for code, _ in ranked[:KOSPI200_SIZE]]


def build_equal_weight_index(
    codes: list[str],
    start: str,
    end: str,
    data_map: dict[str, pd.DataFrame] | None = None,
) -> pd.Series:
    """
    동일가중 지수 시계열 (시작=100).
    일별 구성종목 평균 수익률로 누적.
    """
    if data_map is None:
        warmup = (pd.Timestamp(start) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        data_map = fetch_multiple_daily_quotes(codes, start=warmup, end=end, extended=False)

    closes: dict[str, pd.Series] = {}
    for code in codes:
        df = data_map.get(code)
        if df is None or df.empty:
            continue
        s = df["close"].copy()
        s.index = pd.DatetimeIndex(s.index)
        closes[code] = s

    if not closes:
        return pd.Series(dtype=float)

    panel = pd.DataFrame(closes).sort_index()
    panel = panel.loc[(panel.index >= start) & (panel.index <= end)]
    if panel.empty:
        return pd.Series(dtype=float)

    daily_ret = panel.pct_change().mean(axis=1, skipna=True)
    daily_ret = daily_ret.fillna(0.0)
    index_level = (1 + daily_ret).cumprod() * 100
    if not index_level.empty:
        index_level.iloc[0] = 100.0
    return index_level


def build_kospi200_equal_weight_benchmark(
    start: str,
    end: str,
    as_of: pd.Timestamp | None = None,
    data_map: dict[str, pd.DataFrame] | None = None,
    exclude_codes: frozenset[str] | set[str] | None = None,
) -> pd.Series:
    """KOSPI200 전체 동일가중 벤치마크 (전략 제외종목과 무관)."""
    if data_map is None:
        codes = _static_kospi200_codes(exclude_strategy=False)
        warmup = (pd.Timestamp(start) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        data_map = fetch_multiple_daily_quotes(codes, start=warmup, end=end, extended=False)
    else:
        ref_date = as_of or pd.Timestamp(end)
        codes = select_kospi200_codes(ref_date, data_map, exclude=exclude_codes, for_strategy=False)

    return build_equal_weight_index(codes, start, end, data_map)