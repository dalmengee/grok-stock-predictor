"""키움 로컬 SQLite DB에서 주가 데이터를 읽습니다.

이 모듈은 DB를 읽기 전용(mode=ro)으로만 연결합니다.
INSERT/UPDATE/DELETE 및 DB 파일 수정은 하지 않습니다.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DEFAULT_DB_DIR = Path("/Users/dalmengee/claude_kiwoom_20260521/data/db")
MARKET_LABELS = {"0": "KOSPI", "10": "KOSDAQ"}

_PERIOD_DAYS = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "3y": 1095,
    "5y": 1825,
}


def get_db_dir() -> Path:
    """DB 디렉터리 경로를 반환합니다."""
    env = os.environ.get("KIWOOM_DB_DIR")
    return Path(env) if env else DEFAULT_DB_DIR


def _db_path(name: str) -> Path:
    path = get_db_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"DB 파일을 찾을 수 없습니다: {path}")
    return path


def _connect_readonly(name: str) -> sqlite3.Connection:
    """DB를 읽기 전용으로 연결합니다. 쓰기·수정이 불가능합니다."""
    path = _db_path(name).resolve()
    # immutable=1: WAL 모드 DB도 읽기만 허용, 저널/WAL 파일 생성 없음
    uri = f"file://{path.as_posix()}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def is_korean_code(ticker: str) -> bool:
    """한국 종목 코드인지 판별합니다."""
    code = normalize_korean_code(ticker)
    return code is not None


def normalize_korean_code(ticker: str) -> str | None:
    """Yahoo/키움 형식을 6자리 종목코드로 정규화합니다."""
    raw = ticker.strip().upper()
    if raw.endswith(".KS") or raw.endswith(".KQ"):
        raw = raw.rsplit(".", 1)[0]
    if re.fullmatch(r"[0-9A-Z]{6}", raw):
        return raw
    return None


def period_to_start_date(period: str, end: datetime | None = None) -> str:
    """period 문자열을 시작일(YYYY-MM-DD)로 변환합니다."""
    end = end or datetime.now()
    days = _PERIOD_DAYS.get(period, 365)
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d")


def fetch_daily_quote(
    code: str,
    period: str = "2y",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """단일 종목 일봉 데이터를 반환합니다."""
    normalized = normalize_korean_code(code)
    if not normalized:
        raise ValueError(f"유효하지 않은 한국 종목 코드: {code}")

    if start is None:
        start = period_to_start_date(period)
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    conn = _connect_readonly("daily_quote.sqlite3")
    try:
        rows = conn.execute(
            """
            SELECT dt, open, high, low, close, trade_qty
            FROM daily_quote
            WHERE code = ? AND dt >= ? AND dt <= ?
            ORDER BY dt
            """,
            (normalized, start, end),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"'{normalized}' 종목의 DB 데이터가 없습니다.")

    df = pd.DataFrame(rows, columns=["dt", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def fetch_multiple_daily_quotes(
    codes: list[str],
    period: str = "1y",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """여러 종목 일봉 데이터를 한 번에 반환합니다."""
    normalized = []
    for code in codes:
        n = normalize_korean_code(code)
        if n:
            normalized.append(n)
    if not normalized:
        return {}

    if start is None:
        start = period_to_start_date(period)
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    placeholders = ",".join("?" * len(normalized))
    conn = _connect_readonly("daily_quote.sqlite3")
    try:
        rows = conn.execute(
            f"""
            SELECT code, dt, open, high, low, close, trade_qty
            FROM daily_quote
            WHERE code IN ({placeholders}) AND dt >= ? AND dt <= ?
            ORDER BY code, dt
            """,
            (*normalized, start, end),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {}

    df = pd.DataFrame(rows, columns=["code", "dt", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["dt"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result: dict[str, pd.DataFrame] = {}
    for code, group in df.groupby("code"):
        cleaned = group.set_index("dt")[["open", "high", "low", "close", "volume"]].dropna()
        if not cleaned.empty:
            result[str(code)] = cleaned
    return result


def get_company_info(code: str) -> dict:
    """master.sqlite3에서 종목 정보를 반환합니다."""
    normalized = normalize_korean_code(code)
    if not normalized:
        raise ValueError(f"유효하지 않은 한국 종목 코드: {code}")

    conn = _connect_readonly("master.sqlite3")
    try:
        row = conn.execute(
            """
            SELECT code, name, market_code, up_name, up_size_name, is_active
            FROM companies
            WHERE code = ?
            """,
            (normalized,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return {
            "name": normalized,
            "currency": "KRW",
            "exchange": "KRX",
            "sector": "",
            "market": "",
            "code": normalized,
        }

    market_code = row[2] or ""
    return {
        "name": row[1],
        "currency": "KRW",
        "exchange": "KRX",
        "sector": row[3] or "",
        "market": MARKET_LABELS.get(market_code, market_code),
        "size": row[4] or "",
        "is_active": bool(row[5]),
        "code": row[0],
    }


def list_active_companies(markets: tuple[str, ...] = ("0", "10")) -> list[dict]:
    """활성 종목 목록을 반환합니다."""
    placeholders = ",".join("?" * len(markets))
    conn = _connect_readonly("master.sqlite3")
    try:
        rows = conn.execute(
            f"""
            SELECT code, name, market_code, up_name, up_size_name
            FROM companies
            WHERE is_active = 1 AND market_code IN ({placeholders})
            ORDER BY name
            """,
            markets,
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "code": code,
            "name": name,
            "market": MARKET_LABELS.get(market_code, market_code),
            "sector": up_name or "",
            "size": up_size_name or "",
        }
        for code, name, market_code, up_name, up_size_name in rows
    ]


def get_latest_quote_date() -> str | None:
    """daily_quote DB의 최신 일자를 반환합니다."""
    conn = _connect_readonly("daily_quote.sqlite3")
    try:
        row = conn.execute("SELECT MAX(dt) FROM daily_quote").fetchone()
    finally:
        conn.close()
    return row[0] if row and row[0] else None