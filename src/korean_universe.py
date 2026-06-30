"""한국 주식 종목 유니버스."""

from __future__ import annotations

from dataclasses import dataclass

from .local_db import list_active_companies


@dataclass(frozen=True)
class KoreanStock:
    ticker: str
    name: str
    sector: str
    market: str


KOSPI_LARGE_CODES = [
    "005930", "000660", "373220", "207940", "005380", "000270", "068270",
    "035420", "035720", "051910", "006400", "105560", "055550", "086790",
    "032830", "003550", "034730", "015760", "009150", "012330", "028260",
    "017670", "033780", "096770", "018260", "034020", "003670", "010140",
    "009540", "011200",
]

KOSDAQ_GROWTH_CODES = [
    "247540", "086520", "196170", "263750", "293490", "041510",
    "145020", "357780", "112040", "328130",
]


def _build_universe(codes: list[str]) -> list[KoreanStock]:
    """코드 목록을 KoreanStock 리스트로 변환합니다. DB에서 이름/섹터를 보강합니다."""
    companies = {c["code"]: c for c in list_active_companies()}
    stocks: list[KoreanStock] = []
    for code in codes:
        info = companies.get(code)
        if info:
            stocks.append(
                KoreanStock(
                    ticker=code,
                    name=info["name"],
                    sector=info["sector"] or "기타",
                    market=info["market"],
                )
            )
        else:
            stocks.append(KoreanStock(code, code, "기타", "KRX"))
    return stocks


KOSPI_LARGE = _build_universe(KOSPI_LARGE_CODES)
KOSDAQ_GROWTH = _build_universe(KOSDAQ_GROWTH_CODES)

UNIVERSES = {
    "kospi_large": ("코스피 대형주", KOSPI_LARGE),
    "kosdaq": ("코스닥 성장주", KOSDAQ_GROWTH),
    "all": ("코스피 + 코스닥", KOSPI_LARGE + KOSDAQ_GROWTH),
}


def get_universe(name: str = "kospi_large") -> tuple[str, list[KoreanStock]]:
    """종목 유니버스를 반환합니다."""
    if name in ("kospi200", "kospi200_ex"):
        from src.backtest.kospi200 import get_kospi200_universe

        return get_kospi200_universe(exclude_strategy=(name == "kospi200_ex"))
    if name not in UNIVERSES:
        available = ", ".join([*UNIVERSES, "kospi200", "kospi200_ex"])
        raise ValueError(f"알 수 없는 유니버스: {name}. 사용 가능: {available}")
    label, stocks = UNIVERSES[name]
    return label, stocks


def get_stock_map(name: str = "kospi_large") -> dict[str, KoreanStock]:
    """티커 → 종목 정보 맵을 반환합니다."""
    _, stocks = get_universe(name)
    return {s.ticker: s for s in stocks}