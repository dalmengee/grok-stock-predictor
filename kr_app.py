"""오늘의 한국 주식 매수 스크리너 웹 앱."""

from __future__ import annotations

import streamlit as st

from src.korean_universe import UNIVERSES
from src.local_db import get_db_dir, get_latest_quote_date
from src.screener import picks_to_dataframe, screen_korean_stocks

st.set_page_config(
    page_title="오늘의 한국 주식",
    page_icon="🇰🇷",
    layout="wide",
)

RECOMMENDATION_COLOR = {
    "강력 매수": "#16a34a",
    "매수": "#2563eb",
    "관망": "#ca8a04",
    "회피": "#dc2626",
}

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3a5f; }
    .sub-header { color: #64748b; margin-bottom: 1.5rem; }
    .pick-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🇰🇷 오늘의 한국 주식 매수 스크리너</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">기술적 지표 + ML 예측으로 오늘 살 만한 종목을 순위별로 보여줍니다</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("설정")
    universe = st.selectbox(
        "종목군",
        list(UNIVERSES),
        format_func=lambda x: UNIVERSES[x][0],
        index=0,
    )
    top_n = st.slider("표시 개수", min_value=5, max_value=30, value=10)
    period = st.selectbox("데이터 기간", ["6mo", "1y", "2y"], index=1)
    forecast_days = st.slider("ML 예측 기간 (일)", 1, 10, 5)
    run_btn = st.button("스크리닝 실행", type="primary", use_container_width=True)

    st.divider()
    latest_dt = get_latest_quote_date()
    if latest_dt:
        st.caption(f"DB 최신 일봉: {latest_dt}")
    st.caption(f"데이터 경로: `{get_db_dir()}`")

    st.divider()
    st.markdown("**점수 기준 (100점)**")
    st.markdown("- ML 예측: 30점")
    st.markdown("- 추세: 25점")
    st.markdown("- 모멘텀: 20점")
    st.markdown("- 거래량: 15점")
    st.markdown("- 단기 강도: 10점")

if run_btn:
    with st.spinner("종목 분석 중... (1~2분 소요)"):
        try:
            result = screen_korean_stocks(
                universe=universe,
                period=period,
                forecast_days=forecast_days,
                top_n=top_n,
            )
            st.session_state["screening"] = result
        except Exception as e:
            st.error(f"오류: {e}")
            st.stop()

if "screening" in st.session_state:
    result = st.session_state["screening"]
    buy_picks = [p for p in result.all_picks if p.recommendation in ("강력 매수", "매수")]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("분석일", result.date)
    with col2:
        st.metric("분석 종목", f"{result.analyzed_count}개")
    with col3:
        st.metric("매수 후보", f"{len(buy_picks)}개")
    with col4:
        st.metric("분석 실패", f"{len(result.failed_tickers)}개")

    st.subheader("오늘의 매수 후보")

    if buy_picks:
        for p in buy_picks[:5]:
            color = RECOMMENDATION_COLOR[p.recommendation]
            st.markdown(
                f'<div class="pick-card">'
                f'<span style="color:{color};font-weight:700;">{p.recommendation}</span> '
                f'<strong>{p.name}</strong> ({p.ticker}) &nbsp;|&nbsp; '
                f'₩{p.current_price:,.0f} &nbsp;|&nbsp; '
                f'점수 <strong>{p.score}</strong>/100 &nbsp;|&nbsp; '
                f'ML 5일 <strong>{p.predicted_return*100:+.1f}%</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"{p.name} 상세 신호"):
                for signal in p.signals:
                    st.markdown(f"- {signal}")
                st.markdown("**점수 breakdown**")
                for key, val in p.score_breakdown.items():
                    st.markdown(f"- {key}: {val}점")
    else:
        st.info("현재 매수 추천 종목이 없습니다. 관망을 권장합니다.")

    st.subheader("전체 순위")
    df = picks_to_dataframe(result.picks)

    def highlight_recommendation(val: str) -> str:
        color = RECOMMENDATION_COLOR.get(val, "#000")
        return f"color: {color}; font-weight: bold"

    st.dataframe(
        df.style.map(highlight_recommendation, subset=["추천"]),
        hide_index=True,
        use_container_width=True,
    )

    if result.failed_tickers:
        with st.expander("분석 실패 종목"):
            st.write(", ".join(result.failed_tickers))

    st.warning("⚠️ 본 분석은 참고용이며, 투자 결정의 유일한 근거로 사용하지 마세요.")
else:
    st.info("왼쪽에서 종목군을 선택하고 '스크리닝 실행'을 눌러주세요.")