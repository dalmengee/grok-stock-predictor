"""주식 스크리너 페이지."""

from __future__ import annotations

import streamlit as st

from src.korean_universe import UNIVERSES
from src.local_db import get_db_dir, get_latest_quote_date
from src.screener import picks_to_dataframe, screen_korean_stocks
from ui.styles import RECOMMENDATION_COLOR, SHARED_CSS

st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown('<p class="main-header">🇰🇷 주식 스크리너</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">기술적 지표 + ML 예측으로 매수 후보 순위</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("설정")
    universe = st.selectbox("종목군", list(UNIVERSES), format_func=lambda x: UNIVERSES[x][0], index=0)
    top_n = st.slider("표시 개수", 5, 30, 10)
    period = st.selectbox("데이터 기간", ["6mo", "1y", "2y"], index=1)
    forecast_days = st.slider("ML 예측 기간 (일)", 1, 10, 5)
    run_btn = st.button("스크리닝 실행", type="primary", use_container_width=True)
    st.divider()
    latest_dt = get_latest_quote_date()
    if latest_dt:
        st.caption(f"DB 최신 일봉: {latest_dt}")
    st.caption(f"데이터: `{get_db_dir()}`")
    st.divider()
    st.markdown("**점수 (100점)**")
    st.markdown("- ML 30 · 추세 25 · 모멘텀 20 · 거래량 15 · 단기 10")

if run_btn:
    with st.spinner("종목 분석 중..."):
        try:
            st.session_state["screening"] = screen_korean_stocks(
                universe=universe, period=period, forecast_days=forecast_days, top_n=top_n,
            )
        except Exception as e:
            st.error(f"오류: {e}")
            st.stop()

if "screening" in st.session_state:
    result = st.session_state["screening"]
    buy_picks = [p for p in result.all_picks if p.recommendation in ("강력 매수", "매수")]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석일", result.date)
    col2.metric("분석 종목", f"{result.analyzed_count}개")
    col3.metric("매수 후보", f"{len(buy_picks)}개")
    col4.metric("분석 실패", f"{len(result.failed_tickers)}개")

    st.subheader("매수 후보")
    if buy_picks:
        for p in buy_picks[:5]:
            color = RECOMMENDATION_COLOR[p.recommendation]
            st.markdown(
                f'<div class="pick-card">'
                f'<span style="color:{color};font-weight:700;">{p.recommendation}</span> '
                f'<strong>{p.name}</strong> ({p.ticker}) | '
                f'₩{p.current_price:,.0f} | 점수 {p.score}/100 | ML {p.predicted_return*100:+.1f}%'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"{p.name} 상세"):
                for signal in p.signals:
                    st.markdown(f"- {signal}")
    else:
        st.info("현재 매수 추천 종목이 없습니다.")

    st.subheader("전체 순위")
    df = picks_to_dataframe(result.picks)
    st.dataframe(
        df.style.map(
            lambda v: f"color: {RECOMMENDATION_COLOR.get(v, '#000')}; font-weight: bold",
            subset=["추천"],
        ),
        hide_index=True,
        use_container_width=True,
    )
    if result.failed_tickers:
        with st.expander("분석 실패 종목"):
            st.write(", ".join(result.failed_tickers))
    st.warning("⚠️ 참고용 분석입니다.")
else:
    st.info("사이드바에서 '스크리닝 실행'을 눌러주세요.")