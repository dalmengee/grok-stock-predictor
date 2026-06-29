"""홈 페이지."""

from __future__ import annotations

import streamlit as st

from src.local_db import get_db_dir, get_latest_quote_date
from ui.styles import SHARED_CSS

st.markdown(SHARED_CSS, unsafe_allow_html=True)

st.markdown('<p class="main-header">📊 주식 투자 도구</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">예측 · 스크리닝 · 투자 의사결정을 한 곳에서</p>',
    unsafe_allow_html=True,
)

latest = get_latest_quote_date()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("DB 최신 일봉", latest or "-")
with col2:
    st.metric("한국 주식", "로컬 DB")
with col3:
    st.metric("해외 주식", "Yahoo Finance")

st.markdown("### 페이지 안내")
st.markdown("""
| 페이지 | 설명 |
|--------|------|
| **종목 예측** | ML 기반 개별 종목 주가 예측 및 차트 |
| **주식 스크리너** | 코스피/코스닥 종목 매수 후보 순위 |
| **투자 의사결정** | 수급·리스크 분석, 브리핑, 포트폴리오 |
""")

st.caption(f"데이터 경로: `{get_db_dir()}`")
st.warning("⚠️ 모든 분석은 참고용이며, 투자 결정과 손실은 본인 책임입니다.")