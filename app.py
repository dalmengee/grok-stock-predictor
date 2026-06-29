"""주식 투자 통합 Streamlit 앱 — 단일 실행으로 모든 페이지 접근."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="주식 투자 도구",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("ui/home.py", title="홈", icon="🏠", default=True),
    st.Page("ui/predict_page.py", title="종목 예측", icon="📈"),
    st.Page("ui/screener_page.py", title="주식 스크리너", icon="🇰🇷"),
    st.Page("ui/invest_page.py", title="투자 의사결정", icon="💼"),
]

pg = st.navigation(pages)
pg.run()