"""@deprecated — streamlit run app.py 를 사용하세요."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="리다이렉트", page_icon="🇰🇷")
st.warning("이 앱은 `app.py`로 통합되었습니다.")
st.code("streamlit run app.py", language="bash")
st.page_link("app.py", label="통합 앱 열기", icon="📊")