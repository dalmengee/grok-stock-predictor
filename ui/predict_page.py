"""종목 예측 페이지."""

from __future__ import annotations

import streamlit as st

from src.data_fetcher import fetch_stock_data, get_ticker_info
from src.model import train_and_predict
from src.visualizer import create_price_chart
from ui.styles import SHARED_CSS

st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown('<p class="main-header">📈 종목 예측</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">머신러닝과 기술적 지표를 활용한 주가 예측</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("설정")
    ticker = st.text_input("종목 심볼", value="005930", help="예: 005930, 000660, AAPL")
    period = st.selectbox("데이터 기간", ["1y", "2y", "3y", "5y"], index=1)
    forecast_days = st.slider("예측 기간 (일)", min_value=1, max_value=30, value=5)
    model_name = st.selectbox(
        "모델",
        ["random_forest", "gradient_boosting"],
        format_func=lambda x: "Random Forest" if x == "random_forest" else "Gradient Boosting",
    )
    predict_btn = st.button("예측 실행", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**종목 예시**")
    st.markdown("- 한국: `005930`, `000660` (로컬 DB)")
    st.markdown("- 미국: `AAPL`, `TSLA`, `MSFT`")

if predict_btn:
    with st.spinner(f"{ticker} 데이터를 불러오고 예측 중..."):
        try:
            info = get_ticker_info(ticker)
            df = fetch_stock_data(ticker, period=period)
            result = train_and_predict(df, ticker=ticker, forecast_days=forecast_days, model_name=model_name)
            st.session_state["predict_result"] = result
            st.session_state["predict_info"] = info
        except Exception as e:
            st.error(f"오류: {e}")
            st.stop()

if "predict_result" in st.session_state:
    result = st.session_state["predict_result"]
    info = st.session_state.get("predict_info", {})
    symbol = "₩" if info.get("currency") == "KRW" else "$"

    st.subheader(f"{info.get('name', result.ticker)} ({result.ticker})")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("현재가", f"{symbol}{result.current_price:,.2f}")
    with col2:
        st.metric(
            f"{result.forecast_days}일 후 예측가",
            f"{symbol}{result.predicted_price:,.2f}",
            f"{result.predicted_return * 100:+.2f}%",
        )
    with col3:
        direction_class = {"상승": "up", "하락": "down", "보합": "flat"}[result.direction]
        st.markdown(
            f'<p style="font-size:0.9rem;color:#64748b;">예측 방향</p>'
            f'<p class="{direction_class}" style="font-size:1.8rem;font-weight:700;">{result.direction}</p>',
            unsafe_allow_html=True,
        )
    with col4:
        st.metric("모델 R²", f"{result.metrics['r2']:.4f}")

    chart_col, info_col = st.columns([2, 1])
    with chart_col:
        st.plotly_chart(create_price_chart(result), use_container_width=True)
    with info_col:
        st.markdown("#### 모델 성능")
        st.markdown(f"- MAE: {symbol}{result.metrics['mae']:,.2f}")
        st.markdown(f"- RMSE: {symbol}{result.metrics['rmse']:,.2f}")
        st.markdown(f"- R²: {result.metrics['r2']:.4f}")
        st.markdown("#### 특성 중요도")
        st.dataframe(result.feature_importance.head(8), hide_index=True, use_container_width=True)
    st.warning("⚠️ 본 예측은 참고용이며, 투자 결정의 유일한 근거로 사용하지 마세요.")
else:
    st.info("사이드바에서 종목을 입력하고 '예측 실행'을 눌러주세요.")