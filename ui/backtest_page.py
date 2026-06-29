"""백테스트 페이지."""

from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import streamlit as st

from src.backtest.config import BacktestConfig
from src.backtest.engine import run_backtest
from src.backtest.validation import run_walk_forward
from ui.styles import SHARED_CSS

st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown('<p class="main-header">🔬 전략 백테스트</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">시장 국면 전환 · 리스크 관리 · 변동성 기반 포지션 사이징</p>',
    unsafe_allow_html=True,
)

tab_adaptive, tab_compare, tab_validate = st.tabs(["적응형 전략", "전략 비교", "워크포워드"])

with st.sidebar:
    st.header("설정")
    universe = st.selectbox("종목군", ["kospi_large", "kosdaq", "all"], format_func=lambda x: {
        "kospi_large": "코스피 대형주", "kosdaq": "코스닥", "all": "전체",
    }[x])
    start = st.date_input("시작일", value=date(2024, 1, 1))
    end = st.date_input("종료일", value=date(2025, 6, 30))
    cash = st.number_input("초기자본", 10_000_000, step=1_000_000)
    rebalance = st.slider("리밸런싱(일)", 3, 15, 5)
    crisis_dd = st.slider("위기 DD 차단 %", 8, 20, 12)

with tab_adaptive:
    st.markdown("""
    **적응형 전략**
    - **상승장**: 모멘텀+수급, 90% 투자
    - **횡보장**: 선별 매수, 55% 투자
    - **하락장**: 과매도 반등만, 25% 투자
    - **위기**: DD 12% 초과 시 현금 대기
    """)
    if st.button("백테스트 실행", type="primary"):
        with st.spinner("시뮬레이션 중..."):
            try:
                cfg = BacktestConfig(
                    start_date=str(start), end_date=str(end),
                    initial_cash=cash, adaptive=True,
                    rebalance_days=rebalance,
                    crisis_dd_threshold=crisis_dd / 100,
                )
                st.session_state["bt_result"] = run_backtest(universe=universe, config=cfg)
            except Exception as e:
                st.error(str(e))

    if "bt_result" in st.session_state:
        r = st.session_state["bt_result"]
        m = r.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총수익률", f"{m.total_return_pct:+.1f}%")
        c2.metric("KOSPI 대비", f"{m.alpha_pct:+.1f}%p")
        c3.metric("샤프", f"{m.sharpe_ratio:.2f}")
        c4.metric("MDD", f"{m.max_drawdown_pct:.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r.equity_curve.index, y=r.equity_curve.values, name="전략", line=dict(color="#2563eb")))
        fig.add_trace(go.Scatter(x=r.benchmark_curve.index, y=r.benchmark_curve.values, name="KOSPI", line=dict(dash="dash", color="#94a3b8")))
        fig.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        if not r.regime_log.empty:
            st.subheader("시장 국면")
            st.bar_chart(r.regime_log["regime"].value_counts())
            if not r.exposure_log.empty:
                st.caption(f"평균 주식 비중: {r.exposure_log.mean()*100:.1f}%")

        if m.alpha_pct > 0:
            st.success("KOSPI 대비 초과수익")
        elif m.total_return_pct > 0:
            st.warning("수익은 있으나 벤치마크 열세")
        else:
            st.error("손실 — 파라미터 추가 조정 필요")

with tab_compare:
    if st.button("단순 vs 적응형 비교", type="primary"):
        with st.spinner("비교 중..."):
            base = dict(start_date=str(start), end_date=str(end), initial_cash=cash, rebalance_days=rebalance)
            results = {}
            for name, adaptive in [("단순 로테이션", False), ("적응형", True)]:
                cfg = BacktestConfig(**base, adaptive=adaptive, crisis_dd_threshold=crisis_dd / 100)
                results[name] = run_backtest(universe=universe, config=cfg)
            st.session_state["compare"] = results

    if "compare" in st.session_state:
        rows = []
        for name, r in st.session_state["compare"].items():
            m = r.metrics
            rows.append({
                "전략": name, "수익률%": m.total_return_pct, "알파%p": m.alpha_pct,
                "샤프": m.sharpe_ratio, "MDD%": m.max_drawdown_pct, "거래": m.total_trades,
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

with tab_validate:
    train_ratio = st.slider("In-Sample 비율", 0.5, 0.8, 0.7, 0.05)
    if st.button("워크포워드 검증", type="primary"):
        with st.spinner("검증 중..."):
            try:
                st.session_state["wf"] = run_walk_forward(
                    universe=universe, start_date=str(start), end_date=str(end),
                    train_ratio=train_ratio,
                    base_config=BacktestConfig(adaptive=True, crisis_dd_threshold=crisis_dd / 100),
                )
            except Exception as e:
                st.error(str(e))
    if "wf" in st.session_state:
        wf = st.session_state["wf"]
        c1, c2 = st.columns(2)
        with c1:
            st.metric("IS 수익률", f"{wf.is_metrics.total_return_pct:+.1f}%")
            st.metric("IS 알파", f"{wf.is_metrics.alpha_pct:+.1f}%p")
        with c2:
            st.metric("OOS 수익률", f"{wf.oos_metrics.total_return_pct:+.1f}%")
            st.metric("OOS 알파", f"{wf.oos_metrics.alpha_pct:+.1f}%p")
        st.info(wf.summary) if wf.robust else st.warning(wf.summary)

st.warning("⚠️ 과거 성과가 미래 수익을 보장하지 않습니다.")