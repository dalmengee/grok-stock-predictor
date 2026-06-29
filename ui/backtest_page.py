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
    '<p class="sub-header">과거 데이터로 전략 성과를 검증하고 KOSPI 대비 초과수익을 확인합니다</p>',
    unsafe_allow_html=True,
)

tab_run, tab_validate = st.tabs(["백테스트", "워크포워드 검증"])

with st.sidebar:
    st.header("전략 파라미터")
    universe = st.selectbox("종목군", ["kospi_large", "kosdaq", "all"], format_func=lambda x: {
        "kospi_large": "코스피 대형주", "kosdaq": "코스닥", "all": "전체",
    }[x])
    start = st.date_input("시작일", value=date(2024, 1, 1))
    end = st.date_input("종료일", value=date(2025, 6, 30))
    cash = st.number_input("초기자본", 10_000_000, step=1_000_000)
    positions = st.slider("최대 보유", 1, 10, 5)
    entry = st.slider("진입 점수", 40, 80, 60)
    exit_score = st.slider("청산 점수", 30, 60, 45)
    stop = st.slider("손절 %", 3, 15, 8)
    profit = st.slider("익절 %", 5, 30, 15)
    rebalance = st.slider("리밸런싱(일)", 1, 20, 5)

with tab_run:
    if st.button("백테스트 실행", type="primary"):
        with st.spinner("시뮬레이션 중... (30초~1분)"):
            try:
                cfg = BacktestConfig(
                    start_date=str(start),
                    end_date=str(end),
                    initial_cash=cash,
                    max_positions=positions,
                    entry_score=entry,
                    exit_score=exit_score,
                    stop_loss_pct=stop / 100,
                    take_profit_pct=profit / 100,
                    rebalance_days=rebalance,
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
        c3.metric("샤프비율", f"{m.sharpe_ratio:.2f}")
        c4.metric("최대낙폭", f"{m.max_drawdown_pct:.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r.equity_curve.index, y=r.equity_curve.values,
            name="전략", line=dict(color="#2563eb", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=r.benchmark_curve.index, y=r.benchmark_curve.values,
            name="KOSPI Buy&Hold", line=dict(color="#94a3b8", dash="dash"),
        ))
        fig.update_layout(height=400, template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("승률", f"{m.win_rate_pct:.1f}%")
        c6.metric("손익비", f"{m.profit_factor:.2f}")
        c7.metric("거래", f"{m.total_trades}회")
        c8.metric("최종자산", f"₩{m.final_value:,.0f}")

        sells = [t for t in r.trades if t["side"] == "sell"]
        if sells:
            st.subheader("매도 내역")
            st.dataframe(sells[-20:], hide_index=True, use_container_width=True)

        if m.alpha_pct > 0 and m.sharpe_ratio > 0.5:
            st.success("KOSPI 대비 초과수익 — 과거 기준 전략 유효성 있음")
        elif m.total_return_pct > 0:
            st.warning("수익은 있으나 벤치마크 대비 열세")
        else:
            st.error("손실 구간 — 파라미터 재검토 필요")
    else:
        st.info("파라미터 설정 후 '백테스트 실행'을 눌러주세요.")

with tab_validate:
    train_ratio = st.slider("In-Sample 비율", 0.5, 0.8, 0.7, 0.05)
    if st.button("워크포워드 검증", type="primary"):
        with st.spinner("검증 중..."):
            try:
                st.session_state["wf_result"] = run_walk_forward(
                    universe=universe,
                    start_date=str(start),
                    end_date=str(end),
                    train_ratio=train_ratio,
                )
            except Exception as e:
                st.error(str(e))

    if "wf_result" in st.session_state:
        wf = st.session_state["wf_result"]
        st.markdown(f"**분할일:** {wf.split_date}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### In-Sample")
            st.metric("수익률", f"{wf.is_metrics.total_return_pct:+.1f}%")
            st.metric("샤프", f"{wf.is_metrics.sharpe_ratio:.2f}")
            st.metric("알파", f"{wf.is_metrics.alpha_pct:+.1f}%p")
        with col2:
            st.markdown("#### Out-of-Sample")
            st.metric("수익률", f"{wf.oos_metrics.total_return_pct:+.1f}%")
            st.metric("샤프", f"{wf.oos_metrics.sharpe_ratio:.2f}")
            st.metric("알파", f"{wf.oos_metrics.alpha_pct:+.1f}%p")
        if wf.robust:
            st.success(wf.summary)
        else:
            st.warning(wf.summary)

st.warning("⚠️ 과거 성과가 미래 수익을 보장하지 않습니다.")