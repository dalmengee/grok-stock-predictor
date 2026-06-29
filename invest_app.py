"""투자 의사결정 Streamlit 앱."""

from __future__ import annotations

import streamlit as st

from src.invest_analysis import analyze_for_investment, suggest_order_amount
from src.local_db import get_db_dir, get_latest_quote_date, list_active_companies
from src.market_brief import brief_buy_recommendations, generate_market_brief
from src.portfolio import add_holding, analyze_portfolio, load_portfolio, remove_holding, set_cash

st.set_page_config(page_title="투자 의사결정", page_icon="💼", layout="wide")

ACTION_COLOR = {
    "매수": "#16a34a",
    "분할 매수": "#2563eb",
    "관망": "#ca8a04",
    "보유/관망": "#64748b",
    "회피": "#dc2626",
    "손절 검토": "#dc2626",
    "익절 검토": "#16a34a",
}

st.title("💼 투자 의사결정 도구")
st.caption("기술적 분석 + 수급 + ML 예측 + 포트폴리오 관리")

latest = get_latest_quote_date()
if latest:
    st.sidebar.caption(f"DB 최신: {latest}")
st.sidebar.caption(f"데이터: `{get_db_dir()}`")

tab_brief, tab_analyze, tab_portfolio = st.tabs(["오늘의 브리핑", "종목 분석", "포트폴리오"])

pf = load_portfolio()
portfolio_value = pf.cash + sum(h.quantity * h.avg_price for h in pf.holdings)

with tab_brief:
    col1, col2 = st.columns([1, 3])
    with col1:
        universe = st.selectbox("종목군", ["kospi_large", "kosdaq", "all"], format_func=lambda x: {
            "kospi_large": "코스피 대형주", "kosdaq": "코스닥", "all": "전체"
        }[x])
        if st.button("브리핑 생성", type="primary"):
            with st.spinner("분석 중..."):
                st.session_state["brief"] = generate_market_brief(universe=universe, portfolio_value=portfolio_value)

    if "brief" in st.session_state:
        brief = st.session_state["brief"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("시장 분위기", brief.market_mood)
        m2.metric("포트폴리오", f"₩{brief.portfolio_summary['total_value']:,.0f}")
        m3.metric("평가손익", f"₩{brief.portfolio_summary['total_pnl']:,.0f}",
                  f"{brief.portfolio_summary['total_pnl_pct']:+.1f}%")
        m4.metric("보유 종목", f"{brief.portfolio_summary['holdings_count']}개")

        recs = brief_buy_recommendations(brief, portfolio_value)
        if recs:
            st.subheader("매수 아이디어")
            for r in recs:
                color = ACTION_COLOR.get(r["action"], "#333")
                st.markdown(
                    f"**<span style='color:{color}'>{r['action']}</span>** — "
                    f"{r['name']} ({r['code']}) · 점수 {r['score']}",
                    unsafe_allow_html=True,
                )
                o = r["order"]
                if o["quantity"]:
                    st.markdown(
                        f"매수 제안: **{o['quantity']}주** (₩{o['amount']:,.0f}) · "
                        f"손절 ₩{o['stop_loss']:,.0f} · 목표 ₩{o['target_price']:,.0f}"
                    )
                for reason in r["reasons"]:
                    st.markdown(f"- {reason}")
        else:
            st.info("현재 적극 매수 아이디어가 없습니다.")

        if brief.portfolio_actions:
            st.subheader("보유 종목 액션")
            for a in brief.portfolio_actions:
                st.markdown(f"**{a['name']}** ({a['code']}): {a['action']} — 수익률 {a['pnl_pct']:+.1f}%")
                if a.get("alert"):
                    st.warning(a["alert"])
    else:
        st.info("'브리핑 생성'을 눌러 오늘의 투자 판단을 확인하세요.")

with tab_analyze:
    companies = list_active_companies()
    options = {c["code"]: f"{c['name']} ({c['code']})" for c in companies}
    code = st.selectbox("종목", list(options.keys()), format_func=lambda x: options[x])
    pv = st.number_input("포트폴리오 총액 (원)", value=int(portfolio_value), step=1_000_000)

    if st.button("분석 실행", type="primary"):
        with st.spinner("분석 중..."):
            try:
                st.session_state["report"] = analyze_for_investment(code, portfolio_value=pv)
            except Exception as e:
                st.error(str(e))

    if "report" in st.session_state:
        r = st.session_state["report"]
        order = suggest_order_amount(r, pv)
        color = ACTION_COLOR.get(r.action, "#333")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("판단", r.action)
        c2.metric("점수", f"{r.score}/100")
        c3.metric("현재가", f"₩{r.current_price:,.0f}")
        c4.metric("리스크", r.risk_level)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### 가격 전략")
            st.markdown(f"- **진입가:** ₩{r.entry_price:,.0f}")
            st.markdown(f"- **손절가:** ₩{r.stop_loss:,.0f}")
            st.markdown(f"- **목표가:** ₩{r.target_price:,.0f}")
            st.markdown(f"- **비중 제안:** {r.position_pct}%")
            if order["quantity"]:
                st.markdown(f"- **매수 제안:** {order['quantity']}주 (₩{order['amount']:,.0f})")

        with col_r:
            st.markdown("#### 수급 (5일)")
            for k, v in r.flow_5d.items():
                st.markdown(f"- {k}: {v:+,}주")
            st.markdown("#### 리스크")
            st.markdown(f"- 변동성: {r.volatility_20d}%")
            st.markdown(f"- 최대낙폭: {r.max_drawdown_60d}%")
            st.markdown(f"- RSI: {r.rsi:.0f}")

        if r.reasons:
            st.markdown("#### 근거")
            for reason in r.reasons:
                st.markdown(f"- {reason}")
        if r.cautions:
            st.markdown("#### 주의")
            for c in r.cautions:
                st.warning(c)

with tab_portfolio:
    summary = analyze_portfolio()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 자산", f"₩{summary.total_value:,.0f}")
    c2.metric("현금", f"₩{summary.cash:,.0f}")
    c3.metric("주식", f"₩{summary.stock_value:,.0f}")
    c4.metric("손익", f"₩{summary.total_pnl:,.0f}", f"{summary.total_pnl_pct:+.1f}%")

    if summary.holdings:
        rows = []
        for h in summary.holdings:
            rows.append({
                "종목": h.name, "코드": h.code, "수량": h.quantity,
                "평단": h.avg_price, "현재가": h.current_price,
                "손익%": f"{h.pnl_pct:+.1f}%", "비중%": f"{h.weight:.1f}",
                "판단": h.action, "손절": h.stop_loss, "목표": h.target_price,
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

    st.subheader("포트폴리오 편집")
    edit_col1, edit_col2 = st.columns(2)
    with edit_col1:
        st.markdown("**종목 추가**")
        add_code = st.text_input("코드", "005930", key="add_code")
        add_qty = st.number_input("수량", 1, step=1, key="add_qty")
        add_price = st.number_input("매수가", 0, step=1000, key="add_price")
        if st.button("추가"):
            add_holding(add_code, add_qty, add_price)
            st.success("추가 완료")
            st.rerun()
    with edit_col2:
        st.markdown("**종목 매도 / 현금 설정**")
        sell_code = st.text_input("매도 코드", key="sell_code")
        sell_qty = st.number_input("매도 수량 (0=전량)", 0, step=1, key="sell_qty")
        if st.button("매도"):
            remove_holding(sell_code, sell_qty or None)
            st.success("매도 반영")
            st.rerun()
        cash_amount = st.number_input("현금 설정", int(summary.cash), step=1_000_000, key="cash_amt")
        if st.button("현금 저장"):
            set_cash(cash_amount)
            st.success("저장 완료")
            st.rerun()

st.warning("⚠️ 본 도구는 투자 참고용이며, 모든 투자 결정과 손실은 본인 책임입니다.")