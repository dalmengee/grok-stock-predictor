"""공통 Streamlit 스타일."""

SHARED_CSS = """
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1e3a5f;
    margin-bottom: 0.2rem;
}
.sub-header {
    color: #64748b;
    margin-bottom: 1.5rem;
}
.pick-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.up { color: #16a34a; }
.down { color: #dc2626; }
.flat { color: #ca8a04; }
</style>
"""

RECOMMENDATION_COLOR = {
    "강력 매수": "#16a34a",
    "매수": "#2563eb",
    "관망": "#ca8a04",
    "회피": "#dc2626",
}

ACTION_COLOR = {
    "매수": "#16a34a",
    "분할 매수": "#2563eb",
    "관망": "#ca8a04",
    "보유/관망": "#64748b",
    "회피": "#dc2626",
    "손절 검토": "#dc2626",
    "익절 검토": "#16a34a",
}