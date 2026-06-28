"""차트 시각화 모듈."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .features import add_technical_indicators
from .model import PredictionResult


def create_price_chart(result: PredictionResult) -> go.Figure:
    """주가 및 예측 차트를 생성합니다."""
    data = add_technical_indicators(result.historical)
    recent = data.tail(120)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("주가 추이", "거래량"),
    )

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["close"],
            name="종가",
            line=dict(color="#2563eb", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["sma_20"],
            name="SMA 20",
            line=dict(color="#f59e0b", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["sma_50"],
            name="SMA 50",
            line=dict(color="#10b981", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )

    last_date = recent.index[-1]
    future_date = last_date + pd.Timedelta(days=result.forecast_days)
    fig.add_trace(
        go.Scatter(
            x=[last_date, future_date],
            y=[result.current_price, result.predicted_price],
            name="예측",
            mode="lines+markers",
            line=dict(color="#ef4444", width=2, dash="dot"),
            marker=dict(size=10),
        ),
        row=1,
        col=1,
    )

    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in recent["close"].diff().fillna(0)]
    fig.add_trace(
        go.Bar(x=recent.index, y=recent["volume"], name="거래량", marker_color=colors, opacity=0.6),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    return fig


def save_matplotlib_chart(result: PredictionResult, output_path: str) -> None:
    """matplotlib 차트를 파일로 저장합니다."""
    data = add_technical_indicators(result.historical).tail(120)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#f8fafc")

    ax1.plot(data.index, data["close"], label="종가", color="#2563eb", linewidth=2)
    ax1.plot(data.index, data["sma_20"], label="SMA 20", color="#f59e0b", linestyle="--")
    ax1.plot(data.index, data["sma_50"], label="SMA 50", color="#10b981", linestyle="--")

    last_date = data.index[-1]
    future_date = last_date + pd.Timedelta(days=result.forecast_days)
    ax1.plot(
        [last_date, future_date],
        [result.current_price, result.predicted_price],
        "o--",
        color="#ef4444",
        linewidth=2,
        label=f"{result.forecast_days}일 후 예측",
    )

    ax1.set_title(f"{result.ticker} 주가 예측", fontsize=14, fontweight="bold")
    ax1.set_ylabel("가격")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.bar(data.index, data["volume"], color="#94a3b8", alpha=0.7)
    ax2.set_ylabel("거래량")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()