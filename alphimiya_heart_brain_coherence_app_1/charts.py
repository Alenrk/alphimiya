"""
دوال بناء الرسوم البيانية التفاعلية (Plotly) وبطاقات المؤشرات لتطبيق
توافق القلب-العقل وطفرات جاما وتزامن الفصين ومؤشر التكيف العصبي.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NEON = {
    "cyan": "#22d3ee",
    "violet": "#a855f7",
    "magenta": "#ec4899",
    "amber": "#f59e0b",
    "green": "#34d399",
    "blue": "#60a5fa",
    "red": "#f87171",
}

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Cairo, Tajawal, sans-serif", color="#d7e0f5"),
    margin=dict(l=10, r=10, t=50, b=10),
)


def gauge_chart(value: float, title: str, color: str, suffix: str = "%", max_value: float = 100) -> go.Figure:
    display_value = 0 if (value is None or (isinstance(value, float) and np.isnan(value))) else value
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=display_value,
            number={"suffix": suffix, "font": {"size": 34, "color": color}},
            title={"text": title, "font": {"size": 15, "color": "#c7cfe6"}},
            gauge={
                "axis": {"range": [0, max_value], "tickcolor": "#4b5568"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.12)",
                "steps": [
                    {"range": [0, max_value * 0.4], "color": "rgba(248,113,113,0.15)"},
                    {"range": [max_value * 0.4, max_value * 0.75], "color": "rgba(245,158,11,0.15)"},
                    {"range": [max_value * 0.75, max_value], "color": "rgba(52,211,153,0.15)"},
                ],
            },
        )
    )
    fig.update_layout(height=230, **DARK_LAYOUT)
    return fig


def gamma_burst_chart(band_df: pd.DataFrame, bursts_df: pd.DataFrame, channel: str = "Gamma_mean") -> go.Figure:
    fig = go.Figure()
    if channel in band_df.columns:
        d = band_df[["t_sec", channel]].dropna().sort_values("t_sec")
        fig.add_trace(go.Scatter(
            x=d["t_sec"], y=d[channel], mode="lines", name="قوة موجة غاما",
            line=dict(color=NEON["violet"], width=1.5),
        ))

    color_map = {"متوسطة": NEON["amber"], "قوية": "#fb923c", "استثنائية": NEON["magenta"]}
    if bursts_df is not None and len(bursts_df):
        for intensity, grp in bursts_df.groupby("intensity"):
            fig.add_trace(go.Scatter(
                x=grp["t_sec"], y=grp["value"], mode="markers",
                name=f"طفرة {intensity}",
                marker=dict(
                    size=[10 + z * 1.5 for z in grp["z_score"]],
                    color=color_map.get(intensity, NEON["cyan"]),
                    symbol="star", line=dict(width=1, color="white"),
                ),
                text=[f"t={t:.1f}s | z={z:.1f}" for t, z in zip(grp["t_sec"], grp["z_score"])],
                hoverinfo="text",
            ))

    fig.update_layout(
        height=380, xaxis_title="الزمن منذ بداية الجلسة (ثانية)", yaxis_title="قوة غاما",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        **DARK_LAYOUT,
    )
    return fig


def hemispheric_timeseries_chart(hemi_result: dict) -> go.Figure:
    bands_order = ["Theta", "Alpha", "Beta", "Gamma"]
    colors = {"Theta": NEON["cyan"], "Alpha": NEON["green"], "Beta": NEON["amber"], "Gamma": NEON["magenta"]}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=["الجبهي (AF7 × AF8)", "الخلفي (TP9 × TP10)"])

    for row, key in ((1, "frontal"), (2, "posterior")):
        pair = hemi_result.get(key)
        if not pair:
            continue
        for band in bands_order:
            info = pair.per_band.get(band)
            if not info or len(info["t_sec"]) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=info["t_sec"], y=info["coherence_pct"], mode="lines",
                    name=f"{band}", legendgroup=band, showlegend=(row == 1),
                    line=dict(color=colors.get(band, NEON["blue"]), width=1.8),
                ),
                row=row, col=1,
            )

    fig.update_yaxes(title_text="% تزامن", range=[0, 100])
    fig.update_xaxes(title_text="الزمن منذ بداية الجلسة (ثانية)", row=2, col=1)
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        **DARK_LAYOUT,
    )
    return fig


def hrv_coherence_chart(coherence_ts) -> go.Figure:
    fig = go.Figure()
    if coherence_ts is not None and len(coherence_ts.t_sec):
        fig.add_trace(go.Scatter(
            x=coherence_ts.t_sec, y=coherence_ts.coherence_pct, mode="lines+markers",
            line=dict(color=NEON["blue"], width=2.2), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.10)",
            name="نسبة توافق HRV",
        ))
    fig.update_layout(
        height=320, xaxis_title="الزمن منذ بداية الجلسة (ثانية)", yaxis_title="% توافق HRV",
        yaxis_range=[0, 100], **DARK_LAYOUT,
    )
    return fig


def neuroplasticity_chart(result, session_duration: float) -> go.Figure:
    fig = go.Figure()
    t = result.t_sec

    fig.add_hrect(
        y0=result.alert_threshold, y1=100, fillcolor=NEON["green"], opacity=0.08, line_width=0,
        annotation_text="منطقة التكيّف العصبي المثالي", annotation_position="top left",
        annotation_font_color=NEON["green"],
    )

    comp_specs = [
        ("component_heart_brain", "توافق القلب-العقل", NEON["magenta"]),
        ("component_hemispheric", "تزامن الفصين", NEON["cyan"]),
        ("component_gamma", "نشاط طفرات غاما", NEON["amber"]),
    ]
    for attr, label, color in comp_specs:
        vals = getattr(result, attr)
        fig.add_trace(go.Scatter(
            x=t, y=vals, mode="lines", name=label, line=dict(color=color, width=1.2, dash="dot"), opacity=0.6,
        ))

    fig.add_trace(go.Scatter(
        x=t, y=result.score, mode="lines+markers", name="مؤشر التكيّف العصبي المركّب",
        line=dict(color=NEON["violet"], width=3), marker=dict(size=7),
    ))

    if len(result.peak_moments):
        fig.add_trace(go.Scatter(
            x=result.peak_moments["t_sec"], y=result.peak_moments["score"], mode="markers",
            name="لحظات الذروة", marker=dict(size=14, color=NEON["green"], symbol="diamond",
                                              line=dict(width=2, color="white")),
        ))

    fig.add_hline(y=result.alert_threshold, line_dash="dash", line_color=NEON["green"], opacity=0.6)

    fig.update_layout(
        height=440, xaxis_title="الزمن منذ بداية الجلسة (ثانية)", yaxis_title="%",
        yaxis_range=[0, 100],
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        **DARK_LAYOUT,
    )
    return fig
