"""
كاشف طفرات جاما (Gamma Burst Detector).

يرصد القفزات الفجائية في قوة موجة غاما (>30Hz تقريباً، وهو نطاق Gamma كما
يحسبه Muse) باستخدام إحصاء مقاوم للقيم المتطرفة (median + MAD) ضمن نافذة
متحركة، بدلاً من متوسط/انحراف معياري تقليديين حساسين جداً للتشويش العضلي.
كل طفرة تتجاوز عتبة Z-score المحددة تُسجَّل بطابعها الزمني وشدتها وتُصنَّف
إلى ثلاث درجات: متوسطة / قوية / استثنائية.

تنويه: تسمية هذه اللحظات بـ"استبصار" أو "وعي فائق" اصطلاح استكشافي مستوحى من
أدبيات التأمل الشائعة، وليس تصنيفاً عصبياً مثبتاً سريرياً.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

INTENSITY_LEVELS = [
    (5.0, "استثنائية", "🟣"),
    (4.0, "قوية", "🟠"),
    (3.0, "متوسطة", "🟡"),
]


@dataclass
class GammaBurst:
    t_sec: float
    value: float
    z_score: float
    intensity: str
    icon: str
    channel: str


def _rolling_median_mad(x: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """متوسط وسيطي منزلق + الانحراف المطلق الوسيطي (MAD) — أكثر مقاومة للقيم الشاذة من mean/std."""
    s = pd.Series(x)
    med = s.rolling(win, center=True, min_periods=max(3, win // 3)).median().to_numpy()
    mad = (
        s.rolling(win, center=True, min_periods=max(3, win // 3))
        .apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
        .to_numpy()
    )
    mad = np.where(mad < 1e-6, 1e-6, mad)
    return med, mad


def detect_gamma_bursts(
    band_df: pd.DataFrame,
    channel: str = "Gamma_mean",
    z_threshold: float = 3.0,
    min_separation_sec: float = 1.0,
    baseline_window_sec: float = 10.0,
) -> list[GammaBurst]:
    """يرصد طفرات غاما الفجائية عبر عتبة Z-score مبنية على وسيط/MAD متحرك."""
    if band_df is None or channel not in band_df.columns or len(band_df) < 10:
        return []

    d = band_df[["t_sec", channel]].dropna().sort_values("t_sec").reset_index(drop=True)
    if len(d) < 10:
        return []

    t = d["t_sec"].to_numpy()
    x = d[channel].to_numpy().astype(float)

    median_dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    win = max(5, int(round(baseline_window_sec / max(median_dt, 1e-3))))
    if win >= len(x):
        win = max(5, len(x) // 2)

    med, mad = _rolling_median_mad(x, win)
    z = 0.6745 * (x - med) / mad
    z = np.nan_to_num(z, nan=0.0)

    min_distance = max(1, int(round(min_separation_sec / max(median_dt, 1e-3))))
    peaks, props = signal.find_peaks(z, height=z_threshold, distance=min_distance)

    bursts = []
    for idx in peaks:
        zv = float(z[idx])
        level_label, level_icon = "متوسطة", "🟡"
        for thresh, label, icon in INTENSITY_LEVELS:
            if zv >= thresh:
                level_label, level_icon = label, icon
                break
        bursts.append(
            GammaBurst(
                t_sec=float(t[idx]), value=float(x[idx]), z_score=zv,
                intensity=level_label, icon=level_icon, channel=channel,
            )
        )
    return bursts


def bursts_to_dataframe(bursts: list[GammaBurst]) -> pd.DataFrame:
    if not bursts:
        return pd.DataFrame(columns=["t_sec", "value", "z_score", "intensity", "icon", "channel"])
    return pd.DataFrame([b.__dict__ for b in bursts])


def burst_rate_per_minute(bursts: list[GammaBurst], session_duration_sec: float) -> float:
    if session_duration_sec <= 0:
        return 0.0
    return len(bursts) * 60.0 / session_duration_sec
