"""
مؤشر "لحظة التكيف العصبي" (Neuroplasticity Score) — معادلة مركّبة تدمج:
  • تزامن الفصين (Hemispheric Coherence)
  • نشاط طفرات جاما (Gamma Burst Activity)
  • توافق القلب والعقل (Heart-Brain Coherence)

في مؤشر زمني واحد (0–100%) عبر نوافذ متحركة على طول الجلسة، مع تحديد
"اللحظات الذروة" التي تتجاوز عتبة تنبيهية معينة — وهي، حسب فرضية الأبحاث
المستوحاة من أعمال د. جو ديسبنزا وغيره، لحظات مرشّحة لإعادة تشكيل عصبي.

تنويه: هذا مؤشر مركّب استكشافي من تصميم هذا التطبيق وليس مقياساً سريرياً
معتمداً لـ"اللدونة العصبية" الفعلية.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"heart_brain": 0.4, "hemispheric": 0.3, "gamma": 0.3}
DEFAULT_ALERT_THRESHOLD = 75.0
DEFAULT_GAMMA_RATE_CAP_PER_MIN = 12.0


@dataclass
class NeuroplasticityResult:
    t_sec: np.ndarray
    score: np.ndarray
    component_heart_brain: np.ndarray
    component_hemispheric: np.ndarray
    component_gamma: np.ndarray
    weights: dict
    alert_threshold: float
    peak_moments: pd.DataFrame
    overall_mean: float
    notes: list = field(default_factory=list)


def _interp_series(t_src: np.ndarray, v_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    if t_src is None or len(t_src) == 0:
        return np.full_like(t_grid, np.nan, dtype=float)
    if len(t_src) == 1:
        return np.full_like(t_grid, v_src[0], dtype=float)
    order = np.argsort(t_src)
    t_sorted, v_sorted = np.asarray(t_src)[order], np.asarray(v_src)[order]
    out = np.interp(t_grid, t_sorted, v_sorted, left=np.nan, right=np.nan)
    return out


def _hemispheric_to_series(hemi_result: dict) -> tuple[np.ndarray, np.ndarray]:
    """يجمع كل سلاسل تزامن الفصين (كل موجة × كل زوج أقطاب) في سلسلة واحدة عبر المتوسط."""
    all_t, all_v = [], []
    for pair_key in ("frontal", "posterior"):
        pair = hemi_result.get(pair_key)
        if pair is None:
            continue
        for band_info in pair.per_band.values():
            all_t.append(band_info["t_sec"])
            all_v.append(band_info["coherence_pct"])
    if not all_t:
        return np.array([]), np.array([])
    # نبني شبكة موحدة من كل النقاط المرصودة، ثم نتوسط عبر التداخل الزمني بسيط
    t_concat = np.concatenate(all_t)
    v_concat = np.concatenate(all_v)
    order = np.argsort(t_concat)
    return t_concat[order], v_concat[order]


def _gamma_rate_series(
    bursts_t: np.ndarray, t_grid: np.ndarray, window_sec: float, rate_cap: float
) -> np.ndarray:
    half = window_sec / 2.0
    out = np.zeros_like(t_grid, dtype=float)
    if len(bursts_t) == 0:
        return out
    bursts_t = np.asarray(bursts_t)
    for i, c in enumerate(t_grid):
        count = np.sum((bursts_t >= c - half) & (bursts_t < c + half))
        rate_per_min = count * 60.0 / window_sec
        out[i] = min(100.0, rate_per_min / rate_cap * 100.0)
    return out


def compute_neuroplasticity_score(
    heart_brain_result,
    hemispheric_result: dict,
    gamma_bursts_t: list[float],
    session_duration: float,
    weights: dict = None,
    window_sec: float = 30.0,
    step_sec: float = 10.0,
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
    gamma_rate_cap: float = DEFAULT_GAMMA_RATE_CAP_PER_MIN,
) -> NeuroplasticityResult:
    weights = weights or DEFAULT_WEIGHTS
    notes = []

    if session_duration < window_sec:
        window_sec = max(session_duration, 10.0)

    t_grid = np.arange(window_sec / 2, max(session_duration - window_sec / 2, window_sec / 2) + 1e-6, step_sec)
    if len(t_grid) == 0:
        t_grid = np.array([session_duration / 2.0])

    # مكوّن توافق القلب والعقل
    hb_t, hb_v = np.array([]), np.array([])
    if heart_brain_result is not None and heart_brain_result.coherence_ts is not None:
        hb_t = heart_brain_result.coherence_ts.t_sec
        hb_v = heart_brain_result.coherence_ts.coherence_pct
    if len(hb_t) == 0 and heart_brain_result is not None and heart_brain_result.synchrony_ts is not None:
        hb_t = heart_brain_result.synchrony_ts["t_sec"].to_numpy()
        hb_v = heart_brain_result.synchrony_ts["synchrony_pct"].to_numpy()
    comp_heart = _interp_series(hb_t, hb_v, t_grid)
    if np.all(np.isnan(comp_heart)) and heart_brain_result is not None and not np.isnan(heart_brain_result.overall_pct):
        comp_heart = np.full_like(t_grid, heart_brain_result.overall_pct)
        notes.append("مكوّن توافق القلب والعقل ثابت (قيمة إجمالية واحدة) لعدم كفاية البيانات لسلسلة زمنية.")
    elif np.all(np.isnan(comp_heart)):
        notes.append("تعذّر حساب مكوّن توافق القلب والعقل — سيُستبعد من المعادلة.")

    # مكوّن تزامن الفصين
    hemi_t, hemi_v = _hemispheric_to_series(hemispheric_result) if hemispheric_result else (np.array([]), np.array([]))
    comp_hemi = _interp_series(hemi_t, hemi_v, t_grid)
    if np.all(np.isnan(comp_hemi)):
        notes.append("تعذّر حساب مكوّن تزامن الفصين — سيُستبعد من المعادلة.")

    # مكوّن نشاط طفرات جاما
    comp_gamma = _gamma_rate_series(np.array(gamma_bursts_t), t_grid, window_sec, gamma_rate_cap)

    # الدمج المرجّح — نستبعد أي مكوّن غير متاح تماماً بدلاً من معاملته كصفر
    score = np.zeros_like(t_grid, dtype=float)
    for i in range(len(t_grid)):
        vals, ws = [], []
        for comp, w in ((comp_heart, weights.get("heart_brain", 0)),
                        (comp_hemi, weights.get("hemispheric", 0)),
                        (comp_gamma, weights.get("gamma", 0))):
            v = comp[i]
            if not np.isnan(v):
                vals.append(v)
                ws.append(w)
        score[i] = np.average(vals, weights=ws) if vals else np.nan

    valid = ~np.isnan(score)
    overall_mean = float(np.mean(score[valid])) if valid.any() else float("nan")

    peak_mask = valid & (score >= alert_threshold)
    peak_df = pd.DataFrame({
        "t_sec": t_grid[peak_mask],
        "score": score[peak_mask],
    })

    return NeuroplasticityResult(
        t_sec=t_grid, score=score,
        component_heart_brain=comp_heart, component_hemispheric=comp_hemi, component_gamma=comp_gamma,
        weights=weights, alert_threshold=alert_threshold,
        peak_moments=peak_df, overall_mean=overall_mean, notes=notes,
    )
