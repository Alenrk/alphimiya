"""
قياس تزامن الفصين (Hemispheric Coherence) بين القناتين الجبهيتين (AF7 مقابل
AF8) والقناتين الخلفيتين/الصدغيتين (TP9 مقابل TP10).

تُستخدم طريقتان بحسب توفر البيانات وجودتها، ويختار التطبيق تلقائياً الأنسب
لكل نطاق تردد (موجة) على حدة:

  • "طيفية دقيقة": تُحسب من الإشارة الخام (RAW_*) باستخدام التماسك الطيفي
    (magnitude-squared coherence, عبر scipy.signal.coherence) ضمن نوافذ
    متحركة — لكن فقط للنطاقات التي يسمح معدل العينات الفعلي المُقاس
    بحلّها ترددياً (نظرية Nyquist).
  • "تقديرية من قوة الموجات": ارتباط منزلق (rolling correlation) بين
    سلاسل قوة كل موجة للقناتين — تعمل دائماً كخيار احتياطي موثوق.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal

from data_loader import BAND_FREQ_RANGES

NYQUIST_SAFETY_FACTOR = 2.2  # هامش أمان فوق نظرية Nyquist (fs >= factor × الحد الأعلى للنطاق)


@dataclass
class PairCoherenceResult:
    pair_label: str                 # "الجبهي (AF7-AF8)" أو "الخلفي (TP9-TP10)"
    per_band: dict                  # band -> {"method":..., "t_sec":..., "coherence_pct":..., "mean_pct":...}
    overall_pct: float
    notes: list = field(default_factory=list)


def _raw_band_coherence(
    eeg_raw_df: pd.DataFrame, fs: float, col_a: str, col_b: str, band: str,
    window_sec: float = 2.0, step_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    d = eeg_raw_df[["t_sec", col_a, col_b]].dropna().sort_values("t_sec")
    if len(d) < fs * window_sec:
        return None

    t = d["t_sec"].to_numpy()
    a = d[col_a].to_numpy().astype(float)
    b = d[col_b].to_numpy().astype(float)

    win_pts = max(8, int(round(window_sec * fs)))
    step_pts = max(1, int(round(step_sec * fs)))
    low, high = BAND_FREQ_RANGES[band]

    centers, vals = [], []
    for start in range(0, len(a) - win_pts, step_pts):
        seg_a = a[start:start + win_pts]
        seg_b = b[start:start + win_pts]
        freqs, coh = signal.coherence(seg_a, seg_b, fs=fs, nperseg=min(win_pts, 128))
        mask = (freqs >= low) & (freqs <= high)
        if mask.any():
            centers.append(t[start + win_pts // 2])
            vals.append(float(np.mean(coh[mask])))

    if not centers:
        return None
    return np.array(centers), np.array(vals) * 100.0


def _bandpower_rolling_correlation(
    band_df: pd.DataFrame, col_a: str, col_b: str, window_sec: float = 10.0
) -> tuple[np.ndarray, np.ndarray] | None:
    d = band_df[["t_sec", col_a, col_b]].dropna().sort_values("t_sec")
    if len(d) < 8:
        return None

    t = d["t_sec"].to_numpy()
    a = d[col_a].to_numpy().astype(float)
    b = d[col_b].to_numpy().astype(float)

    median_dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    win = max(5, int(round(window_sec / max(median_dt, 1e-3))))
    if win >= len(a):
        win = max(5, len(a) // 2)
    step = max(1, win // 2)

    centers, vals = [], []
    for start in range(0, len(a) - win, step):
        seg_a = a[start:start + win]
        seg_b = b[start:start + win]
        if np.std(seg_a) > 1e-9 and np.std(seg_b) > 1e-9:
            r = np.corrcoef(seg_a, seg_b)[0, 1]
            if not np.isnan(r):
                centers.append(t[start + win // 2])
                vals.append((r + 1.0) / 2.0 * 100.0)  # تطبيع من [-1,1] إلى [0,100]%

    if not centers:
        return None
    return np.array(centers), np.array(vals)


def compute_pair_coherence(
    pair_label: str,
    electrode_a: str,
    electrode_b: str,
    eeg_raw_df: pd.DataFrame | None,
    eeg_raw_fs: float | None,
    band_df: pd.DataFrame,
    bands: list[str] = ("Theta", "Alpha", "Beta", "Gamma"),
) -> PairCoherenceResult:
    notes = []
    per_band = {}

    for band in bands:
        low, high = BAND_FREQ_RANGES[band]
        can_use_raw = (
            eeg_raw_df is not None
            and eeg_raw_fs is not None
            and eeg_raw_fs >= high * NYQUIST_SAFETY_FACTOR
            and f"RAW_{electrode_a}" in eeg_raw_df.columns
            and f"RAW_{electrode_b}" in eeg_raw_df.columns
        )
        result = None
        method = None
        if can_use_raw:
            result = _raw_band_coherence(eeg_raw_df, eeg_raw_fs, f"RAW_{electrode_a}", f"RAW_{electrode_b}", band)
            method = "طيفية دقيقة (إشارة خام)"
        if result is None:
            col_a, col_b = f"{band}_{electrode_a}", f"{band}_{electrode_b}"
            if col_a in band_df.columns and col_b in band_df.columns:
                result = _bandpower_rolling_correlation(band_df, col_a, col_b)
                method = "تقديرية (ارتباط قوة الموجة)"

        if result is None:
            notes.append(f"تعذّر حساب تزامن موجة {band} بين {electrode_a} و{electrode_b} — بيانات غير كافية.")
            continue

        t_sec, pct = result
        per_band[band] = {"method": method, "t_sec": t_sec, "coherence_pct": pct, "mean_pct": float(np.mean(pct))}

    if per_band:
        overall = float(np.mean([v["mean_pct"] for v in per_band.values()]))
    else:
        overall = float("nan")
        notes.append("تعذّر حساب أي قيمة لتزامن هذا الزوج من الأقطاب.")

    return PairCoherenceResult(pair_label=pair_label, per_band=per_band, overall_pct=overall, notes=notes)


def compute_hemispheric_coherence(
    eeg_raw_df: pd.DataFrame | None,
    eeg_raw_fs: float | None,
    band_df: pd.DataFrame,
    bands: list[str] = ("Theta", "Alpha", "Beta", "Gamma"),
) -> dict:
    frontal = compute_pair_coherence("الجبهي (AF7 × AF8)", "AF7", "AF8", eeg_raw_df, eeg_raw_fs, band_df, bands)
    posterior = compute_pair_coherence("الخلفي (TP9 × TP10)", "TP9", "TP10", eeg_raw_df, eeg_raw_fs, band_df, bands)

    vals = [v for v in (frontal.overall_pct, posterior.overall_pct) if not np.isnan(v)]
    overall = float(np.mean(vals)) if vals else float("nan")

    return {"frontal": frontal, "posterior": posterior, "overall_pct": overall}
