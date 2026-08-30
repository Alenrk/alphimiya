"""
حساب مؤشر "توافق القلب والعقل" (Heart-Brain Coherence).

المنهجية (مبسّطة ومُوثّقة بوضوح، مستوحاة من نهج معهد HeartMath لكنها ليست
الخوارزمية المسجّلة الأصلية):

  1) استخراج سلسلة الفواصل بين النبضات (IBI) إما من ذروات إشارة PPG_IR الخام
     (إذا كان معدل العينات الفعلي كافياً)، أو من عمود Heart_Rate الجاهز كبديل
     دائماً متاح (60 / معدل النبض = IBI بالثواني).
  2) إعادة أخذ العينات لسلسلة IBI على شبكة زمنية منتظمة (4Hz) — ممارسة معيارية
     في تحليل HRV قبل التحويل الطيفي.
  3) حساب كثافة الطيف الطاقي (Welch PSD) ضمن نوافذ زمنية متحركة، واستخراج
     "نسبة التوافق" = قوة الذروة ضمن نطاق التوافق (0.04–0.26Hz) مقسومة على
     باقي القوة الكلية — مبدأ مشابه لمؤشر HeartMath coherence ratio.
  4) حساب "التزامن مع الجبهة الدماغية": الارتباط المتحرك (rolling correlation)
     بين معدل النبض وقوة موجتي ألفا+ثيتا الجبهيتين (AF7/AF8).
  5) الدمج النهائي = مزيج مرجّح (قابل للتعديل) بين نسبة توافق القلب وتزامن
     القلب-الدماغ، مُطبَّع إلى نسبة مئوية 0–100%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal

COHERENCE_BAND = (0.04, 0.26)   # نطاق التوافق (هرتز) — مشابه لنطاق HeartMath
TOTAL_BAND = (0.0033, 0.4)      # نطاق الطاقة الكلية المعتمد لتحليل HRV قصير المدى
MIN_PPG_FS_FOR_PEAKS = 15.0     # الحد الأدنى لمعدل عينات PPG لاعتماد كشف الذروات مباشرة
RESAMPLE_FS = 4.0               # معدل إعادة أخذ العينات القياسي لتحليل HRV


@dataclass
class HRVResult:
    method: str                      # "ppg_peaks" أو "heart_rate_column"
    ibi_t: np.ndarray                # الأزمنة الأصلية لعينات IBI (ثانية)
    ibi_sec: np.ndarray               # قيم IBI بالثواني
    mean_hr: float
    sdnn: float                       # الانحراف المعياري لفواصل NN (مؤشر HRV كلاسيكي)
    rmssd: float                      # الجذر التربيعي لمتوسط مربعات الفروق المتتالية
    quality_note: str = ""


@dataclass
class CoherenceTimeSeries:
    t_sec: np.ndarray
    coherence_ratio: np.ndarray
    coherence_pct: np.ndarray
    window_sec: float
    insufficient_data: bool = False


@dataclass
class HeartBrainResult:
    hrv: HRVResult | None
    coherence_ts: CoherenceTimeSeries | None
    synchrony_pct_mean: float
    synchrony_ts: pd.DataFrame | None
    overall_pct: float
    weights: tuple
    notes: list = field(default_factory=list)


def _bandpass(sig_vals: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    low_n = max(0.001, low / nyq)
    high_n = min(0.99, high / nyq)
    if low_n >= high_n:
        return sig_vals - np.nanmean(sig_vals)
    b, a = signal.butter(order, [low_n, high_n], btype="band")
    vals = np.nan_to_num(sig_vals, nan=np.nanmean(sig_vals))
    try:
        return signal.filtfilt(b, a, vals)
    except Exception:
        return vals - np.mean(vals)


def extract_ibi_from_ppg(ppg_df: pd.DataFrame, fs: float) -> HRVResult | None:
    """يستخرج فواصل النبض من ذروات إشارة PPG_IR الخام إذا كان معدل العينات كافياً."""
    if ppg_df is None or len(ppg_df) < 10 or fs is None or fs < MIN_PPG_FS_FOR_PEAKS:
        return None
    if "PPG_IR" not in ppg_df.columns:
        return None

    d = ppg_df.dropna(subset=["PPG_IR"]).sort_values("t_sec")
    if len(d) < 10:
        return None

    t = d["t_sec"].to_numpy()
    x = d["PPG_IR"].to_numpy().astype(float)
    filtered = _bandpass(x, fs, 0.7, min(3.5, fs / 2 * 0.9))

    min_distance = max(1, int(fs * 60 / 200))  # لا يزيد معدل القلب المفترض عن 200 نبضة/دقيقة
    peaks, _ = signal.find_peaks(filtered, distance=min_distance, prominence=np.std(filtered) * 0.3)
    if len(peaks) < 6:
        return None

    peak_times = t[peaks]
    ibi = np.diff(peak_times)
    ibi_t = peak_times[1:]

    # استبعاد فواصل غير فيزيولوجية (نبض أقل من 30 أو أعلى من 220 بالدقيقة)
    valid = (ibi > 60 / 220) & (ibi < 60 / 30)
    ibi_t, ibi = ibi_t[valid], ibi[valid]
    if len(ibi) < 5:
        return None

    hr_inst = 60.0 / ibi
    return HRVResult(
        method="ppg_peaks",
        ibi_t=ibi_t,
        ibi_sec=ibi,
        mean_hr=float(np.mean(hr_inst)),
        sdnn=float(np.std(ibi * 1000)),
        rmssd=float(np.sqrt(np.mean(np.diff(ibi * 1000) ** 2))) if len(ibi) > 1 else float("nan"),
        quality_note="مُستخرج من ذروات إشارة PPG_IR الخام",
    )


def extract_ibi_from_heart_rate(hr_df: pd.DataFrame) -> HRVResult | None:
    """يشتق فواصل IBI تقريبية من عمود Heart_Rate الجاهز (60 / معدل النبض)."""
    if hr_df is None or len(hr_df) < 5:
        return None
    d = hr_df.dropna(subset=["Heart_Rate"]).sort_values("t_sec")
    d = d[(d["Heart_Rate"] > 30) & (d["Heart_Rate"] < 220)]
    if len(d) < 5:
        return None

    t = d["t_sec"].to_numpy()
    hr = d["Heart_Rate"].to_numpy().astype(float)
    ibi = 60.0 / hr

    return HRVResult(
        method="heart_rate_column",
        ibi_t=t,
        ibi_sec=ibi,
        mean_hr=float(np.mean(hr)),
        sdnn=float(np.std(ibi * 1000)),
        rmssd=float(np.sqrt(np.mean(np.diff(ibi * 1000) ** 2))) if len(ibi) > 1 else float("nan"),
        quality_note="مُشتق من عمود Heart_Rate الجاهز من الجهاز (دقة زمنية أقل من PPG الخام)",
    )


def get_best_hrv(ppg_df: pd.DataFrame, ppg_fs: float | None, hr_df: pd.DataFrame) -> HRVResult | None:
    ppg_result = extract_ibi_from_ppg(ppg_df, ppg_fs) if ppg_fs else None
    if ppg_result is not None:
        return ppg_result
    return extract_ibi_from_heart_rate(hr_df)


def compute_coherence_timeseries(
    hrv: HRVResult, window_sec: float = 60.0, step_sec: float = 15.0
) -> CoherenceTimeSeries:
    """يحسب نسبة توافق HRV ضمن نوافذ متحركة عبر الجلسة (أسلوب Welch PSD)."""
    if hrv is None or len(hrv.ibi_t) < 5:
        return CoherenceTimeSeries(np.array([]), np.array([]), np.array([]), window_sec, insufficient_data=True)

    t0, t1 = hrv.ibi_t.min(), hrv.ibi_t.max()
    duration = t1 - t0

    # إن كانت الجلسة أقصر من نافذة واحدة، استخدم كامل المدة كنافذة واحدة فقط
    eff_window = min(window_sec, max(duration, 10.0))
    if duration < 20:
        return CoherenceTimeSeries(np.array([]), np.array([]), np.array([]), window_sec, insufficient_data=True)

    grid_t = np.arange(t0, t1, 1.0 / RESAMPLE_FS)
    if len(grid_t) < 8:
        return CoherenceTimeSeries(np.array([]), np.array([]), np.array([]), window_sec, insufficient_data=True)

    # إعادة أخذ عينات IBI على شبكة منتظمة (استيفاء خطي، تجنباً لتذبذبات كثيرة الحدود مع بيانات قليلة)
    order = np.argsort(hrv.ibi_t)
    ibi_interp = np.interp(grid_t, hrv.ibi_t[order], hrv.ibi_sec[order])

    centers, ratios, pct = [], [], []
    half = eff_window / 2.0
    step = max(step_sec, eff_window / 4)
    c = t0 + half
    while c <= t1 - half + 1e-6:
        m = (grid_t >= c - half) & (grid_t < c + half)
        seg = ibi_interp[m]
        if len(seg) >= 16:
            seg = seg - np.mean(seg)
            nperseg = min(len(seg), 256)
            freqs, psd = signal.welch(seg, fs=RESAMPLE_FS, nperseg=nperseg)

            band_mask = (freqs >= COHERENCE_BAND[0]) & (freqs <= COHERENCE_BAND[1])
            total_mask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])

            if band_mask.any() and total_mask.any():
                peak_idx_local = np.argmax(psd[band_mask])
                peak_freq = freqs[band_mask][peak_idx_local]
                near_peak = np.abs(freqs - peak_freq) <= 0.015
                peak_power = float(np.sum(psd[near_peak & total_mask]))
                total_power = float(np.sum(psd[total_mask]))
                remainder = max(total_power - peak_power, 1e-9)
                ratio = peak_power / remainder
                centers.append(c)
                ratios.append(ratio)
                pct.append(100.0 * ratio / (ratio + 1.0))
        c += step

    if not centers:
        return CoherenceTimeSeries(np.array([]), np.array([]), np.array([]), window_sec, insufficient_data=True)

    return CoherenceTimeSeries(
        t_sec=np.array(centers), coherence_ratio=np.array(ratios), coherence_pct=np.array(pct),
        window_sec=eff_window, insufficient_data=False,
    )


def compute_eeg_hr_synchrony(
    hrv: HRVResult, band_df: pd.DataFrame, window_sec: float = 30.0
) -> tuple[float, pd.DataFrame | None]:
    """
    الارتباط المتحرك بين معدل ضربات القلب اللحظي وقوة موجتي ألفا+ثيتا الجبهية.
    يعيد: (متوسط نسبة التزامن %, جدول زمني للتزامن).
    """
    if hrv is None or len(hrv.ibi_t) < 5 or band_df is None or len(band_df) < 5:
        return float("nan"), None
    if "Alpha_frontal" not in band_df.columns or "Theta_frontal" not in band_df.columns:
        return float("nan"), None

    hr_inst = 60.0 / hrv.ibi_sec
    hr_series = pd.Series(hr_inst, index=hrv.ibi_t).sort_index()

    eeg = band_df[["t_sec", "Alpha_frontal", "Theta_frontal"]].dropna().sort_values("t_sec")
    if len(eeg) < 5:
        return float("nan"), None
    eeg_signal = (eeg["Alpha_frontal"] + eeg["Theta_frontal"]).to_numpy()
    eeg_t = eeg["t_sec"].to_numpy()

    # إسقاط كلا التسلسلين على شبكة زمنية مشتركة موحّدة (1 هرتز كافية لهذا الغرض)
    t0 = max(hr_series.index.min(), eeg_t.min())
    t1 = min(hr_series.index.max(), eeg_t.max())
    if t1 - t0 < 20:
        return float("nan"), None

    grid = np.arange(t0, t1, 1.0)
    hr_grid = np.interp(grid, hr_series.index.to_numpy(), hr_series.to_numpy())
    eeg_grid = np.interp(grid, eeg_t, eeg_signal)

    win_pts = max(int(window_sec), 10)
    corrs, centers = [], []
    for i in range(0, len(grid) - win_pts, max(1, win_pts // 2)):
        a = hr_grid[i:i + win_pts]
        b = eeg_grid[i:i + win_pts]
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            r = np.corrcoef(a, b)[0, 1]
            if not np.isnan(r):
                corrs.append(r)
                centers.append(grid[i + win_pts // 2])

    if not corrs:
        return float("nan"), None

    corrs = np.array(corrs)
    pct = 100.0 * (np.abs(corrs))
    df_out = pd.DataFrame({"t_sec": centers, "correlation": corrs, "synchrony_pct": pct})
    return float(np.mean(pct)), df_out


def compute_heart_brain_coherence(
    ppg_df: pd.DataFrame,
    ppg_fs: float | None,
    hr_df: pd.DataFrame,
    band_df: pd.DataFrame,
    weight_hrv: float = 0.6,
    weight_sync: float = 0.4,
    window_sec: float = 60.0,
) -> HeartBrainResult:
    notes = []
    hrv = get_best_hrv(ppg_df, ppg_fs, hr_df)
    if hrv is None:
        notes.append("تعذّر استخراج بيانات نبض كافية (لا PPG عالي المعدل ولا Heart_Rate كافٍ).")
        return HeartBrainResult(None, None, float("nan"), None, float("nan"), (weight_hrv, weight_sync), notes)

    notes.append(f"طريقة استخراج النبض: {hrv.quality_note}")
    coherence_ts = compute_coherence_timeseries(hrv, window_sec=window_sec)
    if coherence_ts.insufficient_data:
        notes.append("مدة الجلسة قصيرة جداً لحساب نسبة توافق HRV بدقة (يُفضَّل 60 ثانية فأكثر).")

    sync_mean, sync_ts = compute_eeg_hr_synchrony(hrv, band_df)
    if sync_ts is None:
        notes.append("تعذّر حساب تزامن القلب مع الجبهة الدماغية (بيانات ألفا/ثيتا الجبهية غير كافية).")

    hrv_pct = float(np.mean(coherence_ts.coherence_pct)) if len(coherence_ts.coherence_pct) else float("nan")

    parts, weights_used = [], []
    if not np.isnan(hrv_pct):
        parts.append(hrv_pct)
        weights_used.append(weight_hrv)
    if not np.isnan(sync_mean):
        parts.append(sync_mean)
        weights_used.append(weight_sync)

    if parts:
        overall = float(np.average(parts, weights=weights_used))
    else:
        overall = float("nan")
        notes.append("تعذّر حساب مؤشر توافق القلب والعقل النهائي لعدم كفاية البيانات.")

    return HeartBrainResult(
        hrv=hrv,
        coherence_ts=coherence_ts,
        synchrony_pct_mean=sync_mean,
        synchrony_ts=sync_ts,
        overall_pct=overall,
        weights=(weight_hrv, weight_sync),
        notes=notes,
    )
