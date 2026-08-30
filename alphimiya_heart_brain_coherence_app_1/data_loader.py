"""
وحدة تحميل واستخراج تيارات البيانات من ملف Mind Monitor CSV (Muse 2).

ملف Mind Monitor الواحد يحزم عدة "تيارات" بيانات مختلفة الوتيرة في نفس الجدول:
قوى الموجات (Band Powers) وإشارات القلب (PPG/Heart_Rate) وجودة الاتصال (HSI) عادة
تصل معاً على نفس الصفوف، بينما الإشارة الخام (RAW) وأحداث الرمش/شد الفك (Elements)
قد تصل على صفوف مستقلة. هذه الوحدة تتعامل مع الحالتين دون افتراض مسبق: كل تيار
يُستخرج بشكل مستقل بحسب الأعمدة غير الفارغة في كل صف، ثم يُحاذى زمنياً عند الحاجة.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

ELECTRODES = ["TP9", "AF7", "AF8", "TP10"]
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
BAND_COLS = [f"{b}_{e}" for b in BANDS for e in ELECTRODES]
RAW_COLS = [f"RAW_{e}" for e in ELECTRODES]
PPG_COLS = ["PPG_Ambient", "PPG_IR", "PPG_Red"]
HSI_COLS = [f"HSI_{e}" for e in ELECTRODES]

# نطاقات التردد التقريبية المستخدمة داخلياً من قبل Muse لحساب قوى الموجات (هرتز)
BAND_FREQ_RANGES = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta": (12.0, 30.0),
    "Gamma": (30.0, 44.0),
}


@dataclass
class Streams:
    raw_df: pd.DataFrame              # كل صفوف الملف كما وردت
    band_df: pd.DataFrame             # صفوف قوى الموجات (+ PPG/HR/HSI إن كانت على نفس الصف)
    eeg_raw_df: pd.DataFrame          # صفوف الإشارة الخام RAW_*
    ppg_df: pd.DataFrame              # صفوف إشارة PPG (Ambient/IR/Red)
    hr_df: pd.DataFrame               # صفوف معدل ضربات القلب Heart_Rate
    hsi_df: pd.DataFrame              # صفوف جودة الاتصال HSI
    markers_df: pd.DataFrame          # صفوف الأحداث (رمش/شد فك/أخرى)
    start_time: pd.Timestamp = None
    end_time: pd.Timestamp = None
    eeg_raw_fs: float = None          # التردد الفعلي المُقاس للإشارة الخام (هرتز) أو None
    ppg_fs: float = None              # التردد الفعلي المُقاس لإشارة PPG (هرتز) أو None
    warnings: list = field(default_factory=list)


def _estimate_fs(t_sec: pd.Series, min_points: int = 8) -> float | None:
    """يقدّر معدل العينات الفعلي (هرتز) من فروق الطابع الزمني — لا نفترض قيمة ثابتة مسبقاً."""
    if len(t_sec) < min_points:
        return None
    diffs = np.diff(np.sort(t_sec.to_numpy()))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return None
    return 1.0 / median_dt


def load_mind_monitor_csv(file) -> Streams:
    df = pd.read_csv(file)

    warnings = []
    if "TimeStamp" not in df.columns:
        raise ValueError("الملف لا يحتوي على عمود TimeStamp — تأكد أنه ملف Mind Monitor صحيح.")

    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    df = df.dropna(subset=["TimeStamp"]).sort_values("TimeStamp").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("لم يتم العثور على صفوف صالحة بعد قراءة الأعمدة الزمنية.")

    start_time = df["TimeStamp"].iloc[0]
    end_time = df["TimeStamp"].iloc[-1]
    df["t_sec"] = (df["TimeStamp"] - start_time).dt.total_seconds()

    def extract(cols):
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.DataFrame(columns=["TimeStamp", "t_sec"] + cols), []
        mask = df[present].notna().any(axis=1)
        sub = df.loc[mask, ["TimeStamp", "t_sec"] + present].copy()
        return sub, present

    band_df, band_present = extract(BAND_COLS)
    eeg_raw_df, raw_present = extract(RAW_COLS)
    ppg_df, ppg_present = extract(PPG_COLS)
    hr_df, hr_present = extract(["Heart_Rate"])
    hsi_df, hsi_present = extract(HSI_COLS)
    markers_df = df[df["Elements"].notna()].copy() if "Elements" in df.columns else pd.DataFrame(columns=df.columns)

    if not band_present:
        warnings.append("لم يتم العثور على أعمدة قوى الموجات (Band Powers) في الملف.")
    if not raw_present:
        warnings.append("لا توجد بيانات إشارة خام (RAW) — سيتم الاعتماد على قوى الموجات فقط لحساب تزامن الفصين.")
    if not ppg_present:
        warnings.append("لا توجد بيانات نبض (PPG) — سيُعتمد على عمود Heart_Rate فقط لحساب توافق القلب والعقل.")
    if not hr_present:
        warnings.append("لا يوجد عمود Heart_Rate في الملف — لن يمكن حساب توافق القلب والعقل.")

    if len(band_df) == 0 and len(hr_df) == 0:
        raise ValueError("لم يتم العثور على أي بيانات EEG أو معدل ضربات قلب صالحة للتحليل في هذا الملف.")

    eeg_raw_fs = _estimate_fs(eeg_raw_df["t_sec"]) if len(eeg_raw_df) else None
    ppg_fs = _estimate_fs(ppg_df["t_sec"]) if len(ppg_df) else None

    return Streams(
        raw_df=df,
        band_df=band_df,
        eeg_raw_df=eeg_raw_df,
        ppg_df=ppg_df,
        hr_df=hr_df,
        hsi_df=hsi_df,
        markers_df=markers_df,
        start_time=start_time,
        end_time=end_time,
        eeg_raw_fs=eeg_raw_fs,
        ppg_fs=ppg_fs,
        warnings=warnings,
    )


def add_band_row_means(band_df: pd.DataFrame) -> pd.DataFrame:
    """يضيف عمود متوسط لكل موجة عبر الأقطاب الأربعة، وأعمدة متوسط جبهي (AF7/AF8) وخلفي (TP9/TP10)."""
    df = band_df.copy()
    for band in BANDS:
        cols_all = [f"{band}_{e}" for e in ELECTRODES if f"{band}_{e}" in df.columns]
        if cols_all:
            df[f"{band}_mean"] = df[cols_all].mean(axis=1)
        frontal = [c for c in (f"{band}_AF7", f"{band}_AF8") if c in df.columns]
        if frontal:
            df[f"{band}_frontal"] = df[frontal].mean(axis=1)
    return df


def filter_band_df_by_hsi(band_df: pd.DataFrame, hsi_threshold: float = 2.0) -> tuple[pd.DataFrame, dict]:
    """يستبعد صفوف قوى الموجات التي تكون فيها جودة اتصال أي قطب أسوأ من العتبة (نفس منطق التطبيق الأول)."""
    present = [c for c in HSI_COLS if c in band_df.columns]
    info = {"total_rows": len(band_df)}
    if not present:
        info.update(removed_rows=0, remaining_rows=len(band_df))
        return band_df.copy(), info
    max_hsi = band_df[present].max(axis=1)
    good = max_hsi <= hsi_threshold
    info.update(removed_rows=int((~good).sum()), remaining_rows=int(good.sum()))
    return band_df[good].copy(), info


def tag_quality_by_nearest_hsi(
    target_df: pd.DataFrame, hsi_df: pd.DataFrame, electrodes: list[str], tolerance_sec: float = 3.0
) -> pd.DataFrame:
    """
    يُلحق بكل صف من target_df أقرب قراءة HSI زمنياً لكل قطب مطلوب (ضمن نافذة تسامح)،
    مفيد عندما تكون RAW/PPG على صفوف منفصلة عن صفوف HSI.
    """
    present = [f"HSI_{e}" for e in electrodes if f"HSI_{e}" in hsi_df.columns]
    if len(hsi_df) == 0 or not present or len(target_df) == 0:
        out = target_df.copy()
        for c in present:
            out[c] = np.nan
        return out

    left = target_df.sort_values("t_sec")
    right = hsi_df[["t_sec"] + present].sort_values("t_sec")
    merged = pd.merge_asof(
        left, right, on="t_sec", direction="nearest", tolerance=tolerance_sec
    )
    return merged
