"""
يبني الفقرات السردية "المحمّسة" (العناوين والملخص فقط) لتقرير PDF —
بالاعتماد حصراً على الأرقام الفعلية المحسوبة لهذه الجلسة، دون أي مبالغة
أو ادعاءات غير مدعومة بالبيانات. الجداول والملاحظات المنهجية في التقرير
نفسه تبقى بأسلوب علمي محايد (تُبنى مباشرة في pdf_report.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def fmt_time(t_sec: float) -> str:
    if t_sec != t_sec:  # NaN
        return "—"
    t_sec = max(0, int(round(t_sec)))
    m, s = divmod(t_sec, 60)
    return f"{m:02d}:{s:02d}"


@dataclass
class ReportNarrative:
    headline: str
    summary: str
    peak_story: list = field(default_factory=list)
    closing: str = ""


def _grade(pct: float) -> str:
    """وصف نوعي مختصر لنسبة مئوية — يُستخدم فقط داخل الجمل التحفيزية."""
    if pct != pct:
        return ""
    if pct >= 75:
        return "مرتفعة ومبشّرة جداً"
    if pct >= 55:
        return "جيدة وفي تحسّن"
    if pct >= 35:
        return "معتدلة، وهي نقطة انطلاق جيدة"
    return "في بداياتها — وكل رحلة تبدأ بخطوة"


def build_narrative(
    *,
    heart_brain,
    hemispheric: dict,
    bursts: list,
    neuro,
    session_duration: float,
    alert_threshold: float,
) -> ReportNarrative:
    hb_pct = heart_brain.overall_pct if heart_brain is not None else float("nan")
    hemi_pct = hemispheric.get("overall_pct", float("nan")) if hemispheric else float("nan")
    neuro_pct = neuro.overall_mean if neuro is not None else float("nan")

    available = {
        "توافق القلب والعقل": hb_pct,
        "تزامن الفصين": hemi_pct,
        "مؤشر التكيف العصبي": neuro_pct,
    }
    valid = {k: v for k, v in available.items() if v == v}

    n_bursts = len(bursts) if bursts else 0
    n_peaks = len(neuro.peak_moments) if (neuro is not None and neuro.peak_moments is not None) else 0

    # ------- العنوان الرئيسي -------
    headline = "رحلتك اليوم نحو التوافق الداخلي — تقرير جلسة ALPHIMIYA®"

    # ------- فقرة الملخص المحمّسة -------
    duration_txt = fmt_time(session_duration)
    parts = [
        f"أحسنت! لقد أكملت للتو جلسة استغرقت {duration_txt} (دقيقة:ثانية) من الحضور الداخلي، "
        f"وجهازك سجّل كل لحظة منها بدقة لنكتشف معاً ما الذي حدث في عقلك وقلبك."
    ]

    if valid:
        best_label = max(valid, key=valid.get)
        best_val = valid[best_label]
        parts.append(
            f"أبرز ما لفت الانتباه هذه المرة هو {best_label}، الذي بلغ {best_val:.0f}% — "
            f"وهو مستوى {_grade(best_val)}."
        )
    else:
        parts.append(
            "لم تتوفر بيانات كافية لحساب المؤشرات المركّبة بثقة في هذه الجلسة القصيرة أو المشوَّشة قليلاً، "
            "لكن هذا لا يقلّل من قيمة وقتك الذي قضيته — كل تسجيل يمنحك بيانات أدق للمرة القادمة."
        )

    if n_bursts > 0:
        rate = n_bursts * 60.0 / session_duration if session_duration > 0 else 0.0
        parts.append(
            f"كما رصد التطبيق {n_bursts} طفرة من طفرات موجة غاما الفجائية (بمعدل {rate:.1f} طفرة/دقيقة) — "
            "لحظات استكشافية مثيرة تستحق أن تعود إليها في قسم القصص أدناه."
        )
    else:
        parts.append(
            "لم تُرصد طفرات غاما فجائية واضحة بالعتبة الحالية هذه المرة — جرّب خفض عتبة الحساسية من "
            "اللوحة الجانبية إن أردت اكتشاف تحركات أدق، أو استمتع ببساطة بثبات الجلسة."
        )

    if n_peaks > 0:
        parts.append(
            f"والأهم: وصلت جلستك إلى منطقة \"التكيف العصبي المثالي\" (فوق عتبة {alert_threshold:.0f}%) "
            f"في {n_peaks} لحظة مسجّلة — تفاصيلها الكاملة في القسم المخصص لأبرز لحظات ذروتك بالأسفل!"
        )
    elif neuro_pct == neuro_pct:
        parts.append(
            f"لم تتجاوز الجلسة عتبة التنبيه ({alert_threshold:.0f}%) هذه المرة، وهذا طبيعي تماماً — "
            "التوافق العصبي يُبنى بالتكرار والانتظام، وكل جلسة تضيف طبقة جديدة من التمرّس."
        )

    summary = " ".join(parts)

    closing = (
        "استمر في هذه الممارسة بانتظام — فوفق أدبيات التأمل والدراسات الملهمة حول إعادة التشكيل العصبي، "
        "الاتساق عبر الزمن هو ما يصنع الفرق الحقيقي، لا جلسة واحدة بمفردها."
    )

    return ReportNarrative(headline=headline, summary=summary, peak_story=[], closing=closing)


def build_peak_story(
    *,
    bursts: list,
    neuro,
    top_n: int = 5,
) -> list[str]:
    """
    يبني قائمة جمل سردية محمّسة لأبرز اللحظات (أقوى طفرات غاما + لحظات ذروة
    التكيف العصبي) مرتّبة زمنياً — كل جملة مبنية على رقم فعلي من الجلسة.
    """
    items = []

    if bursts:
        top_bursts = sorted(bursts, key=lambda b: b.z_score, reverse=True)[:top_n]
        for b in top_bursts:
            items.append({
                "t_sec": b.t_sec,
                "text": (
                    f"عند الدقيقة {fmt_time(b.t_sec)}، رصد الجهاز طفرة غاما {b.intensity} "
                    f"(Z-score = {b.z_score:.1f}) على قناة {b.channel} — لحظة استبصار متلألئة تستحق التأمل فيها."
                ),
            })

    if neuro is not None and neuro.peak_moments is not None and len(neuro.peak_moments):
        top_peaks = neuro.peak_moments.sort_values("score", ascending=False).head(top_n)
        for _, row in top_peaks.iterrows():
            items.append({
                "t_sec": float(row["t_sec"]),
                "text": (
                    f"عند الدقيقة {fmt_time(row['t_sec'])}، بلغ مؤشر التكيف العصبي المركّب "
                    f"{row['score']:.0f}% — إحدى أجمل لحظات انسجامك الداخلي في هذه الجلسة."
                ),
            })

    items.sort(key=lambda x: x["t_sec"])
    return [it["text"] for it in items]
