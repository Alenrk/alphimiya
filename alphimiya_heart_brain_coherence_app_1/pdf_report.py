"""
مولّد تقرير PDF لجلسة ALPHIMIYA® (الفيمياء) — توافق القلب والعقل، طفرات
جاما، تزامن الفصين، ومؤشر التكيف العصبي المركّب.

العناوين وفقرة الملخص/القصص مكتوبة بأسلوب محمّس وتحفيزي، بينما الجداول
الرقمية والملاحظات المنهجية تبقى بأسلوب علمي محايد ودقيق — بخط Amiri
المُدمج مع تشكيل وترتيب RTL صحيح (البيانات الرقمية بالإنجليزية).
"""

from __future__ import annotations

import os
import re
from datetime import datetime

import arabic_reshaper
from bidi.algorithm import get_display
from fpdf import FPDF

from report_narrative import fmt_time

FONT_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")
FONT_REGULAR = os.path.join(FONT_DIR, "Amiri-Regular.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "Amiri-Bold.ttf")

INK = (24, 26, 38)
MUTED = (108, 114, 130)
VIOLET = (124, 58, 237)
CYAN = (8, 137, 158)
MAGENTA = (185, 28, 99)
AMBER = (176, 116, 18)
GOOD = (13, 130, 90)
LINE = (223, 225, 234)
CAVEAT = (150, 116, 30)


def _reshape(text: str) -> str:
    return arabic_reshaper.reshape(str(text))


def _visual(text: str) -> str:
    return get_display(_reshape(text))


_AR_RE = re.compile(r"[؀-ۿ]")


def _visual_if_arabic(text) -> str:
    """يطبّق التشكيل وترتيب RTL فقط على القيم التي تحتوي حروفاً عربية،
    ويترك القيم الرقمية/الإنجليزية البحتة كما هي (تُعرض LTR بشكل طبيعي)."""
    text = str(text)
    if _AR_RE.search(text):
        return _visual(text)
    return text


class ArabicReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.add_font("Amiri", "", FONT_REGULAR)
        self.add_font("Amiri", "B", FONT_BOLD)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(16, 16, 16)

    def footer(self):
        self.set_y(-14)
        self.set_font("Amiri", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, _visual(f"صفحة {self.page_no()} — ALPHIMIYA (الفيمياء)"), align="C")

    # ---------- عناصر نصية عربية بترتيب RTL صحيح ----------

    def ar_line(self, text: str, size=12, style="", color=INK, h=7, align="R"):
        self.set_font("Amiri", style, size)
        self.set_text_color(*color)
        self.cell(0, h, _visual(text), align=align, new_x="LMARGIN", new_y="NEXT")

    def ar_paragraph(self, text: str, size=11.5, style="", color=INK, line_h=6.8):
        """يلف فقرة عربية طويلة على عدة أسطر بترتيب RTL سليم."""
        self.set_font("Amiri", style, size)
        self.set_text_color(*color)
        max_w = self.w - self.l_margin - self.r_margin

        shaped_full = _reshape(text)
        words = shaped_full.split(" ")
        lines, current = [], ""
        for w in words:
            trial = (current + " " + w).strip() if current else w
            if self.get_string_width(trial) <= max_w:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)

        for ln in lines:
            self.cell(0, line_h, get_display(ln), align="R", new_x="LMARGIN", new_y="NEXT")

    def section_title(self, text: str, color=VIOLET):
        self.ln(3)
        self.set_draw_color(*color)
        self.set_line_width(0.6)
        y = self.get_y()
        self.line(self.l_margin, y + 7.5, self.w - self.r_margin, y + 7.5)
        self.ar_line(text, size=14, style="B", color=color, h=8)
        self.ln(2)

    def metric_row(self, label: str, value: str):
        self.set_font("Amiri", "", 11)
        self.set_text_color(*MUTED)
        self.cell(60, 7.2, _visual_if_arabic(value), align="L")
        self.set_text_color(*INK)
        self.set_font("Amiri", "B", 11)
        self.cell(0, 7.2, _visual(label + " :"), align="R", new_x="LMARGIN", new_y="NEXT")


def _fmt_pct(v: float) -> str:
    if v != v:
        return "— (بيانات غير كافية)"
    return f"{v:.1f}%"


def build_pdf(
    *,
    session_meta: dict,
    narrative,
    peak_story: list[str],
    heart_brain,
    hemispheric: dict,
    bursts_df,
    burst_rate: float,
    neuro,
    session_duration: float,
    alert_threshold: float,
    chart_image_path: str | None = None,
) -> bytes:
    pdf = ArabicReportPDF()
    pdf.add_page()

    # ------- العنوان الرئيسي (أسلوب محمّس) -------
    pdf.set_font("Amiri", "B", 20)
    pdf.set_text_color(*VIOLET)
    pdf.cell(0, 13, _visual("ALPHIMIYA (الفيمياء)"), align="C")
    pdf.ln(9)
    pdf.set_font("Amiri", "B", 16)
    pdf.set_text_color(*INK)
    pdf.cell(0, 11, _visual(narrative.headline), align="C")
    pdf.ln(9)
    pdf.set_font("Amiri", "", 10.5)
    pdf.set_text_color(*MUTED)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 6, _visual(f"تاريخ إنشاء التقرير: {generated_at}"), align="C")
    pdf.ln(10)

    # ------- ملخص محمّس -------
    pdf.section_title("ملخص رحلتك في هذه الجلسة")
    pdf.ar_paragraph(narrative.summary, size=11.5)
    pdf.ln(2)
    pdf.ar_paragraph(narrative.closing, size=10.5, style="B", color=VIOLET)

    # ------- معلومات الجلسة -------
    pdf.section_title("معلومات الجلسة")
    pdf.metric_row("بداية الجلسة", session_meta.get("start", "—"))
    pdf.metric_row("نهاية الجلسة", session_meta.get("end", "—"))
    pdf.metric_row("مدة الجلسة", f"{session_duration:.0f} ثانية ({fmt_time(session_duration)})")
    pdf.metric_row("عتبة جودة الاتصال (HSI <=)", session_meta.get("hsi_threshold", "—"))
    pdf.metric_row("قراءات EEG النظيفة", session_meta.get("clean_rows", "—"))
    pdf.metric_row("تردد RAW / PPG المُقاس", session_meta.get("fs_info", "—"))

    # ------- المؤشرات الأربعة -------
    pdf.section_title("١) توافق القلب والعقل", color=MAGENTA)
    if heart_brain is not None and heart_brain.hrv is not None:
        pdf.metric_row("النسبة الإجمالية", _fmt_pct(heart_brain.overall_pct))
        pdf.metric_row("متوسط معدل النبض", f"{heart_brain.hrv.mean_hr:.0f} bpm")
        pdf.metric_row("SDNN", f"{heart_brain.hrv.sdnn:.1f} ms")
        rmssd = heart_brain.hrv.rmssd
        pdf.metric_row("RMSSD", f"{rmssd:.1f} ms" if rmssd == rmssd else "—")
        method_label = "ذروات PPG الخام" if heart_brain.hrv.method == "ppg_peaks" else "عمود Heart_Rate"
        pdf.metric_row("مصدر بيانات النبض", method_label)
        if heart_brain.synchrony_ts is not None:
            pdf.metric_row("تزامن القلب مع الجبهة الدماغية", _fmt_pct(heart_brain.synchrony_pct_mean))
    else:
        pdf.ar_paragraph("تعذّر استخراج بيانات نبض كافية من هذا الملف لحساب هذا المؤشر.", size=10.5, color=MUTED)

    pdf.section_title("٢) طفرات جاما (لحظات الاستبصار)", color=AMBER)
    n_bursts = len(bursts_df) if bursts_df is not None else 0
    pdf.metric_row("عدد الطفرات المرصودة", f"{n_bursts}")
    pdf.metric_row("معدل الطفرات / دقيقة", f"{burst_rate:.1f}")
    if n_bursts:
        intens_counts = bursts_df["intensity"].value_counts().to_dict()
        for label in ("استثنائية", "قوية", "متوسطة"):
            if label in intens_counts:
                pdf.metric_row(f"عدد طفرات ({label})", str(intens_counts[label]))
    else:
        pdf.ar_paragraph("لم تُرصد طفرات غاما بالعتبة الحالية في هذه الجلسة.", size=10.5, color=MUTED)

    pdf.section_title("٣) تزامن الفصين", color=CYAN)
    frontal = hemispheric.get("frontal") if hemispheric else None
    posterior = hemispheric.get("posterior") if hemispheric else None
    pdf.metric_row("النسبة الإجمالية", _fmt_pct(hemispheric.get("overall_pct", float("nan"))) if hemispheric else "—")
    if frontal is not None:
        pdf.metric_row(frontal.pair_label, _fmt_pct(frontal.overall_pct))
    if posterior is not None:
        pdf.metric_row(posterior.pair_label, _fmt_pct(posterior.overall_pct))

    pdf.section_title("٤) مؤشر التكيف العصبي المركّب", color=VIOLET)
    if neuro is not None:
        pdf.metric_row("المتوسط العام للجلسة", _fmt_pct(neuro.overall_mean))
        pdf.metric_row("عتبة التنبيه", f"{alert_threshold:.0f}%")
        pdf.metric_row("عدد لحظات الذروة المسجّلة", f"{len(neuro.peak_moments) if neuro.peak_moments is not None else 0}")
        w = neuro.weights or {}
        pdf.metric_row(
            "أوزان الدمج",
            f"قلب-عقل {w.get('heart_brain', 0):.2f} / فصين {w.get('hemispheric', 0):.2f} / غاما {w.get('gamma', 0):.2f}",
        )

    # ------- الرسم البياني -------
    if chart_image_path and os.path.exists(chart_image_path):
        pdf.add_page()
        pdf.section_title("الرسم البياني الزمني — مؤشر التكيف العصبي المركّب")
        page_w = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.image(chart_image_path, x=pdf.l_margin, w=page_w)
        pdf.ln(3)

    # ------- قصة أبرز لحظات الذروة (أسلوب محمّس) -------
    pdf.section_title("أبرز لحظات الذروة في جلستك")
    if peak_story:
        for line in peak_story:
            pdf.ar_paragraph("• " + line, size=11, color=INK)
            pdf.ln(1.5)
    else:
        pdf.ar_paragraph(
            "لم تسجّل هذه الجلسة لحظات ذروة بارزة بالعتبات الحالية — وهذا لا يقلّل من قيمة وقتك، "
            "فالاستمرارية بانتظام هي ما يصنع الفرق مع الوقت. جرّب خفض عتبات الحساسية من اللوحة الجانبية "
            "لاكتشاف تفاصيل أدق، أو استمتع بجلستك القادمة.",
            size=11, color=INK,
        )

    # ------- ملاحظات منهجية (أسلوب علمي محايد) -------
    pdf.section_title("ملاحظات منهجية مهمة", color=CAVEAT)
    pdf.ar_paragraph(
        "جميع المؤشرات في هذا التقرير (توافق القلب والعقل، تزامن الفصين، طفرات جاما، ومؤشر التكيف العصبي "
        "المركّب) هي حسابات استكشافية وتعليمية مصمَّمة خصيصاً لهذا التطبيق، مستوحاة من أدبيات التأمل "
        "وأبحاث HRV/EEG العامة — وليست خوارزميات طبية معتمدة أو مقاييس سريرية موثَّقة لـ\"اللدونة العصبية\" "
        "الفعلية. تتأثر دقتها بجودة اتصال الجهاز، الحركة، ومعدل تصدير البيانات في Mind Monitor. يُنصح "
        "بقراءة النتائج كاتجاهات نسبية عبر عدة جلسات، لا كقياس طبي أو نفسي نهائي.",
        size=9.5, color=CAVEAT,
    )
    pdf.ln(4)

    return bytes(pdf.output())
