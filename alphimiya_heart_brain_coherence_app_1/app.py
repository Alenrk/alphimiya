"""
الفيمياء — ALPHIMIYA®
تطبيق تفاعلي متقدم: توافق القلب والعقل، طفرات جاما، تزامن الفصين،
ومؤشر التكيف العصبي المركّب — من بيانات Muse 2 عبر Mind Monitor.
"""

import glob
import os
import shutil
import tempfile
from datetime import datetime

import streamlit as st

from data_loader import (
    load_mind_monitor_csv, add_band_row_means, filter_band_df_by_hsi, BANDS,
)
from heart_coherence import compute_heart_brain_coherence
from gamma_bursts import detect_gamma_bursts, bursts_to_dataframe, burst_rate_per_minute
from hemispheric_coherence import compute_hemispheric_coherence
from neuroplasticity import compute_neuroplasticity_score, DEFAULT_WEIGHTS
from charts import gauge_chart, gamma_burst_chart, hemispheric_timeseries_chart, hrv_coherence_chart, neuroplasticity_chart, NEON
from report_narrative import build_narrative, build_peak_story
from pdf_report import build_pdf


def ensure_browser_for_chart_export() -> bool:
    """
    يحاول تجهيز متصفح Chrome/Chromium محلي كي تستطيع مكتبة kaleido تصدير
    الرسوم البيانية كصور PNG لتضمينها داخل تقرير PDF. يبحث في المسارات
    الشائعة (بما فيها متصفح Playwright إن وُجد) قبل أن يستسلم بهدوء.
    يعيد True إن وجد متصفحاً صالحاً، وإلا False (وعندها يُبنى التقرير بدون الرسم).
    """
    if os.environ.get("BROWSER_PATH"):
        return True

    candidates = [
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/opt/pw-browsers/chromium",
    ]
    candidates += glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")

    for path in candidates:
        if path and os.path.exists(path):
            os.environ["BROWSER_PATH"] = path
            return True
    return False


st.set_page_config(
    page_title="الفيمياء ALPHIMIYA® | توافق القلب والعقل",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;900&family=Tajawal:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Cairo', 'Tajawal', sans-serif !important; }

[data-testid="stAppViewContainer"], [data-testid="stSidebar"], .main { direction: rtl; }
[data-testid="stSidebar"] *, .main * { text-align: right; }
[data-testid="stMetricValue"], .stPlotlyChart, code, pre { direction: ltr; }
[data-testid="stSlider"] { direction: ltr; }
[data-testid="stSlider"] label { direction: rtl; text-align: right; }
div[data-testid="stMarkdownContainer"] p { text-align: right; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 85% -10%, rgba(168,85,247,0.14) 0%, transparent 45%),
        radial-gradient(circle at 10% 10%, rgba(34,211,238,0.10) 0%, transparent 40%),
        linear-gradient(180deg, #05070d 0%, #070912 45%, #05060b 100%);
    color: #e7ecf5;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0c16 0%, #05060b 100%);
    border-left: 1px solid rgba(168,85,247,0.15);
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
    color: #c7cfe6 !important;
}

.app-hero {
    background: linear-gradient(120deg, rgba(168,85,247,0.16), rgba(34,211,238,0.08));
    border: 1px solid rgba(168,85,247,0.28);
    border-radius: 18px;
    padding: 22px 28px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.app-hero::after {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(circle at 90% 0%, rgba(236,72,153,0.18), transparent 55%);
    pointer-events: none;
}
.app-hero h1 {
    font-size: 1.85rem; margin: 0 0 6px 0;
    background: linear-gradient(90deg, #d8b4fe, #22d3ee, #f0abfc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-hero p { color: #9aa4c3; margin: 0; font-size: 0.95rem; }
.brand-kicker {
    display: inline-block; direction: ltr;
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.18em;
    color: #22d3ee; border: 1px solid rgba(34,211,238,0.4);
    background: rgba(34,211,238,0.08);
    border-radius: 999px; padding: 3px 12px; margin-bottom: 10px;
}
.brand-kicker span { color: #f0abfc; }

.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(168,85,247,0.14);
    border-radius: 16px;
    padding: 18px 20px;
    margin: 12px 0;
}
.note-item {
    border-right: 3px solid #a855f7;
    padding: 8px 14px; margin-bottom: 8px;
    background: rgba(168,85,247,0.06);
    border-radius: 8px; line-height: 1.85; font-size: 0.92rem; color: #c7cfe6;
}
.caveat-box {
    border: 1px dashed rgba(245,197,66,0.5);
    background: rgba(245,197,66,0.06);
    color: #f5d78a; border-radius: 10px;
    padding: 12px 16px; font-size: 0.86rem; line-height: 1.85; margin-top: 10px;
}
.method-badge {
    display:inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700; margin-inline-start: 6px;
    background: rgba(34,211,238,0.14); color: #22d3ee; border: 1px solid rgba(34,211,238,0.35);
}
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg,#a855f7,#22d3ee);
    color: white; border: none; border-radius: 10px; font-weight: 700; padding: 0.6rem 1.2rem;
}
hr { border-color: rgba(168,85,247,0.15); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _cached_load(file_bytes: bytes):
    import io
    return load_mind_monitor_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _cached_heart_brain(ppg_df, ppg_fs, hr_df, band_df, weight_hrv, weight_sync, window_sec):
    return compute_heart_brain_coherence(ppg_df, ppg_fs, hr_df, band_df, weight_hrv, weight_sync, window_sec)


@st.cache_data(show_spinner=False)
def _cached_hemispheric(eeg_raw_df, eeg_raw_fs, band_df, bands):
    return compute_hemispheric_coherence(eeg_raw_df, eeg_raw_fs, band_df, list(bands))


with st.sidebar:
    st.markdown("### ⚙️ لوحة التحكم")
    uploaded_file = st.file_uploader("📂 ارفع ملف CSV من Mind Monitor", type=["csv"])

    st.markdown("---")
    st.markdown("#### 🧹 تنظيف البيانات")
    hsi_threshold = st.slider("الحد الأقصى المسموح لجودة الاتصال (HSI)", 1.0, 4.0, 2.0, 1.0)

    st.markdown("---")
    st.markdown("#### ❤️ توافق القلب والعقل")
    hrv_window = st.slider("نافذة تحليل HRV (ثانية)", 30, 180, 60, 10)
    w_hrv = st.slider("وزن نسبة توافق HRV", 0.0, 1.0, 0.6, 0.1)
    w_sync = round(1.0 - w_hrv, 1)
    st.caption(f"وزن تزامن القلب مع الجبهة الدماغية: {w_sync}")

    st.markdown("---")
    st.markdown("#### ⚡ كاشف طفرات جاما")
    gamma_z = st.slider("عتبة الشدة (Z-score)", 1.5, 5.0, 3.0, 0.5)
    gamma_min_sep = st.slider("أقل فاصل بين طفرتين (ثانية)", 0.2, 5.0, 1.0, 0.2)

    st.markdown("---")
    st.markdown("#### 🧠 تزامن الفصين")
    hemi_bands = st.multiselect("الموجات المشمولة", BANDS, default=["Theta", "Alpha", "Beta", "Gamma"])

    st.markdown("---")
    st.markdown("#### 🌌 مؤشر التكيف العصبي")
    w_heart = st.slider("وزن توافق القلب-العقل", 0.0, 1.0, DEFAULT_WEIGHTS["heart_brain"], 0.05)
    w_hemi = st.slider("وزن تزامن الفصين", 0.0, 1.0, DEFAULT_WEIGHTS["hemispheric"], 0.05)
    w_gamma = st.slider("وزن نشاط طفرات غاما", 0.0, 1.0, DEFAULT_WEIGHTS["gamma"], 0.05)
    alert_threshold = st.slider("عتبة التنبيه (%)",  20, 95, 75, 5)

st.markdown(
    """
    <div class="app-hero">
        <div class="brand-kicker">ALPHIMIYA<span>®</span></div>
        <h1>✨ الفيمياء — توافق القلب والعقل · طفرات جاما · تزامن الفصين</h1>
        <p>تحليل متقدم للتيقظ والتكيف العصبي من بيانات Muse 2 — مستوحى من أبحاث الفيمياء والدراسات المتقدمة على الدماغ أثناء التأمل، ومقدَّم كجزء من منهجية ALPHIMIYA® لدمج الخيمياء الداخلية بالعلوم العصبية.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info("👋 ابدأ برفع ملف CSV مُصدَّر من تطبيق Mind Monitor (يفضَّل تفعيل تسجيل PPG وHeart Rate والإشارة الخام).")
    st.markdown(
        """
        <div class="section-card">
        <b>ما الذي يحتاجه هذا التطبيق؟</b><br><br>
        1) جلسة Mind Monitor مسجَّلة مع جهاز Muse 2 (يفضَّل 3–5 دقائق فأكثر لدقة أعلى في تحليل HRV).<br>
        2) تفعيل خيارات "Include PPG" و"Include Raw EEG" في إعدادات Mind Monitor قبل التسجيل يمنحك دقة أعلى
        (التطبيق يعمل أيضاً بدونها معتمداً على بدائل تقديرية أضعف قليلاً، ويوضح لك ذلك بنفسه).<br>
        3) ارفع ملف CSV من اللوحة الجانبية، وستظهر كل المؤشرات تلقائياً.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    streams = _cached_load(uploaded_file.getvalue())
except Exception as e:
    st.error(f"⚠️ تعذّرت قراءة الملف: {e}")
    st.stop()

for w in streams.warnings:
    st.warning(w)

band_df_filtered, hsi_info = filter_band_df_by_hsi(streams.band_df, hsi_threshold=hsi_threshold)
band_df = add_band_row_means(band_df_filtered)

if len(band_df) < 4:
    st.error("⚠️ عدد قراءات EEG المتبقية بعد التصفية قليل جداً. حاول رفع عتبة HSI من اللوحة الجانبية.")
    st.stop()

session_duration = float(streams.raw_df["t_sec"].max())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("⏱️ مدة الجلسة", f"{session_duration:.0f} ث")
c2.metric("📄 قراءات EEG النظيفة", f"{len(band_df):,}")
c3.metric("💓 قراءات القلب", f"{len(streams.hr_df):,}")
c4.metric("📡 تردد RAW المُقاس", f"{streams.eeg_raw_fs:.1f} Hz" if streams.eeg_raw_fs else "—")
c5.metric("💗 تردد PPG المُقاس", f"{streams.ppg_fs:.1f} Hz" if streams.ppg_fs else "—")

st.markdown("---")

with st.spinner("جارٍ حساب المؤشرات المتقدمة..."):
    heart_brain = _cached_heart_brain(streams.ppg_df, streams.ppg_fs, streams.hr_df, band_df, w_hrv, w_sync, float(hrv_window))
    hemispheric = _cached_hemispheric(streams.eeg_raw_df, streams.eeg_raw_fs, band_df, tuple(hemi_bands) if hemi_bands else tuple(BANDS))
    bursts = detect_gamma_bursts(band_df, z_threshold=gamma_z, min_separation_sec=gamma_min_sep)
    bursts_df = bursts_to_dataframe(bursts)
    neuro = compute_neuroplasticity_score(
        heart_brain, hemispheric, [b.t_sec for b in bursts], session_duration,
        weights={"heart_brain": w_heart, "hemispheric": w_hemi, "gamma": w_gamma},
        alert_threshold=float(alert_threshold),
    )

def _gauge_with_fallback(col, value, title, color):
    with col:
        st.plotly_chart(gauge_chart(value, title, color), use_container_width=True)
        if value != value:  # NaN
            st.caption("⚠️ بيانات غير كافية لحساب هذا المؤشر في هذه الجلسة")


st.markdown("#### 🌌 لوحة المؤشرات الرئيسية")
g1, g2, g3 = st.columns(3)
_gauge_with_fallback(g1, heart_brain.overall_pct, "توافق القلب والعقل", NEON["magenta"])
_gauge_with_fallback(g2, hemispheric["overall_pct"], "تزامن الفصين", NEON["cyan"])
_gauge_with_fallback(g3, neuro.overall_mean, "مؤشر التكيف العصبي", NEON["violet"])

if neuro.overall_mean == neuro.overall_mean and neuro.overall_mean >= alert_threshold:
    st.success(f"🔔 هذه الجلسة بلغت في المتوسط منطقة التكيف العصبي المثالي (≥ {alert_threshold}%)!")

st.markdown("---")

# ============================================================
#  تقرير PDF — قسم مستقل وبارز أعلى الصفحة
# ============================================================
st.markdown(
    """
    <div class="section-card">
    <h4 style="margin-top:0;">📥 تقرير جلستك الكامل — بأسلوب محمّس يحتفي برحلتك!</h4>
    <p style="color:#9aa4c3; font-size:0.92rem;">
    تقرير PDF يجمع المؤشرات الأربعة كاملة، رسماً بيانياً لمؤشر التكيف العصبي، وقصة أبرز لحظات
    ذروتك في هذه الجلسة — مع ملخص تحفيزي وملاحظات منهجية دقيقة.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("✨ توليد تقرير PDF لهذه الجلسة"):
    with st.spinner("جارٍ صياغة تقريرك..."):
        narrative = build_narrative(
            heart_brain=heart_brain, hemispheric=hemispheric, bursts=bursts, neuro=neuro,
            session_duration=session_duration, alert_threshold=float(alert_threshold),
        )
        peak_story = build_peak_story(bursts=bursts, neuro=neuro)

        chart_path = None
        if ensure_browser_for_chart_export():
            try:
                tmp_dir = tempfile.mkdtemp()
                chart_path = os.path.join(tmp_dir, "neuro_chart.png")
                chart_fig = neuroplasticity_chart(neuro, session_duration)
                chart_fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white")
                chart_fig.write_image(chart_path, width=1400, height=560, scale=2)
            except Exception as e:
                st.warning(f"تعذّر إنشاء صورة الرسم البياني داخل التقرير ({e}) — سيتم إنشاء التقرير بدونها.")
                chart_path = None
        else:
            st.info(
                "ℹ️ لم يتم العثور على متصفح Chrome/Chromium مثبّت على هذا الجهاز، لذا سيُنشأ "
                "التقرير بدون تضمين صورة الرسم البياني."
            )

        session_meta = {
            "start": streams.start_time.strftime("%Y-%m-%d %H:%M:%S") if streams.start_time else "—",
            "end": streams.end_time.strftime("%Y-%m-%d %H:%M:%S") if streams.end_time else "—",
            "hsi_threshold": f"{hsi_threshold:.0f}",
            "clean_rows": f"{len(band_df):,}",
            "fs_info": (
                f"{streams.eeg_raw_fs:.1f} Hz / {streams.ppg_fs:.1f} Hz"
                if streams.eeg_raw_fs and streams.ppg_fs else "—"
            ),
        }

        pdf_bytes = build_pdf(
            session_meta=session_meta,
            narrative=narrative,
            peak_story=peak_story,
            heart_brain=heart_brain,
            hemispheric=hemispheric,
            bursts_df=bursts_df,
            burst_rate=burst_rate_per_minute(bursts, session_duration),
            neuro=neuro,
            session_duration=session_duration,
            alert_threshold=float(alert_threshold),
            chart_image_path=chart_path,
        )

    st.success("✅ تقريرك جاهز!")
    st.download_button(
        "⬇️ تنزيل تقرير الجلسة (PDF)",
        data=pdf_bytes,
        file_name=f"alphimiya_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )

st.markdown("---")

tab_heart, tab_gamma, tab_hemi, tab_neuro = st.tabs([
    "❤️ توافق القلب والعقل", "⚡ طفرات جاما", "🧠 تزامن الفصين", "🌌 مؤشر التكيف العصبي",
])

with tab_heart:
    if heart_brain.hrv is None:
        st.warning("تعذّر استخراج بيانات نبض كافية من هذا الملف لحساب توافق القلب والعقل.")
    else:
        method_label = "ذروات PPG الخام" if heart_brain.hrv.method == "ppg_peaks" else "عمود Heart_Rate"
        st.markdown(f'<span class="method-badge">مصدر النبض: {method_label}</span>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("متوسط معدل النبض", f"{heart_brain.hrv.mean_hr:.0f} bpm")
        m2.metric("SDNN", f"{heart_brain.hrv.sdnn:.1f} ms")
        m3.metric("RMSSD", f"{heart_brain.hrv.rmssd:.1f} ms" if heart_brain.hrv.rmssd == heart_brain.hrv.rmssd else "—")

        st.plotly_chart(hrv_coherence_chart(heart_brain.coherence_ts), use_container_width=True)

        if heart_brain.synchrony_ts is not None:
            st.markdown(f"**متوسط تزامن القلب مع الجبهة الدماغية:** {heart_brain.synchrony_pct_mean:.1f}%")

    for note in heart_brain.notes:
        st.markdown(f'<div class="note-item">{note}</div>', unsafe_allow_html=True)

with tab_gamma:
    b1, b2 = st.columns(2)
    b1.metric("عدد الطفرات المرصودة", f"{len(bursts)}")
    b2.metric("معدل الطفرات/دقيقة", f"{burst_rate_per_minute(bursts, session_duration):.1f}")

    st.plotly_chart(gamma_burst_chart(band_df, bursts_df), use_container_width=True)

    if len(bursts_df):
        st.markdown("**جدول الطفرات المرصودة:**")
        show = bursts_df.copy()
        show["t_sec"] = show["t_sec"].round(2)
        show["z_score"] = show["z_score"].round(2)
        st.dataframe(
            show[["t_sec", "intensity", "z_score", "channel"]].rename(
                columns={"t_sec": "الزمن (ث)", "intensity": "الشدة", "z_score": "Z-score", "channel": "القناة"}
            ),
            use_container_width=True, height=280,
        )
    else:
        st.info("لم تُرصد أي طفرات جاما بالعتبة الحالية — جرّب خفض عتبة Z-score من اللوحة الجانبية.")

with tab_hemi:
    hf, hp = st.columns(2)
    hf.metric(hemispheric["frontal"].pair_label, f"{hemispheric['frontal'].overall_pct:.1f}%" if hemispheric["frontal"].overall_pct == hemispheric["frontal"].overall_pct else "—")
    hp.metric(hemispheric["posterior"].pair_label, f"{hemispheric['posterior'].overall_pct:.1f}%" if hemispheric["posterior"].overall_pct == hemispheric["posterior"].overall_pct else "—")

    st.plotly_chart(hemispheric_timeseries_chart(hemispheric), use_container_width=True)

    for pair_key in ("frontal", "posterior"):
        pair = hemispheric[pair_key]
        methods = {info["method"] for info in pair.per_band.values()}
        for m in methods:
            st.markdown(f'<span class="method-badge">{pair.pair_label}: {m}</span>', unsafe_allow_html=True)
        for note in pair.notes:
            st.markdown(f'<div class="note-item">{note}</div>', unsafe_allow_html=True)

with tab_neuro:
    st.plotly_chart(neuroplasticity_chart(neuro, session_duration), use_container_width=True)

    if len(neuro.peak_moments):
        st.markdown(f"**⭐ لحظات الذروة (تجاوزت عتبة {alert_threshold}%):**")
        peaks_show = neuro.peak_moments.copy()
        peaks_show["t_sec"] = peaks_show["t_sec"].round(1)
        peaks_show["score"] = peaks_show["score"].round(1)
        st.dataframe(
            peaks_show.rename(columns={"t_sec": "الزمن (ث)", "score": "المؤشر %"}),
            use_container_width=True, height=220,
        )
    else:
        st.info("لم تصل الجلسة إلى عتبة التنبيه المحددة في أي لحظة — جرّب خفض العتبة من اللوحة الجانبية.")

    for note in neuro.notes:
        st.markdown(f'<div class="note-item">{note}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="caveat-box">
    ⚠️ تنويه علمي: جميع المؤشرات في هذا التطبيق (توافق القلب والعقل، تزامن الفصين، طفرات جاما،
    ومؤشر التكيف العصبي المركّب) هي حسابات استكشافية وتعليمية مصمَّمة خصيصاً لهذا التطبيق،
    مستوحاة من أدبيات التأمل وأبحاث HRV/EEG العامة — وليست خوارزميات طبية معتمدة أو مقاييس
    سريرية موثَّقة لـ"اللدونة العصبية" الفعلية. تتأثر دقتها بجودة اتصال الجهاز، الحركة، ومعدل
    تصدير البيانات في Mind Monitor. يُنصح بقراءة النتائج كاتجاهات نسبية عبر عدة جلسات، لا كقياس
    طبي أو نفسي نهائي.
    </div>
    <p style="text-align:center; color:#5b6478; font-size:0.78rem; margin-top:18px;" dir="ltr">
        ALPHIMIYA® — الفيمياء
    </p>
    """,
    unsafe_allow_html=True,
)
