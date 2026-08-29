"""
የጡት ካንሰር ትንበያ — የድር መተግበሪያ (Streamlit) — Styled Version
================================================
Run with:  streamlit run breast_cancer_web_app_amharic.py

Loads the pre-trained model from breast_cancer_model.pkl (run
train_and_save_model.py first if that file doesn't exist yet).
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import cv2
import os
import io
from datetime import datetime

from image_features import extract_features_from_image

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

MODEL_PATH = "breast_cancer_model.pkl"
LOGO_PATH = "logo.png"
FONT_REGULAR_PATH = "NotoSansEthiopic-Regular.ttf"
FONT_BOLD_PATH = "NotoSansEthiopic-Bold.ttf"

# Register Amharic-capable font for PDF generation (falls back to
# Helvetica automatically if the font files aren't found)
FONT_NAME = "Helvetica"
FONT_NAME_BOLD = "Helvetica-Bold"
if os.path.exists(FONT_REGULAR_PATH) and os.path.exists(FONT_BOLD_PATH):
    pdfmetrics.registerFont(TTFont("NotoEthiopic", FONT_REGULAR_PATH))
    pdfmetrics.registerFont(TTFont("NotoEthiopic-Bold", FONT_BOLD_PATH))
    FONT_NAME = "NotoEthiopic"
    FONT_NAME_BOLD = "NotoEthiopic-Bold"

FIELD_INFO = {
    "radius_mean": ("አማካይ ራዲየስ", "Radius", "ከመሃል እስከ ጠርዝ ያለው አማካይ ርቀት", "📏"),
    "texture_mean": ("አማካይ ሸካራነት", "Texture", "በግራጫ ሚዛን እሴቶች ውስጥ ያለ ልዩነት", "🎨"),
    "perimeter_mean": ("አማካይ ዙሪያ", "Perimeter", "የኒውክሊየስ አማካይ ዙሪያ", "⭕"),
    "area_mean": ("አማካይ ስፋት", "Area", "የኒውክሊየስ አማካይ ስፋት", "🔲"),
    "concavity_mean": ("አማካይ ጥልቀት", "Concavity", "የጠለቁ ክፍሎች ክብደት", "🌊"),
    "concave points_mean": ("አማካይ የጥልቀት ነጥቦች", "Concave Points", "የጠለቁ ክፍሎች ብዛት", "📍"),
    "radius_worst": ("ከፍተኛ ራዲየስ", "Worst Radius", "የተለካው ትልቁ ራዲየስ", "📐"),
    "concave points_worst": ("ከፍተኛ የጥልቀት ነጥቦች", "Worst Concave Points", "የተለኩ ከፍተኛ የጥልቀት ነጥቦች", "🎯"),
}

st.set_page_config(page_title="የጡት ካንሰር ትንበያ", page_icon="🩺", layout="centered")

# ----------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Noto+Sans+Ethiopic:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans Ethiopic', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #f0f4f8 0%, #ffffff 35%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-card {
        background: linear-gradient(135deg, #6C63FF 0%, #4B4EFC 45%, #2E9CCA 100%);
        border-radius: 24px;
        padding: 32px 28px;
        text-align: center;
        box-shadow: 0 20px 40px -12px rgba(76, 81, 191, 0.45);
        margin-bottom: 24px;
    }
    .hero-title {
        color: white;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 10px 0 4px 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        font-weight: 400;
    }
    .accuracy-pill {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        backdrop-filter: blur(6px);
        color: white;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 14px;
        border: 1px solid rgba(255,255,255,0.35);
    }

    .info-card {
        background: white;
        border-radius: 18px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px -6px rgba(0,0,0,0.08);
        border: 1px solid #eef0f4;
        margin-bottom: 20px;
        color: #4a5568;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .section-label {
        font-weight: 700;
        font-size: 1.1rem;
        color: #2d3748;
        margin: 6px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    div[data-testid="stNumberInput"] label {
        font-weight: 600 !important;
        color: #2d3748 !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stNumberInput"] input {
        border-radius: 12px !important;
        border: 1.5px solid #e2e8f0 !important;
        padding: 10px !important;
        font-weight: 600;
        color: #1a202c;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 3px rgba(108,99,255,0.15) !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #FF6B6B 0%, #EE4266 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 24px -8px rgba(238, 66, 102, 0.55) !important;
        transition: transform 0.15s ease !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
    }

    .result-card {
        border-radius: 20px;
        padding: 26px;
        text-align: center;
        margin: 20px 0;
        animation: fadeIn 0.4s ease;
    }
    .result-benign {
        background: linear-gradient(135deg, #D4F8E8 0%, #B7F0D6 100%);
        border: 2px solid #34C77B;
    }
    .result-malignant {
        background: linear-gradient(135deg, #FFE0E0 0%, #FFC9C9 100%);
        border: 2px solid #EE4266;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .result-benign .result-title { color: #1B8A54; }
    .result-malignant .result-title { color: #C0223F; }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .footer-note {
        text-align: center;
        color: #a0aec0;
        font-size: 0.78rem;
        margin-top: 30px;
        line-height: 1.6;
    }
    .footer-warn {
        text-align: center;
        color: #e53e3e;
        font-size: 0.8rem;
        background: #fff5f5;
        border-radius: 12px;
        padding: 10px 16px;
        border: 1px solid #fed7d7;
        margin-top: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def _is_ethiopic(ch):
    return "\u1200" <= ch <= "\u139F" or "\u2D80" <= ch <= "\u2DDF" or "\uAB00" <= ch <= "\uAB2F"


def _split_runs(text):
    """Splits text into runs of (substring, is_ethiopic) so each can use the right font."""
    runs = []
    cur, cur_eth = "", None
    for ch in text:
        eth = _is_ethiopic(ch) if ch != " " else cur_eth
        if cur_eth is None:
            cur_eth = eth
        if eth != cur_eth and ch != " ":
            runs.append((cur, cur_eth))
            cur, cur_eth = ch, eth
        else:
            cur += ch
    if cur:
        runs.append((cur, cur_eth))
    return runs


def draw_mixed_text(c, x, y, text, size, bold=False, align="left"):
    """Draws text that may mix Amharic and Latin/number characters, auto-switching
    fonts per run since the Ethiopic font has no Latin/digit glyphs and vice versa."""
    eth_font = FONT_NAME_BOLD if bold else FONT_NAME
    lat_font = "Helvetica-Bold" if bold else "Helvetica"
    runs = _split_runs(text)

    total_width = sum(
        c.stringWidth(t, eth_font if is_eth else lat_font, size) for t, is_eth in runs
    )
    if align == "center":
        cx = x - total_width / 2
    elif align == "right":
        cx = x - total_width
    else:
        cx = x

    for t, is_eth in runs:
        font = eth_font if is_eth else lat_font
        c.setFont(font, size)
        c.drawString(cx, y, t)
        cx += c.stringWidth(t, font, size)


def generate_pdf_report(user_values, field_info, pred, proba, accuracy, logo_path):
    """Builds a one-page PDF summary of the prediction and returns it as bytes."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 20 * mm
    y = height - margin

    # Logo
    if os.path.exists(logo_path):
        try:
            logo_w = 30 * mm
            logo_h = 30 * mm
            c.drawImage(logo_path, (width - logo_w) / 2, y - logo_h, width=logo_w, height=logo_h,
                        preserveAspectRatio=True, mask="auto")
            y -= logo_h + 8 * mm
        except Exception:
            pass

    # Title
    c.setFillColor(colors.black)
    draw_mixed_text(c, width / 2, y, "የጡት ካንሰር ትንበያ ሪፖርት", 16, bold=True, align="center")
    y -= 7 * mm
    draw_mixed_text(c, width / 2, y, "Breast Cancer Prediction Report", 11, align="center")
    y -= 10 * mm

    c.setStrokeColor(colors.HexColor("#6C63FF"))
    c.setLineWidth(1.2)
    c.line(margin, y, width - margin, y)
    y -= 10 * mm

    # Date
    c.setFillColor(colors.HexColor("#4a5568"))
    draw_mixed_text(c, margin, y, f"ቀን / Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 10)
    y -= 10 * mm

    # Result box
    is_malignant = pred == 1
    box_color = colors.HexColor("#FFC9C9") if is_malignant else colors.HexColor("#B7F0D6")
    border_color = colors.HexColor("#EE4266") if is_malignant else colors.HexColor("#34C77B")
    text_color = colors.HexColor("#C0223F") if is_malignant else colors.HexColor("#1B8A54")

    box_h = 22 * mm
    c.setFillColor(box_color)
    c.setStrokeColor(border_color)
    c.setLineWidth(1.5)
    c.roundRect(margin, y - box_h, width - 2 * margin, box_h, 6, fill=1, stroke=1)

    c.setFillColor(text_color)
    result_am = "ትንበያ፡ አደገኛ" if is_malignant else "ትንበያ፡ ጤናማ"
    result_en = "Prediction: Malignant" if is_malignant else "Prediction: Benign"
    draw_mixed_text(c, width / 2, y - 9 * mm, f"{result_am}  ({result_en})", 14, bold=True, align="center")

    conf_text = (
        f"የካንሰርነት እርግጠኝነት: {proba[1]*100:.1f}%   |   የጤናማነት እርግጠኝነት: {proba[0]*100:.1f}%"
    )
    draw_mixed_text(c, width / 2, y - 16 * mm, conf_text, 10, align="center")
    y -= box_h + 12 * mm

    # Measurements table
    c.setFillColor(colors.HexColor("#2d3748"))
    draw_mixed_text(c, margin, y, "የዕጢ መለኪያዎች / Tumor Measurements", 12, bold=True)
    y -= 8 * mm

    row_h = 8 * mm
    for key, value in user_values.items():
        am_label, en_label, desc, icon = field_info.get(key, (key, key, "", ""))
        c.setFillColor(colors.HexColor("#f7fafc"))
        c.rect(margin, y - row_h + 2, width - 2 * margin, row_h - 2, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#2d3748"))
        draw_mixed_text(c, margin + 4 * mm, y - row_h + 5, f"{am_label} ({en_label})", 10)
        draw_mixed_text(c, width - margin - 4 * mm, y - row_h + 5, f"{value:.4f}", 10, bold=True, align="right")
        y -= row_h

    y -= 6 * mm
    c.setFillColor(colors.HexColor("#718096"))
    draw_mixed_text(c, margin, y, f"የሞዴል ትክክለኛነት / Model accuracy: {accuracy*100:.2f}%", 9)
    y -= 12 * mm

    # Disclaimer
    c.setFillColor(colors.HexColor("#fff5f5"))
    c.setStrokeColor(colors.HexColor("#fed7d7"))
    disclaimer_h = 14 * mm
    c.roundRect(margin, y - disclaimer_h, width - 2 * margin, disclaimer_h, 4, fill=1, stroke=1)
    c.setFillColor(colors.HexColor("#e53e3e"))
    draw_mixed_text(
        c, width / 2, y - 6 * mm,
        "ለትምህርት ዓላማ ብቻ የተዘጋጀ — የሕክምና ምርመራ አይደለም። ሁልጊዜ ብቁ ሐኪም ያማክሩ።",
        8, align="center",
    )
    draw_mixed_text(
        c, width / 2, y - 11 * mm,
        "For educational purposes only — not a medical diagnosis. Always consult a qualified doctor.",
        8, align="center",
    )
    y -= disclaimer_h + 8 * mm

    # Footer
    c.setFillColor(colors.HexColor("#a0aec0"))
    draw_mixed_text(c, width / 2, margin / 2, "Developed by Natan Sabawi", 8, align="center")

    c.save()
    buffer.seek(0)
    return buffer


# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(
        f"'{MODEL_PATH}' ማግኘት አልተቻለም። እባክዎ መጀመሪያ train_and_save_model.py "
        "ያሂዱ (በተመሳሳይ ማህደር ውስጥ) ለመፍጠር።"
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
all_features = bundle["all_features"]
medians = bundle["medians"]
key_features = bundle["key_features"]
accuracy = bundle["accuracy"]

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
if os.path.exists(LOGO_PATH):
    logo_col1, logo_col2, logo_col3 = st.columns([1, 1.1, 1])
    with logo_col2:
        st.image(LOGO_PATH, use_container_width=True)

st.markdown(
    f"""
    <div class="hero-card">
        <div style="font-size:2.6rem;">🩺</div>
        <div class="hero-title">የጡት ካንሰር ትንበያ</div>
        <div class="hero-subtitle">Breast Cancer Prediction System</div>
        <div class="accuracy-pill">✨ የሞዴል ትክክለኛነት: {accuracy*100:.2f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
        ከዚህ በታች የዕጢ መለኪያዎችን ያስገቡ እና <b>አደገኛ (Malignant)</b> ወይም
        <b>ጤናማ (Benign)</b> የሚል ወዲያውኑ ትንበያ ያግኙ። መስኮች በተለመዱ የመረጃ
        ስብስብ እሴቶች አስቀድመው ተሞልተዋል — በትክክለኛ መለኪያዎች ማንኛውንም ማስተካከል ይችላሉ።
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Input mode selector + image upload (Beta)
# ----------------------------------------------------------------------
if "prefill" not in st.session_state:
    st.session_state.prefill = None

input_mode = st.radio(
    "የመለኪያ ግቤት ዘዴ / Input method",
    ["✏️ በእጅ ማስገቢያ (Manual)", "📷 ምስል መስቀል (Upload Image — Beta)"],
    horizontal=True,
)

if input_mode == "📷 ምስል መስቀል (Upload Image — Beta)":
    st.info(
        "⚠️ **የሙከራ ባህሪ (Experimental).** ይህ ከተሰቀለ ማይክሮስኮፕ ምስል 8ቱን መለኪያዎች "
        "በምስል ማቀናበሪያ (image processing) ይገምታል፣ ከዚያም ሞዴሉ ከሰለጠነበት እሴት መጠን ጋር "
        "እንዲመጣጠን በስታስቲክስ ያስተካክላል። ይህ ኦሪጅናል የክሊኒካል ዳታሴቱ የተሰራበት የተስተካከለ "
        "ዘዴ አይደለም — ውጤቱን እንደ ግምታዊ የትምህርት ማሳያ ብቻ ይውሰዱት።\n\n"
        "This is an experimental approximation, not a calibrated clinical "
        "measurement — treat results as educational only."
    )

    uploaded_file = st.file_uploader(
        "ማይክሮስኮፕ ምስል ይስቀሉ / Upload a microscopy image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("ምስሉን ማንበብ አልተቻለም። እባክዎ የተለየ JPG/PNG ይሞክሩ።")
        else:
            with st.spinner("የሕዋስ ኒውክሊየሶችን በመለየት ላይ... / Detecting cell nuclei..."):
                try:
                    result, annotated_bgr, n_nuclei = extract_features_from_image(image_bgr)
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                    st.image(
                        annotated_rgb,
                        caption=f"{n_nuclei} ኒውክሊየስ-መሰል ክፍሎች ተገኝተዋል / {n_nuclei} nucleus-like regions detected",
                        use_container_width=True,
                    )
                    st.success("የተገመቱ መለኪያዎች ከታች ተሞልተዋል — ከመተንበይዎ በፊት ማስተካከል ይችላሉ።")
                    st.session_state.prefill = result
                except ValueError as e:
                    st.warning(str(e))
                    st.session_state.prefill = None
                except Exception:
                    st.error(
                        "ይህን ምስል በመተንተን ላይ ችግር ተፈጥሯል። እባክዎ የተለየ ምስል ይሞክሩ ወይም "
                        "ወደ በእጅ ማስገቢያ ይቀይሩ። / Something went wrong analyzing this "
                        "image — try a different image or switch to Manual Entry."
                    )
                    st.session_state.prefill = None

prefill = st.session_state.prefill

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
st.markdown('<div class="section-label">📋 የዕጢ መለኪያዎች (Tumor Measurements)</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    user_values = {}
    columns = [col1, col2]

    for i, key in enumerate(key_features):
        am_label, en_label, desc, icon = FIELD_INFO.get(key, (key, key, "", "🔹"))
        col = columns[i % 2]
        default_val = prefill[key] if prefill is not None else medians[key]
        with col:
            user_values[key] = st.number_input(
                f"{icon} {am_label} ({en_label})",
                value=float(default_val),
                help=desc,
                format="%.4f",
            )

    submitted = st.form_submit_button("🔍 ተንብይ (Predict)", use_container_width=True)

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if submitted:
    row = dict(medians)
    row.update(user_values)

    vector = pd.DataFrame([[row[f] for f in all_features]], columns=all_features)
    vector_scaled = scaler.transform(vector)

    pred = model.predict(vector_scaled)[0]
    proba = model.predict_proba(vector_scaled)[0]

    if pred == 1:
        st.markdown(
            f"""
            <div class="result-card result-malignant">
                <div style="font-size:2.4rem;">⚠️</div>
                <div class="result-title">ትንበያ፡ አደገኛ (Malignant)</div>
                <div style="color:#7a2e3d; font-size:0.9rem;">የካንሰርነት እርግጠኝነት: {proba[1]*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-benign">
                <div style="font-size:2.4rem;">✅</div>
                <div class="result-title">ትንበያ፡ ጤናማ (Benign)</div>
                <div style="color:#2f6b4f; font-size:0.9rem;">የጤናማነት እርግጠኝነት: {proba[0]*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    c1.metric("🟢 የጤናማነት እርግጠኝነት", f"{proba[0]*100:.1f}%")
    c2.metric("🔴 የካንሰርነት እርግጠኝነት", f"{proba[1]*100:.1f}%")

    st.progress(float(proba[1]), text="የካንሰር አደጋ ደረጃ (Malignancy risk score)")

    if prefill is not None:
        st.caption(
            "ℹ️ ይህ ትንበያ ከተሰቀለው ምስል የተገመቱ መለኪያዎችን ተጠቅሟል — የሙከራ ግምት እንጂ "
            "የተስተካከለ ክሊኒካዊ መለኪያ አይደለም።"
        )

    # --- PDF download ---
    pdf_buffer = generate_pdf_report(user_values, FIELD_INFO, pred, proba, accuracy, LOGO_PATH)
    st.download_button(
        label="📄 ሪፖርት አውርድ (Download PDF Report)",
        data=pdf_buffer,
        file_name=f"breast_cancer_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-warn">⚠️ ለትምህርት ዓላማ ብቻ የተዘጋጀ — የሕክምና ምርመራ አይደለም። ሁልጊዜ ብቁ ሐኪም ያማክሩ።</div>
    <div class="footer-note">Developed by <b>Natan Sabawi</b></div>
    """,
    unsafe_allow_html=True,
)
