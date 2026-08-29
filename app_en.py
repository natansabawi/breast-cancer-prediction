"""
Breast Cancer Prediction — Web App (Streamlit)
================================================
Run with:  streamlit run app_en.py

Loads the pre-trained model from breast_cancer_model.pkl (run
train_and_save_model.py first if that file doesn't exist yet).

Two input modes:
  - Manual Entry: type the 8 key measurements directly
  - Upload Image: estimate the 8 measurements from a microscopy image
    using classical image processing (see image_features.py)
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import cv2
import os

from image_features import extract_features_from_image

MODEL_PATH = "breast_cancer_model.pkl"

FIELD_INFO = {
    "radius_mean": ("Mean Radius", "Average distance from center to edge"),
    "texture_mean": ("Mean Texture", "Variation in gray-scale values"),
    "perimeter_mean": ("Mean Perimeter", "Average perimeter of the nucleus"),
    "area_mean": ("Mean Area", "Average area of the nucleus"),
    "concavity_mean": ("Mean Concavity", "Severity of concave portions"),
    "concave points_mean": ("Mean Concave Points", "Number of concave portions"),
    "radius_worst": ("Worst Radius", "Largest radius measured"),
    "concave points_worst": ("Worst Concave Points", "Most concave points measured"),
}

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="centered")

# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Could not find '{MODEL_PATH}'. Run `train_and_save_model.py` "
        "first (in the same folder) to create it."
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
# Header
# ----------------------------------------------------------------------
st.title("🩺 Breast Cancer Prediction")
st.caption(f"Model accuracy on test data: **{accuracy*100:.2f}%**")

st.divider()

# ----------------------------------------------------------------------
# Input mode selector
# ----------------------------------------------------------------------
mode = st.radio(
    "How would you like to provide measurements?",
    ["✏️ Manual Entry", "📷 Upload Image (Beta)"],
    horizontal=True,
)

if "prefill" not in st.session_state:
    st.session_state.prefill = None

# ----------------------------------------------------------------------
# IMAGE MODE
# ----------------------------------------------------------------------
if mode == "📷 Upload Image (Beta)":
    st.info(
        "⚠️ **Experimental feature.** This estimates the 8 measurements from an "
        "uploaded microscopy image using classical image processing (nucleus "
        "detection + geometry), then statistically maps them onto the scale the "
        "model was trained on. It is **not** the calibrated method used to build "
        "the original clinical dataset — treat results as a rough, educational "
        "approximation only, especially on images that aren't clear, high-contrast "
        "microscopy shots of individually visible cell nuclei."
    )

    uploaded_file = st.file_uploader(
        "Upload a microscopy / histopathology image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Couldn't read that image file. Please try a different JPG/PNG.")
        else:
            with st.spinner("Detecting cell nuclei and estimating measurements..."):
                try:
                    result, annotated_bgr, n_nuclei = extract_features_from_image(image_bgr)
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                    st.image(
                        annotated_rgb,
                        caption=f"Detected {n_nuclei} nucleus-like region(s) (outlined in green)",
                        use_container_width=True,
                    )

                    st.success(
                        "Estimated measurements below — feel free to review and adjust "
                        "them before predicting."
                    )
                    st.session_state.prefill = result
                except ValueError as e:
                    st.warning(str(e))
                    st.session_state.prefill = None
                except Exception:
                    st.error(
                        "Something went wrong analyzing this image. Try a different "
                        "image, or switch to Manual Entry below."
                    )
                    st.session_state.prefill = None

st.divider()

# ----------------------------------------------------------------------
# Input form (used for both modes — image mode pre-fills it)
# ----------------------------------------------------------------------
prefill = st.session_state.prefill

if prefill is not None:
    st.write("Review / adjust the estimated measurements, then predict:")
else:
    st.write(
        "Enter tumor measurements below and get an instant prediction of "
        "**Malignant** or **Benign**. Fields are pre-filled with typical "
        "dataset values — edit any of them with real measurements."
    )

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    user_values = {}
    columns = [col1, col2]

    for i, key in enumerate(key_features):
        label, desc = FIELD_INFO.get(key, (key, ""))
        col = columns[i % 2]
        default_val = prefill[key] if prefill is not None else medians[key]
        with col:
            user_values[key] = st.number_input(
                label,
                value=float(default_val),
                help=desc,
                format="%.4f",
            )

    submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

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

    st.divider()

    if pred == 1:
        st.error(f"### Prediction: Malignant")
    else:
        st.success(f"### Prediction: Benign")

    c1, c2 = st.columns(2)
    c1.metric("Benign confidence", f"{proba[0]*100:.1f}%")
    c2.metric("Malignant confidence", f"{proba[1]*100:.1f}%")

    st.progress(float(proba[1]), text="Malignancy risk score")

    if prefill is not None:
        st.caption(
            "ℹ️ This prediction used measurements estimated from your uploaded "
            "image — an experimental approximation, not a calibrated clinical "
            "measurement."
        )

st.divider()
st.caption(
    "⚠️ For educational purposes only — not a medical diagnosis. "
    "Always consult a qualified doctor."
)
