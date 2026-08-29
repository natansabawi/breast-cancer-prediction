"""
Breast Cancer Prediction — Web App (Streamlit)
================================================
Run with:  streamlit run breast_cancer_web_app.py

Loads the pre-trained model from breast_cancer_model.pkl (run
train_and_save_model.py first if that file doesn't exist yet).
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

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
st.write(
    "Enter tumor measurements below and get an instant prediction of "
    "**Malignant** or **Benign**. Fields are pre-filled with typical "
    "dataset values — edit any of them with real measurements."
)

st.divider()

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    user_values = {}
    columns = [col1, col2]

    for i, key in enumerate(key_features):
        label, desc = FIELD_INFO.get(key, (key, ""))
        col = columns[i % 2]
        with col:
            user_values[key] = st.number_input(
                label,
                value=float(medians[key]),
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

st.divider()
st.caption(
    "⚠️ For educational purposes only — not a medical diagnosis. "
    "Always consult a qualified doctor."
)
