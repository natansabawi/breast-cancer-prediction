# Breast Cancer Prediction — Deployment Package

This folder is ready to push to GitHub and deploy on Streamlit Community Cloud (free).

## Contents
- `app_en.py` — English Streamlit app
- `app_am.py` — Amharic Streamlit app (generates a bilingual PDF report)
- `image_features.py` — shared module: estimates the 8 key measurements from
  an uploaded microscopy image using classical image processing (nucleus
  detection via thresholding/watershed + geometric measurement, then
  quantile-mapped onto the training data's scale). **Experimental /
  educational approximation** — see the in-app disclaimer.
- `feature_quantiles.json` — per-feature percentile table (from the training
  data) used by `image_features.py` for the quantile mapping
- `breast_cancer_model.pkl` — trained SVM model + scaler bundle
- `logo.png` — used by the Amharic app
- `NotoSansEthiopic-Regular.ttf` / `NotoSansEthiopic-Bold.ttf` — fonts used
  for the Amharic PDF report
- `requirements.txt` — Python dependencies

## Image-based input (Beta)
Both apps now offer a second input mode: upload a microscopy/histopathology
image instead of typing the 8 measurements by hand. The app detects
nucleus-like regions, measures their geometry (radius, texture, perimeter,
area, concavity, concave points), and maps those measurements onto the same
scale the model was trained on. This is **not** the calibrated method used to
build the original clinical dataset, so treat it as an educational
approximation — it works best on clear, high-contrast images with visibly
separated cell nuclei.

## Deploy (Streamlit Community Cloud, free)
1. Push this folder to a new **public** GitHub repo.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", pick the repo/branch, and set **Main file path** to `app_en.py`.
   Deploy — this gives you the English site.
4. Click "New app" again on the same repo, but set **Main file path** to `app_am.py`.
   Deploy — this gives you the Amharic site (separate URL).

Each app gets its own free `*.streamlit.app` URL, and both auto-redeploy whenever
you push changes to the repo.
