# Breast Cancer Prediction — Deployment Package

This folder is ready to push to GitHub and deploy on Streamlit Community Cloud (free).

## Contents
- `app_en.py` — English Streamlit app
- `app_am.py` — Amharic Streamlit app (generates a bilingual PDF report)
- `breast_cancer_model.pkl` — trained SVM model + scaler bundle
- `logo.png` — used by the Amharic app
- `fonts/` — Ethiopic fonts used for the PDF report
- `requirements.txt` — Python dependencies

## Deploy (Streamlit Community Cloud, free)
1. Push this folder to a new **public** GitHub repo.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", pick the repo/branch, and set **Main file path** to `app_en.py`.
   Deploy — this gives you the English site.
4. Click "New app" again on the same repo, but set **Main file path** to `app_am.py`.
   Deploy — this gives you the Amharic site (separate URL).

Each app gets its own free `*.streamlit.app` URL, and both auto-redeploy whenever
you push changes to the repo.
