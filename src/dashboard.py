"""Streamlit dashboard for browsing pre-computed Evidently drift reports.

Run monitor.py first to generate reports/<site>.json + .html, then:
    streamlit run src/dashboard.py
"""

from pathlib import Path

import streamlit.components.v1 as components
import pandas as pd
import streamlit as st

from drift_report import (
    SITES,
    extract_classification_accuracy,
    extract_drift_summary,
    extract_feature_drift_table,
    load_report_json,
)

FEATURE_GLOSSARY = {
    "age": "Age (years)",
    "sex": "Biological sex: 1 = male, 0 = female",
    "cp": "Chest pain type: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic",
    "trestbps": "Resting blood pressure (mm Hg on admission)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl: 1 = true, 0 = false",
    "restecg": "Resting ECG: 0 = normal, 1 = ST-T wave abnormality, 2 = probable/definite left ventricular hypertrophy",
    "thalach": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise-induced angina: 1 = yes, 0 = no",
    "oldpeak": "ST depression induced by exercise, relative to rest (mm)",
    "slope": "Slope of the peak exercise ST segment: 1 = upsloping, 2 = flat, 3 = downsloping",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thalassemia screen: 3 = normal, 6 = fixed defect, 7 = reversible defect",
    "target": "True diagnosis: 1 = heart disease present, 0 = absent",
    "prediction": "Model's predicted target, same encoding",
}

DRIFT_GLOSSARY = {
    "Stattest": "The statistical test Evidently auto-selected for that column (e.g. Kolmogorov-Smirnov for numeric, chi-squared for categorical).",
    "Drift score": "The p-value from that test. Lower means stronger evidence that the distribution changed; a column is flagged drifted once the p-value falls below the test's significance level (0.05 by default).",
    "Dataset drift": "Evidently's overall flag for the site: True once more than half of columns are individually flagged as drifted.",
    "Current accuracy": "How often the model's prediction matches the true target at that site — only available here because every UCI site happens to have labels.",
}

st.set_page_config(page_title="Heart Disease Drift Dashboard", layout="wide")
st.title("Heart Disease UCI — Drift Monitoring Dashboard")
st.caption("Reference: Cleveland cohort. Current: each site below, compared independently.")

with st.expander("What do these columns and terms mean?"):
    st.markdown(
        "Full write-up with dataset source, citation, and more detail: "
        "`docs/DATA_DICTIONARY.md` in the project root."
    )
    st.markdown("**Feature columns**")
    st.table(pd.DataFrame(FEATURE_GLOSSARY.items(), columns=["column", "meaning"]))
    st.markdown("**Drift terms**")
    st.table(pd.DataFrame(DRIFT_GLOSSARY.items(), columns=["term", "meaning"]))


@st.cache_data
def cached_report(site: str) -> dict:
    return load_report_json(site)


site = st.sidebar.selectbox("Site (vs. Cleveland reference)", SITES)
try:
    report = cached_report(site)
except FileNotFoundError:
    st.info("Reports are not available yet. Run ingest.py, train.py, and monitor.py to generate them.")
    st.stop()

summary = extract_drift_summary(report)
accuracy = extract_classification_accuracy(report)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Drifted features", f"{summary.get('n_drifted', '—')}/{summary.get('n_features', '—')}")
col2.metric("Share drifted", f"{summary.get('share_drifted', 0):.0%}")
col3.metric("Dataset drift flagged", "Yes" if summary.get("dataset_drift") else "No")
col4.metric("Current accuracy", f"{accuracy:.2%}" if accuracy is not None else "—")

st.subheader("Per-feature drift")
st.caption("Drifted features first, then by p-value ascending, so the strongest evidence of drift is at the top.")
st.dataframe(extract_feature_drift_table(report), use_container_width=True)

st.subheader("Accuracy across all sites")
accuracy_rows = []
for s in SITES:
    a = extract_classification_accuracy(cached_report(s))
    accuracy_rows.append({"site": s, "accuracy": a})
st.bar_chart(pd.DataFrame(accuracy_rows).set_index("site"))


st.caption(
    "Cleveland reference predictions are in-sample: the model was trained on those same rows. "
    "Reference accuracy is not a held-out baseline. Cross-site accuracy differences alone "
    "do not establish that drift caused a performance change."
)

st.subheader("Full Evidently report")
report_path = Path("reports") / f"{site}.html"
if report_path.exists():
    report_html = report_path.read_text(encoding="utf-8")
    st.download_button("Download full report", report_html, file_name=f"{site}.html", mime="text/html")
    with st.expander("View full report"):
        components.html(report_html, height=800, scrolling=True)
else:
    st.info("The HTML report is missing. Run monitor.py to regenerate it.")
