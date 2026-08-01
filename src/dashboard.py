"""Streamlit dashboard for browsing pre-computed Evidently drift reports.

Run monitor.py first to generate reports/<site>.json + .html, then:
    streamlit run src/dashboard.py
"""

import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

SITES = ("hungarian", "switzerland", "va")

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
    "Drift score": "The p-value (or distance metric) from that test. A column is flagged drifted once its score crosses the test's threshold.",
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
def load_report_json(site: str) -> dict:
    with open(f"reports/{site}.json") as f:
        return json.load(f)


def extract_drift_summary(report: dict) -> dict:
    for metric in report["metrics"]:
        if metric["metric"] == "DataDriftTable":
            result = metric["result"]
            return {
                "n_drifted": result["number_of_drifted_columns"],
                "n_features": result["number_of_columns"],
                "share_drifted": result["share_of_drifted_columns"],
                "dataset_drift": result["dataset_drift"],
            }
    return {}


def extract_feature_drift_table(report: dict) -> pd.DataFrame:
    for metric in report["metrics"]:
        if metric["metric"] == "DataDriftTable":
            per_column = metric["result"]["drift_by_columns"]
            rows = [
                {
                    "feature": name,
                    "drift_detected": info["drift_detected"],
                    "stattest": info.get("stattest_name"),
                    "score": info.get("drift_score"),
                }
                for name, info in per_column.items()
            ]
            return pd.DataFrame(rows).sort_values("score", ascending=False)
    return pd.DataFrame()


def extract_classification_accuracy(report: dict) -> float | None:
    for metric in report["metrics"]:
        if metric["metric"] == "ClassificationQualityMetric":
            return metric["result"]["current"]["accuracy"]
    return None


site = st.sidebar.selectbox("Site (vs. Cleveland reference)", SITES)
report = load_report_json(site)

summary = extract_drift_summary(report)
accuracy = extract_classification_accuracy(report)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Drifted features", f"{summary.get('n_drifted', '—')}/{summary.get('n_features', '—')}")
col2.metric("Share drifted", f"{summary.get('share_drifted', 0):.0%}")
col3.metric("Dataset drift flagged", "Yes" if summary.get("dataset_drift") else "No")
col4.metric("Current accuracy", f"{accuracy:.2%}" if accuracy is not None else "—")

st.subheader("Per-feature drift")
st.dataframe(extract_feature_drift_table(report), use_container_width=True)

st.subheader("Accuracy across all sites")
accuracy_rows = []
for s in SITES:
    r = load_report_json(s)
    a = extract_classification_accuracy(r)
    accuracy_rows.append({"site": s, "accuracy": a})
st.bar_chart(pd.DataFrame(accuracy_rows).set_index("site"))

st.subheader("Full Evidently report")
with open(f"reports/{site}.html") as f:
    components.html(f.read(), height=800, scrolling=True)
