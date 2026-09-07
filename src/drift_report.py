"""Read Evidently report JSON into plain dataframes and summaries.

Kept free of Streamlit so the extraction logic can be tested directly.

Every stattest Evidently selects for these datasets is p-value based
(Kolmogorov-Smirnov, chi-square, Z-test), so a lower score always means
stronger evidence that the distribution changed. Ordering therefore has to
be ascending; sorting descending would surface the least-drifted columns
first.
"""

import json

import pandas as pd

SITES = ("hungarian", "switzerland", "va")


def load_report_json(site: str, reports_dir: str = "reports") -> dict:
    with open(f"{reports_dir}/{site}.json") as f:
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
            return (
                pd.DataFrame(rows)
                .sort_values(["drift_detected", "score"], ascending=[False, True])
                .reset_index(drop=True)
            )
    return pd.DataFrame()


def extract_classification_accuracy(report: dict) -> float | None:
    for metric in report["metrics"]:
        if metric["metric"] == "ClassificationQualityMetric":
            return metric["result"]["current"]["accuracy"]
    return None
