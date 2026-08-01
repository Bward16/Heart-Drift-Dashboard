"""Generate Evidently drift/quality/performance reports comparing each
'production' site against the Cleveland reference distribution.

Writes one HTML + JSON snapshot per site into reports/, which dashboard.py
reads rather than recomputing reports on every page load.
"""

import os

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

COLUMN_MAPPING = ColumnMapping(
    target="target",
    prediction="prediction",
    numerical_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
)


def run_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset(),
        ClassificationPreset(),
    ])
    report.run(reference_data=reference, current_data=current, column_mapping=COLUMN_MAPPING)
    return report


def main() -> None:
    os.makedirs("reports", exist_ok=True)
    reference = pd.read_csv("data/reference_scored.csv")

    for site in ("hungarian", "switzerland", "va"):
        current = pd.read_csv(f"data/current_{site}_scored.csv")
        report = run_report(reference, current)
        report.save_html(f"reports/{site}.html")
        report.save_json(f"reports/{site}.json")
        print(f"Wrote reports/{site}.html and reports/{site}.json")


if __name__ == "__main__":
    main()
