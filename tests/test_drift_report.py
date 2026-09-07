"""Checks on how Evidently reports are turned into dashboard tables.

The ordering tests exist because every stattest used here is p-value based, so
sorting descending silently puts the least-drifted columns at the top.
"""

from pathlib import Path

import pytest

from drift_report import (
    SITES,
    extract_classification_accuracy,
    extract_drift_summary,
    extract_feature_drift_table,
    load_report_json,
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def build_report(columns: dict) -> dict:
    return {
        "metrics": [
            {
                "metric": "DataDriftTable",
                "result": {
                    "number_of_drifted_columns": sum(
                        1 for c in columns.values() if c["drift_detected"]
                    ),
                    "number_of_columns": len(columns),
                    "share_of_drifted_columns": 0.0,
                    "dataset_drift": False,
                    "drift_by_columns": columns,
                },
            }
        ]
    }


def test_lowest_p_value_is_listed_first():
    report = build_report(
        {
            "chol": {"drift_detected": False, "stattest_name": "K-S p_value", "drift_score": 0.67},
            "age": {"drift_detected": True, "stattest_name": "K-S p_value", "drift_score": 0.0001},
            "cp": {"drift_detected": True, "stattest_name": "chi-square p_value", "drift_score": 0.02},
        }
    )

    table = extract_feature_drift_table(report)

    assert list(table["feature"]) == ["age", "cp", "chol"]


def test_drifted_columns_come_before_undrifted_ones():
    report = build_report(
        {
            "undrifted_low": {"drift_detected": False, "stattest_name": "K-S p_value", "drift_score": 0.06},
            "drifted_high": {"drift_detected": True, "stattest_name": "Z-test p_value", "drift_score": 0.049},
        }
    )

    table = extract_feature_drift_table(report)

    assert list(table["drift_detected"]) == [True, False]
    assert table.loc[0, "feature"] == "drifted_high"


def test_missing_drift_metric_yields_an_empty_table():
    assert extract_feature_drift_table({"metrics": []}).empty


@pytest.mark.skipif(not REPORTS_DIR.exists(), reason="reports/ not generated yet")
@pytest.mark.parametrize("site", SITES)
def test_real_reports_are_ordered_by_strength_of_evidence(site):
    report = load_report_json(site, reports_dir=str(REPORTS_DIR))
    table = extract_feature_drift_table(report)

    assert not table.empty

    drifted = table[table["drift_detected"]]
    undrifted = table[~table["drift_detected"]]

    assert list(drifted.index) == list(range(len(drifted)))
    assert drifted["score"].is_monotonic_increasing
    assert undrifted["score"].is_monotonic_increasing

    if not drifted.empty:
        assert drifted["score"].max() < 0.05
    if not undrifted.empty:
        assert undrifted["score"].min() >= 0.05


@pytest.mark.skipif(not REPORTS_DIR.exists(), reason="reports/ not generated yet")
@pytest.mark.parametrize("site", SITES)
def test_summary_and_accuracy_are_extracted(site):
    report = load_report_json(site, reports_dir=str(REPORTS_DIR))
    summary = extract_drift_summary(report)

    assert summary["n_features"] > 0
    assert 0 <= summary["n_drifted"] <= summary["n_features"]

    accuracy = extract_classification_accuracy(report)
    assert accuracy is None or 0.0 <= accuracy <= 1.0
