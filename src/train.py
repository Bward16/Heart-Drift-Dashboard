"""Train a simple classifier on the Cleveland reference distribution.

The model is intentionally simple (logistic regression) since the point of this
project is monitoring, not modeling. Predictions are attached back onto the
reference and current dataframes so Evidently can also track prediction/target
drift and performance decay, not just raw feature drift.
"""

import os

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def add_predictions(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prediction"] = model.predict(df[FEATURE_COLS])
    return df


def main() -> None:
    os.makedirs("reports", exist_ok=True)
    reference = pd.read_csv("data/reference.csv")

    model = build_pipeline()
    model.fit(reference[FEATURE_COLS], reference["target"])
    joblib.dump(model, "reports/model.pkl")

    reference = add_predictions(model, reference)
    reference.to_csv("data/reference_scored.csv", index=False)

    for site in ("hungarian", "switzerland", "va"):
        current = pd.read_csv(f"data/current_{site}.csv")
        current = add_predictions(model, current)
        current.to_csv(f"data/current_{site}_scored.csv", index=False)

    print("Trained model saved to reports/model.pkl; scored datasets written to data/")


if __name__ == "__main__":
    main()
