"""Download and harmonize the UCI Heart Disease multi-site dataset.

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
Four sites, each a separate file with the same 14 raw columns and '?' for missing values.
Cleveland is the clean, widely-used 303-row set -> treated as the reference/training distribution.
Hungary / Switzerland / VA are messier and structurally different -> treated as "production" batches.
"""

import io
import os

import pandas as pd
import requests

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"

SITES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def fetch_site(site: str) -> pd.DataFrame:
    filename = SITES[site]
    resp = requests.get(f"{BASE_URL}/{filename}", timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(
        io.StringIO(resp.text),
        header=None,
        names=COLUMNS,
        na_values="?",
    )
    df["site"] = site
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # UCI 'num' is 0 (no disease) to 4 (severity) -> binarize for a simple classifier
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df


def main() -> None:
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    frames = {}
    for site in SITES:
        df = fetch_site(site)
        df.to_csv(f"{raw_dir}/{site}.csv", index=False)
        frames[site] = clean(df)
        print(f"{site}: {len(df)} rows, {df.isna().sum().sum()} missing cells")

    frames["cleveland"].to_csv("data/reference.csv", index=False)
    for site in ("hungarian", "switzerland", "va"):
        frames[site].to_csv(f"data/current_{site}.csv", index=False)

    print("Wrote data/reference.csv and data/current_<site>.csv")


if __name__ == "__main__":
    main()
