# Heart Disease UCI — Drift Monitoring Dashboard

An MLOps-style drift monitoring dashboard built on the UCI Heart Disease dataset.
The dataset's four hospital sources (Cleveland, Hungary, Switzerland, VA Long Beach)
stand in for a reference training distribution vs. three "production" sites, giving
real distributional shift instead of synthetic noise.

## Design

- **Reference**: Cleveland cohort (303 rows, the "clean" set most tutorials use) — this is what the model was trained on.
- **Current**: Hungarian / Switzerland / VA cohorts — messier, different measurement practices per site, different disease prevalence.
- **Model**: logistic regression, intentionally simple — the point of this project is monitoring, not modeling.
- **Drift detection**: [Evidently](https://github.com/evidentlyai/evidently) (`DataDriftPreset`, `DataQualityPreset`, `TargetDriftPreset`, `ClassificationPreset`), pinned to `<0.5` since its API changed significantly in later versions.
- **Dashboard**: Streamlit, reads pre-computed Evidently JSON/HTML snapshots rather than recomputing on page load — mirrors how real monitoring separates the batch job from the UI.

Because labels exist for every site here, the dashboard also tracks model
classification accuracy per site alongside drift — most real production
dashboards only have the drift side, since labels lag. Worth calling out as a
simplification if you present this.

## Evaluation limitation

Cleveland predictions are currently generated on the same rows used to fit the model. Its accuracy is in-sample and must not be treated as a held-out baseline. Cross-site accuracy differences do not by themselves establish a causal effect of drift. A reproducible held-out evaluation is planned; existing reports retain their original methodology.

Run `pytest` to verify report extraction and ordering. The dashboard also provides the full HTML reports for inspection and download.

## Reference

Not sure what a column name or a drift term on the dashboard means? See
[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) — covers every feature
(units, encoding, clinical meaning), the dataset source/citation, and a
glossary of the Evidently drift terms (stattest, drift score, dataset drift, etc).

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```bash
python src/ingest.py    # downloads + harmonizes the 4 UCI sources into data/
python src/train.py     # trains logistic regression on Cleveland, scores all sites
python src/monitor.py   # generates reports/<site>.{html,json} via Evidently
streamlit run src/dashboard.py
```

## Pipeline stages ↔ MLOps roles

The four commands above are four independent, decoupled stages — each one
writes artifacts to disk that the next stage reads, rather than sharing
in-memory state. That decoupling is itself the MLOps-relevant design choice:
it's what lets each stage run on its own schedule, on its own machine,
independently of the others in a real deployment.

```
ingest.py   →   train.py   →   monitor.py   →   dashboard.py
(data          (training       (monitoring       (serving /
 pipeline)      pipeline)       pipeline)          observability UI)
```

- **`ingest.py` — data pipeline / ETL.** Pulls raw data from an external
  source, harmonizes schema across four inconsistent files, and writes clean
  artifacts (`data/raw/*.csv`, `data/reference.csv`, `data/current_<site>.csv`).
  In production this stage would run on a schedule as new data lands; here
  it's a stand-in for that, run once against a static historical dataset.
- **`train.py` — training pipeline.** Fits only on `data/reference.csv`,
  same as a real job only ever seeing its training split. Writes two kinds of
  artifacts: the model itself (`reports/model.pkl`, via `joblib` — a crude
  stand-in for a model registry like MLflow) and *scored* datasets
  (`data/*_scored.csv`, i.e. every site with `prediction` attached). That
  second artifact is what lets the monitoring stage check prediction/target
  drift and accuracy without re-running the model.
- **`monitor.py` — monitoring / observability pipeline.** The batch job in a
  drift-monitoring setup — in production this runs periodically (on a
  schedule, or whenever a new scored batch lands), not on every dashboard
  view. It writes static snapshots (`reports/<site>.json` + `.html`) rather
  than computing anything live, since the drift statistics are somewhat
  expensive to compute and don't need to be recomputed per viewer.
- **`dashboard.py` — serving layer.** Reads only the JSON/HTML `monitor.py`
  already produced — never touches raw data, Evidently, or the model
  directly. Same role as a Grafana dashboard sitting in front of a metrics
  database rather than computing metrics itself.

## Structure

```
heart-drift-dashboard/
├── data/
│   ├── raw/                       # original per-site UCI files
│   ├── reference.csv              # Cleveland, cleaned
│   ├── reference_scored.csv       # Cleveland + model predictions (generated)
│   ├── current_<site>.csv         # Hungary / Switzerland / VA, cleaned
│   └── current_<site>_scored.csv  # + model predictions (generated)
├── src/
│   ├── ingest.py
│   ├── train.py
│   ├── monitor.py
│   └── dashboard.py
├── docs/
│   └── DATA_DICTIONARY.md    # feature definitions, dataset source, drift/stattest glossary
├── reports/                   # Evidently snapshots + model.pkl (generated, gitignored)
└── requirements.txt
```

## Known open items

- Column harmonization in `ingest.py` assumes the standard 14-column UCI schema
  (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num).
  Worth spot-checking against the actual downloaded files before trusting the drift numbers.
- No CI/scheduling yet — reports are generated on demand, not on a recurring batch.
- No feedback loop — a real pipeline would have drift/accuracy numbers trigger
  something (an alert, an automatic retrain) when they cross a threshold.
  Here, a human reads the dashboard and decides what to do.

## License

Licensed under the [Apache License 2.0](LICENSE).

Releases up to and including commit `13bc7d8` were published under the MIT
license and remain available under those terms.
