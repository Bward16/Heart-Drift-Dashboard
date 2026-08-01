# Data Dictionary & Glossary

Reference for the columns you'll see in `data/*.csv`, the dashboard, and the
Evidently reports — plus the drift-monitoring terms the dashboard uses.

## Source

- **Dataset**: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Original study**: Detrano, R., et al. (1989). *International application of a new
  probability algorithm for the diagnosis of coronary artery disease.*
  American Journal of Cardiology, 64(5), 304–310.
  [doi:10.1016/0002-9149(89)90524-9](https://doi.org/10.1016/0002-9149(89)90524-9)
- **Sites**: four hospitals each ran their own version of the workup —
  Cleveland Clinic Foundation (used here as the reference set), Hungarian
  Institute of Cardiology (Budapest), University Hospital Zurich (Switzerland),
  and V.A. Medical Center (Long Beach, CA).
- This project uses the UCI "processed" files (`processed.<site>.data`), which
  already reduce the original 76 raw attributes down to the 14 most commonly
  used ones — that's the schema below.

## Feature columns

Values below are exactly what's present in `data/reference.csv` after
`ingest.py` — original UCI codes, unchanged (no re-indexing).

| Column | Meaning | Values / units |
|---|---|---|
| `age` | Age | Years (29–77 in the reference set) |
| `sex` | Biological sex | `1` = male, `0` = female |
| `cp` | Chest pain type | `1` = typical angina, `2` = atypical angina, `3` = non-anginal pain, `4` = asymptomatic |
| `trestbps` | Resting blood pressure | mm Hg, on admission to hospital |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | `1` = true, `0` = false |
| `restecg` | Resting electrocardiographic results | `0` = normal, `1` = ST-T wave abnormality, `2` = probable/definite left ventricular hypertrophy (Estes' criteria) |
| `thalach` | Maximum heart rate achieved | bpm, during exercise test |
| `exang` | Exercise-induced angina | `1` = yes, `0` = no |
| `oldpeak` | ST depression induced by exercise, relative to rest | mm of ST depression (ECG) |
| `slope` | Slope of the peak exercise ST segment | `1` = upsloping, `2` = flat, `3` = downsloping |
| `ca` | Number of major vessels colored by fluoroscopy | `0`–`3` |
| `thal` | Thalassemia (blood disorder screen, via nuclear stress test) | `3` = normal, `6` = fixed defect, `7` = reversible defect |
| `target` | Presence of heart disease (derived from raw `num`) | `1` = disease present, `0` = absent — this project binarizes UCI's original 0–4 severity scale (`num > 0`) since the model here is a simple binary classifier |
| `prediction` | Model's predicted `target` | Same encoding as `target`, added by `train.py` |

`ca` and `thal` are the two columns with the most missing values (`?` in the
raw files, especially in Switzerland/VA) — worth keeping in mind when reading
their drift numbers, since sparse data makes drift statistics noisier.

## Understanding target, drift, and the statistical test — in plain terms

These three ideas get conflated easily, so worth spelling out explicitly.

### What `target` is (and isn't)

`target` is the *diagnosis label* — `1` if that patient was diagnosed with
heart disease, `0` if not. It's the one column the whole project revolves
around: `train.py` fits a model to predict it, and `prediction` is that
model's guess. It is **not** about blood pressure specifically — `trestbps`
(resting blood pressure) is just one of 13 *input* columns the model looks at
to make that guess, same as `age`, `chol`, `thal`, etc. None of the inputs
are "the target"; `target` is the separate, 14th column being predicted.

### What "drift" is (and isn't)

Drift asks one narrow question per column: **"does this column's distribution
of values look different at this site compared to Cleveland?"** That's it.
It says nothing about whether the column is useful for prediction, and
nothing about whether the model's accuracy has changed — a column can drift
heavily and have zero effect on accuracy, or barely drift and coincide with a
big accuracy drop. They're measured completely independently in this
dashboard (drift from the input data alone; accuracy from comparing
`prediction` to `target`).

Why check it at all, then? Because the model was *fit* to the shape of the
Cleveland data. If a new site's inputs look statistically different — a
different age spread, different measurement units, a different local disease
rate — there's no guarantee the patterns the model learned still apply there,
even before you have enough labels to measure accuracy directly. In real
deployments (unlike this demo) labels usually lag by weeks or months, so
drift is often the *only* signal you have in the meantime — it's a leading
indicator, where accuracy is a lagging one.

### What "the test used" (stattest) means, and why it matters

Evidently doesn't eyeball the data — for each column it runs an actual
statistical hypothesis test comparing the reference sample to the current
sample, and the test itself depends on the column's type:

- **Numeric columns** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) get
  the **Kolmogorov–Smirnov (K-S) test**. It compares the full shape of two
  distributions (mean, spread, skew — everything at once) without assuming
  they're normally distributed.
- **Categorical columns** (`sex`, `cp`, `restecg`, `slope`, `thal`, etc.) get
  a **chi-squared test**, which compares how often each category shows up in
  one sample vs. the other.

Using the right test per column matters: running a numeric test on `thal`'s
codes (`3`/`6`/`7`) would treat them as if `7` is "more" than `3` in some
meaningful magnitude, when really they're unordered labels (normal / fixed
defect / reversible defect). Evidently auto-picks the test based on the
column's data type and how many distinct values it has, so you don't have to
choose manually.

Both tests output a **p-value** — roughly, "if these two samples actually
came from the same underlying distribution, how likely is it we'd see a
difference this large just by chance?" A low p-value means "very unlikely to
be chance," so Evidently calls it drifted. Worked example, pulled straight
from `reports/hungarian.json`:

| feature | test used | p-value | drifted? |
|---|---|---|---|
| `trestbps` | K-S | 0.58 | No — blood pressure distribution looks statistically similar to Cleveland |
| `age` | K-S | 2.8 × 10⁻¹⁷ | **Yes** — Hungary's age distribution is meaningfully different |
| `thal` | chi-squared | 1.0 × 10⁻¹⁰ | **Yes** — the mix of normal/fixed/reversible defect codes differs |

(Evidently's default threshold is p < 0.05 — anything below that counts as
drifted.)

## Drift-monitoring terms (dashboard / Evidently)

| Term | Meaning |
|---|---|
| **Reference** | The baseline distribution a site is compared against — here, the Cleveland cohort the model was trained on. |
| **Current** | The "live" batch being checked for drift — here, one of Hungary / Switzerland / VA. |
| **Drift** | A statistically significant change in a column's distribution between reference and current, independent of whether the model's accuracy has actually changed. |
| **Stattest** | The statistical test Evidently chose for a given column (e.g. Kolmogorov–Smirnov for continuous numeric columns, chi-squared for categorical ones) — it auto-selects based on column type and cardinality. |
| **Drift score** | The p-value (or distance metric, depending on the test) from that stattest. Evidently flags `drift_detected = True` when the score crosses its threshold (p < 0.05 by default for most tests). |
| **Share of drifted columns / Dataset drift** | Evidently flags the *whole dataset* as drifted once enough individual columns drift (default: >50% of columns) — a heuristic, not a formal statistical test in itself. |
| **Target drift** | Whether the distribution of the true `target` label has shifted — tells you if disease prevalence itself differs at that site, separate from feature drift. |
| **Classification performance / accuracy** | How well the model's `prediction` matches the true `target` at that site. Only measurable here because all four UCI sites happen to have labels — in most real deployments, labels lag, so this metric usually isn't available in real time (see the README's "known open items"). |

## Further reading

- [Evidently AI docs — Data Drift](https://docs.evidentlyai.com/presets/data-drift) — how stattests are chosen and thresholds set.
- [UCI Heart Disease dataset page](https://archive.ics.uci.edu/dataset/45/heart+disease) — full attribute list (including the 62 raw attributes not used here) and citation info.
