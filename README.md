---

# Ligue 1 Expected Goals (xG) Model

An interpretable, calibrated **logistic regression** model that predicts the probability of a shot becoming a goal in the French **Ligue 1**.

<p align="center">
<img width="1191" height="674" alt="image" src="https://github.com/user-attachments/assets/0557cd19-ac02-42d4-b2b0-7b1453fcd123" />

</p>

---

## Table of Contents

* [Overview](#overview)
* [What is xG?](#what-is-xg)
* [Business Context](#business-context)
* [Repository Structure](#repository-structure)
* [Data Dictionary](#data-dictionary)
* [Data Quality & Assumptions](#data-quality--assumptions)
* [Modeling Approach](#modeling-approach)

  * [Design Principles](#design-principles)
  * [Model Choice](#model-choice)
  * [Preprocessing](#preprocessing)
  * [Splitting Strategy](#splitting-strategy)
  * [Calibration](#calibration)
  * [Metrics](#metrics)
  * [Penalty Handling](#penalty-handling)
* [Code Walkthrough](#code-walkthrough)

  * [Imports & Configuration](#imports--configuration)
  * [Load & Prepare Data](#load--prepare-data)
  * [Feature Engineering](#feature-engineering)
  * [Pipeline & Model](#pipeline--model)
  * [Group-Aware Split, Train, Evaluate](#group-aware-split-train-evaluate)
  * [Calibration Plots & PR Curve](#calibration-plots--pr-curve)
  * [Save Artifacts](#save-artifacts)
* [Quick Start](#quick-start)
* [Training & Inference](#training--inference)
* [Evaluation & Results](#evaluation--results)
* [Visuals](#visuals)
* [Pitfalls & Best Practices](#pitfalls--best-practices)
* [Reproducibility & Model Card](#reproducibility--model-card)
* [Roadmap](#roadmap)
* [FAQ](#faq)
* [Citations](#citations)
* [License](#license)

---

## Overview

This repository implements an **Expected Goals (xG)** model for **Ligue 1** using historical shot event data (e.g., Understat).
It uses **logistic regression** to produce **per-shot probabilities** (0–1). You can aggregate these to **match**, **team**, or **player** xG for analysis and reporting.

This project:

* **Fetches / processes** Ligue 1 shot data
* **Trains & calibrates** a logistic regression model
* **Evaluates** with proper scoring rules (LogLoss, Brier)
* **Generates** pitch visualizations & calibration diagnostics
* **Saves** reproducible artifacts (model + preprocessors)

---

## What is xG?

**Expected Goals (xG)** estimates the probability a shot will result in a goal, based on historical outcomes of similar shots.

xG is widely **credited to Sam Green (Opta)**, who analyzed **300,000+ shots** to estimate the likelihood of scoring from a given location and situation. This work helped establish xG as a cornerstone metric in football analytics.

To learn more, see the **Friends of Tracking** YouTube playlist:
👉 [https://www.youtube.com/playlist?list=PL38nJNjpdbRwXk8Yswn4s8lKavZOQa3gD](https://www.youtube.com/playlist?list=PL38nJNjpdbRwXk8Yswn4s8lKavZOQa3gD)

**Key factors:**

* **Shot location**: distance & angle to goal
* **Shot type**: header, volley, left/right foot
* **Play situation**: open play, counter, set piece
* **Context**: defensive pressure, assist type (if available)

---

## Business Context

**Who uses xG?**

* **Coaches & Analysts** — assess *chance quality*, not just goals
* **Recruiters & Scouts** — identify players who get into good finishing positions
* **Media** — enrich match reports with objective chance evaluation
* **Betting/Forecasting** — improved probability baselines for match models

---

## Repository Structure

```
Ligue1-xG-Model/
│
├── data/
│   ├── shots/                 # Raw shot event data (not tracked; add samples only)
│   └── processed/             # Cleaned / engineered datasets
│
├── artifacts/                 # Saved model, encoders, scalers, predictions (gitignored)
│
├── figures/                   # Visualizations & pitch plots (png)
│
├── notebooks/
│   └── Logistic_Regression_xg.ipynb
│
├── scripts/
│   ├── train_model.py         # CLI training script (optional)
│   └── predict.py             # CLI inference script (optional)
│
├── README.md                  # This documentation
├── requirements.txt           # Dependencies
└── .gitignore
```

---

## Data Dictionary

> Coordinate system must be consistent across all seasons and sources (origin, units, attack direction).

| Column Name      | Type  | Description                                                       |
| ---------------- | ----- | ----------------------------------------------------------------- |
| `match_id`       | int   | Unique match identifier (also used for group-aware splitting)     |
| `player`         | str   | Shooter’s name                                                    |
| `x`              | float | X-coordinate (pitch length scaled; document range/origin)         |
| `y`              | float | Y-coordinate (pitch width scaled)                                 |
| `distance`       | float | Distance from shot location to goal center (meters)               |
| `angle`          | float | Opening angle to the goal mouth (radians or degrees; state which) |
| `shot_type`      | str   | Body part/technique (Header, Right Foot, Left Foot, Volley, etc.) |
| `situation`      | str   | Open Play, Free Kick, Corner, Throw-in, Fast Break, etc.          |
| `under_pressure` | int   | 1 if shooter under pressure; 0 otherwise (if available)           |
| `goal`           | int   | Target: 1 = goal, 0 = not a goal                                  |
| `xg` (predicted) | float | Model output probability (0–1)                                    |

---

## Data Quality & Assumptions

* **Missing values**: Handle explicitly (drop or impute) before training.
* **Outliers**: Check for off-pitch coordinates, zero distances, invalid angles.
* **Duplicates**: Deduplicate events by unique IDs where possible.
* **Coordinate standardization**: Confirm pitch orientation and normalize units.
* **Class balance**: Goals are rare (\~10% of shots); optimize for *probabilities*, not accuracy.

---

## Modeling Approach

### Design Principles

* **Interpretability** → coefficients/odds ratios are meaningful to analysts.
* **Calibration-first** → aggregated xG must sum sensibly across units (shots → matches → seasons).
* **Leakage-safe** → group/time splits replicate deployment conditions.
* **Reproducibility** → version data, code, and artifacts.

### Model Choice

* **Logistic Regression** (L2 regularization, `solver='saga'`, `max_iter=1000`)

  * Stable on small/medium datasets
  * Easy to calibrate (isotonic or Platt)
  * Fast and interpretable

### Preprocessing

* `StandardScaler` for numeric features
* `OneHotEncoder(drop='first')` for categorical variables
* Combined via `ColumnTransformer` inside a `Pipeline`

### Splitting Strategy

* **Group-aware split** (`GroupShuffleSplit`) on `match_id` to avoid leakage
* Optional **temporal split** (e.g., Rounds 1–28 train; 29–34 test) for realism

### Calibration

* **Isotonic regression** (`CalibratedClassifierCV(method="isotonic")`) on held-out folds
* Improves reliability, crucial when summing probabilities over many shots

### Metrics

* **Log Loss** (primary) — proper scoring rule for probabilities
* **Brier Score** (secondary) — mean squared error of probabilities
* Optional: **Reliability diagram**, **PR/ROC curves** for diagnostics

### Penalty Handling

* Exclude penalties during training; assign a **fixed probability** at inference (≈ **0.76**)
* Rationale: penalties dominate calibration otherwise and are better handled as a constant

---

## Code Walkthrough

> These snippets mirror the core pieces of the notebook so reviewers can understand the pipeline without opening `.ipynb`.

### Imports & Configuration

```python
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss
```

### Load & Prepare Data

```python
DATA_PATH = Path("data/processed/ligue1_shots.csv")
df = pd.read_csv(DATA_PATH)

# Exclude penalties for training (assign constant at inference)
df = df[df["situation"].ne("Penalty")].copy()

# Basic sanity checks
req_cols = ["match_id","x","y","distance","angle","shot_type","situation","goal"]
missing = [c for c in req_cols if c not in df.columns]
assert not missing, f"Missing required columns: {missing}"
assert set(df["goal"].unique()) <= {0,1}, "Target must be binary 0/1"
```

### Feature Engineering

```python
num_cols = ["x","y","distance","angle"]
cat_cols = ["shot_type","situation"]

X = df[num_cols + cat_cols]
y = df["goal"].astype(int).values
groups = df["match_id"].values
```

### Pipeline & Model

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ]
)

base_lr = LogisticRegression(
    penalty="l2", solver="saga", max_iter=1000, C=1.0, random_state=42
)

# Wrap LR in a calibrator for reliable probabilities
clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("calibrated_lr", CalibratedClassifierCV(
        base_estimator=base_lr, method="isotonic", cv=3
    ))
])
```

### Group-Aware Split, Train, Evaluate

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)[:, 1]

print("LogLoss:", log_loss(y_test, y_proba))
print("Brier  :", brier_score_loss(y_test, y_proba))
```

### Calibration Plots & PR Curve

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

# Reliability (calibration) curve
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
plt.figure()
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0,1],[0,1],"--",label="Perfectly calibrated")
plt.xlabel("Predicted probability")
plt.ylabel("Observed goal rate")
plt.title("Reliability Diagram (xG)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/reliability_curve.png", dpi=160)

# Precision–Recall
prec, rec, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall (AP = {ap:.3f})")
plt.tight_layout()
plt.savefig("figures/pr_curve.png", dpi=160)
```

### Save Artifacts

```python
ARTIFACTS = Path("artifacts"); ARTIFACTS.mkdir(exist_ok=True)
joblib.dump(clf, ARTIFACTS / "xg_model_ligue1.joblib")
print("Saved:", ARTIFACTS / "xg_model_ligue1.joblib")
```

---

## Quick Start

### Environment

```bash
git clone https://github.com/YourUser/Ligue1-xG-Model.git
cd Ligue1-xG-Model
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt (example):**

```txt
pandas
numpy
scikit-learn
matplotlib
joblib
```

**.gitignore (suggested):**

```
__pycache__/
.ipynb_checkpoints/
env/
venv/
.venv/
*.log
artifacts/
data/*.csv
data/*.json
figures/*.svg
```

---

## Training & Inference

### Train (script)

```bash
python scripts/train_model.py
```

### Predict on new shots (script)

```bash
python scripts/predict.py --input data/processed/match_shots.csv --output artifacts/predictions.csv
```

### Predict (inline)

```python
import joblib, pandas as pd

model = joblib.load("artifacts/xg_model_ligue1.joblib")
df_new = pd.read_csv("data/processed/new_shots.csv")

features = ["x","y","distance","angle","shot_type","situation"]
df_new["xg"] = model.predict_proba(df_new[features])[:, 1]

# Fixed xG for penalties (if present)
PEN_XG_CONST = 0.76
is_pen = df_new["situation"].eq("Penalty")
df_new.loc[is_pen, "xg"] = PEN_XG_CONST

df_new.to_csv("artifacts/predictions_new_shots.csv", index=False)
```

---

## Evaluation & Results

* **Calibration**: reliability curve close to the diagonal across bins
* **Log Loss / Brier**: reported in training logs (replicable with same seed)
* **Comparison**: aligns with Understat xG broadly; improvements expected where features better capture angle/distance nuances

> Add your actual scores and sample plots to `figures/` and reference them here.
<img width="722" height="595" alt="image" src="https://github.com/user-attachments/assets/14e3ed19-2400-47d7-a2a3-c8ba6e6cd5e6" />




---

## Visuals

| Visualization                                           | Description                                     |
| ------------------------------------------------------- | ----------------------------------------------- |
| ![Attacking Half](https://tinyurl.com/4z7bwbr8) | All shots in attacking half, colored by outcome |
| ![Full Pitch](https://tinyurl.com/mv9ndyck)               | All shots, full pitch view                      |
| ![Hexbin Density](https://tinyurl.com/3n58cme3)     | Shot density heatmap                            |
| ![Reliability](https://tinyurl.com/37b6k8n7)           | Predicted vs observed goal rate (calibration)   |
| ![PR Curve](figures/pr_curve.png)                       | Precision–Recall curve                          |
| ![Model Card](figures/xg_model_card.png)                | Model performance summary                       |

---

## Pitfalls & Best Practices

**Avoid**

* Mixing shots from the **same match** across train/test
* Training with **penalties** included (distorts calibration)
* Using **accuracy/AUC** as primary metrics for probability models

**Do**

* Use **group** or **time-based** splits
* Evaluate with **LogLoss** and **Brier**, plus **reliability**
* Save and version **artifacts** (model, encoders, scalers)
* Document **coordinate system** and **feature definitions**

---

## Reproducibility & Model Card

* **Random seeds**: set for model and splits
* **Environment**: save `pip freeze > requirements-lock.txt`
* **Data hash**: store dataset hash/row count in logs
* **Artifacts**: save model + preprocessors + plots
* **Model card** (suggested fields):

  * Project: Ligue 1 xG
  * Data: source, seasons, rows, features, exclusions (penalties)
  * Model: LogisticRegression (L2, saga), calibrated (isotonic)
  * Metrics: LogLoss, Brier, reliability summary
  * Intended use: per-shot probability, aggregate to players/teams
  * Limitations: lacks pressure velocity, keeper positioning, etc.
  * Ethics: no personal data; reporting is aggregate/statistical

---

## Roadmap

* [ ] Add time-based validation split (e.g., Rounds 1–28 train; 29–34 test)
* [ ] Feature expansion: big-chance flag, assist type, pressure proxy
* [ ] Add **xA** (expected assists) and shot-chain features
* [ ] Interactive dashboard (Streamlit / Power BI)
* [ ] Publish model card and season reports

---

## FAQ

**Why logistic regression over XGBoost/NN?**
Interpretable, fast, less prone to overfitting with small-to-medium structured datasets, and easy to calibrate for reliable probabilities.

**Should I use `class_weight="balanced"`?**
Not for xG. It often **hurts calibration**. Prefer regularization + proper scoring + calibration.

**How do you handle penalties?**
Exclude in training; assign a **fixed xG ≈ 0.76** at inference.

**Can I adapt this to other leagues?**
Yes. Ensure **coordinate normalization**, match feature engineering, retrain, and recalibrate.

---

## Citations

* **Sam Green (Opta)** — early xG methodology based on 300,000+ shots
* **Understat** — public football event/shot data: [https://understat.com/](https://understat.com/)
* **Friends of Tracking (YouTube)** — xG learning resources: [https://www.youtube.com/playlist?list=PL38nJNjpdbRwXk8Yswn4s8lKavZOQa3gD](https://www.youtube.com/playlist?list=PL38nJNjpdbRwXk8Yswn4s8lKavZOQa3gD)
* **scikit-learn** — Pedregosa et al. (2011), JMLR: [https://scikit-learn.org/](https://scikit-learn.org/)
* **mplsoccer** — pitch plotting: [https://mplsoccer.readthedocs.io/](https://mplsoccer.readthedocs.io/)

---

## License

This project is licensed under the **[MIT License](LICENSE)**.

**Summary:** You are free to **use, modify, and distribute** this project for any purpose (including commercial) with proper attribution. The project is provided **“as is” without warranty**, and the author is **not liable** for damages arising from its use.

---



---

