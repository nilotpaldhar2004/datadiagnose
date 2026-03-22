# DataDiagnose

**A zero-dependency Python library that audits your dataset and prescribes fixes before you train your model.**

[![Tests](https://github.com/nilotpaldhar2004/datadiagnose/actions/workflows/tests.yml/badge.svg)](https://github.com/nilotpaldhar2004/datadiagnose/actions)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-1.0.1-orange.svg)](https://github.com/nilotpaldhar2004/datadiagnose)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/nilotpaldhar2004/datadiagnose/blob/main/LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](https://github.com/nilotpaldhar2004/datadiagnose/blob/main/pyproject.toml)

---

## The Problem This Solves

Every beginner data scientist goes through the same painful experience. You download a dataset, you are excited to build your first model, you train it — and the results are terrible. 50% accuracy. Predictions that make no sense. Hours of confusion.

In most cases the model is not the problem. **The data is broken.** There are missing values pulling your model in the wrong direction. An outlier like an age of 900 years is destroying your statistics. Your target column has 95% of rows saying "no", so your model just learned to say "no" for everything.

Experienced data scientists know to check for these things before touching a model. They run a full diagnostic on the dataset first. **DataDiagnose automates that diagnostic in one function call.**

---

## What It Does

Give DataDiagnose any dataset and it returns:

- A **health score** from 0 to 100 showing how clean your data is
- A list of every **problem detected**, with severity (CRITICAL / HIGH / MEDIUM / LOW)
- A specific **fix suggestion** for every problem
- **Model recommendations** based on your data characteristics
- **Feature engineering hints** based on your column names
- A **Pandas DataFrame** of all column statistics with one call *(new in v1.0.1)*

Eight problems are detected automatically:

| Problem | What It Means |
|---|---|
| Missing Values | Null, empty, or NaN values in any column |
| Outliers | Extreme values detected by both IQR and Z-score methods |
| Skewness | Lopsided distributions that hurt linear models |
| Class Imbalance | One class vastly outnumbering others in your target |
| Data Leakage | Columns that secretly contain the answer |
| Duplicate Rows | Identical rows that bias your model |
| Constant Columns | Columns with zero variation — zero information |
| High Cardinality | ID-like columns with almost all unique values |

---

## What Is New in v1.0.1

### 1. Pass a pandas DataFrame directly

You no longer need to call `.to_dict()` yourself. Pass a DataFrame straight into `diagnose()`:

```python
import pandas as pd
import datadiagnose as dd

df = pd.read_csv("my_data.csv")

# v1.0.0 — old way (still works)
report = dd.diagnose(df.to_dict(orient="list"), target_col="target")

# v1.0.1 — new way (simpler)
report = dd.diagnose(df, target_col="target")
```

### 2. Get a stats DataFrame with one call

New function `get_stats_df()` runs the full diagnosis and returns all column statistics as a Pandas DataFrame — perfect for Jupyter notebooks:

```python
import datadiagnose as dd

df_stats = dd.get_stats_df(df, target_col="target")
print(df_stats)
```

The output is a clean table where every column in your dataset is a header and every statistic (type, mean, std, missing%, skewness, outlier count, etc.) is a row. Easy to read, easy to sort, easy to export.

### 3. Smarter class imbalance detection

In v1.0.0, regression targets with many unique values (like house prices or salaries) were sometimes incorrectly flagged as imbalanced. This is now fixed — DataDiagnose correctly skips class imbalance checks on continuous numeric regression targets.

---

## Installation

DataDiagnose has **zero runtime dependencies** for its core features. It runs entirely on Python's standard library.

```bash
# Once published on PyPI
pip install datadiagnose
```

For now, copy the `datadiagnose/` folder into your project and import directly.

To use `get_stats_df()` or pass a DataFrame directly, pandas must be installed:

```bash
pip install pandas
```

---

## Quick Start

```python
from datadiagnose import diagnose

dataset = {
    "age":    [25, 30, None, 22, 900, 28],
    "income": [50000, 60000, None, 48000, 52000, 61000],
    "city":   ["KOL", "MUM", "KOL", "DEL", "KOL", "KOL"],
    "target": [1, 0, 1, 0, 1, 0],
}

report = diagnose(dataset, target_col="target")
print(report)
```

Output:

```
==============================================================
  DATADIAGNOSE REPORT — DATASET
==============================================================
  Rows    : 6
  Columns : 4
  Score   : 84/100   Needs Work
--------------------------------------------------------------
  Issues Found (2)

  1. MEDIUM
     Missing Values in 'age'
     -> 16.7% of values are missing.
     Fix: Fill 'age' with median (numeric) or mode (categorical).

  2. MEDIUM
     Missing Values in 'income'
     -> 16.7% of values are missing.
     Fix: Fill 'income' with median (numeric) or mode (categorical).
...
```

---

## Works With pandas — Three Ways

```python
import pandas as pd
import datadiagnose as dd

df = pd.read_csv("titanic.csv")

# Way 1 — Pass DataFrame directly (new in v1.0.1)
report = dd.diagnose(df, target_col="Survived")

# Way 2 — Convert manually (original method, still works)
report = dd.diagnose(df.to_dict(orient="list"), target_col="Survived")

# Way 3 — Get all stats as a DataFrame instantly (new in v1.0.1)
df_stats = dd.get_stats_df(df, target_col="Survived")
print(df_stats)
```

---

## Full API Reference

### `diagnose(dataset, target_col=None, dataset_name="dataset")`

The main function. Runs all eight detectors and returns a `DiagnosisReport`.

**New in v1.0.1:** accepts pandas DataFrames directly without conversion.

```python
report = diagnose(dataset, target_col="label", dataset_name="Titanic")

report.score          # int — health score 0-100
report.issues         # list of Issue objects
report.suggestions    # list of fix strings
report.model_types    # list of recommended model names
report.column_reports # dict — {col_name: ColumnReport}
report.n_rows         # int
report.n_cols         # int
```

---

### `get_stats_df(dataset, target_col=None)` — *new in v1.0.1*

Runs a full diagnosis and returns all column statistics as a transposed Pandas DataFrame. Metrics are rows, your dataset columns are the headers.

```python
import datadiagnose as dd

df_stats = dd.get_stats_df(my_dataset, target_col="target")
# Returns DataFrame — perfect for Jupyter notebooks
# Requires: pip install pandas
```

---

### `report.to_df()` — *new in v1.0.1*

Method directly on the `DiagnosisReport` object. Does the same thing as `get_stats_df()` but you call it after you already have a report.

```python
report   = diagnose(my_dataset, target_col="target")
df_stats = report.to_df()
```

---

### `quick_scan(dataset, target_col=None)`

One-liner that runs the diagnosis and immediately prints the report to your terminal. Returns the `DiagnosisReport` object in case you need it.

```python
quick_scan(dataset, target_col="label")
```

---

### `health_score(dataset, target_col=None)`

Returns only the integer score (0–100). Perfect for quality gates in automated pipelines — block model training if data quality is too low.

```python
score = health_score(dataset, target_col="label")

if score < 70:
    raise ValueError(f"Data quality too low: {score}/100. Fix issues first.")
```

---

### `list_issues(dataset, target_col=None)`

Returns a concise list of `(severity, title)` tuples for quick scanning without reading the full report.

```python
for severity, title in list_issues(dataset, "label"):
    print(f"[{severity}] {title}")
```

---

### `get_suggestions(dataset, target_col=None)`

Returns only the list of actionable fix suggestions as plain strings.

```python
for tip in get_suggestions(dataset, "label"):
    print("-", tip)
```

---

### `column_summary(dataset, col_name, target_col=None)`

Returns the `ColumnReport` for one specific column — all the statistics computed for that column during diagnosis.

```python
rep = column_summary(dataset, "age")
print(rep.details)
# {'type': 'numeric', 'mean': '27.4', 'std': '3.2', 'missing_pct': '10.0%', ...}
```

---

## Understanding the Health Score

Every dataset starts at 100. Each detected issue deducts points based on its severity:

| Severity | Points Lost | Example |
|---|---|---|
| CRITICAL | 25 | Data leakage confirmed, >60% missing values |
| HIGH | 15 | >30% missing, class imbalance ratio 10:1+ |
| MEDIUM | 8 | Moderate skewness, some outliers, mild imbalance |
| LOW | 3 | A few minor outliers, slight skewness |

| Score | Status | What To Do |
|---|---|---|
| 80 – 100 | Healthy | Data is ready for modelling |
| 50 – 79 | Needs Work | Fix HIGH and CRITICAL issues before training |
| 0 – 49 | Critical | Do not train models yet |

---

## Project Structure

```
datadiagnose/
│
├── datadiagnose/          <- The Python package
│   ├── __init__.py        <- Public API
│   ├── core.py            <- Main diagnose() engine + get_stats_df()
│   ├── detectors.py       <- All 8 detector functions
│   ├── models.py          <- DiagnosisReport, Issue, ColumnReport + to_df()
│   └── utils.py           <- Pure math helpers (no dependencies)
│
├── tests/
│   ├── __init__.py
│   ├── sample_data.py     <- 11 sample datasets with known problems
│   ├── test_detectors.py  <- 60 unit tests for each detector
│   └── test_core.py       <- 80 integration tests for the full API
│
├── examples/
│   ├── basic_usage.py          <- Start here — every function shown
│   ├── pandas_integration.py   <- How to use with pandas DataFrames
│   └── student_dataset_demo.py <- Full workflow, step by step
│
├── .github/workflows/tests.yml <- Auto-run tests on every push
├── .flake8                     <- Code style configuration
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## Running the Tests

DataDiagnose has 140 tests covering every detector, every public function, and edge cases. Tests use only Python's built-in `unittest` — no pytest required (though pytest works too).

```bash
# Run all 140 tests at once
python -m unittest discover -s tests -v

# Run just the detector tests
python -m unittest tests.test_detectors -v

# Run just the core API tests
python -m unittest tests.test_core -v

# Run one specific test class
python -m unittest tests.test_detectors.TestMissingValueDetector -v

# Run one specific test
python -m unittest tests.test_detectors.TestMissingValueDetector.test_critical_missing_severity -v
```

All 140 tests should pass with output ending in:

```
----------------------------------------------------------------------
Ran 140 tests in 0.08s

OK
```

---

## Running the Examples

```bash
# Simplest introduction — run this first
python examples/basic_usage.py

# How to use with pandas DataFrames
python examples/pandas_integration.py

# A full realistic data cleaning workflow, step by step
python examples/student_dataset_demo.py
```

---

## Why Zero Runtime Dependencies?

DataDiagnose uses only Python's built-in `math`, `statistics`, and `collections` modules for its core logic. Pandas is **optional** and only needed if you want `get_stats_df()` or `report.to_df()`. This design was deliberate:

**Works everywhere** — any Python 3.7+ environment, nothing extra to install for the core library.

**No version conflicts** — requiring numpy or pandas would create compatibility problems for users who already have specific versions pinned in their projects.

**Educational** — every algorithm (IQR outlier detection, Pearson correlation, skewness) is written from scratch in plain Python. You can read the source code and learn exactly how each calculation works, not just how to call a black box.

**Lightweight** — five Python files, around 1000 lines, no compiled extensions, no C dependencies.

---

## Design Decisions

**Why does it detect problems but not fix them automatically?**

Automatically modifying data without human judgment is dangerous. Filling missing values with the wrong strategy can make your model worse, not better. Whether you should drop a column or impute it — and what value to impute with — depends on your domain knowledge and what the data actually means in your context. DataDiagnose gives you the diagnosis and a specific recommendation for each issue. The decision and the action remain yours.

**Why does it accept a dict-of-lists as its native format?**

The original design needed zero dependencies, which meant no pandas. A plain Python dict is the simplest possible format. As of v1.0.1, DataFrames are also accepted directly. Internally, the library always converts to dict-of-lists format because column-wise operations on a dict are faster than on a DataFrame for this use case.

---

## How to Contribute

Contributions are welcome. Here are some ideas from the roadmap:

- HTML report export — a self-contained HTML file with charts and colour-coded issue cards
- Correlation matrix — detect multicollinearity between feature columns
- Web dashboard — a Flask/FastAPI endpoint where you upload a CSV and get a diagnosis in the browser
- More detectors — mixed type columns, date range anomalies, text language detection

To contribute:

1. Fork the repository on GitHub
2. Create a branch: `git checkout -b feature/my-improvement`
3. Write your code and add tests for it
4. Confirm all 140 existing tests still pass: `python -m unittest discover -s tests -v`
5. Open a pull request with a clear description of what you changed and why

---

## Changelog

### v1.0.1
- **New:** `diagnose()` now accepts pandas DataFrames directly — no `.to_dict()` needed
- **New:** `get_stats_df(dataset, target_col)` — returns full column statistics as a Pandas DataFrame
- **New:** `report.to_df()` — method on DiagnosisReport to get stats table as DataFrame
- **Fixed:** Class imbalance detector no longer incorrectly flags continuous regression targets (e.g. house prices, salaries)

### v1.0.0 — Initial Release
- Eight detectors: missing values, outliers, skewness, class imbalance, data leakage, duplicate rows, constant columns, high cardinality
- Feature engineering hints based on column name patterns (datetime, text, geo)
- Model recommendation engine for regression, binary, and multiclass classification
- Health score system (0–100) with severity-based point deductions
- Full public API: `diagnose`, `quick_scan`, `health_score`, `list_issues`, `get_suggestions`, `column_summary`
- 140 unit and integration tests
- Zero runtime dependencies

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for the full text.

In plain English: you can use this code for anything, including commercial projects, as long as you keep the copyright notice with the author's name in any copy you distribute.

Copyright (c) 2026 **Nilotpal Dhar**

---

## Author

**Nilotpal Dhar**

Built as a beginner Python project to learn how data science diagnostics work from first principles. Every algorithm in this library — IQR outlier detection, Pearson correlation, skewness calculation — is implemented from scratch in plain Python so that reading the source code teaches you how the mathematics actually works, not just how to call a function.

If this library helped you, please star the repository on GitHub.
If you found a bug or have a feature idea, open an issue — all feedback is welcome.

[GitHub Repository](https://github.com/nilotpaldhar2004/datadiagnose)
