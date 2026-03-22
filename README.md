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

Eight problems are detected automatically:

| Problem | What It Means |
|---|---|
|  Missing Values | Null, empty, or NaN values in any column |
|  Outliers | Extreme values detected by IQR and Z-score methods |
|  Skewness | Lopsided distributions that hurt linear models |
|  Class Imbalance | One class vastly outnumbering others in your target |
|  Data Leakage | Columns that secretly contain the answer |
|  Duplicate Rows | Identical rows that bias your model |
|  Constant Columns | Columns with zero variation — zero information |
|  High Cardinality | ID-like columns with almost all unique values |

---

## Installation

DataDiagnose has **zero external dependencies**. It runs on pure Python standard library. No pandas, no numpy, no scikit-learn required.

```bash
# Once published on PyPI
pip install datadiagnose
```

For now, copy `datadiagnose/` into your project folder and import directly.

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
  Score   : 84/100     Needs Work
--------------------------------------------------------------
    Issues Found (2)

  1. 🟡 MEDIUM
     Missing Values in 'age'
     → 16.7% of values are missing.
      Fix: Fill 'age' with median (numeric) or mode (categorical).

  2. 🟡 MEDIUM
     Missing Values in 'income'
     → 16.7% of values are missing.
      Fix: Fill 'income' with median (numeric) or mode (categorical).
...
```

---

## Works With Pandas Too

DataDiagnose is not a pandas replacement — it works alongside it. Convert your DataFrame in one line:

```python
import pandas as pd
from datadiagnose import diagnose

df     = pd.read_csv("my_data.csv")
report = diagnose(df.to_dict(orient="list"), target_col="target")

print(f"Health score: {report.score}/100")
```

---

## Full API

### `diagnose(dataset, target_col=None, dataset_name="dataset")`

The main function. Runs all eight detectors and returns a `DiagnosisReport`.

```python
report = diagnose(dataset, target_col="label", dataset_name="Titanic")

report.score          # int — health score 0-100
report.issues         # list of Issue objects
report.suggestions    # list of fix strings
report.model_types    # list of recommended model names
report.column_reports # dict of per-column statistics
```

### `quick_scan(dataset, target_col=None)`

One-liner that runs the diagnosis and immediately prints the report.

```python
quick_scan(dataset, target_col="label")
```

### `health_score(dataset, target_col=None)`

Returns only the integer score. Perfect for quality gates in automated pipelines.

```python
score = health_score(dataset, target_col="label")

if score < 70:
    raise ValueError(f"Data quality too low: {score}/100. Fix issues first.")
```

### `list_issues(dataset, target_col=None)`

Returns a concise list of `(severity, title)` tuples.

```python
for severity, title in list_issues(dataset, "label"):
    print(f"[{severity}] {title}")
```

### `get_suggestions(dataset, target_col=None)`

Returns only the actionable fix suggestions as strings.

```python
for tip in get_suggestions(dataset, "label"):
    print("-", tip)
```

### `column_summary(dataset, col_name, target_col=None)`

Deep-dives into one specific column.

```python
rep = column_summary(dataset, "age")
print(rep.details)
# {'type': 'numeric', 'mean': '27.4', 'std': '3.2', ...}
```

---

## Understanding the Health Score

Every dataset starts at 100. Each detected issue deducts points based on severity:

| Severity | Points Lost | Example |
|---|---|---|
| 🔴 CRITICAL | 25 | Data leakage, >60% missing values |
| 🟠 HIGH | 15 | >30% missing, severe class imbalance |
| 🟡 MEDIUM | 8 | Moderate skewness, some outliers |
| ⚪ LOW | 3 | A few minor outliers |

| Score | Status | What To Do |
|---|---|---|
| 80 – 100 |  Healthy | Data is ready for modelling |
| 50 – 79 |  Needs Work | Fix HIGH and CRITICAL issues first |
| 0 – 49 |  Critical | Do not train models yet |

---

## Project Structure

```
datadiagnose/
│
├── datadiagnose/          ← The Python package
│   ├── __init__.py        ← Public API
│   ├── core.py            ← Main diagnose() engine
│   ├── detectors.py       ← All 8 detector functions
│   ├── models.py          ← DiagnosisReport, Issue, ColumnReport classes
│   └── utils.py           ← Pure math helpers (no dependencies)
│
├── tests/
│   ├── sample_data.py     ← 11 sample datasets with known problems
│   ├── test_detectors.py  ← 60 unit tests for each detector
│   └── test_core.py       ← 80 integration tests for the full API
│
├── examples/
│   ├── basic_usage.py          ← Start here — every function shown
│   ├── pandas_integration.py   ← How to use with pandas DataFrames
│   └── student_dataset_demo.py ← Full workflow, step by step
│
├── docs/
│   └── DataDiagnose_Documentation.pdf
│
├── .github/workflows/tests.yml ← Auto-run tests on every push
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## Running the Tests

DataDiagnose has 140 tests covering every detector, every public function, and edge cases. Tests use only Python's built-in `unittest` — no pytest required (though pytest works too).

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run just the detector tests
python -m unittest tests.test_detectors -v

# Run just the core API tests
python -m unittest tests.test_core -v
```

All 140 tests should pass with output ending:

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

# A full realistic data cleaning workflow
python examples/student_dataset_demo.py
```

---

## Why Zero Dependencies?

DataDiagnose uses only Python's built-in `math`, `statistics`, and `collections` modules. This was a deliberate decision:

1. **Works everywhere** — any Python 3.7+ environment, no pip install needed beyond the library itself
2. **No version conflicts** — adding numpy or pandas as dependencies would create compatibility issues for people who already have specific versions installed
3. **Educational** — every algorithm (IQR, Pearson correlation, skewness) is implemented from scratch in readable Python, so you can read the code and learn exactly how it works
4. **Lightweight** — the entire library is five Python files totalling around 1000 lines

---

## Design Decisions

**Why does it only detect problems and not fix them automatically?**

Automatically fixing data without human judgment is dangerous. Filling missing values with the wrong strategy can make your model *worse*. Whether you should drop a column or impute it, and what value to impute with, depends on domain knowledge — your understanding of what the data means. DataDiagnose gives you the information and recommendation. The decision is yours.

**Why is it a dict-of-lists and not a DataFrame?**

Accepting a plain Python dict means the library works with no dependencies at all. If you have a pandas DataFrame, converting it takes one line: `df.to_dict(orient="list")`. Supporting DataFrames directly would require adding pandas as a dependency, which defeats the zero-dependency design.

---

## How to Contribute

Contributions are welcome. Here are some ideas from the roadmap:

- HTML report export (generate a self-contained HTML file with charts)
- Correlation matrix analysis (detect multicollinearity between features)
- Direct pandas DataFrame support without conversion
- Web dashboard (Flask/FastAPI endpoint to upload CSV and get diagnosis)

To contribute:

1. Fork the repository on GitHub
2. Create a branch: `git checkout -b feature/my-new-detector`
3. Write your code and tests
4. Make sure all 140 existing tests still pass
5. Open a pull request with a clear description

---

## Changelog

### v1.0.0 — Initial Release
- Eight detectors: missing values, outliers, skewness, class imbalance, data leakage, duplicate rows, constant columns, high cardinality
- Feature engineering hints based on column name patterns
- Model recommendation engine
- Health score system (0–100)
- Full public API: `diagnose`, `quick_scan`, `health_score`, `list_issues`, `get_suggestions`, `column_summary`
- 140 unit and integration tests
- Zero external dependencies

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for the full text.

In plain English: you can use this code for anything, including commercial projects, as long as you keep the copyright notice with my name in any copy you distribute.

Copyright (c) 2026 **Nilotpal Dhar**

---

## Author

**Nilotpal Dhar**

Built as a beginner Python project to learn how data science diagnostics work from first principles. Every algorithm in this library — IQR outlier detection, Pearson correlation, skewness calculation — is implemented from scratch in plain Python so that reading the code teaches you how the maths actually works.

If this library helped you, star the repository on GitHub. If you found a bug or have a feature idea, open an issue.
