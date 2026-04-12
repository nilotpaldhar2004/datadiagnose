"""
examples/student_dataset_demo.py
=================================
A complete, realistic data science workflow using DataDiagnose.

This is the most detailed example. It walks through every step
a real data scientist would take from raw data to a clean,
model-ready dataset — using DataDiagnose to guide every decision.

The story:
    You are a student at a university. You have been given a dataset
    of student exam results and asked to build a model that predicts
    whether a student will pass or fail. Before you can build the
    model, you need to understand and clean the data.
    DataDiagnose is your tool for that first step.

What this script demonstrates
-------------------------------
    1.  Loading the raw dataset
    2.  First diagnosis — see everything that is wrong
    3.  Understanding each issue one by one
    4.  Fixing missing values (the right way, not the lazy way)
    5.  Handling the outlier in age
    6.  Removing the constant column
    7.  Removing the leaky column
    8.  Addressing class imbalance (explanation only)
    9.  Second diagnosis — verify improvements
    10. Exporting the clean dataset
    11. A comparison table — before vs after

How to run
----------
    python examples/student_dataset_demo.py

Author  : Nilotpal Dhar
License : MIT
"""

import sys
import os
import math
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datadiagnose import diagnose, health_score, list_issues, get_suggestions


# ─────────────────────────────────────────────────────────────
# UTILITY HELPERS
# Pure Python versions of common pandas operations.
# We write them ourselves so this demo works with zero dependencies.
# ─────────────────────────────────────────────────────────────

def col_median(values):
    """Return the median of a list, ignoring None/NaN."""
    nums = [float(v) for v in values
            if v is not None and v != '' and not (isinstance(v, float) and math.isnan(v))]
    if not nums:
        return None
    s   = sorted(nums)
    n   = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def col_mode(values):
    """Return the most common non-null value in a list."""
    nn = [v for v in values if v is not None and v != '']
    if not nn:
        return None
    return Counter(str(v) for v in nn).most_common(1)[0][0]


def cap_outliers(values, lower_pct=1, upper_pct=99):
    """Cap outliers at the given percentile boundaries."""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return values
    s     = sorted(nums)
    n     = len(s)
    lo_i  = max(0, int(n * lower_pct / 100) - 1)
    hi_i  = min(n - 1, int(n * upper_pct / 100))
    lo, hi = s[lo_i], s[hi_i]
    return [
        None if v is None else max(lo, min(hi, float(v)))
        for v in values
    ]


def section(title, subtitle=''):
    """Print a formatted section header."""
    print("\n" + "═" * 64)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("═" * 64)


def subsection(title):
    print(f"\n  ── {title} ──")


def bullet(text, indent=4):
    print(f"{' ' * indent}•  {text}")


# ─────────────────────────────────────────────────────────────
# STEP 1 — THE RAW DATASET
# ─────────────────────────────────────────────────────────────

section(
    "STEP 1 — The Raw Dataset",
    "This is the data we were given. It has not been touched yet."
)

# This is a realistic student performance dataset.
# It has been deliberately crafted to contain every common
# type of data problem so we can demonstrate fixing them all.

raw_dataset = {

    # Student ID — high cardinality, useless as a feature
    "student_id": [
        "STU001","STU002","STU003","STU004","STU005",
        "STU006","STU007","STU008","STU009","STU010",
        "STU011","STU012","STU013","STU014","STU015",
        "STU016","STU017","STU018","STU019","STU020",
    ],

    # Age — has 3 missing values and one huge outlier (900)
    "age": [
        18,   19,   None, 21,   900,
        20,   19,   18,   None, 22,
        20,   21,   18,   19,   None,
        20,   22,   19,   21,   18,
    ],

    # Study hours per day — has 2 missing values
    "study_hours": [
        2.5,  5.0,  3.0,  None, 4.5,
        6.0,  3.5,  2.0,  4.0,  None,
        5.5,  3.0,  2.5,  4.5,  3.5,
        5.0,  4.0,  3.0,  2.5,  4.0,
    ],

    # Attendance percentage — has 1 missing value
    "attendance_pct": [
        85,   90,   78,   92,   88,
        None, 75,   95,   82,   87,
        91,   79,   84,   93,   77,
        89,   86,   94,   83,   88,
    ],

    # Previous exam score (0–100) — complete, slight right skew
    "prev_score": [
        55,   72,   61,   78,   80,
        90,   58,   66,   74,   82,
        70,   63,   59,   85,   68,
        77,   73,   88,   62,   75,
    ],

    # City — very imbalanced (Kolkata appears 15 out of 20 times)
    "city": [
        "Kolkata","Kolkata","Kolkata","Delhi",  "Kolkata",
        "Kolkata","Mumbai", "Kolkata","Kolkata","Kolkata",
        "Delhi",  "Kolkata","Kolkata","Kolkata","Mumbai",
        "Kolkata","Kolkata","Kolkata","Delhi",  "Kolkata",
    ],

    # Country — constant column, every row is "India"
    "country": ["India"] * 20,

    # Result flag — same values as 'passed', leaky column name
    "result_flag": [
        0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
        1, 1, 0, 1, 0, 1, 1, 1, 1, 0,
    ],

    # Target — what we want to predict (pass = 1, fail = 0)
    # Slightly imbalanced: 14 passes vs 6 fails
    "passed": [
        0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
        1, 1, 0, 1, 0, 1, 1, 1, 1, 0,
    ],
}

print(f"\n  Dataset has {len(raw_dataset['student_id'])} rows and "
      f"{len(raw_dataset)} columns.")
print(f"\n  Columns: {list(raw_dataset.keys())}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — FIRST DIAGNOSIS
# ─────────────────────────────────────────────────────────────

section(
    "STEP 2 — First Diagnosis",
    "Running DataDiagnose on the raw data to find all problems."
)

report_before = diagnose(
    raw_dataset,
    target_col="passed",
    dataset_name="Student Performance (Raw)"
)

print(report_before)

score_before = report_before.score
print(f"\n  Initial health score: {score_before}/100")


# ─────────────────────────────────────────────────────────────
# STEP 3 — UNDERSTANDING EACH ISSUE
# ─────────────────────────────────────────────────────────────

section(
    "STEP 3 — Understanding Each Issue",
    "Going through every detected problem and explaining what it means."
)

print(f"\n  Total issues found: {len(report_before.issues)}\n")

for i, issue in enumerate(report_before.issues, 1):
    emoji = {"critical":"🔴","high":"🟠","medium":"🟡","low":"⚪"}.get(
        issue.severity, "•")
    print(f"  {i}. {emoji} [{issue.severity.upper():8}] {issue.title}")
    print(f"     What it means: {issue.description}")
    if issue.fix:
        print(f"     How to fix it: {issue.fix}")
    print()


# ─────────────────────────────────────────────────────────────
# STEP 4 — FIX MISSING VALUES
# ─────────────────────────────────────────────────────────────

section(
    "STEP 4 — Fixing Missing Values",
    "Filling missing values with the right strategy for each column."
)

# Work on a copy so we can compare before and after
dataset = {col: list(vals) for col, vals in raw_dataset.items()}

# Count missing before
missing_before = {
    col: sum(1 for v in dataset[col] if v is None)
    for col in dataset
}

subsection("Missing values BEFORE fixing")
for col, count in missing_before.items():
    if count > 0:
        pct = count / len(dataset[col]) * 100
        print(f"    {col:<20} {count} missing  ({pct:.1f}%)")

# ── Fix age: fill with median ────────────────────────────────
age_median = col_median(dataset["age"])
print(f"\n  Filling 'age' missing values with median = {age_median}")
dataset["age"] = [
    age_median if v is None else v
    for v in dataset["age"]
]

# ── Fix study_hours: fill with median ───────────────────────
sh_median = col_median(dataset["study_hours"])
print(f"  Filling 'study_hours' missing values with median = {sh_median}")
dataset["study_hours"] = [
    sh_median if v is None else v
    for v in dataset["study_hours"]
]

# ── Fix attendance_pct: fill with median ────────────────────
att_median = col_median(dataset["attendance_pct"])
print(f"  Filling 'attendance_pct' missing values with median = {att_median}")
dataset["attendance_pct"] = [
    att_median if v is None else v
    for v in dataset["attendance_pct"]
]

# Count missing after
missing_after = {
    col: sum(1 for v in dataset[col] if v is None)
    for col in dataset
}

subsection("Missing values AFTER fixing")
total_missing_after = sum(missing_after.values())
if total_missing_after == 0:
    print("     All missing values have been filled.")
else:
    for col, count in missing_after.items():
        if count > 0:
            print(f"    {col}: {count} still missing")


# ─────────────────────────────────────────────────────────────
# STEP 5 — FIX THE OUTLIER IN AGE
# ─────────────────────────────────────────────────────────────

section(
    "STEP 5 — Fixing the Outlier in 'age'",
    "The value 900 is clearly a data entry error (should be 20 or similar)."
)

print("\n  Age values before capping:")
print(f"  {dataset['age']}")

print(f"\n  Minimum age: {min(dataset['age'])}")
print(f"  Maximum age: {max(dataset['age'])}")
print(f"\n  The value 900 is 30× higher than any realistic age.")
print(f"  We will cap all age values at the 99th percentile.\n")

dataset["age"] = cap_outliers(dataset["age"], lower_pct=1, upper_pct=99)

print(f"  Age values AFTER capping:")
print(f"  {[round(v, 1) if v else None for v in dataset['age']]}")
print(f"\n  Maximum age is now: {max(v for v in dataset['age'] if v):.1f}")


# ─────────────────────────────────────────────────────────────
# STEP 6 — REMOVE THE CONSTANT COLUMN
# ─────────────────────────────────────────────────────────────

section(
    "STEP 6 — Removing 'country' (Constant Column)",
    "Every row is 'India'. This column carries zero information."
)

print(f"\n  Unique values in 'country': {set(dataset['country'])}")
print(f"\n  Because every single row has the same value, this column")
print(f"  tells a machine learning model absolutely nothing.")
print(f"  It just wastes memory. Dropping it now.")

del dataset["country"]
print(f"\n   'country' column dropped.")
print(f"  Remaining columns: {list(dataset.keys())}")


# ─────────────────────────────────────────────────────────────
# STEP 7 — REMOVE THE LEAKY COLUMN
# ─────────────────────────────────────────────────────────────

section(
    "STEP 7 — Removing 'result_flag' (Data Leakage)",
    "This column has a suspicious name AND correlates perfectly with the target."
)

print(f"\n  Column 'result_flag' values: {dataset['result_flag']}")
print(f"  Column 'passed' values:      {dataset['passed']}")
print(f"\n  They are identical! This is data leakage.")
print(f"  'result_flag' was created AFTER we knew the exam result,")
print(f"  meaning it contains the answer. The model would simply")
print(f"  learn to read 'result_flag' and ignore everything else.")
print(f"  In production, this column would not exist yet.")

del dataset["result_flag"]
print(f"\n   'result_flag' column dropped.")
print(f"  Remaining columns: {list(dataset.keys())}")


# ─────────────────────────────────────────────────────────────
# STEP 8 — DROP THE ID COLUMN
# ─────────────────────────────────────────────────────────────

section(
    "STEP 8 — Dropping 'student_id' (High Cardinality)",
    "Every row has a unique ID. This column is useless as a feature."
)

print(f"\n  'student_id' has {len(set(dataset['student_id']))} unique values")
print(f"  out of {len(dataset['student_id'])} rows — 100% unique.")
print(f"\n  A model cannot learn any pattern from an ID column because")
print(f"  every value is different. It also takes up memory and can")
print(f"  accidentally introduce ordering bias.")

del dataset["student_id"]
print(f"\n    'student_id' column dropped.")
print(f"  Remaining columns: {list(dataset.keys())}")


# ─────────────────────────────────────────────────────────────
# STEP 9 — CLASS IMBALANCE NOTE
# ─────────────────────────────────────────────────────────────

section(
    "STEP 9 — Class Imbalance in 'passed'",
    "14 passes vs 6 fails. We acknowledge it but handle it at model training time."
)

pass_count = dataset["passed"].count(1)
fail_count = dataset["passed"].count(0)
total      = len(dataset["passed"])

print(f"\n  Passes (1): {pass_count}  ({pass_count/total*100:.1f}%)")
print(f"  Fails  (0): {fail_count}  ({fail_count/total*100:.1f}%)")
print(f"  Ratio: {pass_count/fail_count:.1f}:1")
print(f"""
  Why we do NOT try to fix this with DataDiagnose:
  DataDiagnose is a diagnostic tool. It tells us the imbalance
  exists. The actual fix (SMOTE, class_weight, etc.) happens
  at the model training stage, not the data cleaning stage.

  When training, we would use:
      model = RandomForestClassifier(class_weight='balanced')

  And we would evaluate using F1-score, not accuracy:
      from sklearn.metrics import f1_score
      print(f1_score(y_true, y_pred))
""")


# ─────────────────────────────────────────────────────────────
# STEP 10 — SECOND DIAGNOSIS
# ─────────────────────────────────────────────────────────────

section(
    "STEP 10 — Second Diagnosis",
    "Running DataDiagnose again on the cleaned dataset."
)

report_after = diagnose(
    dataset,
    target_col="passed",
    dataset_name="Student Performance (Cleaned)"
)

print(report_after)

score_after = report_after.score
print(f"\n  Score BEFORE cleaning: {score_before}/100")
print(f"  Score AFTER  cleaning: {score_after}/100")
print(f"  Improvement:          +{score_after - score_before} points  "
      f"{'✅' if score_after > score_before else '⚠️'}")


# ─────────────────────────────────────────────────────────────
# STEP 11 — FINAL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────

section(
    "STEP 11 — Before vs After Comparison",
    "A summary of every change made and its effect."
)

changes = [
    ("Missing values in 'age'",         "3 missing (15%)",   " Filled with median"),
    ("Missing values in 'study_hours'", "2 missing (10%)",   " Filled with median"),
    ("Missing in 'attendance_pct'",     "1 missing  (5%)",   " Filled with median"),
    ("Outlier in 'age'",                "Value 900 (3000%)", " Capped at 99th percentile"),
    ("Constant column 'country'",       "All 'India'",       " Dropped"),
    ("Leaky column 'result_flag'",      "Corr = 1.0 w/target"," Dropped"),
    ("High cardinality 'student_id'",   "100% unique IDs",   " Dropped"),
    ("Class imbalance in 'passed'",     "14:6 = 2.3:1",      "  Handle at training time"),
    ("City imbalance",                  "KOL 75% of rows",   "  Handle with encoding"),
]

col1 = max(len(c[0]) for c in changes) + 2
col2 = max(len(c[1]) for c in changes) + 2
col3 = max(len(c[2]) for c in changes) + 2

header = f"  {'Issue':<{col1}} {'Before':<{col2}} {'Action Taken':<{col3}}"
print(f"\n{header}")
print(f"  {'─' * (col1 + col2 + col3 + 4)}")
for issue, before, action in changes:
    print(f"  {issue:<{col1}} {before:<{col2}} {action:<{col3}}")

print(f"\n  {'─' * (col1 + col2 + col3 + 4)}")
print(f"\n  Health score:  {score_before}/100  →  {score_after}/100  "
      f"(+{score_after - score_before} improvement)")
print(f"\n  Remaining columns: {list(dataset.keys())}")
print(f"  Remaining rows:    {len(dataset['passed'])} (none dropped)")


# ─────────────────────────────────────────────────────────────
# STEP 12 — WHAT TO DO NEXT
# ─────────────────────────────────────────────────────────────

section(
    "STEP 12 — What To Do Next",
    "Your data is now clean. Here are the next steps."
)

print(f"""
  Your dataset health score is now {score_after}/100.
  The critical and high issues have been resolved.
  Here is what to do next:

  1. ENCODE CATEGORICAL COLUMNS
     The 'city' column still contains text.
     Before feeding it to a model, convert it to numbers:
         from sklearn.preprocessing import LabelEncoder
         le = LabelEncoder()
         dataset['city_encoded'] = le.fit_transform(dataset['city'])

  2. SPLIT INTO FEATURES AND TARGET
         X = [all columns except 'passed']
         y = dataset['passed']

  3. TRAIN A BASELINE MODEL
     Start with the simplest model first:
         from sklearn.linear_model import LogisticRegression
         model = LogisticRegression(class_weight='balanced')
         model.fit(X_train, y_train)

  4. EVALUATE WITH THE RIGHT METRIC
     Because of class imbalance, use F1-score not accuracy:
         from sklearn.metrics import f1_score
         print(f1_score(y_test, model.predict(X_test)))

  5. ITERATE
     If the baseline model is not good enough, try
     RandomForestClassifier or XGBoostClassifier.
     DataDiagnose already recommended these for you.
""")

print("=" * 64)
print("  End of student_dataset_demo.py")
print(f"  Final dataset: {len(dataset)} columns, "
      f"{len(dataset['passed'])} rows, score {score_after}/100")
print("=" * 64 + "\n")

