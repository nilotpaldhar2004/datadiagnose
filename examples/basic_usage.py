"""
examples/basic_usage.py
=======================
The simplest possible introduction to DataDiagnose.

This script shows every public function in the library,
one by one, with plain English explanations before each one.
Run this file first if you have never used DataDiagnose before.

How to run
----------
    # From the root of the project
    python examples/basic_usage.py

Author  : Nilotpal Dhar
License : MIT
"""

import sys
import os

# ── Allow running from the examples/ folder directly ─────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datadiagnose import (
    diagnose,
    quick_scan,
    health_score,
    list_issues,
    get_suggestions,
    column_summary,
    get_stats_df,
)


# ─────────────────────────────────────────────────────────────
# THE DATASET
# ─────────────────────────────────────────────────────────────
# This is a made-up student dataset with several
# deliberately planted problems:
#
#   age       → has a missing value (None) and one
#               huge outlier (900 — a clear data entry error)
#   income    → has two missing values
#   city      → heavily imbalanced (KOL appears 7 out of 10 times)
#   result    → name contains "result" — suspicious for leakage
#   passed    → the target column (what we want to predict)
#   const_col → every single row is 0 — useless constant column
#
# ─────────────────────────────────────────────────────────────

dataset = {
    "student_id": [1,    2,    3,    4,    5,    6,    7,    8,    9,    10],
    "age":        [18,   19,   None, 21,   900,  20,   19,   18,   22,   20 ],
    "income":     [30000,45000,None, 48000,52000,61000,None, 34000,47000,55000],
    "city":       ["KOL","MUM","KOL","DEL","KOL","KOL","KOL","KOL","KOL","MUM"],
    "score":      [45,   78,   62,   55,   80,   90,   50,   42,   75,   68 ],
    "result":     [0,    1,    1,    1,    1,    1,    0,    0,    1,    1  ],
    "passed":     [0,    1,    1,    1,    1,    1,    0,    0,    1,    1  ],
    "const_col":  [0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],
}

TARGET = "passed"

print("=" * 60)
print("  DataDiagnose — Basic Usage Example")
print("  Author: Nilotpal Dhar")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# EXAMPLE 1 — diagnose()
# The main function. Returns a full DiagnosisReport object.
# Pass your dataset and the name of the column you want to predict.
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("EXAMPLE 1 — diagnose()  (full report)")
print("─" * 60)

report = diagnose(dataset, target_col=TARGET, dataset_name="Student Dataset")

# Just printing the report object gives you the full summary
print(report)


# ─────────────────────────────────────────────────────────────
# EXAMPLE 2 — quick_scan()
# The one-liner shortcut — it runs diagnose() and prints
# the report immediately. Useful when you just want to
# see results fast without storing the report object.
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("EXAMPLE 2 — quick_scan()  (diagnose + print in one call)")
print("─" * 60)

# Identical to: report = diagnose(...); print(report)
report2 = quick_scan(dataset, target_col=TARGET)

# The report object is still returned if you need it
print(f"\nquick_scan() also returned a report: {repr(report2)}")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 3 — health_score()
# Returns just the integer score (0–100).
# Perfect for a quick sanity check or for using in an
# if-statement before model training.
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("EXAMPLE 3 — health_score()  (just the number)")
print("─" * 60)

score = health_score(dataset, target_col=TARGET)
print(f"\nDataset health score: {score}/100")

# A common pattern — block model training if data is too dirty
if score < 70:
    print(f" Score is {score}/100 — fix the issues before training!")
else:
    print(f" Score is {score}/100 — data is good enough to train.")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 4 — list_issues()
# Returns a concise list of (severity, title) tuples.
# Great for quickly scanning what problems exist without
# reading the full report narrative.
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("EXAMPLE 4 — list_issues()  (just the issue titles)")
print("─" * 60)

issues = list_issues(dataset, target_col=TARGET)

print(f"\nFound {len(issues)} issues:\n")
for severity, title in issues:
    # Pick an emoji based on severity level
    emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "⚪"}.get(severity, "•")
    print(f"  {emoji}  [{severity:8}]  {title}")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 5 — get_suggestions()
# Returns only the list of fix suggestions as plain strings.
# Each suggestion is a specific, actionable recommendation.
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("EXAMPLE 5 — get_suggestions()  (just the fix list)")
print("─" * 60)

suggestions = get_suggestions(dataset, target_col=TARGET)

print(f"\n{len(suggestions)} fix suggestions:\n")
for i, suggestion in enumerate(suggestions, 1):
    print(f"  {i:2}. {suggestion}")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 6 — column_summary()
# Returns the ColumnReport for a single column — all the
# statistics DataDiagnose computed for that one column.
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("EXAMPLE 6 — column_summary()  (deep-dive into one column)")
print("─" * 60)

for col_name in ["age", "city", "const_col"]:
    rep = column_summary(dataset, col_name, target_col=TARGET)
    if rep:
        print(f"\n  Column: '{col_name}'")
        for key, value in rep.details.items():
            print(f"    {key:<20} {value}")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 7 — Inspecting the DiagnosisReport object
# The report object has many useful attributes you can
# access directly in your own code.
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("EXAMPLE 7 — DiagnosisReport attributes")
print("─" * 60)

print(f"""
  report.score          = {report.score}         (health score 0-100)
  report.n_rows         = {report.n_rows}        (total rows)
  report.n_cols         = {report.n_cols}         (total columns)
  report.issues         = {len(report.issues)} items  (Issue objects)
  report.suggestions    = {len(report.suggestions)} items  (fix strings)
  report.model_types    = {len(report.model_types)} items  (model name strings)
  report.column_reports = {len(report.column_reports)} keys   (ColumnReport per column)
""")

# Loop over Issue objects directly
print("  Issue objects:")
for issue in report.issues:
    print(f"    [{issue.severity.upper():8}]  {issue.title}")
    if issue.fix:
        print(f"               Fix: {issue.fix[:70]}...")

# Loop over recommended models
print(f"\n  Recommended models:")
for model in report.model_types:
    print(f"    •  {model}")


# ─────────────────────────────────────────────────────────────
# EXAMPLE 8 — Using as a quality gate
# A common real-world pattern — use health_score() to decide
# whether it is safe to proceed with model training.
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("EXAMPLE 8 — Quality gate pattern")
print("─" * 60)

def check_before_training(data, target, min_score=70):
    """
    Run DataDiagnose and block training if quality is too low.
    Returns True if safe to proceed, False otherwise.
    """
    score  = health_score(data, target_col=target)
    issues = list_issues(data, target_col=target)

    print(f"\n  Dataset health score: {score}/100")

    if score < min_score:
        print(f"  Score {score} is below threshold {min_score}.")
        print(f"  Fix these issues first:")
        critical_and_high = [(s, t) for s, t in issues
                             if s in ("CRITICAL", "HIGH")]
        for sev, title in critical_and_high:
            print(f"    [{sev}] {title}")
        return False

    print(f" Score {score} meets threshold {min_score}. Safe to train!")
    return True


safe = check_before_training(dataset, target=TARGET, min_score=70)
print(f"\n  Training allowed: {safe}")

print("\n" + "─" * 60)
print("EXAMPLE 9 — get_stats_df()  (all stats as a DataFrame)")
print("─" * 60)

# ─────────────────────────────────────────────────────────────
# EXAMPLE 9 — get_stats_df()
# New in v1.0.1: runs a full diagnosis and returns all column
# statistics as a Pandas DataFrame — perfect for Jupyter notebooks.
# Requires pandas to be installed: pip install pandas
# ─────────────────────────────────────────────────────────────

print("""
  get_stats_df() runs diagnose() internally and returns
  all column statistics as a Pandas DataFrame.

  Each COLUMN in your dataset becomes a header.
  Each ROW is a statistic (type, mean, std, missing_pct ...).

  Requires pandas:  pip install pandas
""")

try:
    df_stats = get_stats_df(dataset, target_col=TARGET)
    print("  Column statistics as a DataFrame:\n")
    print(df_stats.to_string())
    print(f"\n  Shape: {df_stats.shape[0]} stats × {df_stats.shape[1]} columns")
    print("\n  You can also call report.to_df() if you already have a report:")
    print("  df_stats = report.to_df()")
except ImportError:
    print("  pandas is not installed — skipping get_stats_df() demo.")
    print("  Install it with:  pip install pandas")

print("\n" + "=" * 60)
print("  End of basic_usage.py")
print("  Next: try pandas_integration.py or student_dataset_demo.py")
print("=" * 60 + "\n")
