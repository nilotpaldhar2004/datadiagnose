"""
examples/pandas_integration.py
===============================
How to use DataDiagnose with pandas DataFrames.

In the real world most people load data with pandas.
This script shows every way to connect pandas and DataDiagnose
— from loading a CSV, to running the diagnosis, to fixing the
issues pandas-style, to checking the score improved.

No real CSV file is needed — we create a realistic DataFrame
from scratch using only Python's standard library so you can
run this without downloading anything.

How to run
----------
    python examples/pandas_integration.py

    # If you have pandas installed:
    pip install pandas
    python examples/pandas_integration.py

Author  : Nilotpal Dhar
License : MIT
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datadiagnose import diagnose, health_score, quick_scan, get_suggestions, get_stats_df

print("=" * 62)
print("  DataDiagnose + pandas Integration Example")
print("  Author: Nilotpal Dhar")
print("=" * 62)


# ─────────────────────────────────────────────────────────────
# CHECK IF PANDAS IS AVAILABLE
# DataDiagnose itself does not need pandas at all.
# But this example shows how to connect the two when you
# are already using pandas in your project.
# ─────────────────────────────────────────────────────────────

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    print("\n  pandas found — running full pandas integration demo.")
except ImportError:
    PANDAS_AVAILABLE = False
    print("\n  pandas not installed.")
    print("    Install it with: pip install pandas")
    print("    Running DataDiagnose with plain Python dicts instead.\n")


# ─────────────────────────────────────────────────────────────
# THE RAW DATA
# Simulates what you would get after pd.read_csv("sales_data.csv")
# A realistic e-commerce sales dataset with common problems.
# ─────────────────────────────────────────────────────────────

RAW_DATA = {
    "order_id":        [f"ORD{str(i).zfill(4)}" for i in range(1, 31)],
    "customer_age":    [
        25, 34, None, 28, 45, 30, None, 52, 27, 39,
        33, None, 41, 26, 55, 31, 29, None, 44, 38,
        27, 35, 30, None, 48, 22, 36, 43, 29, 31,
    ],
    "purchase_amount": [
        120.5,  340.0,  89.99,  None, 210.0,  450.0, 75.5,   None,  315.0, 190.0,
        None,   520.0,  140.0,  95.0, None,   280.0, 410.0,  165.0, None,  230.0,
        88.0,   None,   390.0,  175.0,None,   310.0, 135.0,  None,  220.0, 490.0,
    ],
    "category": [
        "Electronics","Clothing","Electronics","Books","Clothing",
        "Electronics","Books","Electronics","Clothing","Books",
        "Electronics","Clothing","Books","Electronics","Clothing",
        "Books","Electronics","Clothing","Books","Electronics",
        "Clothing","Books","Electronics","Clothing","Books",
        "Electronics","Clothing","Books","Electronics","Clothing",
    ],
    "city": [
        "Kolkata","Kolkata","Kolkata","Kolkata","Mumbai",
        "Kolkata","Kolkata","Kolkata","Kolkata","Kolkata",
        "Delhi","Kolkata","Kolkata","Kolkata","Mumbai",
        "Kolkata","Kolkata","Kolkata","Delhi","Kolkata",
        "Kolkata","Kolkata","Kolkata","Mumbai","Kolkata",
        "Kolkata","Delhi","Kolkata","Kolkata","Kolkata",
    ],
    "visit_count": [
        3, 8, 2, 5, 12, 1, 4, 7, 2, 6,
        3, 9, 1, 5, 11, 2, 8, 3, 6, 4,
        1, 7, 3, 5, 10, 2, 4, 8, 1, 6,
    ],
    "website_visits_total": [
        300, 800, 200, 500, 1200, 100, 400, 700, 200, 600,
        300, 9999, 100, 500, 1100, 200, 800, 300, 600, 400,
        100, 700, 300, 500, 1000, 200, 400, 800, 100, 600,
    ],
    "purchased": [
        1, 1, 0, 1, 1, 0, 0, 1, 0, 1,
        0, 1, 0, 1, 1, 0, 1, 0, 1, 1,
        0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
    ],
}


# ─────────────────────────────────────────────────────────────
# SECTION A — WITH PANDAS
# ─────────────────────────────────────────────────────────────

if PANDAS_AVAILABLE:

    print("\n" + "─" * 62)
    print("SECTION A — Load DataFrame and diagnose")
    print("─" * 62)

    # ── Step 1: Create a DataFrame (as if loaded from CSV) ───
    df = pd.DataFrame(RAW_DATA)

    print(f"\nDataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\nData types:\n{df.dtypes.to_string()}")

    # ── Step 2: Convert to DataDiagnose format ───────────────
    # .to_dict(orient="list")    → dict-of-lists  (preferred)
    # .to_dict(orient="records") → list-of-dicts  (also works)

    print("\n" + "─" * 62)
    print("SECTION B — Convert DataFrame → DataDiagnose")
    print("─" * 62)

    dataset_dict = df.to_dict(orient="list")
    print(f"\n  df.to_dict(orient='list') gives a dict with {len(dataset_dict)} keys.")
    print(f"  Keys: {list(dataset_dict.keys())}")

    # ── Step 3: Run the diagnosis ────────────────────────────
    print("\n" + "─" * 62)
    print("SECTION C — Run diagnosis on the DataFrame data")
    print("─" * 62)

    report = diagnose(dataset_dict, target_col="purchased",
                      dataset_name="E-Commerce Sales")
    print(f"\n  Health Score: {report.score}/100  ({report.status()})")
    print(f"  Issues found: {len(report.issues)}")
    print(f"  Suggestions:  {len(report.suggestions)}")
    print()
    print(report.summary())

    # ── Step 4: Fix the issues pandas-style ─────────────────
    print("\n" + "─" * 62)
    print("SECTION D — Fix issues using pandas")
    print("─" * 62)

    score_before = health_score(dataset_dict, target_col="purchased")
    print(f"\n  Score BEFORE cleaning: {score_before}/100")

    df_clean = df.copy()

    # Fix 1: Drop the order_id column (high cardinality — useless as feature)
    print("\n  Fix 1: Dropping high-cardinality 'order_id' column...")
    df_clean = df_clean.drop(columns=["order_id"])

    # Fix 2: Fill missing customer_age with median
    print("  Fix 2: Filling missing 'customer_age' with median...")
    df_clean["customer_age"] = df_clean["customer_age"].fillna(
        df_clean["customer_age"].median()
    )

    # Fix 3: Fill missing purchase_amount with median
    print("  Fix 3: Filling missing 'purchase_amount' with median...")
    df_clean["purchase_amount"] = df_clean["purchase_amount"].fillna(
        df_clean["purchase_amount"].median()
    )

    # Fix 4: Cap the outlier in website_visits_total
    print("  Fix 4: Capping outlier in 'website_visits_total'...")
    cap_value = df_clean["website_visits_total"].quantile(0.95)
    df_clean["website_visits_total"] = df_clean["website_visits_total"].clip(
        upper=cap_value
    )

    # ── Step 5: Re-diagnose the cleaned data ─────────────────
    print("\n  Re-running diagnosis on cleaned DataFrame...")
    dataset_clean = df_clean.to_dict(orient="list")
    score_after   = health_score(dataset_clean, target_col="purchased")

    print(f"\n  Score BEFORE cleaning: {score_before}/100")
    print(f"  Score AFTER  cleaning: {score_after}/100")
    print(f"  Improvement:          +{score_after - score_before} points")

    # ── Step 6: Compare missing values before and after ──────
    print("\n" + "─" * 62)
    print("SECTION E — Missing value comparison")
    print("─" * 62)

    print(f"\n  Missing values BEFORE:\n{df.isnull().sum().to_string()}")
    print(f"\n  Missing values AFTER:\n{df_clean.isnull().sum().to_string()}")

    # ── Step 7: Read suggestions and apply programmatically ──
    print("\n" + "─" * 62)
    print("SECTION F — Read suggestions programmatically")
    print("─" * 62)

    suggestions = get_suggestions(dataset_dict, target_col="purchased")
    print(f"\n  Suggestions from DataDiagnose:")
    for i, s in enumerate(suggestions, 1):
        print(f"    {i:2}. {s}")

    # ── Step 8: get_stats_df() — stats as a DataFrame ────────
    print("\n" + "─" * 62)
    print("SECTION G — get_stats_df()  (all column stats as DataFrame)")
    print("─" * 62)

    print("""
  get_stats_df() is a convenience function that runs the full
  diagnosis and returns all column statistics as a Pandas
  DataFrame in one call — perfect for Jupyter notebooks.

  Rows    = statistics  (type, mean, std, missing_pct ...)
  Columns = your dataset columns (customer_age, city ...)
    """)

    df_stats = get_stats_df(dataset_dict, target_col="purchased")
    print("  Stats DataFrame (transposed — metrics as rows):\n")
    print(df_stats.to_string())
    print(f"\n  Shape: {df_stats.shape[0]} metrics × {df_stats.shape[1]} columns")

    print("""
  You can also call .to_df() on an existing report object:
      report   = diagnose(dataset_dict, target_col="purchased")
      df_stats = report.to_df()
    """)


# ─────────────────────────────────────────────────────────────
# SECTION B — WITHOUT PANDAS (fallback)
# Same demonstration using only plain Python dicts.
# DataDiagnose works identically — no pandas needed.
# ─────────────────────────────────────────────────────────────

else:

    print("\n" + "─" * 62)
    print("Running DataDiagnose with plain Python dicts (no pandas)")
    print("─" * 62)

    print(f"\nDataset: {len(RAW_DATA['order_id'])} rows × {len(RAW_DATA)} columns")

    report = diagnose(RAW_DATA, target_col="purchased",
                      dataset_name="E-Commerce Sales")

    print(f"\nHealth Score : {report.score}/100  ({report.status()})")
    print(f"Issues       : {len(report.issues)}")
    print()

    print("Issues detected:")
    for issue in report.issues:
        print(f"  [{issue.severity.upper():8}] {issue.title}")
        print(f"             → {issue.description}")

    print("\nFix suggestions:")
    for i, s in enumerate(get_suggestions(RAW_DATA, "purchased"), 1):
        print(f"  {i:2}. {s}")


# ─────────────────────────────────────────────────────────────
# SECTION C — TIPS FOR PANDAS USERS
# ─────────────────────────────────────────────────────────────

print("\n" + "─" * 62)
print("TIPS FOR PANDAS USERS")
print("─" * 62)

print("""
  1. Always diagnose BEFORE any preprocessing.
     Run diagnose(df.to_dict(orient='list')) on your raw data
     BEFORE you start filling missing values or scaling features.
     Otherwise DataDiagnose might miss the original problems.

  2. Diagnose AGAIN after cleaning.
     Compare the scores before and after. If the score went up,
     your cleaning worked. If it didn't, you may have missed
     something or introduced a new problem.

  3. Use pandas for fixing, DataDiagnose for detecting.
     DataDiagnose tells you WHAT is wrong.
     pandas tells you HOW to fix it.
     They work together — DataDiagnose does not replace pandas.

  4. The two conversion methods:
     df.to_dict(orient='list')    → dict-of-lists  (fast, preferred)
     df.to_dict(orient='records') → list-of-dicts  (also works fine)

  5. After encoding categoricals, run diagnose again.
     One-hot encoding can introduce constant columns (all-zero
     columns for rare categories). DataDiagnose will catch those.
""")

print("=" * 62)
print("  End of pandas_integration.py")
print("=" * 62 + "\n")
