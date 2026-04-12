"""

sample_data.py
==============
Ready-made sample datasets for testing DataDiagnose.

Every dataset here is deliberately crafted to contain specific,
known problems so the tests can assert exact expected behaviour.

Each dataset is a plain Python dict-of-lists — no pandas needed.

Datasets
--------
    CLEAN_DATASET            — no problems at all (score should be 100)
    MISSING_VALUES_DATASET   — controlled missing value percentages
    OUTLIER_DATASET          — known outlier values planted at known positions
    SKEWED_DATASET           — heavily right-skewed numeric column
    IMBALANCED_DATASET       — 90/10 class split on target column
    LEAKAGE_DATASET          — column that perfectly correlates with target
    DUPLICATE_DATASET        — exact duplicate rows planted
    CONSTANT_DATASET         — one column that never changes
    HIGH_CARDINALITY_DATASET — column with almost all unique text values
    MESSY_DATASET            — everything wrong at once (kitchen-sink test)
    LISTOFDICT_DATASET       — same as CLEAN but in list-of-dicts format

Author  : Nilotpal Dhar
License : MIT

"""


# ──────────────────────────────────────────────────────────────
# CLEAN DATASET
# No issues. All columns are complete, balanced, and well-behaved.
# diagnose() on this dataset should return score = 100
# and issues = [].
# ──────────────────────────────────────────────────────────────

CLEAN_DATASET = {
    "age": [
        22, 25, 28, 30, 24, 27, 23, 31, 26, 29,
        22, 25, 28, 30, 24, 27, 23, 31, 26, 29,
    ],
    "income": [
        30000, 45000, 52000, 61000, 38000,
        47000, 33000, 65000, 41000, 55000,
        30000, 45000, 52000, 61000, 38000,
        47000, 33000, 65000, 41000, 55000,
    ],
    "score": [
        55, 72, 68, 80, 61, 74, 59, 83, 66, 77,
        55, 72, 68, 80, 61, 74, 59, 83, 66, 77,
    ],
    "target": [
        0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
    ],
}

# ──────────────────────────────────────────────────────────────
# MISSING VALUES DATASET
# Three columns with controlled missing percentages:
#   col_low    →  2/20 =  10% missing  (LOW severity)
#   col_medium →  6/20 =  30% missing  (HIGH severity)
#   col_high   → 14/20 =  70% missing  (CRITICAL severity)
# ──────────────────────────────────────────────────────────────

MISSING_VALUES_DATASET = {
    "id": list(range(1, 21)),

    # 10% missing — LOW
    "col_low": [
        1.0, 2.0, None, 4.0, 5.0,
        6.0, 7.0, 8.0, None, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
    ],

    # 30% missing — HIGH
    "col_medium": [
        1.0, None, None, 4.0, None,
        6.0, None, None, 9.0, None,
        11.0, None, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
    ],

    # 70% missing — CRITICAL
    "col_high": [
        1.0, None, None, None, None,
        None, None, None, None, None,
        None, None, None, None, 15.0,
        None, None, None, 19.0, None,
    ],

    "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
               0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
}

# ──────────────────────────────────────────────────────────────
# OUTLIER DATASET
# col_with_outliers: 18 normal values + 2 extreme outliers (999)
# The IQR fence for normal values [10-30] is roughly [-20, 60].
# 999 is way outside that range → should be flagged.
# ──────────────────────────────────────────────────────────────

OUTLIER_DATASET = {
    "id": list(range(1, 21)),

    # Normal range ~10-30, two extreme outliers planted at index 0 and 19
    "col_with_outliers": [
        999,  # outlier
        15, 18, 22, 17, 25, 20, 13, 24, 19,
        16, 21, 23, 14, 26, 18, 20, 22, 15,
        999,  # outlier
    ],

    # Clean reference column
    "col_clean": [
        10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
        11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
    ],

    "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
               1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
}

# ──────────────────────────────────────────────────────────────
# SKEWED DATASET
# col_right_skewed: exponential-like distribution.
# Most values are small (1-10), a few are very large (500, 1000).
# This creates strong positive (right) skewness.
# ──────────────────────────────────────────────────────────────

SKEWED_DATASET = {
    "id": list(range(1, 21)),

    # Heavy right skew — long tail to the right
    "col_right_skewed": [
        1, 2, 1, 3, 2, 1, 4, 2, 1, 3,
        2, 1, 500, 2, 1, 3, 1, 1000, 2, 1,
    ],

    # Mild left skew for comparison
    "col_left_skewed": [
        99, 98, 97, 99, 100, 98, 97, 96, 99, 100,
        98, 97, 1, 99, 100, 98, 99, 97, 100, 99,
    ],

    "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
               1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
}

# ──────────────────────────────────────────────────────────────
# IMBALANCED DATASET
# target: 18 zeros vs 2 ones → 9:1 ratio → HIGH severity
# target_severe: 19 zeros vs 1 one → 19:1 ratio → CRITICAL
# ──────────────────────────────────────────────────────────────

IMBALANCED_DATASET = {
    "feature_a": [float(i) for i in range(1, 21)],
    "feature_b": [float(20 - i) for i in range(20)],

    # 9:1 ratio → HIGH severity
    "target": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ],

    # 19:1 ratio → CRITICAL severity
    "target_severe": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ],
}

# ──────────────────────────────────────────────────────────────
# DATA LEAKAGE DATASET
# leaked_col: identical to target → correlation = 1.0 → CRITICAL
# suspicious_name: column named "result_label" → HIGH (name heuristic)
# ──────────────────────────────────────────────────────────────

LEAKAGE_DATASET = {
    "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0,
                  6.0, 7.0, 8.0, 9.0, 10.0,
                  11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0],

    # This column is a copy of target → perfect correlation → leakage
    "leaked_col": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                   0, 1, 0, 1, 0, 1, 0, 1, 0, 1],

    # Suspicious name — will be caught by name heuristic
    "result_label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                     0, 1, 0, 1, 0, 1, 0, 1, 0, 1],

    # Legitimate feature — should NOT be flagged
    "legitimate_feature": [23, 45, 12, 67, 34, 89, 56, 11, 78, 43,
                           25, 47, 14, 69, 36, 91, 58, 13, 80, 45],

    "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
               0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
}

# ──────────────────────────────────────────────────────────────
# DUPLICATE ROWS DATASET
# Rows 0-9 are original. Rows 10-14 are exact copies of rows 0-4.
# So 5 out of 15 rows (33%) are duplicates → HIGH severity.
# ──────────────────────────────────────────────────────────────

_base_rows = {
    "age":    [18, 22, 25, 30, 28, 19, 24, 27, 21, 26],
    "income": [30000, 45000, 50000, 65000, 55000,
               32000, 47000, 60000, 40000, 53000],
    "target": [0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
}

DUPLICATE_DATASET = {
    # Original rows + 5 duplicates of first 5 rows appended
    "age":    _base_rows["age"]    + _base_rows["age"][:5],
    "income": _base_rows["income"] + _base_rows["income"][:5],
    "target": _base_rows["target"] + _base_rows["target"][:5],
}

# ──────────────────────────────────────────────────────────────
# CONSTANT COLUMN DATASET
# constant_int : always 42 (integer constant)
# constant_str : always "yes" (string constant)
# varying_col  : normal values (control — should NOT be flagged)
# ──────────────────────────────────────────────────────────────

CONSTANT_DATASET = {
    "id":           list(range(1, 16)),
    "constant_int": [42] * 15,
    "constant_str": ["yes"] * 15,
    "varying_col":  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                     110, 120, 130, 140, 150],
    "target":       [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
}

# ──────────────────────────────────────────────────────────────
# HIGH CARDINALITY DATASET
# email_col : unique email per row → 100% unique → should be flagged
# city_col  : only 3 unique values → should NOT be flagged
# ──────────────────────────────────────────────────────────────

HIGH_CARDINALITY_DATASET = {
    # Almost every value is unique — classic ID/email column
    "email_col": [
        "nilotpal@email.com", "user2@email.com", "user3@email.com",
        "user4@email.com",    "user5@email.com", "user6@email.com",
        "user7@email.com",    "user8@email.com", "user9@email.com",
        "user10@email.com",   "user11@email.com","user12@email.com",
        "user13@email.com",   "user14@email.com","user15@email.com",
    ],
    # Low cardinality — should NOT be flagged
    "city_col": [
        "Kolkata", "Mumbai", "Delhi",
        "Kolkata", "Mumbai", "Delhi",
        "Kolkata", "Mumbai", "Delhi",
        "Kolkata", "Mumbai", "Delhi",
        "Kolkata", "Mumbai", "Delhi",
    ],
    "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
}

# ──────────────────────────────────────────────────────────────
# MESSY DATASET
# The kitchen-sink test — every possible problem in one dataset.
# Used to test that diagnose() finds ALL issues simultaneously.
# ──────────────────────────────────────────────────────────────

MESSY_DATASET = {
    # High cardinality — likely ID column
    "user_id": [f"USR{str(i).zfill(4)}" for i in range(1, 21)],

    # 25% missing values — HIGH severity
    "age": [
        18, None, 22, 25, None, 28, 30, None, 24, 19,
        21, 26, None, 29, 23, 27, 20, None, 31, 25,
    ],

    # Heavy outlier (999) — HIGH severity
    "income": [
        30000, 35000, 40000, 45000, 50000,
        55000, 60000, 65000, 70000, 75000,
        999999, 35000, 40000, 45000, 50000,  # 999999 is the outlier
        55000, 60000, 65000, 70000, 75000,
    ],

    # Constant — zero information
    "country": ["India"] * 20,

    # Data leakage by name
    "outcome_flag": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                     0, 1, 0, 1, 0, 1, 0, 1, 0, 1],

    # Heavily right-skewed
    "website_visits": [
        1, 2, 1, 3, 1, 2, 1, 4, 1, 2,
        1, 3, 1, 2, 1000, 1, 2, 1, 3, 1,
    ],

    # Target with 8:1 imbalance — HIGH severity
    "purchased": [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    ],
}

# ──────────────────────────────────────────────────────────────
# LIST-OF-DICTS FORMAT DATASET
# Identical data to CLEAN_DATASET but in list-of-dicts format.
# Used to test that the input normaliser works correctly.
# ──────────────────────────────────────────────────────────────

LISTOFDICT_DATASET = [
    {"age": 22, "income": 30000, "score": 55, "target": 0},
    {"age": 25, "income": 45000, "score": 72, "target": 1},
    {"age": 28, "income": 52000, "score": 68, "target": 1},
    {"age": 30, "income": 61000, "score": 80, "target": 1},
    {"age": 24, "income": 38000, "score": 61, "target": 0},
    {"age": 27, "income": 47000, "score": 74, "target": 1},
    {"age": 23, "income": 33000, "score": 59, "target": 0},
    {"age": 31, "income": 65000, "score": 83, "target": 1},
    {"age": 26, "income": 41000, "score": 66, "target": 0},
    {"age": 29, "income": 55000, "score": 77, "target": 1},
    {"age": 22, "income": 30000, "score": 55, "target": 0},
    {"age": 25, "income": 45000, "score": 72, "target": 1},
    {"age": 28, "income": 52000, "score": 68, "target": 1},
    {"age": 30, "income": 61000, "score": 80, "target": 1},
    {"age": 24, "income": 38000, "score": 61, "target": 0},
    {"age": 27, "income": 47000, "score": 74, "target": 1},
    {"age": 23, "income": 33000, "score": 59, "target": 0},
    {"age": 31, "income": 65000, "score": 83, "target": 1},
    {"age": 26, "income": 41000, "score": 66, "target": 0},
    {"age": 29, "income": 55000, "score": 77, "target": 1},
]

# ──────────────────────────────────────────────────────────────
# CONVENIENCE — all datasets in one dict for looping in tests
# ──────────────────────────────────────────────────────────────

ALL_DATASETS = {
    "clean":            CLEAN_DATASET,
    "missing":          MISSING_VALUES_DATASET,
    "outlier":          OUTLIER_DATASET,
    "skewed":           SKEWED_DATASET,
    "imbalanced":       IMBALANCED_DATASET,
    "leakage":          LEAKAGE_DATASET,
    "duplicate":        DUPLICATE_DATASET,
    "constant":         CONSTANT_DATASET,
    "high_cardinality": HIGH_CARDINALITY_DATASET,
    "messy":            MESSY_DATASET,
}
