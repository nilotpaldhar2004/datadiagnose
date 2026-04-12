"""

detectors.py
============
All 8 individual detector functions.

Each detector follows the same contract:
    - Receives the raw column data, column name, a ColumnReport
      to write stats into, and the DiagnosisReport to write
      issues and suggestions into.
    - Returns nothing — it mutates the report objects directly.

The detectors are:
    1. detect_missing_values        — None / "" / NaN
    2. detect_outliers              — IQR + Z-score dual method
    3. detect_skewness              — Pearson moment coefficient
    4. detect_class_imbalance       — majority/minority ratio
    5. detect_data_leakage          — name heuristic + correlation
    6. detect_duplicate_rows        — exact row matching
    7. detect_constant_columns      — zero variance
    8. detect_high_cardinality      — too many unique text values

Author  : Nilotpal Dhar
License : MIT

"""

from .models import Issue
from .utils import (
    is_missing, to_numeric_list, non_null_values,
    skewness, iqr_bounds, zscore_outliers,
    pearson_correlation, matches_any_keyword,
    LEAKY_KEYWORDS, DATETIME_KEYWORDS, TEXT_KEYWORDS, GEO_KEYWORDS,
)
from collections import Counter


# 1. MISSING VALUE DETECTOR

def detect_missing_values(data, col_name, col_report, diagnosis):
    """
    Detect and report missing values in a column.

    A missing value is any of: None, empty string "", or float NaN.

    Severity is determined by the percentage of missing values:
        ≥ 60%  → CRITICAL  (column is mostly empty — drop it)
        ≥ 30%  → HIGH      (needs model-based imputation)
        ≥ 10%  → MEDIUM    (median/mode imputation appropriate)
        > 0%   → LOW       (simple fill is fine)
    """
    total = len(data)
    missing = sum(1 for v in data if is_missing(v))
    pct = round(missing / total * 100, 2) if total else 0.0

    col_report.add('missing_count', missing)
    col_report.add('missing_pct',   f'{pct:.1f}%')

    if missing == 0:
        return   # nothing to do

    if pct >= 60:
        severity = 'critical'
        desc = f'{pct:.1f}% of values are missing — column is mostly empty.'
        fix = f"Drop column '{col_name}': df.drop(columns=['{col_name}'])"
    elif pct >= 30:
        severity = 'high'
        desc = f'{pct:.1f}% of values are missing — significant gap.'
        fix = (f"Use IterativeImputer or KNNImputer from sklearn "
               f"for column '{col_name}'.")
    elif pct >= 10:
        severity = 'medium'
        desc = f'{pct:.1f}% of values are missing.'
        fix = (f"Fill '{col_name}' with median (numeric) "
               f"or mode (categorical).")
    else:
        severity = 'low'
        desc = f'{pct:.1f}% of values are missing — small gap, easy to fix.'
        fix = f"df['{col_name}'].fillna(df['{col_name}'].median())"

    diagnosis.add_issue(Issue(
        title=f"Missing Values in '{col_name}'",
        description=desc,
        severity=severity,
        column=col_name,
        fix=fix,
    ))
    diagnosis.add_suggestion(fix)


# 2. OUTLIER DETECTOR
def detect_outliers(data, col_name, col_report, diagnosis, is_target=False):
    """
    Detect outliers in a numeric column using IQR and Z-score methods.

    Method 1 — IQR (Interquartile Range):
        Any value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] is an outlier.
        This is Tukey's classic fence method (1977).

    Method 2 — Z-score:
        Any value with |Z| > 3.0 is an outlier.
        Z = (value - mean) / std

    Updated for v1.0.1:
    - Added is_target flag: If True, suggests model-based fixes (Log-Transform,
      Robust Loss) instead of data-cleaning fixes (Clipping, Dropping).
    """
    nums = to_numeric_list(data)

    # We need at least 10 values to calculate meaningful quartiles
    if nums is None or len(nums) < 10:
        return

    # ── Step A: Calculate Statistics ────────────────────────
    lower, upper, q1, q3, iqr = iqr_bounds(nums)
    if iqr == 0:
        # If there is no spread, IQR-based outlier detection is not applicable
        return
    iqr_out = [x for x in nums if x < lower or x > upper]
    iqr_pct = round(len(iqr_out) / len(nums) * 100, 2)

    # Z-score for secondary verification
    z_out = zscore_outliers(nums)

    # Log metrics to the Column Report
    col_report.add('outliers_iqr', f'{len(iqr_out)} ({iqr_pct:.1f}%)')
    col_report.add('outliers_zscore', len(z_out))
    col_report.add('iqr_fence', f'[{lower:.3f}, {upper:.3f}]')
    col_report.add('q1', f'{q1:.3f}')
    col_report.add('q3', f'{q3:.3f}')

    # Exit early if no outliers exist
    if len(iqr_out) == 0:
        return

    # ── Step B: Determine Severity and Fixes ────────────────

    # HIGH SEVERITY (>15% outliers)
    if iqr_pct >= 15:
        severity = 'high'
        if is_target:
            # Focus on model-level robustness for targets, not data-level clipping
            fix = (f"Target '{col_name}' has extreme outliers. Avoid clipping target labels; "
                   f"instead, use robust estimators like HuberRegressor or RANSACClassifier.")
        else:
            fix = (f"Apply RobustScaler to '{col_name}' or cap values with: "
                   f"df['{col_name}'].clip(lower={lower:.1f}, upper={upper:.1f})")

    # MEDIUM SEVERITY (5% - 15% outliers)
    elif iqr_pct >= 5:
        severity = 'medium'
        if is_target:
            fix = (f"Target '{col_name}' is unstable. Consider predicting "
                   f"log({col_name}) to reduce outlier impact.")
        else:
            fix = (f"Winsorise '{col_name}': "
                   f"df['{col_name}'].clip(df['{col_name}'].quantile(0.01), "
                   f"df['{col_name}'].quantile(0.99))")

    # LOW SEVERITY (<5% outliers)
    else:
        severity = 'low'
        fix = (f"Investigate {len(iqr_out)} potential outliers in '{col_name}' "
               f"manually to check for data entry errors.")

    # ── Step C: Add to Diagnosis Report ─────────────────────
    diagnosis.add_issue(Issue(
        title=f"Outliers Detected in '{col_name}'",
        description=(f"{len(iqr_out)} outliers ({iqr_pct:.1f}%) detected via IQR. "
                     f"Fence: [{lower:.3f}, {upper:.3f}]. "
                     f"Z-score method also found {len(z_out)} outliers."),
        severity=severity,
        column=col_name,
        fix=fix,
    ))
    diagnosis.add_suggestion(fix)


# 3. SKEWNESS DETECTOR

def detect_skewness(data, col_name, col_report, diagnosis, is_target=False):
    """
    Detect skewed distributions in numeric columns using Pearson's coefficient.

    A symmetric distribution has skewness near 0.
    - Positive skew (> 0): Long tail to the right (e.g., Income, Prices).
    - Negative skew (< 0): Long tail to the left (e.g., Age of retirement).

    Updated for v1.0.1:
    - Added is_target flag: Customizes transformation advice.
    - Features (X): Focus on simple transforms like sqrt/log1p.
    - Targets (Y): Focus on error normalization (Box-Cox/Yeo-Johnson).
    """
    nums = to_numeric_list(data)

    # Need sufficient data to calculate skewness reliably
    if nums is None or len(nums) < 10:
        return

    # ── Step A: Calculate Skewness ──────────────────────────
    skew_val = skewness(nums)
    abs_skew = abs(skew_val)
    direction = 'right-skewed (positive)' if skew_val > 0 else 'left-skewed (negative)'

    col_report.add('skewness', f'{skew_val:.4f}')

    # Exit if the distribution is roughly symmetric
    if abs_skew < 0.5:
        col_report.add('skew_status', 'symmetric')
        return

    # ── Step B: Logic for Severity and Fix ───────────────────

    # 1. MILD SKEW (0.5 to 1.0)
    if abs_skew < 1.0:
        severity = 'low'
        label = 'mildly skewed'
        if is_target:
            fix = (f"Target '{col_name}' is mildly skewed. Consider a square root "
                   f"transform to help normalize model errors.")
        else:
            fix = f"Apply np.sqrt(df['{col_name}']) to handle mild {direction} skew."

    # 2. HIGH SKEW (1.0 to 2.0)
    elif abs_skew < 2.0:
        severity = 'medium'
        label = 'highly skewed'
        if is_target:
            fix = (f"Highly skewed target '{col_name}'. Predicting log({col_name}) or "
                   f"using a TransformedTargetRegressor often improves R-squared.")
        else:
            fix = f"Apply np.log1p(df['{col_name}']) to pull in the long tail."

    # 3. EXTREME SKEW (> 2.0)
    else:
        severity = 'high'
        label = 'extremely skewed'
        if is_target:
            fix = (f"Extremely skewed target. Use a PowerTransformer "
                   f"(Box-Cox/Yeo-Johnson) on '{col_name}' to stabilize variance "
                   f"and make residuals normal.")
        else:
            fix = (f"Apply sklearn.preprocessing.PowerTransformer(method='yeo-johnson') "
                   f"to column '{col_name}'.")

    # ── Step C: Update Reports ──────────────────────────────
    col_report.add('skew_status', f'{label} ({direction})')

    diagnosis.add_issue(Issue(
        title=f"Skewed Distribution in '{col_name}'",
        description=(f"'{col_name}' is {label} (skewness = {skew_val:.4f}). "
                     f"This violates the normality assumption for linear models/NNs."),
        severity=severity,
        column=col_name,
        fix=fix,
    ))
    diagnosis.add_suggestion(fix)


# 4. CLASS IMBALANCE DETECTOR
def detect_class_imbalance(data, col_name, col_report, diagnosis,
                           is_target=False):
    """
    Detect class imbalance in categorical or binary columns.

    Computes the ratio of the majority class vs the minority class.

    Updated for v1.0.1:
    - Regression Guard: Automatically exits if it detects a numeric target
      with high cardinality (preventing "SMOTE" advice for price prediction).
    - Refined Thresholds: Focuses only on 2-20 unique classes.
    """
    nn = non_null_values(data)
    if not nn:
        return

    # ── Step A: Guard 1 — Regression target ─────────────────
    # If this is the target and it's numeric with many unique values,
    # it is a REGRESSION task. Class imbalance does not apply.
    nums = to_numeric_list(nn)
    unique_vals = set(str(v) for v in nn)

    if is_target and nums is not None and len(unique_vals) > 20:
        return

    # ── Step A: Guard 2 — Numeric FEATURE columns ───────────
    # This is the critical fix for real-world datasets.
    #
    # Columns like bedrooms=[1,2,3,4], stories=[1,2,3,4], parking=[0,1,2,3]
    # are discrete numeric features — NOT class labels. Flagging them as
    # "imbalanced" is a FALSE POSITIVE that confuses users and tanks the score.
    #
    # A column is a genuine categorical label (worth checking) only when:
    #   - It contains text values (not numbers), OR
    #   - It is the target column (explicitly passed as is_target=True), OR
    #   - It is a binary numeric column (0/1) — which IS a real class label
    #
    # So: if the column is numeric AND not the target AND has more than
    # 2 unique values → it is a discrete feature, skip imbalance check.
    if nums is not None and not is_target and len(unique_vals) > 2:
        # Store the stats for the report without raising an issue
        counts = Counter(str(v) for v in nn)
        n_class = len(counts)
        most_common_n = counts.most_common(1)[0][1]
        least_common_n = counts.most_common()[-1][1]
        ratio = round(most_common_n / max(least_common_n, 1), 2)
        majority_pct = round(most_common_n / len(nn) * 100, 1)
        minority_pct = round(least_common_n / len(nn) * 100, 1)
        col_report.add('n_classes',       n_class)
        col_report.add('majority_pct',    f'{majority_pct:.1f}%')
        col_report.add('minority_pct',    f'{minority_pct:.1f}%')
        col_report.add('imbalance_ratio', f'{ratio:.1f}:1')
        return   # record stats but do NOT raise an issue

    # ── Step B: Identify Classes ────────────────────────────
    counts = Counter(str(v) for v in nn)
    n_class = len(counts)

    # Only check columns with 2-20 classes
    # (1 class = constant, >20 = probably high-cardinality text/ID)
    if n_class < 2 or n_class > 20:
        return

    most_common_n = counts.most_common(1)[0][1]
    least_common_n = counts.most_common()[-1][1]

    # Avoid division by zero and calculate ratio
    ratio = round(most_common_n / max(least_common_n, 1), 2)
    majority_pct = round(most_common_n / len(nn) * 100, 1)
    minority_pct = round(least_common_n / len(nn) * 100, 1)

    # Log stats to ColumnReport
    col_report.add('n_classes', n_class)
    col_report.add('majority_pct', f'{majority_pct:.1f}%')
    col_report.add('minority_pct', f'{minority_pct:.1f}%')
    col_report.add('imbalance_ratio', f'{ratio:.1f}:1')

    # ── Step C: Severity and Fix Logic ──────────────────────
    label = 'target column' if is_target else f"column '{col_name}'"

    if ratio >= 10:
        severity = 'critical'
        fix = (f"Severe imbalance in {label}. Use SMOTE (imblearn) to oversample, "
               f"or use class_weight='balanced'. Evaluate using F1-Score or PRC, NOT accuracy.")
    elif ratio >= 5:
        severity = 'high'
        fix = ("Significant imbalance. Use StratifiedKFold cross-validation and "
               "consider RandomOverSampler or BalancedRandomForestClassifier.")
    elif ratio >= 3:
        severity = 'medium'
        fix = ("Moderate imbalance. Ensure your train-test split is stratified: "
               "train_test_split(..., stratify=y).")
    else:
        # Ratio < 3:1 is generally considered acceptable for most ML models
        return

    # ── Step D: Update Diagnosis ────────────────────────────
    diagnosis.add_issue(Issue(
        title=f"Class Imbalance in '{col_name}'",
        description=(f"In {label}: majority class is {majority_pct:.1f}%, "
                     f"minority class is {minority_pct:.1f}% (Ratio {ratio:.1f}:1)."),
        severity=severity,
        column=col_name,
        fix=fix,
    ))
    diagnosis.add_suggestion(fix)


# ──────────────────────────────────────────────────────────────
# 5. DATA LEAKAGE DETECTOR
# ──────────────────────────────────────────────────────────────

def detect_data_leakage(dataset, col_names, target_col, diagnosis):
    """
    Detect potential data leakage across the entire dataset.

    Updated for v1.0.1:
    - Pre-calculates target numeric list once for performance (O(n) improvement).
    - Statistical Confidence Guard: Correlation severity scales with sample
      size to prevent false positives on tiny datasets.
    """
    # ── Step A: Early Exits ─────────────────────────────────
    if not target_col or target_col not in col_names:
        return

    # PRE-CALCULATE: Convert target to numeric once outside the loop
    # This prevents converting the same target column 100+ times.
    target_raw = dataset[target_col]
    target_nums = to_numeric_list(target_raw)

    # If the target is categorical (e.g., 'Yes'/'No'), we need to
    # map it to 1/0 manually if to_numeric_list returns None.
    if target_nums is None:
        return  # Cannot perform Pearson correlation on non-numeric targets

    # ── Step B: Iterate through features ────────────────────
    for col in col_names:
        if col == target_col:
            continue

        # ── Strategy 1: Name heuristic ──────────────────────
        if matches_any_keyword(col, LEAKY_KEYWORDS):
            diagnosis.add_issue(Issue(
                title=f"Possible Data Leakage (name): '{col}'",
                description=(f"Column '{col}' matches known leaky keywords. "
                             f"It likely contains target-derived info."),
                severity='high',
                column=col,
                fix=(f"Investigate '{col}'. If it is a proxy for the target "
                     f"or captured after the event, drop it."),
            ))
            diagnosis.add_suggestion(
                f"Check '{col}' — name suggests it may be a target proxy.")

        # ── Strategy 2: Correlation check ───────────────────
        col_data = dataset[col]
        col_nums = to_numeric_list(col_data)

        if col_nums is not None:
            # Align rows and skip missing values
            # Using the pre-calculated target_raw here for alignment
            pairs = [
                (float(c), float(t))
                for c, t in zip(col_data, target_raw)
                if not (is_missing(c) or is_missing(t))
            ]

            # Need at least 5 pairs to calculate correlation
            if len(pairs) >= 5:
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                corr = pearson_correlation(xs, ys)

                if corr is not None and abs(corr) > 0.98:
                    # Logic: Only mark as Critical if we have enough samples (>15)
                    # Small samples (e.g. 6 rows) often have high random correlation.
                    severity = 'critical' if len(pairs) > 15 else 'high'

                    diagnosis.add_issue(Issue(
                        title=f"Extreme Correlation: '{col}' ↔ '{target_col}'",
                        description=(f"'{col}' has a correlation of {corr:.4f} with target. "
                                     f"Calculated from {len(pairs)} rows. This feature "
                                     f"likely contains 'future' information."),
                        severity=severity,
                        column=col,
                        fix=(f"Remove '{col}' immediately unless you are 100% "
                             f"certain this data is available during inference."),
                    ))
                    diagnosis.add_suggestion(
                        f"Drop '{col}' — correlation {corr:.4f} is too high for a safe feature.")


# 6. DUPLICATE ROW DETECTOR

def detect_duplicate_rows(dataset, col_names, diagnosis):
    """
    Detect exact duplicate rows in the dataset.

    Two rows are duplicates if ALL their values are identical.

    Why duplicates matter:
        The model sees the same record multiple times during training,
        effectively giving it more weight than it deserves. This
        biases the model toward patterns that appear in duplicate rows.

    Detection method:
        Convert each row to a string tuple.
        Put all tuples in a Python set (sets cannot have duplicates).
        Duplicates = total rows - unique rows.

    Parameters
    ----------
    dataset   : dict       — {col_name: [values]}
    col_names : list[str]  — all column names
    diagnosis : DiagnosisReport
    """
    n_rows = len(dataset[col_names[0]])

    # Convert each row to a tuple of string representations
    row_tuples = [
        tuple(str(dataset[col][i]) for col in col_names)
        for i in range(n_rows)
    ]

    n_unique = len(set(row_tuples))
    n_duplicate = n_rows - n_unique

    if n_duplicate == 0:
        return

    dup_pct = round(n_duplicate / n_rows * 100, 2)
    severity = 'high' if dup_pct > 5 else 'medium' if dup_pct > 1 else 'low'
    fix = f"df.drop_duplicates(inplace=True)  # removes {n_duplicate} duplicate rows"

    diagnosis.add_issue(Issue(
        title='Duplicate Rows Detected',
        description=(f"{n_duplicate} duplicate rows found "
                     f"({dup_pct:.1f}% of {n_rows} total rows)."),
        severity=severity,
        fix=fix,
    ))
    diagnosis.add_suggestion(fix)


# 7. CONSTANT COLUMN DETECTOR


def detect_constant_columns(dataset, col_names, diagnosis):
    """
    Detect columns where every non-null value is identical.

    A constant column carries zero information — every row has
    the same value, so the model cannot learn anything from it.
    It just wastes memory and computation time.

    Common causes:
        - Exporting from a database where a filter column has
          only one value (e.g. country = 'India' for all rows)
        - A data pipeline bug that filled an entire column with
          the same default value
        - A feature that was once variable but became fixed

    Parameters
    ----------
    dataset   : dict       — {col_name: [values]}
    col_names : list[str]  — all column names
    diagnosis : DiagnosisReport
    """
    for col in col_names:
        nn = non_null_values(dataset[col])
        if not nn:
            continue
        unique_vals = set(str(v) for v in nn)
        if len(unique_vals) == 1:
            constant_val = nn[0]
            fix = f"df.drop(columns=['{col}'], inplace=True)  # constant = {constant_val}"
            diagnosis.add_issue(Issue(
                title=f"Constant Column: '{col}'",
                description=(f"Every non-null value in '{col}' is '{constant_val}'. "
                             f"Zero variance — zero information for the model."),
                severity='medium',
                column=col,
                fix=fix,
            ))
            diagnosis.add_suggestion(fix)


# 8. HIGH CARDINALITY DETECTOR
def detect_high_cardinality(dataset, col_names, diagnosis,
                            unique_ratio_threshold=0.9):
    """
    Detect text columns where almost every value is unique.

    High cardinality columns (like user IDs, email addresses,
    or full names) are problematic because:

        1. The model cannot generalise from them — if every row
           has a unique value, there is no pattern to learn.
        2. One-hot encoding them creates an explosion of columns
           (one per unique value), causing memory issues.
        3. They can act as a proxy for row identity, which may
           leak ordering information into the model.

    Detection:
        If (unique non-null values / total non-null values) ≥ 0.9
        AND the column is not numeric → flag as high cardinality.

    We skip numeric columns because a column like 'price' can
    legitimately have many unique values without being an ID column.

    Parameters
    ----------
    dataset                : dict
    col_names              : list[str]
    diagnosis              : DiagnosisReport
    unique_ratio_threshold : float — default 0.9 (90%)
    """
    for col in col_names:
        data = dataset[col]

        # Only check non-numeric columns
        nums = to_numeric_list(data)
        if nums is not None:
            continue   # numeric column — skip

        nn = non_null_values(data)
        if len(nn) < 10:
            continue   # too few values to judge

        unique = set(str(v) for v in nn)
        unique_ratio = len(unique) / len(nn)

        if unique_ratio >= unique_ratio_threshold:
            fix = (f"Drop '{col}' if it is an ID column, or apply "
                   f"TargetEncoder / frequency encoding instead of one-hot encoding.")
            diagnosis.add_issue(Issue(
                title=f"High Cardinality Column: '{col}'",
                description=(f"'{col}' has {len(unique)} unique values out of "
                             f"{len(nn)} non-null rows ({unique_ratio*100:.1f}% unique). "
                             f"Likely an ID or free-text column."),
                severity='medium',
                column=col,
                fix=fix,
            ))
            diagnosis.add_suggestion(fix)


# FEATURE ENGINEERING HINTS
def suggest_feature_engineering(col_names, diagnosis):
    """
    Scan column names for patterns that suggest feature engineering
    opportunities. This does not detect problems — it proactively
    suggests ways to extract more information from existing columns.

    Three patterns detected:
        DateTime columns  → extract year, month, day, weekday, hour
        Text columns      → apply TF-IDF, word count, sentiment score
        Geo columns       → compute distance, cluster regions

    Parameters
    ----------
    col_names : list[str]  — all column names
    diagnosis : DiagnosisReport
    """
    for col in col_names:
        if matches_any_keyword(col, DATETIME_KEYWORDS):
            diagnosis.add_suggestion(
                f"Feature Eng: Extract year/month/day/weekday/hour from "
                f"datetime column '{col}' using pd.to_datetime()."
            )
        elif matches_any_keyword(col, TEXT_KEYWORDS):
            diagnosis.add_suggestion(
                f"Feature Eng: Apply TF-IDF, word count, or sentiment "
                f"analysis to free-text column '{col}'."
            )
        elif matches_any_keyword(col, GEO_KEYWORDS):
            diagnosis.add_suggestion(
                f"Feature Eng: Compute distance to reference point or "
                f"cluster regions from geographic column '{col}'."
            )


# MODEL RECOMMENDATION ENGINE
def suggest_models(dataset, col_names, target_col, diagnosis):
    """
    Recommend appropriate ML model types based on dataset characteristics.

    Decision logic for Task Type:
        1. No target column      → Unsupervised (Clustering/DR)
        2. Numeric target (>20 unique) → Regression
        3. 2 unique values       → Binary Classification
        4. 3-20 unique values    → Multiclass Classification

    Decision logic for Model Choice:
        - Dataset size (<500, <5000, >5000) determines complexity.
        - Dimensionality (>50 cols) triggers feature selection advice.
    """
    n_rows = len(dataset[col_names[0]])
    n_cols = len(col_names)

    # ── Step A: No target → Unsupervised Learning ────────────
    if not target_col or target_col not in col_names:
        diagnosis.model_types += [
            'K-Means Clustering (sklearn.cluster.KMeans)',
            'DBSCAN — density-based clustering for noisy data',
            'PCA — dimensionality reduction / visualization'
        ]
        diagnosis.add_suggestion(
            "No target specified: running in unsupervised mode.")
        return

    # ── Step B: Determine Task Type (Classification vs Regression) ──
    target_data = dataset[target_col]
    target_nn = non_null_values(target_data)
    target_unique = set(str(v) for v in target_nn)
    n_unique = len(target_unique)

    # Strictly check if the target is numeric for regression tasks
    target_nums = to_numeric_list(target_nn)
    is_numeric = target_nums is not None

    if is_numeric and n_unique > 20:
        task = 'regression'
    elif n_unique == 2:
        task = 'binary_classification'
    elif 2 < n_unique <= 20:
        task = 'multiclass_classification'
    else:
        task = 'unknown'

    # ── Step C: Model Recommendations ────────────────────────
    if task == 'regression':
        diagnosis.model_types += [
            'LinearRegression / Ridge — baseline (start here)',
            'RandomForestRegressor — handles non-linear patterns well',
            'XGBoost / LightGBM — usually provides best accuracy'
        ]
    elif task == 'binary_classification':
        diagnosis.model_types += [
            'LogisticRegression — baseline (fast and interpretable)',
            'RandomForestClassifier — robust against outliers',
            'LGBMClassifier / XGBClassifier — top-tier performance'
        ]
    elif task == 'multiclass_classification':
        diagnosis.model_types += [
            'LogisticRegression(multi_class="multinomial")',
            'RandomForestClassifier — handles multiple classes natively',
            'GradientBoostingClassifier — high performance'
        ]
    else:
        diagnosis.model_types.append(
            f"Task ambiguous: Target has {n_unique} unique values. "
            f"Check if it is an ID or clean it."
        )

    # ── Step D: Size and Dimensionality Advice ───────────────

    # 1. Dataset Size
    if n_rows < 500:
        diagnosis.add_suggestion(
            f"Small sample ({n_rows} rows): Use StratifiedKFold cross-validation "
            f"to ensure the model generalizes."
        )
    elif n_rows > 5000:
        diagnosis.add_suggestion(
            f"Large sample ({n_rows} rows): Gradient Boosting (LightGBM/CatBoost) "
            f"will significantly outperform simpler models here.")

    # 2. Dimensionality (Wide datasets)
    if n_cols > 50:
        diagnosis.add_suggestion(
            f"High feature count ({n_cols}): Use Lasso (L1) regression "
            f"or 'SelectFromModel' to reduce noise before training.")

    # 3. Imbalance check helper (if target is binary)
    if task == 'binary_classification' and n_rows > 0:
        # Check if the suggest_models should trigger an evaluation metric warning
        diagnosis.add_suggestion(
            "For classification: evaluate using ROC-AUC or F1-Score instead of Accuracy.")
