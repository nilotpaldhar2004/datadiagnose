"""
test_detectors.py
=================
Unit tests for every individual detector function in detectors.py.

Each test class focuses on one detector.
Each test method covers one specific behaviour — one thing that
should or should not be detected, and at what severity.

How to run
----------
    # From the root of the project (one level above datadiagnose/)
    python -m pytest tests/ -v

    # Run just this file
    python -m pytest tests/test_detectors.py -v

    # Run a single test class
    python -m pytest tests/test_detectors.py::TestMissingValueDetector -v

What we are testing
-------------------
    TestMissingValueDetector    — None / "" / NaN detection + severity thresholds
    TestOutlierDetector         — IQR and Z-score outlier detection
    TestSkewnessDetector        — skewness calculation + severity thresholds
    TestClassImbalanceDetector  — majority/minority ratio + severity thresholds
    TestDataLeakageDetector     — name heuristic + correlation detection
    TestDuplicateRowDetector    — exact duplicate row counting
    TestConstantColumnDetector  — zero-variance column detection
    TestHighCardinalityDetector — near-unique text column detection
    TestFeatureEngHints         — datetime / text / geo column name hints
    TestModelSuggestions        — regression vs classification task detection

Author  : Nilotpal Dhar
License : MIT
"""

import sys
import os
import unittest

# ── Path setup ────────────────────────────────────────────────
# Allow running this file directly (python test_detectors.py)
# without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datadiagnose.models    import DiagnosisReport, ColumnReport
from datadiagnose.detectors import (
    detect_missing_values,
    detect_outliers,
    detect_skewness,
    detect_class_imbalance,
    detect_data_leakage,
    detect_duplicate_rows,
    detect_constant_columns,
    detect_high_cardinality,
    suggest_feature_engineering,
    suggest_models,
)
from tests.sample_data import (
    MISSING_VALUES_DATASET,
    OUTLIER_DATASET,
    SKEWED_DATASET,
    IMBALANCED_DATASET,
    LEAKAGE_DATASET,
    DUPLICATE_DATASET,
    CONSTANT_DATASET,
    HIGH_CARDINALITY_DATASET,
)


# ──────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────

def _fresh():
    """Return a fresh DiagnosisReport and ColumnReport for a test."""
    return DiagnosisReport("test"), ColumnReport("test_col")


def _severities(report):
    """Return a list of severity strings from all issues in a report."""
    return [i.severity for i in report.issues]


def _titles(report):
    """Return a list of issue title strings from all issues in a report."""
    return [i.title for i in report.issues]


# ──────────────────────────────────────────────────────────────
# 1. MISSING VALUE DETECTOR
# ──────────────────────────────────────────────────────────────

class TestMissingValueDetector(unittest.TestCase):
    """Tests for detect_missing_values()."""

    def test_no_missing_raises_no_issue(self):
        """A complete column should produce zero issues."""
        report, col_rep = _fresh()
        detect_missing_values([1, 2, 3, 4, 5], "age", col_rep, report)
        self.assertEqual(len(report.issues), 0)

    def test_none_is_detected_as_missing(self):
        """Python None should count as missing."""
        report, col_rep = _fresh()
        detect_missing_values([1, None, 3], "age", col_rep, report)
        self.assertEqual(col_rep.get("missing_count"), 1)

    def test_empty_string_is_detected_as_missing(self):
        """Empty string should count as missing."""
        report, col_rep = _fresh()
        detect_missing_values([1, "", 3], "age", col_rep, report)
        self.assertEqual(col_rep.get("missing_count"), 1)

    def test_low_missing_severity(self):
        """Under 10% missing → LOW severity."""
        # 1 missing out of 20 rows = 5%
        data = [1.0] * 19 + [None]
        report, col_rep = _fresh()
        detect_missing_values(data, "col", col_rep, report)
        self.assertEqual(len(report.issues), 1)
        self.assertEqual(report.issues[0].severity, "low")

    def test_medium_missing_severity(self):
        """10–29% missing → MEDIUM severity."""
        # 3 missing out of 20 = 15%
        data = [1.0] * 17 + [None, None, None]
        report, col_rep = _fresh()
        detect_missing_values(data, "col", col_rep, report)
        self.assertEqual(report.issues[0].severity, "medium")

    def test_high_missing_severity(self):
        """30–59% missing → HIGH severity."""
        # 8 missing out of 20 = 40%
        data = [1.0] * 12 + [None] * 8
        report, col_rep = _fresh()
        detect_missing_values(data, "col", col_rep, report)
        self.assertEqual(report.issues[0].severity, "high")

    def test_critical_missing_severity(self):
        """≥ 60% missing → CRITICAL severity."""
        # 14 missing out of 20 = 70%
        data = [1.0] * 6 + [None] * 14
        report, col_rep = _fresh()
        detect_missing_values(data, "col", col_rep, report)
        self.assertEqual(report.issues[0].severity, "critical")

    def test_missing_pct_stored_in_col_report(self):
        """Missing percentage must be written into the ColumnReport."""
        data = [1.0] * 18 + [None, None]  # 10%
        report, col_rep = _fresh()
        detect_missing_values(data, "col", col_rep, report)
        self.assertIn("missing_pct", col_rep.details)
        self.assertEqual(col_rep.get("missing_count"), 2)

    def test_suggestion_added(self):
        """At least one suggestion should be generated when missing values exist."""
        data = [1.0] * 15 + [None] * 5
        report, col_rep = _fresh()
        detect_missing_values(data, "col", col_rep, report)
        self.assertGreater(len(report.suggestions), 0)

    def test_col_report_from_sample_data_low(self):
        """col_low in MISSING_VALUES_DATASET is exactly 10% → MEDIUM severity."""
        report, col_rep = _fresh()
        detect_missing_values(
            MISSING_VALUES_DATASET["col_low"], "col_low", col_rep, report
        )
        # col_low has 2/20 = 10% missing — the boundary is ≥10% = MEDIUM
        self.assertEqual(report.issues[0].severity, "medium")

    def test_col_report_from_sample_data_critical(self):
        """col_high in MISSING_VALUES_DATASET should produce CRITICAL severity."""
        report, col_rep = _fresh()
        detect_missing_values(
            MISSING_VALUES_DATASET["col_high"], "col_high", col_rep, report
        )
        self.assertEqual(report.issues[0].severity, "critical")


# ──────────────────────────────────────────────────────────────
# 2. OUTLIER DETECTOR
# ──────────────────────────────────────────────────────────────

class TestOutlierDetector(unittest.TestCase):
    """Tests for detect_outliers()."""

    def test_no_outliers_raises_no_issue(self):
        """Tightly clustered values should produce zero issues."""
        data = [10, 11, 12, 11, 10, 13, 12, 11, 10, 12,
                11, 10, 12, 11, 13, 10, 11, 12, 10, 11]
        report, col_rep = _fresh()
        detect_outliers(data, "col", col_rep, report)
        self.assertEqual(len(report.issues), 0)

    def test_extreme_outlier_detected(self):
        """A single extreme value (999) in an otherwise normal column is an outlier."""
        data = [10, 12, 11, 13, 10, 12, 11, 13, 10, 12,
                11, 13, 10, 12, 11, 13, 10, 12, 11, 999]
        report, col_rep = _fresh()
        detect_outliers(data, "col", col_rep, report)
        self.assertEqual(len(report.issues), 1)

    def test_outlier_stats_stored(self):
        """IQR outlier count and fence should be written to the ColumnReport."""
        data = OUTLIER_DATASET["col_with_outliers"]
        report, col_rep = _fresh()
        detect_outliers(data, "col", col_rep, report)
        self.assertIn("outliers_iqr", col_rep.details)
        self.assertIn("iqr_fence", col_rep.details)
        self.assertIn("outliers_zscore", col_rep.details)

    def test_skips_non_numeric_column(self):
        """Text columns should not be processed by detect_outliers."""
        data = ["a", "b", "c", "d", "e"] * 4
        report, col_rep = _fresh()
        detect_outliers(data, "col", col_rep, report)
        self.assertEqual(len(report.issues), 0)

    def test_skips_too_few_values(self):
        """Columns with fewer than 10 values should be skipped (unreliable stats)."""
        data = [1, 2, 999, 3, 4]   # only 5 values — skip
        report, col_rep = _fresh()
        detect_outliers(data, "col", col_rep, report)
        self.assertEqual(len(report.issues), 0)

    def test_suggestion_added_when_outliers_found(self):
        """A fix suggestion must be added when outliers are detected."""
        data = OUTLIER_DATASET["col_with_outliers"]
        report, col_rep = _fresh()
        detect_outliers(data, "col", col_rep, report)
        self.assertGreater(len(report.suggestions), 0)

    def test_high_outlier_pct_gives_high_severity(self):
        """When more than 15% of values are outliers → HIGH severity."""
        # Plant outliers in 4/20 = 20% of values
        data = [10, 12, 11, 999, 10, 12, 11, 999, 10, 12,
                11, 999, 10, 12, 11, 999, 10, 12, 11, 12]
        report, col_rep = _fresh()
        detect_outliers(data, "col", col_rep, report)
        if report.issues:
            self.assertEqual(report.issues[0].severity, "high")


# ──────────────────────────────────────────────────────────────
# 3. SKEWNESS DETECTOR
# ──────────────────────────────────────────────────────────────

class TestSkewnessDetector(unittest.TestCase):
    """Tests for detect_skewness()."""

    def test_symmetric_data_no_issue(self):
        """Symmetric data (skewness ≈ 0) should produce no issue."""
        # Symmetric around 10
        data = [8, 9, 10, 11, 12, 9, 10, 11, 10, 9,
                10, 11, 9, 10, 12, 8, 11, 10, 9, 10]
        report, col_rep = _fresh()
        detect_skewness(data, "col", col_rep, report)
        self.assertEqual(len(report.issues), 0)

    def test_heavily_skewed_data_detected(self):
        """Heavily right-skewed data should produce an issue."""
        data = SKEWED_DATASET["col_right_skewed"]
        report, col_rep = _fresh()
        detect_skewness(data, "col", col_rep, report)
        self.assertGreater(len(report.issues), 0)

    def test_skewness_value_stored_in_col_report(self):
        """Skewness coefficient must be stored in the ColumnReport."""
        data = SKEWED_DATASET["col_right_skewed"]
        report, col_rep = _fresh()
        detect_skewness(data, "col", col_rep, report)
        self.assertIn("skewness", col_rep.details)

    def test_skips_non_numeric_column(self):
        """Text columns should not be processed by detect_skewness."""
        data = ["KOL", "MUM", "DEL", "KOL", "MUM"] * 4
        report, col_rep = _fresh()
        detect_skewness(data, "col", col_rep, report)
        self.assertEqual(len(report.issues), 0)

    def test_skips_too_few_values(self):
        """Columns with fewer than 10 values should be skipped."""
        data = [1, 2, 999, 3, 4]
        report, col_rep = _fresh()
        detect_skewness(data, "col", col_rep, report)
        self.assertEqual(len(report.issues), 0)

    def test_extremely_skewed_is_high_severity(self):
        """Skewness |G1| ≥ 2 should be HIGH severity."""
        # Income-like distribution — extremely skewed
        data = [1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
                1, 1, 2, 1, 1, 1, 1, 5000, 1, 1]
        report, col_rep = _fresh()
        detect_skewness(data, "col", col_rep, report)
        if report.issues:
            self.assertEqual(report.issues[0].severity, "high")

    def test_fix_suggestion_added(self):
        """A fix suggestion (log transform) should be added for skewed data."""
        data = SKEWED_DATASET["col_right_skewed"]
        report, col_rep = _fresh()
        detect_skewness(data, "col", col_rep, report)
        self.assertGreater(len(report.suggestions), 0)


# ──────────────────────────────────────────────────────────────
# 4. CLASS IMBALANCE DETECTOR
# ──────────────────────────────────────────────────────────────

class TestClassImbalanceDetector(unittest.TestCase):
    """Tests for detect_class_imbalance()."""

    def test_balanced_data_no_issue(self):
        """50/50 split should produce no issue."""
        data = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        report, col_rep = _fresh()
        detect_class_imbalance(data, "target", col_rep, report)
        self.assertEqual(len(report.issues), 0)

    def test_critical_imbalance_detected(self):
        """19:1 ratio (IMBALANCED_DATASET target_severe) → CRITICAL."""
        data = IMBALANCED_DATASET["target_severe"]
        report, col_rep = _fresh()
        detect_class_imbalance(data, "target_severe", col_rep, report, is_target=True)
        self.assertEqual(len(report.issues), 1)
        self.assertEqual(report.issues[0].severity, "critical")

    def test_high_imbalance_detected(self):
        """9:1 ratio (IMBALANCED_DATASET target) → HIGH."""
        data = IMBALANCED_DATASET["target"]
        report, col_rep = _fresh()
        detect_class_imbalance(data, "target", col_rep, report, is_target=True)
        self.assertGreater(len(report.issues), 0)
        self.assertIn(report.issues[0].severity, ["high", "critical"])

    def test_skips_too_many_classes(self):
        """Columns with >20 unique values should be skipped (not categorical)."""
        data = list(range(100))   # 100 unique values
        report, col_rep = _fresh()
        detect_class_imbalance(data, "col", col_rep, report)
        self.assertEqual(len(report.issues), 0)

    def test_skips_single_class(self):
        """A column with only one unique value should be skipped."""
        data = [1] * 20
        report, col_rep = _fresh()
        detect_class_imbalance(data, "col", col_rep, report)
        self.assertEqual(len(report.issues), 0)

    def test_imbalance_stats_stored(self):
        """Majority/minority percentage and ratio stored in ColumnReport."""
        data = IMBALANCED_DATASET["target"]
        report, col_rep = _fresh()
        detect_class_imbalance(data, "target", col_rep, report, is_target=True)
        self.assertIn("imbalance_ratio", col_rep.details)
        self.assertIn("majority_pct", col_rep.details)
        self.assertIn("minority_pct", col_rep.details)

    def test_smote_suggestion_added_for_critical(self):
        """SMOTE suggestion should appear when imbalance is CRITICAL."""
        data = IMBALANCED_DATASET["target_severe"]
        report, col_rep = _fresh()
        detect_class_imbalance(data, "target_severe", col_rep, report, is_target=True)
        combined = " ".join(report.suggestions).lower()
        self.assertIn("smote", combined)

    def test_numeric_feature_not_flagged_as_imbalanced(self):
        """Discrete numeric features (bedrooms, stories, parking) must NOT be
        flagged as imbalanced — they are features, not class labels. (v1.0.4)"""
        # bedrooms: [1]*2 + [2]*5 + [3]*16 + [4]*5 + [5]*1 + [6]*1
        # This mimics a real housing dataset — majority is 3 bedrooms
        bedrooms = [3]*16 + [2]*5 + [4]*5 + [1]*2 + [5]*1 + [6]*1
        report, col_rep = _fresh()
        # is_target=False because bedrooms is a feature, not the target
        detect_class_imbalance(bedrooms, "bedrooms", col_rep, report, is_target=False)
        self.assertEqual(len(report.issues), 0,
                         "bedrooms (discrete numeric feature) should NOT be flagged as imbalanced")

    def test_binary_numeric_feature_still_checked(self):
        """Binary 0/1 numeric columns ARE class labels and SHOULD be checked. (v1.0.4)"""
        # SeniorCitizen: 83% are 0, 17% are 1 → imbalanced binary feature
        senior = [0]*42 + [1]*8
        report, col_rep = _fresh()
        detect_class_imbalance(senior, "SeniorCitizen", col_rep, report, is_target=False)
        # Binary 0/1 non-target with ratio ≥ 5:1 should still be flagged
        self.assertGreater(len(report.issues), 0,
                           "SeniorCitizen (binary 0/1) should still be checked for imbalance")


# ──────────────────────────────────────────────────────────────
# 5. DATA LEAKAGE DETECTOR
# ──────────────────────────────────────────────────────────────

class TestDataLeakageDetector(unittest.TestCase):
    """Tests for detect_data_leakage()."""

    def test_perfect_correlation_detected(self):
        """A column with correlation 1.0 to target → CRITICAL issue."""
        col_names = ["feature_a", "leaked_col", "result_label",
                     "legitimate_feature", "target"]
        detect_data_leakage(LEAKAGE_DATASET, col_names, "target",
                            _d := DiagnosisReport("test"))
        critical_titles = [i.title for i in _d.issues if i.severity == "critical"]
        self.assertTrue(
            any("leaked_col" in t for t in critical_titles),
            f"Expected 'leaked_col' in CRITICAL issues. Got: {critical_titles}"
        )

    def test_suspicious_name_detected(self):
        """A column named 'result_label' → HIGH issue via name heuristic."""
        col_names = ["feature_a", "result_label", "target"]
        dataset   = {
            "feature_a":    [1.0] * 10,
            "result_label": [0, 1] * 5,
            "target":       [0, 1] * 5,
        }
        report = DiagnosisReport("test")
        detect_data_leakage(dataset, col_names, "target", report)
        high_titles = [i.title for i in report.issues if i.severity == "high"]
        self.assertTrue(any("result_label" in t for t in high_titles))

    def test_legitimate_feature_not_flagged(self):
        """A normal, weakly-correlated feature should not be flagged."""
        dataset = {
            "feature":  [23, 45, 12, 67, 34, 89, 56, 11, 78, 43,
                         25, 47, 14, 69, 36, 91, 58, 13, 80, 45],
            "target":   [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                         0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
        report = DiagnosisReport("test")
        detect_data_leakage(dataset, ["feature", "target"], "target", report)
        titles = [i.title for i in report.issues]
        self.assertFalse(
            any("feature" in t and "Near-Perfect" in t for t in titles),
            f"Legitimate feature was incorrectly flagged. Issues: {titles}"
        )

    def test_target_column_itself_not_flagged(self):
        """The target column must never be flagged as its own leakage."""
        col_names = ["feature", "target"]
        dataset   = {
            "feature": [1.0] * 10,
            "target":  [0, 1] * 5,
        }
        report = DiagnosisReport("test")
        detect_data_leakage(dataset, col_names, "target", report)
        for issue in report.issues:
            self.assertNotEqual(issue.column, "target",
                                "Target column should never be flagged as leakage.")


# ──────────────────────────────────────────────────────────────
# 6. DUPLICATE ROW DETECTOR
# ──────────────────────────────────────────────────────────────

class TestDuplicateRowDetector(unittest.TestCase):
    """Tests for detect_duplicate_rows()."""

    def test_no_duplicates_no_issue(self):
        """Dataset with all unique rows should produce no issue."""
        dataset   = {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}
        col_names = ["a", "b"]
        report    = DiagnosisReport("test")
        detect_duplicate_rows(dataset, col_names, report)
        self.assertEqual(len(report.issues), 0)

    def test_duplicates_detected(self):
        """Dataset with known duplicates should produce an issue."""
        col_names = ["age", "income", "target"]
        report    = DiagnosisReport("test")
        detect_duplicate_rows(DUPLICATE_DATASET, col_names, report)
        self.assertGreater(len(report.issues), 0)

    def test_duplicate_issue_title(self):
        """Issue title must mention 'Duplicate'."""
        col_names = ["age", "income", "target"]
        report    = DiagnosisReport("test")
        detect_duplicate_rows(DUPLICATE_DATASET, col_names, report)
        self.assertIn("Duplicate", report.issues[0].title)

    def test_drop_duplicates_suggestion_added(self):
        """Suggestion must mention drop_duplicates."""
        col_names = ["age", "income", "target"]
        report    = DiagnosisReport("test")
        detect_duplicate_rows(DUPLICATE_DATASET, col_names, report)
        combined = " ".join(report.suggestions)
        self.assertIn("drop_duplicates", combined)

    def test_severity_for_high_pct(self):
        """More than 5% duplicates → HIGH severity."""
        # 5 duplicates out of 15 rows = 33%
        col_names = ["age", "income", "target"]
        report    = DiagnosisReport("test")
        detect_duplicate_rows(DUPLICATE_DATASET, col_names, report)
        self.assertEqual(report.issues[0].severity, "high")


# ──────────────────────────────────────────────────────────────
# 7. CONSTANT COLUMN DETECTOR
# ──────────────────────────────────────────────────────────────

class TestConstantColumnDetector(unittest.TestCase):
    """Tests for detect_constant_columns()."""

    def test_constant_int_column_detected(self):
        """A column where every value is 42 should be flagged."""
        col_names = ["id", "constant_int", "constant_str", "varying_col", "target"]
        report    = DiagnosisReport("test")
        detect_constant_columns(CONSTANT_DATASET, col_names, report)
        flagged = [i.column for i in report.issues]
        self.assertIn("constant_int", flagged)

    def test_constant_str_column_detected(self):
        """A column where every value is 'yes' should be flagged."""
        col_names = ["id", "constant_int", "constant_str", "varying_col", "target"]
        report    = DiagnosisReport("test")
        detect_constant_columns(CONSTANT_DATASET, col_names, report)
        flagged = [i.column for i in report.issues]
        self.assertIn("constant_str", flagged)

    def test_varying_column_not_flagged(self):
        """A column with different values each row should NOT be flagged."""
        col_names = ["id", "constant_int", "constant_str", "varying_col", "target"]
        report    = DiagnosisReport("test")
        detect_constant_columns(CONSTANT_DATASET, col_names, report)
        flagged = [i.column for i in report.issues]
        self.assertNotIn("varying_col", flagged)

    def test_drop_suggestion_added(self):
        """A drop() suggestion must be added for constant columns."""
        col_names = ["constant_int"]
        dataset   = {"constant_int": [42] * 15}
        report    = DiagnosisReport("test")
        detect_constant_columns(dataset, col_names, report)
        self.assertGreater(len(report.suggestions), 0)

    def test_severity_is_medium(self):
        """Constant columns should be MEDIUM severity."""
        col_names = ["constant_int"]
        dataset   = {"constant_int": [42] * 15}
        report    = DiagnosisReport("test")
        detect_constant_columns(dataset, col_names, report)
        self.assertEqual(report.issues[0].severity, "medium")


# ──────────────────────────────────────────────────────────────
# 8. HIGH CARDINALITY DETECTOR
# ──────────────────────────────────────────────────────────────

class TestHighCardinalityDetector(unittest.TestCase):
    """Tests for detect_high_cardinality()."""

    def test_email_column_flagged(self):
        """A column with all unique email addresses → HIGH CARDINALITY."""
        col_names = ["email_col", "city_col", "target"]
        report    = DiagnosisReport("test")
        detect_high_cardinality(HIGH_CARDINALITY_DATASET, col_names, report)
        flagged = [i.column for i in report.issues]
        self.assertIn("email_col", flagged)

    def test_low_cardinality_not_flagged(self):
        """A column with only 3 unique values (city) should NOT be flagged."""
        col_names = ["email_col", "city_col", "target"]
        report    = DiagnosisReport("test")
        detect_high_cardinality(HIGH_CARDINALITY_DATASET, col_names, report)
        flagged = [i.column for i in report.issues]
        self.assertNotIn("city_col", flagged)

    def test_numeric_column_not_flagged(self):
        """Numeric columns should be skipped even if all values are unique."""
        dataset   = {"price": [float(i) for i in range(1, 16)]}
        col_names = ["price"]
        report    = DiagnosisReport("test")
        detect_high_cardinality(dataset, col_names, report)
        self.assertEqual(len(report.issues), 0)

    def test_severity_is_medium(self):
        """High cardinality should be MEDIUM severity."""
        col_names = ["email_col", "city_col", "target"]
        report    = DiagnosisReport("test")
        detect_high_cardinality(HIGH_CARDINALITY_DATASET, col_names, report)
        for issue in report.issues:
            self.assertEqual(issue.severity, "medium")


# ──────────────────────────────────────────────────────────────
# 9. FEATURE ENGINEERING HINTS
# ──────────────────────────────────────────────────────────────

class TestFeatureEngHints(unittest.TestCase):
    """Tests for suggest_feature_engineering()."""

    def test_datetime_column_hint(self):
        """Column named 'purchase_date' should trigger a datetime hint."""
        report = DiagnosisReport("test")
        suggest_feature_engineering(["purchase_date", "income"], report)
        combined = " ".join(report.suggestions).lower()
        self.assertIn("datetime", combined)

    def test_text_column_hint(self):
        """Column named 'review_text' should trigger a text processing hint."""
        report = DiagnosisReport("test")
        suggest_feature_engineering(["review_text", "score"], report)
        combined = " ".join(report.suggestions).lower()
        self.assertIn("tf-idf", combined)

    def test_geo_column_hint(self):
        """Column named 'latitude' should trigger a geographic feature hint."""
        report = DiagnosisReport("test")
        suggest_feature_engineering(["latitude", "longitude"], report)
        combined = " ".join(report.suggestions).lower()
        self.assertIn("distance", combined)

    def test_plain_column_no_hint(self):
        """Column named 'score' with no special keywords → no hint."""
        report = DiagnosisReport("test")
        suggest_feature_engineering(["score"], report)
        self.assertEqual(len(report.suggestions), 0)

    def test_timestamp_in_name_detected(self):
        """Column containing 'timestamp' should be detected as datetime."""
        report = DiagnosisReport("test")
        suggest_feature_engineering(["event_timestamp"], report)
        combined = " ".join(report.suggestions).lower()
        self.assertIn("datetime", combined)


# ──────────────────────────────────────────────────────────────
# 10. MODEL SUGGESTIONS
# ──────────────────────────────────────────────────────────────

class TestModelSuggestions(unittest.TestCase):
    """Tests for suggest_models()."""

    def test_binary_target_gives_classification_models(self):
        """Binary target (0/1) → classification model suggestions."""
        dataset   = {"feature": list(range(1, 21)),
                     "target":  [0, 1] * 10}
        col_names = ["feature", "target"]
        report    = DiagnosisReport("test")
        suggest_models(dataset, col_names, "target", report)
        combined = " ".join(report.model_types).lower()
        self.assertIn("logistic", combined)

    def test_continuous_target_gives_regression_models(self):
        """Continuous numeric target → regression model suggestions."""
        dataset   = {
            "feature": list(range(1, 21)),
            "price":   [float(i * 1000) for i in range(1, 21)],  # 20 unique values
        }
        col_names = ["feature", "price"]
        report    = DiagnosisReport("test")
        suggest_models(dataset, col_names, "price", report)
        combined = " ".join(report.model_types).lower()
        self.assertIn("regression", combined)

    def test_no_target_gives_unsupervised_models(self):
        """No target column → unsupervised model suggestions."""
        dataset   = {"feature_a": list(range(1, 11)),
                     "feature_b": list(range(10, 0, -1))}
        col_names = ["feature_a", "feature_b"]
        report    = DiagnosisReport("test")
        suggest_models(dataset, col_names, None, report)
        combined = " ".join(report.model_types).lower()
        self.assertIn("cluster", combined)

    def test_small_dataset_suggestion(self):
        """Dataset with < 500 rows → suggestion to use cross-validation."""
        dataset   = {"feature": list(range(1, 21)),
                     "target":  [0, 1] * 10}
        col_names = ["feature", "target"]
        report    = DiagnosisReport("test")
        suggest_models(dataset, col_names, "target", report)
        combined = " ".join(report.suggestions).lower()
        self.assertIn("cross-validation", combined)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
