"""
test_core.py
============
Unit tests for the main diagnose() engine and all public API functions
in core.py.

These are higher-level integration tests — instead of testing
individual detectors in isolation (that is test_detectors.py),
we test that diagnose() correctly orchestrates everything together
and that the public convenience functions work as expected.

Test classes
------------
    TestInputNormalisation     — dict-of-lists, list-of-dicts, edge cases
    TestDiagnoseReturnType     — correct return type and attributes
    TestDiagnoseCleanData      — clean dataset gives high score
    TestDiagnoseMessyData      — messy dataset finds multiple issues
    TestDiagnoseScoring        — health score decreases with issues
    TestDiagnoseTargetCol      — target_col parameter behaviour
    TestQuickScan              — quick_scan() prints and returns report
    TestHealthScore            — health_score() returns correct int
    TestListIssues             — list_issues() returns correct format
    TestGetSuggestions         — get_suggestions() returns correct format
    TestColumnSummary          — column_summary() returns ColumnReport
    TestEdgeCases              — empty strings, all None, single column

How to run
----------
    python -m pytest tests/test_core.py -v
    python -m pytest tests/ -v               # run all tests

Author  : Nilotpal Dhar
License : MIT
"""

import sys
import os
import unittest
from io import StringIO

# ── Path setup ────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datadiagnose import (
    diagnose,
    quick_scan,
    health_score,
    list_issues,
    get_suggestions,
    column_summary,
    DiagnosisReport,
    Issue,
    ColumnReport,
)
from tests.sample_data import (
    CLEAN_DATASET,
    MESSY_DATASET,
    MISSING_VALUES_DATASET,
    OUTLIER_DATASET,
    SKEWED_DATASET,
    IMBALANCED_DATASET,
    LEAKAGE_DATASET,
    DUPLICATE_DATASET,
    CONSTANT_DATASET,
    HIGH_CARDINALITY_DATASET,
    LISTOFDICT_DATASET,
    ALL_DATASETS,
)


# ──────────────────────────────────────────────────────────────
# 1. INPUT NORMALISATION
# ──────────────────────────────────────────────────────────────

class TestInputNormalisation(unittest.TestCase):
    """diagnose() must accept both dict-of-lists and list-of-dicts."""

    def test_accepts_dict_of_lists(self):
        """Standard dict-of-lists input should work without error."""
        report = diagnose(CLEAN_DATASET)
        self.assertIsInstance(report, DiagnosisReport)

    def test_accepts_list_of_dicts(self):
        """list-of-dicts input (pandas to_dict format) should work."""
        report = diagnose(LISTOFDICT_DATASET)
        self.assertIsInstance(report, DiagnosisReport)

    def test_list_of_dicts_same_result_as_dict(self):
        """Both formats on the same data should give the same score."""
        score_dict = health_score(CLEAN_DATASET)
        score_list = health_score(LISTOFDICT_DATASET)
        self.assertEqual(score_dict, score_list)

    def test_raises_on_empty_dict(self):
        """Empty dict should raise ValueError."""
        with self.assertRaises(ValueError):
            diagnose({})

    def test_raises_on_empty_list(self):
        """Empty list should raise ValueError."""
        with self.assertRaises(ValueError):
            diagnose([])

    def test_raises_on_wrong_type(self):
        """Passing a string or int should raise TypeError."""
        with self.assertRaises(TypeError):
            diagnose("this is not a dataset")
        with self.assertRaises(TypeError):
            diagnose(12345)

    def test_raises_on_unequal_column_lengths(self):
        """Columns with different lengths should raise ValueError."""
        bad_dataset = {
            "col_a": [1, 2, 3],
            "col_b": [1, 2],     # one row short
        }
        with self.assertRaises(ValueError):
            diagnose(bad_dataset)

    def test_dataset_name_appears_in_report(self):
        """The dataset_name parameter should appear in the report."""
        report = diagnose(CLEAN_DATASET, dataset_name="My Test Dataset")
        self.assertEqual(report.dataset_name, "My Test Dataset")


# ──────────────────────────────────────────────────────────────
# 2. RETURN TYPE AND ATTRIBUTES
# ──────────────────────────────────────────────────────────────

class TestDiagnoseReturnType(unittest.TestCase):
    """diagnose() must return a properly structured DiagnosisReport."""

    def setUp(self):
        self.report = diagnose(CLEAN_DATASET, target_col="target")

    def test_returns_diagnosis_report_instance(self):
        self.assertIsInstance(self.report, DiagnosisReport)

    def test_score_is_integer(self):
        self.assertIsInstance(self.report.score, int)

    def test_score_is_between_0_and_100(self):
        self.assertGreaterEqual(self.report.score, 0)
        self.assertLessEqual(self.report.score, 100)

    def test_issues_is_a_list(self):
        self.assertIsInstance(self.report.issues, list)

    def test_suggestions_is_a_list(self):
        self.assertIsInstance(self.report.suggestions, list)

    def test_model_types_is_a_list(self):
        self.assertIsInstance(self.report.model_types, list)

    def test_column_reports_is_a_dict(self):
        self.assertIsInstance(self.report.column_reports, dict)

    def test_n_rows_correct(self):
        expected = len(CLEAN_DATASET["age"])
        self.assertEqual(self.report.n_rows, expected)

    def test_n_cols_correct(self):
        expected = len(CLEAN_DATASET.keys())
        self.assertEqual(self.report.n_cols, expected)

    def test_column_reports_keys_match_dataset(self):
        """Every column in the dataset should have a ColumnReport."""
        for col in CLEAN_DATASET:
            self.assertIn(col, self.report.column_reports)

    def test_each_column_report_is_column_report_instance(self):
        for col, rep in self.report.column_reports.items():
            self.assertIsInstance(rep, ColumnReport,
                                  f"column_reports['{col}'] is not a ColumnReport")

    def test_all_issues_are_issue_instances(self):
        for issue in self.report.issues:
            self.assertIsInstance(issue, Issue)

    def test_all_suggestions_are_strings(self):
        for s in self.report.suggestions:
            self.assertIsInstance(s, str)

    def test_summary_returns_string(self):
        self.assertIsInstance(str(self.report), str)

    def test_summary_contains_dataset_name(self):
        report = diagnose(CLEAN_DATASET, dataset_name="TestName")
        self.assertIn("TESTNAME", report.summary().upper())


# ──────────────────────────────────────────────────────────────
# 3. CLEAN DATASET BEHAVIOUR
# ──────────────────────────────────────────────────────────────

class TestDiagnoseCleanData(unittest.TestCase):
    """A clean, well-formed dataset should produce a high health score."""

    def setUp(self):
        self.report = diagnose(CLEAN_DATASET, target_col="target")

    def test_score_is_high(self):
        """Clean dataset score should be at least 80."""
        self.assertGreaterEqual(self.report.score, 80,
                                f"Expected score ≥ 80, got {self.report.score}")

    def test_no_critical_issues(self):
        """Clean dataset should have zero CRITICAL issues."""
        critical = [i for i in self.report.issues if i.severity == "critical"]
        self.assertEqual(len(critical), 0,
                         f"Unexpected CRITICAL issues: {critical}")

    def test_model_types_not_empty(self):
        """Even a clean dataset should recommend model types."""
        self.assertGreater(len(self.report.model_types), 0)

    def test_column_reports_have_type_field(self):
        """Every column report should have a 'type' field."""
        for col, rep in self.report.column_reports.items():
            self.assertIn("type", rep.details,
                          f"Column '{col}' missing 'type' in ColumnReport")


# ──────────────────────────────────────────────────────────────
# 4. MESSY DATASET — FINDS MULTIPLE ISSUES
# ──────────────────────────────────────────────────────────────

class TestDiagnoseMessyData(unittest.TestCase):
    """The messy dataset must trigger multiple issues of different types."""

    def setUp(self):
        self.report = diagnose(MESSY_DATASET, target_col="purchased",
                               dataset_name="Messy")

    def test_score_is_low(self):
        """Messy dataset should score below 60."""
        self.assertLess(self.report.score, 60,
                        f"Expected score < 60, got {self.report.score}")

    def test_multiple_issues_found(self):
        """At least 4 distinct issues should be detected."""
        self.assertGreaterEqual(len(self.report.issues), 4)

    def test_missing_value_issue_found(self):
        """Missing values in 'age' should be detected."""
        titles = [i.title for i in self.report.issues]
        self.assertTrue(any("Missing" in t for t in titles),
                        f"No missing value issue found. Titles: {titles}")

    def test_constant_column_issue_found(self):
        """The constant 'country' column should be detected."""
        titles = [i.title for i in self.report.issues]
        self.assertTrue(any("Constant" in t for t in titles),
                        f"No constant column issue found. Titles: {titles}")

    def test_leakage_issue_found(self):
        """The 'outcome_flag' column should be detected as potential leakage."""
        titles = [i.title for i in self.report.issues]
        self.assertTrue(any("Leakage" in t or "outcome_flag" in t for t in titles),
                        f"No leakage issue found. Titles: {titles}")

    def test_model_types_suggested(self):
        """Model recommendations should still be provided."""
        self.assertGreater(len(self.report.model_types), 0)

    def test_suggestions_not_empty(self):
        """Fix suggestions should not be empty."""
        self.assertGreater(len(self.report.suggestions), 0)


# ──────────────────────────────────────────────────────────────
# 5. HEALTH SCORE DECREASES WITH ISSUES
# ──────────────────────────────────────────────────────────────

class TestDiagnoseScoring(unittest.TestCase):
    """Each issue must deduct the correct number of points."""

    def test_score_starts_at_100(self):
        """A report with no issues should have score = 100."""
        r = DiagnosisReport("empty")
        self.assertEqual(r.score, 100)

    def test_critical_deducts_25(self):
        """Adding a CRITICAL issue should reduce score by 25."""
        r = DiagnosisReport("test")
        r.add_issue(Issue("Test", "desc", severity="critical"))
        self.assertEqual(r.score, 75)

    def test_high_deducts_15(self):
        """Adding a HIGH issue should reduce score by 15."""
        r = DiagnosisReport("test")
        r.add_issue(Issue("Test", "desc", severity="high"))
        self.assertEqual(r.score, 85)

    def test_medium_deducts_8(self):
        """Adding a MEDIUM issue should reduce score by 8."""
        r = DiagnosisReport("test")
        r.add_issue(Issue("Test", "desc", severity="medium"))
        self.assertEqual(r.score, 92)

    def test_low_deducts_3(self):
        """Adding a LOW issue should reduce score by 3."""
        r = DiagnosisReport("test")
        r.add_issue(Issue("Test", "desc", severity="low"))
        self.assertEqual(r.score, 97)

    def test_score_never_goes_below_zero(self):
        """Score should floor at 0, never go negative."""
        r = DiagnosisReport("test")
        for _ in range(20):
            r.add_issue(Issue("Test", "desc", severity="critical"))
        self.assertEqual(r.score, 0)

    def test_messy_lower_than_clean(self):
        """Messy dataset must score lower than clean dataset."""
        clean_score = health_score(CLEAN_DATASET, "target")
        messy_score = health_score(MESSY_DATASET, "purchased")
        self.assertLess(messy_score, clean_score)

    def test_all_datasets_return_valid_score(self):
        """Every sample dataset must return a score between 0 and 100."""
        for name, dataset in ALL_DATASETS.items():
            score = health_score(dataset)
            self.assertGreaterEqual(score, 0,   f"{name}: score below 0")
            self.assertLessEqual(score,   100,  f"{name}: score above 100")


# ──────────────────────────────────────────────────────────────
# 6. TARGET COLUMN PARAMETER
# ──────────────────────────────────────────────────────────────

class TestDiagnoseTargetCol(unittest.TestCase):
    """target_col parameter should affect model suggestions and leakage detection."""

    def test_no_target_gives_unsupervised_models(self):
        """No target → unsupervised model types suggested."""
        report  = diagnose(CLEAN_DATASET, target_col=None)
        combined = " ".join(report.model_types).lower()
        self.assertIn("cluster", combined)

    def test_binary_target_gives_classification(self):
        """Binary 0/1 target → classification model names."""
        report  = diagnose(CLEAN_DATASET, target_col="target")
        combined = " ".join(report.model_types).lower()
        self.assertIn("logistic", combined)

    def test_leakage_only_checked_when_target_given(self):
        """Data leakage check should only run when target_col is provided."""
        # With target → leakage checked
        report_with = diagnose(LEAKAGE_DATASET, target_col="target")
        # Without target → leakage NOT checked
        report_without = diagnose(LEAKAGE_DATASET, target_col=None)
        critical_with    = len([i for i in report_with.issues
                                if "Correlation" in i.title])
        critical_without = len([i for i in report_without.issues
                                if "Correlation" in i.title])
        self.assertGreater(critical_with, critical_without)

    def test_nonexistent_target_col_handled_gracefully(self):
        """Passing a target_col that does not exist should not crash."""
        report = diagnose(CLEAN_DATASET, target_col="nonexistent_column")
        self.assertIsInstance(report, DiagnosisReport)


# ──────────────────────────────────────────────────────────────
# 7. QUICK SCAN
# ──────────────────────────────────────────────────────────────

class TestQuickScan(unittest.TestCase):
    """quick_scan() must print to stdout and return a DiagnosisReport."""

    def test_returns_diagnosis_report(self):
        """quick_scan() must return a DiagnosisReport, not None."""
        import sys
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            result = quick_scan(CLEAN_DATASET, target_col="target")
        finally:
            sys.stdout = old_stdout
        self.assertIsInstance(result, DiagnosisReport)

    def test_prints_to_stdout(self):
        """quick_scan() must produce output to stdout."""
        captured = StringIO()
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            quick_scan(CLEAN_DATASET, target_col="target")
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        self.assertGreater(len(output), 0, "quick_scan printed nothing")

    def test_output_contains_score(self):
        """Printed output must contain the word 'Score'."""
        captured = StringIO()
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            quick_scan(CLEAN_DATASET)
        finally:
            sys.stdout = old_stdout
        self.assertIn("Score", captured.getvalue())


# ──────────────────────────────────────────────────────────────
# 8. HEALTH SCORE
# ──────────────────────────────────────────────────────────────

class TestHealthScore(unittest.TestCase):
    """health_score() must return an integer between 0 and 100."""

    def test_returns_integer(self):
        score = health_score(CLEAN_DATASET)
        self.assertIsInstance(score, int)

    def test_returns_value_in_range(self):
        score = health_score(CLEAN_DATASET)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_clean_dataset_high_score(self):
        score = health_score(CLEAN_DATASET, "target")
        self.assertGreaterEqual(score, 80)

    def test_messy_dataset_low_score(self):
        score = health_score(MESSY_DATASET, "purchased")
        self.assertLess(score, 60)

    def test_consistent_with_diagnose(self):
        """health_score() must return same value as diagnose().score."""
        report = diagnose(CLEAN_DATASET, target_col="target")
        score  = health_score(CLEAN_DATASET, target_col="target")
        self.assertEqual(report.score, score)


# ──────────────────────────────────────────────────────────────
# 9. LIST ISSUES
# ──────────────────────────────────────────────────────────────

class TestListIssues(unittest.TestCase):
    """list_issues() must return a list of (severity_str, title_str) tuples."""

    def test_returns_list(self):
        result = list_issues(CLEAN_DATASET)
        self.assertIsInstance(result, list)

    def test_each_item_is_tuple_of_two_strings(self):
        result = list_issues(MESSY_DATASET, "purchased")
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            severity, title = item
            self.assertIsInstance(severity, str)
            self.assertIsInstance(title, str)

    def test_severity_is_uppercase(self):
        """Severity strings should be uppercase e.g. 'CRITICAL' not 'critical'."""
        result = list_issues(MESSY_DATASET, "purchased")
        valid  = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        for severity, _ in result:
            self.assertIn(severity, valid,
                          f"Severity '{severity}' is not a valid uppercase severity")

    def test_clean_dataset_has_no_issues(self):
        result = list_issues(CLEAN_DATASET, "target")
        # Clean dataset might still have a couple of minor issues (like feature eng)
        # but should have no CRITICAL issues
        critical = [s for s, t in result if s == "CRITICAL"]
        self.assertEqual(len(critical), 0)

    def test_messy_dataset_has_issues(self):
        result = list_issues(MESSY_DATASET, "purchased")
        self.assertGreater(len(result), 0)

    def test_consistent_with_diagnose(self):
        """list_issues() count should match diagnose().issues count."""
        report = diagnose(MESSY_DATASET, target_col="purchased")
        issues = list_issues(MESSY_DATASET, target_col="purchased")
        self.assertEqual(len(report.issues), len(issues))


# ──────────────────────────────────────────────────────────────
# 10. GET SUGGESTIONS
# ──────────────────────────────────────────────────────────────

class TestGetSuggestions(unittest.TestCase):
    """get_suggestions() must return a list of non-empty strings."""

    def test_returns_list(self):
        result = get_suggestions(CLEAN_DATASET)
        self.assertIsInstance(result, list)

    def test_each_suggestion_is_string(self):
        result = get_suggestions(MESSY_DATASET, "purchased")
        for s in result:
            self.assertIsInstance(s, str)

    def test_no_empty_strings(self):
        result = get_suggestions(MESSY_DATASET, "purchased")
        for s in result:
            self.assertGreater(len(s.strip()), 0, "Empty suggestion found")

    def test_suggestions_not_duplicated(self):
        """Duplicate suggestions should not appear in the list."""
        result = get_suggestions(MESSY_DATASET, "purchased")
        self.assertEqual(len(result), len(set(result)),
                         "Duplicate suggestions found in get_suggestions()")

    def test_consistent_with_diagnose(self):
        """get_suggestions() must match diagnose().suggestions."""
        report      = diagnose(MESSY_DATASET, target_col="purchased")
        suggestions = get_suggestions(MESSY_DATASET, target_col="purchased")
        self.assertEqual(report.suggestions, suggestions)

    def test_missing_data_includes_imputation_suggestion(self):
        """Datasets with missing values should suggest imputation."""
        result   = get_suggestions(MISSING_VALUES_DATASET, "target")
        combined = " ".join(result).lower()
        self.assertTrue(
            "fill" in combined or "impute" in combined or "drop" in combined,
            f"No imputation suggestion found. Suggestions: {result}"
        )


# ──────────────────────────────────────────────────────────────
# 11. COLUMN SUMMARY
# ──────────────────────────────────────────────────────────────

class TestColumnSummary(unittest.TestCase):
    """column_summary() must return a ColumnReport for a named column."""

    def test_returns_column_report(self):
        rep = column_summary(CLEAN_DATASET, "age")
        self.assertIsInstance(rep, ColumnReport)

    def test_returns_none_for_missing_column(self):
        rep = column_summary(CLEAN_DATASET, "nonexistent_column")
        self.assertIsNone(rep)

    def test_column_report_has_type(self):
        rep = column_summary(CLEAN_DATASET, "age")
        self.assertIn("type", rep.details)

    def test_numeric_column_has_mean(self):
        rep = column_summary(CLEAN_DATASET, "age")
        self.assertIn("mean", rep.details)

    def test_numeric_column_has_std(self):
        rep = column_summary(CLEAN_DATASET, "income")
        self.assertIn("std", rep.details)

    def test_numeric_column_has_min_max(self):
        rep = column_summary(CLEAN_DATASET, "income")
        self.assertIn("min", rep.details)
        self.assertIn("max", rep.details)


# ──────────────────────────────────────────────────────────────
# 12. EDGE CASES
# ──────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):
    """Unusual inputs that should not crash the library."""

    def test_all_missing_column(self):
        """A column of all Nones should not crash diagnose()."""
        dataset = {"a": [None, None, None, None, None],
                   "b": [1, 2, 3, 4, 5]}
        report  = diagnose(dataset)
        self.assertIsInstance(report, DiagnosisReport)

    def test_single_column_dataset(self):
        """A dataset with only one column should not crash."""
        dataset = {"only_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        report  = diagnose(dataset)
        self.assertIsInstance(report, DiagnosisReport)

    def test_very_small_dataset(self):
        """A dataset with only 3 rows should not crash."""
        dataset = {"a": [1, 2, 3], "b": [4, 5, 6]}
        report  = diagnose(dataset)
        self.assertIsInstance(report, DiagnosisReport)

    def test_mixed_none_and_empty_string_missing(self):
        """Both None and '' should both be counted as missing."""
        dataset = {
            "col":    [1, None, "", 4, 5, 6, 7, 8, 9, 10],
            "target": [0, 1,    0, 1, 0, 1, 0, 1, 0,  1],
        }
        report = diagnose(dataset, target_col="target")
        rep    = report.column_reports["col"]
        self.assertEqual(rep.get("missing_count"), 2)

    def test_numeric_string_values_treated_as_numeric(self):
        """Strings that look like numbers ('42', '3.14') are numeric."""
        dataset = {
            "prices": ["10.5", "20.0", "15.3", "18.7", "22.1",
                       "11.0", "19.5", "14.8", "21.3", "16.9"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
        report = diagnose(dataset, target_col="target")
        rep    = report.column_reports["prices"]
        self.assertEqual(rep.get("type"), "numeric")

    def test_duplicate_suggestion_not_added_twice(self):
        """The same suggestion string should not appear twice in the list."""
        report = diagnose(MESSY_DATASET, target_col="purchased")
        suggestions = report.suggestions
        self.assertEqual(len(suggestions), len(set(suggestions)),
                         "Duplicate suggestions exist in the report.")

    def test_issue_repr_does_not_crash(self):
        """repr(Issue) should return a string without crashing."""
        issue = Issue("Test Issue", "Test description", "high", "col_name")
        self.assertIsInstance(repr(issue), str)

    def test_report_repr_does_not_crash(self):
        """repr(DiagnosisReport) should return a string without crashing."""
        report = diagnose(CLEAN_DATASET)
        self.assertIsInstance(repr(report), str)

    def test_dataset_with_boolean_values(self):
        """Boolean True/False values in a column should not crash diagnose()."""
        dataset = {
            "is_active": [True, False, True, True, False,
                          True, False, True, False, True],
            "score":     [80, 60, 90, 75, 55, 85, 70, 95, 65, 88],
        }
        report = diagnose(dataset)
        self.assertIsInstance(report, DiagnosisReport)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
