"""
core.py
=======
The main diagnosis engine and all public API functions.

This is the file that ties everything together.
It imports from models.py, utils.py, and detectors.py,
and exposes the clean public API that users call.

Public API
----------
    diagnose()        — full diagnosis, returns DiagnosisReport
    quick_scan()      — diagnose + print
    health_score()    — just the integer score
    list_issues()     — just the issue titles
    get_suggestions() — just the suggestion strings

Author  : Nilotpal Dhar
License : MIT
"""

from .models    import DiagnosisReport, ColumnReport
from .utils     import (
    to_numeric_list, is_categorical, non_null_values,
    mean, median, std,
)
from .detectors import (
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


# ──────────────────────────────────────────────────────────────
# INPUT NORMALISATION
# ──────────────────────────────────────────────────────────────

def _normalise_input(dataset):
    # Detect if input is a Pandas DataFrame or similar object
    if hasattr(dataset, 'to_dict') and callable(getattr(dataset, 'to_dict')):
        # We use orient='list' because it matches your preferred dict-of-lists format
        dataset = dataset.to_dict(orient='list')
    # ---------------------------
    """
    Accept two input formats and return a dict-of-lists.

    Format 1 — dict of lists (preferred):
        {"age": [25, 30, 22], "income": [50000, 60000, 48000]}

    Format 2 — list of dicts (from pandas .to_dict("records")):
        [{"age": 25, "income": 50000}, {"age": 30, "income": 60000}]

    Raises
    ------
    TypeError  : if input is neither a dict nor a list
    ValueError : if the dataset is empty or columns have unequal lengths
    """
    if isinstance(dataset, list):
        if not dataset:
            raise ValueError("Dataset is empty — got an empty list.")
        if not isinstance(dataset[0], dict):
            raise TypeError(
                "List input must be a list of dicts. "
                "Use df.to_dict(orient='records') to convert a pandas DataFrame."
            )
        col_names = list(dataset[0].keys())
        result    = {col: [row.get(col) for row in dataset] for col in col_names}
        return result

    elif isinstance(dataset, dict):
        if not dataset:
            raise ValueError("Dataset is empty — got an empty dict.")
        return dataset

    else:
        raise TypeError(
            f"Expected dict or list of dicts, got {type(dataset).__name__}. "
            "Convert your data: dict-of-lists or list-of-dicts formats are supported."
        )


def _validate(dataset):
    """
    Basic sanity checks on the normalised dict-of-lists dataset.

    Checks:
        - At least one column exists
        - All columns have the same number of rows
        - At least one row exists
    """
    col_names = list(dataset.keys())

    if not col_names:
        raise ValueError("Dataset has no columns.")

    lengths = {col: len(dataset[col]) for col in col_names}
    if len(set(lengths.values())) > 1:
        bad = {c: l for c, l in lengths.items()}
        raise ValueError(
            f"All columns must have the same number of rows. "
            f"Column lengths: {bad}"
        )

    if lengths[col_names[0]] == 0:
        raise ValueError("Dataset has 0 rows.")

    return col_names


# ──────────────────────────────────────────────────────────────
# MAIN DIAGNOSIS ENGINE
# ──────────────────────────────────────────────────────────────

def diagnose(dataset, target_col=None, dataset_name='dataset'):
    """
    Run a full diagnosis on any dataset and return a DiagnosisReport.

    This is the main function — the heart of DataDiagnose.
    It orchestrates all 8 detectors, collects issues and suggestions,
    computes per-column statistics, and returns a complete report.

    Parameters
    ----------
    dataset      : dict or list
        Your data in one of two formats:
            • dict-of-lists: {"col": [val, val, ...], ...}
            • list-of-dicts: [{"col": val, ...}, ...]
        Both formats are accepted. Pandas DataFrames can be passed
        via df.to_dict(orient="list") or df.to_dict(orient="records").

    target_col   : str or None
        The name of the column you want to predict (the label).
        Used to detect data leakage, class imbalance in the target,
        and to recommend the correct model type.
        Pass None for unsupervised learning scenarios.

    dataset_name : str
        A display name for the report header. Default is 'dataset'.

    Returns
    -------
    DiagnosisReport
        An object with:
            .score          int       — health score 0-100
            .issues         list      — all detected Issue objects
            .suggestions    list      — actionable fix strings
            .model_types    list      — recommended model type strings
            .column_reports dict      — per-column statistics
            .n_rows         int
            .n_cols         int

    Examples
    --------
    >>> from datadiagnose import diagnose
    >>> data = {"age": [25, None, 900], "target": [1, 0, 1]}
    >>> report = diagnose(data, target_col="target")
    >>> print(report)
    >>> print(report.score)
    """

    # ── Step 1: Normalise input ──────────────────────────────
    dataset   = _normalise_input(dataset)
    col_names = _validate(dataset)

    n_rows = len(dataset[col_names[0]])
    n_cols = len(col_names)

    # ── Initialise report ────────────────────────────────────
    report          = DiagnosisReport(dataset_name)
    report.n_rows   = n_rows
    report.n_cols   = n_cols

    # ── Step 2: Dataset-level checks ────────────────────────
    # These look at the whole dataset, not individual columns.

    detect_duplicate_rows(dataset, col_names, report)
    detect_constant_columns(dataset, col_names, report)
    detect_high_cardinality(dataset, col_names, report)

    if target_col:
        detect_data_leakage(dataset, col_names, target_col, report)

    # ── Step 3: Per-column analysis ──────────────────────────
    for col in col_names:
        data       = dataset[col]
        col_report = ColumnReport(col)

        # ── Determine column type ────────────────────────────
        nums   = to_numeric_list(data)
        is_cat = is_categorical(data)

        if nums is not None:
            # Numeric column — compute descriptive statistics
            col_report.add('type',   'numeric')
            col_report.add('mean',   f'{mean(nums):.4f}')
            col_report.add('median', f'{median(nums):.4f}')
            col_report.add('std',    f'{std(nums):.4f}')
            col_report.add('min',    f'{min(nums):.4f}')
            col_report.add('max',    f'{max(nums):.4f}')
        elif is_cat:
            nn     = non_null_values(data)
            unique = set(str(v) for v in nn)
            col_report.add('type',          'categorical')
            col_report.add('unique_values', len(unique))
            col_report.add('top_value',     _mode(nn))
        else:
            col_report.add('type', 'text / high-cardinality')

        # ── Run detectors ────────────────────────────────────
        detect_missing_values(data, col, col_report, report)

        if nums is not None:
            detect_outliers(data, col, col_report, report)
            detect_skewness(data, col, col_report, report)
        # ── IMPROVED CLASS IMBALANCE LOGIC ──────────────────
        # 1. Determine if this is the target column
        is_target = (col == target_col)

        # 2. Identify if this is a Regression target (Numeric + many unique values)
        unique_vals = set(str(v) for v in non_null_values(data))
        is_regression_target = (is_target and nums is not None and len(unique_vals) > 20)

        # 3. Only check imbalance if it's NOT a regression target
        if not is_regression_target:
            if is_cat or (nums is not None and len(unique_vals) <= 20):
                detect_class_imbalance(data, col, col_report, report, is_target)

        report.column_reports[col] = col_report

    # ── Step 4: Feature engineering hints ───────────────────
    suggest_feature_engineering(col_names, report)

    # ── Step 5: Model recommendations ───────────────────────
    suggest_models(dataset, col_names, target_col, report)

    return report


# ──────────────────────────────────────────────────────────────
# PRIVATE HELPER
# ──────────────────────────────────────────────────────────────

def _mode(values):
    """Return the most common value in a list (simple mode)."""
    if not values:
        return None
    from collections import Counter
    return Counter(str(v) for v in values).most_common(1)[0][0]


# ──────────────────────────────────────────────────────────────
# PUBLIC CONVENIENCE FUNCTIONS
# ──────────────────────────────────────────────────────────────

def quick_scan(dataset, target_col=None, dataset_name='dataset'):
    """
    Run a full diagnosis AND immediately print the report to the console.

    Identical to diagnose() in every way, except it also calls print().
    Returns the DiagnosisReport in case you need to inspect it further.

    Example
    -------
    >>> quick_scan(my_dataset, target_col='survived')
    """
    report = diagnose(dataset, target_col=target_col, dataset_name=dataset_name)
    print(report.summary())
    return report


def health_score(dataset, target_col=None):
    """
    Return just the integer health score (0 to 100).

    100 = perfectly clean dataset.
    0   = dataset has critical, unresolvable issues.

    Useful for using DataDiagnose as a gate in automated pipelines:

    Example
    -------
    >>> score = health_score(my_data, target_col='target')
    >>> if score < 70:
    ...     raise ValueError(f'Data quality too low: {score}/100')
    """
    return diagnose(dataset, target_col=target_col).score


def list_issues(dataset, target_col=None):
    """
    Return a concise list of (severity, title) tuples for all issues.

    Useful when you want to quickly scan what problems exist
    without reading the full report narrative.

    Returns
    -------
    list of (str, str) — e.g. [('CRITICAL', 'Missing Values in age'), ...]

    Example
    -------
    >>> for severity, title in list_issues(my_data, 'target'):
    ...     print(f'[{severity}] {title}')
    """
    report = diagnose(dataset, target_col=target_col)
    return [(issue.severity.upper(), issue.title) for issue in report.issues]


def get_suggestions(dataset, target_col=None):
    """
    Return only the list of actionable fix suggestions as strings.

    Each suggestion is a specific, actionable recommendation
    tied to a detected issue in the dataset.

    Returns
    -------
    list of str

    Example
    -------
    >>> for tip in get_suggestions(my_data, 'target'):
    ...     print('-', tip)
    """
    return diagnose(dataset, target_col=target_col).suggestions


def column_summary(dataset, col_name, target_col=None):
    """
    Return the ColumnReport for a single column by name.

    Useful when you want to deep-dive into one specific column
    without reading the full report.

    Returns
    -------
    ColumnReport object (or None if column not found)

    Example
    -------
    >>> rep = column_summary(my_data, 'age')
    >>> print(rep.details)
    """
    report = diagnose(dataset, target_col=target_col)
    return report.column_reports.get(col_name)


# ... (existing imports and functions like diagnose, to_numeric_list, etc.) ...

# ──────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPERS
# ──────────────────────────────────────────────────────────────

def get_stats_df(dataset, target_col=None):
    """
    Convenience function to run a full diagnosis and return
    only the statistics table as a Pandas DataFrame.
    """
    # 1. Runs the full logic (Step 1, 2, 3)
    report = diagnose(dataset, target_col=target_col)

    # 2. Uses the new method we added to models.py
    return report.to_df()
