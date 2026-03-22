"""
models.py
=========
Data classes for DataDiagnose.

Contains:
    - Issue           : Represents a single detected problem
    - ColumnReport    : Statistics and flags for one column
    - DiagnosisReport : The full report returned by diagnose()

Author  : Nilotpal Dhar
License : MIT
"""


# ──────────────────────────────────────────────────────────────
# ISSUE
# ──────────────────────────────────────────────────────────────

class Issue:
    """
    Represents a single problem detected in the dataset.

    Attributes
    ----------
    title           : str   — short name of the issue
    description     : str   — full explanation of what was found
    severity        : str   — 'critical', 'high', 'medium', or 'low'
    severity_points : int   — how many health points this deducts
    column          : str   — which column the issue belongs to (or None)
    fix             : str   — the suggested fix for this issue
    """

    # How many health-score points each severity level costs
    SEVERITY_POINTS = {
        'critical': 25,
        'high':     15,
        'medium':    8,
        'low':       3,
    }

    # Emoji badge for each severity (used in the report summary)
    SEVERITY_BADGE = {
        'critical': '🔴 CRITICAL',
        'high':     '🟠 HIGH',
        'medium':   '🟡 MEDIUM',
        'low':      '⚪ LOW',
    }

    def __init__(self, title, description, severity='medium', column=None, fix=''):
        if severity not in self.SEVERITY_POINTS:
            raise ValueError(
                f"Invalid severity '{severity}'. "
                f"Choose from: {list(self.SEVERITY_POINTS.keys())}"
            )
        self.title           = title
        self.description     = description
        self.severity        = severity
        self.severity_points = self.SEVERITY_POINTS[severity]
        self.column          = column
        self.fix             = fix

    def badge(self):
        """Return the coloured severity badge string."""
        return self.SEVERITY_BADGE.get(self.severity, self.severity.upper())

    def __repr__(self):
        col = f" [col='{self.column}']" if self.column else ''
        return f"Issue({self.badge()}{col}: {self.title})"


# ──────────────────────────────────────────────────────────────
# COLUMN REPORT
# ──────────────────────────────────────────────────────────────

class ColumnReport:
    """
    Holds computed statistics and flags for a single column.

    Created by the per-column analysis loop inside diagnose().
    Stored in DiagnosisReport.column_reports[col_name].

    Attributes
    ----------
    name    : str  — column name
    details : dict — key/value statistics (type, mean, missing%, etc.)
    """

    def __init__(self, name):
        self.name    = name
        self.details = {}

    def add(self, key, value):
        """Add a statistic to this column's report."""
        self.details[key] = value

    def get(self, key, default=None):
        """Safely retrieve a statistic."""
        return self.details.get(key, default)

    def __repr__(self):
        return f"ColumnReport('{self.name}', stats={list(self.details.keys())})"


# ──────────────────────────────────────────────────────────────
# DIAGNOSIS REPORT
# ──────────────────────────────────────────────────────────────

class DiagnosisReport:
    """
    The complete report returned by diagnose().

    Attributes
    ----------
    dataset_name   : str            — display name of the dataset
    n_rows         : int            — total number of rows
    n_cols         : int            — total number of columns
    score          : int            — health score 0-100 (starts at 100)
    issues         : list[Issue]    — all detected problems
    suggestions    : list[str]      — actionable fix recommendations
    model_types    : list[str]      — recommended ML model types
    column_reports : dict           — {col_name: ColumnReport}
    """

    def __init__(self, dataset_name='dataset'):
        self.dataset_name   = dataset_name
        self.n_rows         = 0
        self.n_cols         = 0
        self.score          = 100
        self.issues         = []
        self.suggestions    = []
        self.model_types    = []
        self.column_reports = {}

    # ── Mutators ────────────────────────────────────────────

    def add_issue(self, issue: Issue):
        """
        Record a detected issue and deduct its severity points
        from the health score.
        """
        self.issues.append(issue)
        self.score = max(0, self.score - issue.severity_points)

    def add_suggestion(self, text: str):
        """
        Add an actionable suggestion.
        Duplicate suggestions are silently ignored.
        """
        if text and text not in self.suggestions:
            self.suggestions.append(text)

    # ── Convenience accessors ───────────────────────────────

    def issues_by_severity(self, severity: str):
        """Return only issues of the given severity level."""
        return [i for i in self.issues if i.severity == severity]

    def critical_issues(self):
        return self.issues_by_severity('critical')

    def high_issues(self):
        return self.issues_by_severity('high')

    def status(self):
        """Return a human-readable status string based on score."""
        if self.score >= 80:
            return ' Healthy'
        elif self.score >= 50:
            return ' Needs Work'
        else:
            return ' Critical'

    # ── Report formatting ───────────────────────────────────

    def summary(self):
        """
        Return the full plain-text report as a string.
        This is what print(report) shows.
        """
        lines = []
        W = 62   # report width

        lines.append('=' * W)
        lines.append(f"  DATADIAGNOSE REPORT — {self.dataset_name.upper()}")
        lines.append('=' * W)
        lines.append(f"  Rows    : {self.n_rows}")
        lines.append(f"  Columns : {self.n_cols}")
        lines.append(f"  Score   : {self.score}/100   {self.status()}")
        lines.append('-' * W)

        # Issues
        if not self.issues:
            lines.append('  No issues detected. Your dataset looks clean!')
        else:
            lines.append(f" Issues Found ({len(self.issues)})")
            lines.append('')
            for idx, issue in enumerate(self.issues, 1):
                lines.append(f"  {idx}. {issue.badge()}")
                lines.append(f"     {issue.title}")
                lines.append(f"     → {issue.description}")
                if issue.fix:
                    lines.append(f"      Fix: {issue.fix}")
                lines.append('')

        lines.append('-' * W)

        # Suggestions
        if self.suggestions:
            lines.append(f"  Suggestions ({len(self.suggestions)})")
            lines.append('')
            for idx, s in enumerate(self.suggestions, 1):
                lines.append(f"  {idx}. {s}")
            lines.append('')

        lines.append('-' * W)

        # Models
        if self.model_types:
            lines.append('  Recommended Models')
            lines.append('')
            for m in self.model_types:
                lines.append(f"  •  {m}")
            lines.append('')

        lines.append('-' * W)

        # Per-column stats
        lines.append('  Column Statistics')
        lines.append('')
        for col, rep in self.column_reports.items():
            lines.append(f"  [{col}]")
            for k, v in rep.details.items():
                lines.append(f"    {k:<18} {v}")
            lines.append('')

        lines.append('=' * W)
        return '\n'.join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return (
            f"DiagnosisReport(name='{self.dataset_name}', "
            f"score={self.score}, issues={len(self.issues)})"
        )

    # ── Export to Pandas ───────────────────────────────────

    def to_df(self):
        """
        Convert mixed categorical and numeric reports into a single,
        comprehensive Pandas DataFrame.

        Transposes the output so that metrics (Mean, Std, Min, etc.)
        are rows and your dataset columns are the headers.
        """
        try:
            import pandas as pd
        except ImportError:
            # We use a lazy import so the library doesn't depend on pandas
            # unless the user specifically wants a DataFrame.
            raise ImportError(
                "Pandas is required to use .to_df(). "
                "Please install it with: pip install pandas"
            )

        # 1. Create a dictionary of dictionaries from all column reports
        all_stats = {}
        for col_name, col_report in self.column_reports.items():
            all_stats[col_name] = col_report.details

        # 2. Load into DataFrame
        # orient='index' handles the case where columns have different numbers of stats
        df = pd.DataFrame.from_dict(all_stats, orient='index')

        # 3. Transpose so it's a "Tall" report (Metrics vs Features)
        # This prevents the horizontal scrolling problem.
        df = df.transpose()

        return df

