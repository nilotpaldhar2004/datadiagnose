"""
datadiagnose
============
Dataset Auto-Diagnosis Python Library.

A lightweight, zero-dependency Python library that analyses
any tabular dataset and tells you exactly what is wrong with it,
how serious each problem is, and how to fix it.

Zero external dependencies — uses only Python's standard library.

Quick Start
-----------
    import datadiagnose as dd

    dataset = {
        "age":    [25, 30, None, 22, 900],
        "income": [50000, 60000, None, 48000, 52000],
        "target": [1, 0, 1, 0, 1],
    }

    # New in v1.0.1: Get a Pandas DataFrame of stats instantly
    df_stats = dd.get_stats_df(dataset, target_col="target")

    # Or get the full text report
    report = dd.diagnose(dataset, target_col="target")
    print(report)

Public API
----------
    diagnose(dataset, target_col, dataset_name)  → DiagnosisReport
    quick_scan(dataset, target_col)              → prints + returns report
    health_score(dataset, target_col)            → int 0-100
    list_issues(dataset, target_col)             → list of (severity, title)
    get_suggestions(dataset, target_col)         → list of fix strings
    column_summary(dataset, col_name)            → ColumnReport


Author  : Nilotpal Dhar
Version : 1.0.1
License : MIT
GitHub  : https://github.com/nilotpaldhar2004/datadiagnose
"""

# ── Public API imports ────────────────────────────────────────
# FIX: Moved these to the top!
from .core import (
    diagnose,
    get_stats_df,
    quick_scan,
    health_score,
    list_issues,
    get_suggestions,
    column_summary,
)

from .models import (
    DiagnosisReport,
    Issue,
    ColumnReport
)

# ── Version info ──────────────────────────────────────────────
__version__ = '1.0.1'
__author__  = 'Nilotpal Dhar'
__license__ = 'MIT'

# ── Public API ────────────────────────────────────────────────
__all__ = [
    'diagnose',
    'get_stats_df',
    'quick_scan',
    'health_score',
    'list_issues',
    'get_suggestions',
    'column_summary',
    'DiagnosisReport',
    'Issue',
    'ColumnReport',
]


