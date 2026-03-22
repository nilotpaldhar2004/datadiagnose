"""
tests/
======
Unit test suite for DataDiagnose.

Files
-----
    sample_data.py      — ready-made sample datasets with known problems
    test_detectors.py   — unit tests for every individual detector
    test_core.py        — integration tests for diagnose() and public API

How to run all tests
--------------------
    # From the root of the project
    python -m pytest tests/ -v

    # Run with coverage report
    pip install pytest-cov
    python -m pytest tests/ -v --cov=datadiagnose --cov-report=term-missing

Author  : Nilotpal Dhar
License : MIT
"""
