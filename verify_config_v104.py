"""
verify_config_v104.py
=====================
Run this script from the ROOT of your project to verify that
pyproject.toml, .flake8, and tests.yml are all correct for v1.0.4.

How to run
----------
    python verify_config_v104.py

All checks must print OK. If any print FAIL, fix that file first.

Author  : Nilotpal Dhar
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PASS = "\u2705 OK  "
FAIL = "\u274c FAIL"
errors = []


def check(condition, label, fix_hint=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}")
        if fix_hint:
            print(f"          Fix: {fix_hint}")
        errors.append(label)


def read(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full, encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────
# 1. pyproject.toml
# ─────────────────────────────────────────────────────────────
print("\n── pyproject.toml ───────────────────────────────────────")
toml = read("pyproject.toml")

if toml is None:
    print(f"  {FAIL}  pyproject.toml not found in {ROOT}")
    errors.append("pyproject.toml missing")
else:
    check('version = "1.0.4"' in toml,
          'version = "1.0.4"',
          'Change version = "1.0.3" → version = "1.0.4"')

    check("datadiagnose" in toml and 'name = "datadiagnose"' in toml,
          'name = "datadiagnose" present')

    check("dharnilotpal31@gmail.com" in toml,
          "Author email is set")

    check("nilotpaldhar2004/datadiagnose" in toml,
          "GitHub URL is correct")

    check('requires-python = ">=3.7"' in toml,
          'Python 3.7+ minimum set')

    check("dependencies = []" in toml,
          "Zero runtime dependencies declared")

    check('include = ["datadiagnose*"]' in toml,
          "Package discovery includes only datadiagnose/")

    check("testpaths" in toml and '"tests"' in toml,
          "pytest testpaths = tests")


# ─────────────────────────────────────────────────────────────
# 2. .flake8
# ─────────────────────────────────────────────────────────────
print("\n── .flake8 ──────────────────────────────────────────────")
flake8 = read(".flake8")

if flake8 is None:
    print(f"  {FAIL}  .flake8 not found in {ROOT}")
    errors.append(".flake8 missing")
else:
    check("max-line-length = 100" in flake8,
          "max-line-length = 100",
          "Change max-line-length = 130 → max-line-length = 100 to match tests.yml")

    check("venv" in flake8,
          "venv in exclude list",
          "Add 'venv,' to the exclude = section")

    check("__pycache__" in flake8,
          "__pycache__ in exclude list")

    check(".git" in flake8,
          ".git in exclude list")

    check("E203" in flake8,
          "E203 in extend-ignore (allows alignment spacing)")

    check("W503" in flake8,
          "W503 in extend-ignore (line break before binary operator)")


# ─────────────────────────────────────────────────────────────
# 3. .github/workflows/tests.yml
# ─────────────────────────────────────────────────────────────
print("\n── .github/workflows/tests.yml ──────────────────────────")
yml = read(os.path.join(".github", "workflows", "tests.yml"))

if yml is None:
    print(f"  {FAIL}  .github/workflows/tests.yml not found")
    errors.append("tests.yml missing")
else:
    check("--max-line-length 100" in yml,
          "flake8 --max-line-length 100 present",
          "Add --max-line-length 100 to the flake8 step in tests.yml")

    check("--exclude=__pycache__" in yml,
          "flake8 --exclude=__pycache__ present",
          "Add --exclude=__pycache__ to the flake8 step")

    check('"3.8"' in yml and '"3.12"' in yml,
          "Python matrix covers 3.8 → 3.12")

    check("ubuntu-latest" in yml and "macos-latest" in yml and "windows-latest" in yml,
          "OS matrix covers Ubuntu + macOS + Windows")

    check("fail-fast: false" in yml,
          "fail-fast: false (keep running on failure)")

    check("actions/checkout@v4" in yml,
          "Uses checkout@v4 (latest)")

    check("actions/setup-python@v5" in yml,
          "Uses setup-python@v5 (latest)")

    check("pip install -e ." in yml,
          "Installs DataDiagnose in editable mode")

    check("pytest tests/ -v --tb=short" in yml,
          "Full test suite command present")

    check("--cov=datadiagnose" in yml,
          "Coverage measurement enabled")

    check("workflow_dispatch" in yml,
          "Manual trigger (workflow_dispatch) enabled")

    check("quick-check" in yml or "quick_check" in yml,
          "Quick smoke test job present")

    check("examples/basic_usage.py" in yml,
          "basic_usage.py example run in CI")


# ─────────────────────────────────────────────────────────────
# 4. Package __init__.py version check
# ─────────────────────────────────────────────────────────────
print("\n── datadiagnose/__init__.py ─────────────────────────────")
init = read(os.path.join("datadiagnose", "__init__.py"))

if init is None:
    print(f"  {FAIL}  datadiagnose/__init__.py not found")
    errors.append("__init__.py missing")
else:
    check("__version__ = '1.0.4'" in init or '__version__ = "1.0.4"' in init,
          "__version__ = '1.0.4'",
          "Update __version__ string to '1.0.4'")

    check("Version : 1.0.4" in init,
          "Version docstring says 1.0.4")

    check("get_stats_df" in init,
          "get_stats_df exported in __all__")

    check("diagnose" in init and "health_score" in init,
          "Core API functions exported")


# ─────────────────────────────────────────────────────────────
# 5. Consistency check — all versions match
# ─────────────────────────────────────────────────────────────
print("\n── Version consistency across files ─────────────────────")
versions_found = {}

if toml:
    if 'version = "1.0.4"' in toml:
        versions_found["pyproject.toml"] = "1.0.4"
    else:
        import re
        m = re.search(r'version = "(\d+\.\d+\.\d+)"', toml)
        versions_found["pyproject.toml"] = m.group(1) if m else "NOT FOUND"

if init:
    import re
    m = re.search(r"__version__ = ['\"](\d+\.\d+\.\d+)['\"]", init)
    versions_found["__init__.py"] = m.group(1) if m else "NOT FOUND"

for filename, ver in versions_found.items():
    check(ver == "1.0.4",
          f"{filename} version = {ver}",
          f"Update to 1.0.4")

all_same = len(set(versions_found.values())) == 1 and "1.0.4" in versions_found.values()
check(all_same,
      "All version strings are consistent: 1.0.4",
      "Make sure pyproject.toml and __init__.py both say 1.0.4")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if not errors:
    print("  ALL CHECKS PASSED — ready to push v1.0.4 to GitHub")
else:
    print(f"  {len(errors)} CHECK(S) FAILED:")
    for e in errors:
        print(f"    • {e}")
    print("\n  Fix the issues above then run this script again.")
print("=" * 55 + "\n")

sys.exit(0 if not errors else 1)
