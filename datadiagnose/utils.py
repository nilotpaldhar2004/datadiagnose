"""
utils.py
========
Pure math and type-detection helper functions.

These are all internal helpers — they start with an underscore
to signal that you should not call them directly from outside
the package. They are used by detectors.py and core.py.

No external libraries are used here — only Python's built-in
math, statistics, and collections modules.

Author  : Nilotpal Dhar
License : MIT
"""

import math
from collections import Counter


# ──────────────────────────────────────────────────────────────
# TYPE DETECTION
# ──────────────────────────────────────────────────────────────

def is_missing(value):
    """
    Return True if a value should be considered missing.

    Missing values in Python can appear as:
      - None            (the Python null value)
      - ""              (empty string)
      - float('nan')    (Not a Number — used by pandas for missing numerics)
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == '':
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def to_numeric_list(values):
    """
    Try to convert a list of values to a list of floats,
    skipping any missing values.

    Returns the list of floats if ALL non-missing values are numeric.
    Returns None if ANY non-missing value cannot be converted to float.
    This distinguishes truly numeric columns from text columns.

    Examples
    --------
    to_numeric_list([1, 2, None, 4])   → [1.0, 2.0, 4.0]
    to_numeric_list(['a', 'b', 'c'])   → None
    to_numeric_list([1, 2, 'abc'])     → None
    """
    result = []
    for v in values:
        if is_missing(v):
            continue
        try:
            result.append(float(v))
        except (ValueError, TypeError):
            return None   # found a non-numeric value → not a numeric column
    return result if len(result) >= 2 else None


def is_categorical(values, max_unique=50, unique_ratio_threshold=0.05):
    """
    Heuristic: decide whether a column is categorical.

    A column is considered categorical if:
      - The number of unique non-missing values is ≤ max_unique, OR
      - The ratio of unique values to total non-missing values is ≤ threshold

    This catches both small-vocabulary columns (e.g. gender: Male/Female)
    and moderately sized ones (e.g. city names in a region).

    Parameters
    ----------
    values                 : list  — raw column values
    max_unique             : int   — maximum unique values to still call categorical
    unique_ratio_threshold : float — max ratio of unique/total to call categorical
    """
    non_null = [v for v in values if not is_missing(v)]
    if not non_null:
        return False
    unique = set(str(v) for v in non_null)
    ratio  = len(unique) / len(non_null)
    return len(unique) <= max_unique or ratio <= unique_ratio_threshold


def non_null_values(values):
    """Return a list of all non-missing values from a column."""
    return [v for v in values if not is_missing(v)]


# ──────────────────────────────────────────────────────────────
# BASIC STATISTICS  (reimplemented without external libraries)
# ──────────────────────────────────────────────────────────────

def mean(nums):
    """
    Arithmetic mean of a list of numbers.

    mean([1, 2, 3, 4, 5]) → 3.0

    Why not use statistics.mean()? No reason we can't, but
    implementing it ourselves is cleaner and avoids the import.
    """
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def median(nums):
    """
    Median (middle value) of a sorted list.

    For an odd-length list  : returns the middle element.
    For an even-length list : returns the average of the two middle elements.

    median([1, 3, 5])       → 3
    median([1, 2, 3, 4])    → 2.5

    The median is preferred over the mean for imputation because
    it is not affected by outliers.
    """
    if not nums:
        return 0.0
    s   = sorted(nums)
    n   = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def std(nums):
    """
    Sample standard deviation (divides by n-1, not n).

    We use the sample formula (Bessel's correction, n-1) because
    our data is a sample of a larger population — not the entire
    population. This gives a less biased estimate of the true
    population standard deviation.

    std([2, 4, 4, 4, 5, 5, 7, 9]) → 2.138...
    """
    if len(nums) < 2:
        return 0.0
    m   = mean(nums)
    var = sum((x - m) ** 2 for x in nums) / (len(nums) - 1)
    return math.sqrt(var)


def variance(nums):
    """Sample variance (square of sample standard deviation)."""
    s = std(nums)
    return s * s


def skewness(nums):
    """
    Pearson's moment coefficient of skewness.

    Formula: G1 = [n / ((n-1)(n-2))] * Σ[(xi - mean) / std]^3

    This is the same formula used by pandas, numpy, and most
    statistics textbooks. It requires at least 3 data points.

    Interpretation:
        ≈ 0       → symmetric distribution (ideal)
        > 0       → right-skewed (long tail on the right)
        < 0       → left-skewed (long tail on the left)
        |G1| > 1  → significantly skewed
        |G1| > 2  → extremely skewed

    skewness([1, 1, 1, 1, 10])  → strongly positive (right tail)
    """
    n = len(nums)
    if n < 3:
        return 0.0
    m = mean(nums)
    s = std(nums)
    if s == 0:
        return 0.0   # constant column — no skewness meaningful
    correction = n / ((n - 1) * (n - 2))
    return correction * sum(((x - m) / s) ** 3 for x in nums)


def percentile(nums, p):
    """
    Return the p-th percentile of a sorted list (0 ≤ p ≤ 100).

    Uses nearest-rank method — simple, no interpolation.

    percentile([1,2,3,4,5,6,7,8,9,10], 25) → 3   (Q1)
    percentile([1,2,3,4,5,6,7,8,9,10], 75) → 8   (Q3)
    """
    if not nums:
        return 0.0
    s   = sorted(nums)
    n   = len(s)
    idx = max(0, min(n - 1, int(math.ceil(p / 100.0 * n)) - 1))
    return s[idx]


def iqr_bounds(nums):
    """
    Compute the IQR outlier detection fence values.

    The IQR (Interquartile Range) method was established by
    statistician John Tukey in 1977. It works by:
        1. Finding Q1 (25th percentile) and Q3 (75th percentile)
        2. Computing IQR = Q3 - Q1
        3. Setting lower fence = Q1 - 1.5 * IQR
        4. Setting upper fence = Q3 + 1.5 * IQR

    Any value outside [lower_fence, upper_fence] is an outlier.
    The multiplier 1.5 is Tukey's standard — 3.0 is used for
    detecting only the most extreme outliers.

    Returns
    -------
    (lower_fence, upper_fence, q1, q3, iqr_value)
    """
    q1  = percentile(nums, 25)
    q3  = percentile(nums, 75)
    iqr = q3 - q1
    return (
        q1 - 1.5 * iqr,   # lower fence
        q3 + 1.5 * iqr,   # upper fence
        q1,
        q3,
        iqr,
    )


def zscore_outliers(nums, threshold=3.0):
    """
    Detect outliers using the Z-score method.

    The Z-score of a value measures how many standard deviations
    it is away from the mean:
        Z = (x - mean) / std

    A Z-score above 3.0 (or below -3.0) is considered an outlier
    in most practical applications. The threshold 3.0 corresponds
    to roughly 99.7% of values in a normal distribution.

    Returns list of outlier values (not indices).
    """
    if len(nums) < 3:
        return []
    m = mean(nums)
    s = std(nums)
    if s == 0:
        return []
    return [x for x in nums if abs((x - m) / s) > threshold]


def pearson_correlation(x, y):
    """
    Compute the Pearson correlation coefficient between two lists.

    Returns a value between -1 and +1:
        +1 → perfect positive linear relationship
         0 → no linear relationship
        -1 → perfect negative linear relationship

    Returns None if the correlation cannot be computed
    (e.g. too few values, or zero variance in either list).

    Used by the data leakage detector to catch columns that
    are suspiciously perfectly correlated with the target.

    Formula
    -------
        r = Σ[(xi - mx)(yi - my)] / (sqrt(Σ(xi-mx)²) * sqrt(Σ(yi-my)²))
    """
    # Only use rows where both values are non-missing and numeric
    pairs = [
        (float(xi), float(yi))
        for xi, yi in zip(x, y)
        if not is_missing(xi) and not is_missing(yi)
    ]
    n = len(pairs)
    if n < 3:
        return None

    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n

    numerator = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    denom_x   = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    denom_y   = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))

    if denom_x == 0 or denom_y == 0:
        return None   # one variable is constant — correlation undefined

    return numerator / (denom_x * denom_y)


# ──────────────────────────────────────────────────────────────
# COLUMN NAME PATTERN MATCHING
# ──────────────────────────────────────────────────────────────

# Keywords that suggest a column contains datetime data
DATETIME_KEYWORDS = {
    'date', 'time', 'year', 'month', 'day',
    'created', 'updated', 'timestamp', 'datetime', 'dob', 'birth',
}

# Keywords that suggest a column contains free text
TEXT_KEYWORDS = {
    'name', 'description', 'desc', 'comment', 'text',
    'review', 'title', 'message', 'note', 'feedback', 'summary',
}

# Keywords that suggest a column contains geographic data
GEO_KEYWORDS = {
    'lat', 'lon', 'latitude', 'longitude', 'location',
    'city', 'zip', 'postal', 'address', 'country', 'region', 'state',
}

# Keywords that suggest a column may leak target information
LEAKY_KEYWORDS = {
    'result', 'outcome', 'label', 'target', 'output',
    'prediction', 'pred', 'answer', 'final', 'ground_truth',
    'actual', 'true_label', 'y_true',
}


def column_keywords(col_name):
    """
    Normalise a column name and return its lowercase words.

    Handles snake_case, camelCase, and hyphen-separated names.
    'purchase_date' → {'purchase', 'date'}
    'createdAt'     → {'createdat'}   (simple lower — good enough)
    """
    # Replace common separators with space, lower, split
    normalised = col_name.lower().replace('_', ' ').replace('-', ' ')
    return set(normalised.split())


def matches_any_keyword(col_name, keyword_set):
    """
    Return True if the column name contains any word from keyword_set.

    Also checks if the full normalised name CONTAINS any keyword
    as a substring (catches things like 'timestamp' inside 'event_timestamp').
    """
    col_lower = col_name.lower()
    words     = column_keywords(col_name)
    return bool(words & keyword_set) or any(kw in col_lower for kw in keyword_set)
