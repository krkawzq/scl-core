"""
SCL Statistics Module.

This module provides statistical operations for sparse matrices, including:

    - Descriptive statistics (sum, mean, variance, standard deviation)
    - Statistical tests (Mann-Whitney U, t-test)
    - Group-wise statistics

All functions support multiple input formats through the overload dispatch
system, including native SCL types, scipy sparse matrices, and numpy arrays.

Example:
    >>> import scl.statistics as stats
    >>> from scl import SclCSR
    >>>
    >>> mat = SclCSR.from_arrays([1, 2, 3], [0, 1, 0], [0, 2, 3], (2, 3))
    >>> row_means = stats.mean(mat, axis=1)
    >>> print(row_means.to_list())  # [1.0, 1.0]

The module also works with scipy:
    >>> import scipy.sparse as sp
    >>> scipy_mat = sp.random(100, 50, density=0.1)
    >>> col_vars = stats.var(scipy_mat, axis=0)
"""

from scl.statistics.descriptive import (
    sum,
    mean,
    var,
    std,
    min,
    max,
    nnz_count,
)

from scl.statistics.tests import (
    mwu_test,
    ttest,
    welch_ttest,
)

from scl.statistics.group import (
    group_mean,
    group_var,
    group_stats,
    count_group_sizes,
)

from scl.statistics.distance import (
    mmd,
)

__all__ = [
    # Descriptive
    "sum",
    "mean",
    "var",
    "std",
    "min",
    "max",
    "nnz_count",
    # Tests
    "mwu_test",
    "ttest",
    "welch_ttest",
    # Group
    "group_mean",
    "group_var",
    "group_stats",
    "count_group_sizes",
    # Distance
    "mmd",
]
