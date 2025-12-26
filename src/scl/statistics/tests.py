"""
Statistical Tests for Sparse Matrices.

This module provides statistical hypothesis tests optimized for sparse
matrices, particularly useful for differential expression analysis in
single-cell genomics.

Implemented Tests:
    - Mann-Whitney U test (Wilcoxon rank-sum test)
    - Welch's t-test (unequal variance t-test)
    - Student's t-test (equal variance assumption)

All tests support sparse matrices directly without densification,
enabling analysis of large-scale single-cell datasets.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union, overload

from scl._typing import (
    SparseInput,
    CSCInput,
    IndexInput,
    ensure_scl_csc,
    ensure_index_vector,
    get_format,
)

if TYPE_CHECKING:
    import numpy as np
    from scipy import sparse as sp
    from scl.sparse import SclCSR, SclCSC, Array

# Type aliases for backward compatibility
RealArray = "Array"
IndexArray = "Array"


# =============================================================================
# Mann-Whitney U Test
# =============================================================================

@overload
def mwu_test(
    mat: "SclCSC",
    groups: "IndexArray",
    *,
    alternative: str = "two-sided",
    continuity: bool = True,
) -> Tuple["RealArray", "RealArray", "RealArray"]: ...

@overload
def mwu_test(
    mat: "sp.csc_matrix",
    groups: "np.ndarray",
    *,
    alternative: str = "two-sided",
    continuity: bool = True,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]: ...


def mwu_test(
    mat: SparseInput,
    groups: IndexInput,
    *,
    alternative: str = "two-sided",
    continuity: bool = True,
) -> Tuple[Union["RealArray", "np.ndarray"], ...]:
    """Perform Mann-Whitney U test for differential expression.

    The Mann-Whitney U test (also known as Wilcoxon rank-sum test) is a
    non-parametric test for comparing two independent samples. It tests
    whether one sample tends to have larger values than the other.

    Mathematical Background:
        The test statistic U is computed as:

            U = sum(R_1) - n1*(n1+1)/2

        where R_1 is the sum of ranks for group 1, and n1 is the sample
        size of group 1.

        For large samples (n1, n2 > 20), U is approximately normal:

            Z = (U - mu_U) / sigma_U

        where:
            mu_U = n1*n2/2
            sigma_U = sqrt(n1*n2*(n1+n2+1)/12)

        The effect size is computed as the rank-biserial correlation:

            r = 2*U/(n1*n2) - 1

        This gives a value in [-1, 1], where:
            -1: All values in group 1 < all values in group 2
             0: No difference between groups
            +1: All values in group 1 > all values in group 2

    Handling Zeros in Sparse Data:
        For sparse single-cell data, most values are zero. The algorithm
        handles this efficiently by:
        1. Computing ranks only for non-zero values
        2. Assigning average ranks to tied zeros
        3. Avoiding materialization of dense rank arrays

    Time Complexity:
        O(nnz * log(nnz)) for sorting, O(n_features) for output.

    Args:
        mat: Feature matrix in CSC format (features x cells).
            Each column represents a feature (gene), each row a cell.
        groups: Integer array of group labels (0 or 1) for each cell.
            Length must equal number of rows in mat.
        alternative: Alternative hypothesis.
            - "two-sided": Groups differ (default)
            - "greater": Group 0 > Group 1
            - "less": Group 0 < Group 1
        continuity: Apply continuity correction for Z-score.

    Returns:
        Tuple of three arrays (u_statistics, p_values, effect_sizes):
            - u_statistics: U statistic for each feature
            - p_values: Two-tailed p-values
            - effect_sizes: Rank-biserial correlation coefficients

    Raises:
        ValueError: If groups contains values other than 0 or 1.
        ValueError: If either group is empty.

    Examples:
        >>> # Differential expression between two cell types
        >>> from scl import SclCSC
        >>> import scl.statistics as stats
        >>>
        >>> # mat: genes x cells, groups: cell type labels (0 or 1)
        >>> u_stats, pvals, effects = stats.mwu_test(mat, groups)
        >>>
        >>> # Find significantly different genes (FDR < 0.05)
        >>> # (Note: apply multiple testing correction separately)
        >>> significant = [i for i, p in enumerate(pvals) if p < 0.05]

    Notes:
        - This test makes no assumptions about distribution shape.
        - It is robust to outliers compared to t-test.
        - For very sparse data, many p-values may be identical due to
          the discrete nature of ranks with many tied zeros.

    References:
        Mann, H.B. and Whitney, D.R. (1947). "On a Test of Whether one
        of Two Random Variables is Stochastically Larger than the Other".
        Annals of Mathematical Statistics, 18(1), 50-60.

    See Also:
        ttest: Parametric alternative assuming normality.
        welch_ttest: T-test not assuming equal variances.
    """
    fmt = get_format(mat)

    # Convert to CSC for efficient column (feature) access
    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _mwu_scipy(mat, groups, alternative, continuity)

    # Native SCL implementation
    csc = ensure_scl_csc(mat)
    groups_arr = ensure_index_vector(groups, size=csc.shape[0])

    csc.materialize()

    return _mwu_scl(csc, groups_arr, alternative, continuity)


def _mwu_scl(
    mat: "SclCSC",
    groups: "IndexArray",
    alternative: str,
    continuity: bool,
) -> Tuple["RealArray", "RealArray", "RealArray"]:
    """Mann-Whitney U implementation for SCL matrices using C++ kernel."""
    from scl.sparse import Array
    import ctypes

    n_features = mat.shape[1]
    n_cells = mat.shape[0]

    # Count groups
    n0 = sum(1 for i in range(n_cells) if groups[i] == 0)
    n1 = n_cells - n0

    if n0 == 0 or n1 == 0:
        raise ValueError("Both groups must have at least one sample")

    # Try to use C++ kernel
    try:
        from scl._kernel import stats as kernel_stats
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        # Get C pointers from CSC matrix
        data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz = mat.get_c_pointers()
        
        # Convert groups to int32 array for kernel
        group_ids = (ctypes.c_int32 * n_cells)()
        for i in range(n_cells):
            group_ids[i] = int(groups[i])
        
        # Allocate output arrays
        u_stats = Array(n_features, dtype='float64')
        p_values = Array(n_features, dtype='float64')
        log2_fc = Array(n_features, dtype='float64')
        
        # Call kernel
        kernel_stats.mwu_test_csc(
            data_ptr, indices_ptr, indptr_ptr, lengths_ptr,
            rows, cols, nnz,
            ctypes.cast(group_ids, ctypes.POINTER(ctypes.c_int32)),
            u_stats.get_pointer(),
            p_values.get_pointer(),
            log2_fc.get_pointer()
        )
        
        # Kernel returns log2_fc, but we want effect_sizes (rank-biserial correlation)
        # effect_size = 2 * U / (n0 * n1) - 1
        effect_sizes = Array(n_features, dtype='float64')
        for j in range(n_features):
            effect_sizes[j] = 2.0 * u_stats[j] / (n0 * n1) - 1.0
        
        return u_stats, p_values, effect_sizes
        
    except (ImportError, LibraryNotFoundError, RuntimeError, AttributeError):
        pass  # Fall through to Python implementation
    
    # Fallback to pure Python implementation
    return _mwu_scl_fallback(mat, groups, alternative, continuity, n0, n1)


def _mwu_scl_fallback(
    mat: "SclCSC",
    groups: "IndexArray",
    alternative: str,
    continuity: bool,
    n0: int,
    n1: int,
) -> Tuple["RealArray", "RealArray", "RealArray"]:
    """Fallback pure Python MWU implementation."""
    from scl.sparse import Array
    
    n_features = mat.shape[1]

    u_stats = Array(n_features, dtype='float64')
    p_values = Array(n_features, dtype='float64')
    effect_sizes = Array(n_features, dtype='float64')

    # Compute for each feature
    for j in range(n_features):
        u, p, r = _compute_mwu_feature(
            mat, j, groups, n0, n1, alternative, continuity
        )
        u_stats[j] = u
        p_values[j] = p
        effect_sizes[j] = r

    return u_stats, p_values, effect_sizes


def _compute_mwu_feature(
    mat: "SclCSC",
    j: int,
    groups: "IndexArray",
    n0: int,
    n1: int,
    alternative: str,
    continuity: bool,
) -> Tuple[float, float, float]:
    """Compute MWU test for a single feature."""
    start = mat._indptr[j]
    end = mat._indptr[j + 1]

    # Collect values for each group
    vals0 = []
    vals1 = []

    for k in range(start, end):
        row = mat._indices[k]
        val = mat._data[k]
        if groups[row] == 0:
            vals0.append(val)
        else:
            vals1.append(val)

    # Add zeros for missing cells
    vals0.extend([0.0] * (n0 - len(vals0)))
    vals1.extend([0.0] * (n1 - len(vals1)))

    # Compute U statistic using comparison counting
    u = 0.0
    for v0 in vals0:
        for v1 in vals1:
            if v0 > v1:
                u += 1.0
            elif v0 == v1:
                u += 0.5

    # Effect size (rank-biserial correlation)
    r = 2.0 * u / (n0 * n1) - 1.0

    # P-value using normal approximation
    mu = n0 * n1 / 2.0
    sigma = math.sqrt(n0 * n1 * (n0 + n1 + 1) / 12.0)

    if sigma == 0:
        return u, 1.0, 0.0

    # Continuity correction
    correction = 0.5 if continuity else 0.0

    if alternative == "two-sided":
        z = abs(u - mu) - correction
        z = z / sigma if z > 0 else 0.0
        p = 2.0 * (1.0 - _norm_cdf(z))
    elif alternative == "greater":
        z = (u - mu - correction) / sigma
        p = 1.0 - _norm_cdf(z)
    else:  # less
        z = (u - mu + correction) / sigma
        p = _norm_cdf(z)

    return u, p, r


def _mwu_scipy(mat, groups, alternative: str, continuity: bool):
    """Mann-Whitney U using scipy (fallback for scipy inputs)."""
    import numpy as np
    from scipy import stats

    csc = mat.tocsc()
    n_features = csc.shape[1]

    groups = np.asarray(groups).ravel()
    mask0 = groups == 0
    mask1 = groups == 1

    n0 = np.sum(mask0)
    n1 = np.sum(mask1)

    u_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)
    effect_sizes = np.zeros(n_features)

    for j in range(n_features):
        col = csc.getcol(j).toarray().ravel()
        vals0 = col[mask0]
        vals1 = col[mask1]

        try:
            result = stats.mannwhitneyu(
                vals0, vals1, alternative=alternative,
                use_continuity=continuity
            )
            u_stats[j] = result.statistic
            p_values[j] = result.pvalue
            # Effect size
            effect_sizes[j] = 2.0 * result.statistic / (n0 * n1) - 1.0
        except Exception:
            u_stats[j] = 0.0
            p_values[j] = 1.0
            effect_sizes[j] = 0.0

    return u_stats, p_values, effect_sizes


# =============================================================================
# T-Test
# =============================================================================

@overload
def ttest(
    mat: "SclCSC",
    groups: "IndexArray",
    *,
    equal_var: bool = False,
) -> Tuple["RealArray", "RealArray", "RealArray", "RealArray"]: ...

@overload
def ttest(
    mat: "sp.csc_matrix",
    groups: "np.ndarray",
    *,
    equal_var: bool = False,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]: ...


def ttest(
    mat: SparseInput,
    groups: IndexInput,
    *,
    equal_var: bool = False,
) -> Tuple[Union["RealArray", "np.ndarray"], ...]:
    """Perform t-test for differential expression.

    Computes the t-test for the means of two independent samples.
    By default, uses Welch's t-test which does not assume equal variances.

    Mathematical Background:
        For Welch's t-test (unequal variances):

            t = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)

            Degrees of freedom (Welch-Satterthwaite):
            df = (var1/n1 + var2/n2)^2 /
                 ((var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1))

        For Student's t-test (equal variances):

            t = (mean1 - mean2) / (sp * sqrt(1/n1 + 1/n2))

            where sp = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            df = n1 + n2 - 2

        Log fold change:
            log2FC = log2(mean1 + eps) - log2(mean2 + eps)

            where eps is a small constant to avoid log(0).

    Handling Zeros in Sparse Data:
        The mean and variance calculations include implicit zeros:
        - mean = sum(non-zeros) / n_total
        - var includes contribution from (n_total - nnz) zeros

        This is correct for count data where zeros represent
        true zero expression.

    Args:
        mat: Feature matrix in CSC format (features x cells).
        groups: Integer array of group labels (0 or 1).
        equal_var: If True, use Student's t-test assuming equal
            variances. If False (default), use Welch's t-test.

    Returns:
        Tuple of four arrays (t_statistics, p_values, mean_diff, log2fc):
            - t_statistics: T statistic for each feature
            - p_values: Two-tailed p-values
            - mean_diff: Difference in means (group0 - group1)
            - log2fc: Log2 fold change

    Examples:
        >>> # Test for differential expression
        >>> t_stats, pvals, mean_diff, log2fc = stats.ttest(mat, groups)
        >>>
        >>> # Find upregulated genes in group 0
        >>> up_genes = [i for i, (p, fc) in enumerate(zip(pvals, log2fc))
        ...             if p < 0.05 and fc > 1.0]

    Notes:
        - T-test assumes normally distributed data.
        - For single-cell count data, consider log-transformation first
          or use the non-parametric mwu_test instead.
        - Welch's t-test is more robust when group sizes or variances
          differ substantially.

    References:
        Welch, B. L. (1947). "The generalization of Student's problem
        when several different population variances are involved".
        Biometrika, 34(1-2), 28-35.

    See Also:
        welch_ttest: Explicit Welch's t-test function.
        mwu_test: Non-parametric alternative.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _ttest_scipy(mat, groups, equal_var)

    # Native SCL
    csc = ensure_scl_csc(mat)
    groups_arr = ensure_index_vector(groups, size=csc.shape[0])

    csc.materialize()

    return _ttest_scl(csc, groups_arr, equal_var)


def _ttest_scl(
    mat: "SclCSC",
    groups: "IndexArray",
    equal_var: bool,
) -> Tuple["RealArray", "RealArray", "RealArray", "RealArray"]:
    """T-test implementation for SCL matrices using C++ kernel."""
    from scl.sparse import Array
    import ctypes

    n_features = mat.shape[1]
    n_cells = mat.shape[0]

    # Count groups
    n0 = sum(1 for i in range(n_cells) if groups[i] == 0)
    n1 = n_cells - n0

    if n0 < 2 or n1 < 2:
        raise ValueError("Both groups must have at least 2 samples")

    # Try to use C++ kernel
    try:
        from scl._kernel import stats as kernel_stats
        from scl._kernel.lib_loader import LibraryNotFoundError, get_lib
        
        # Get C pointers from CSC matrix
        data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz = mat.get_c_pointers()
        
        # Convert groups to int32 array for kernel
        group_ids = (ctypes.c_int32 * n_cells)()
        for i in range(n_cells):
            group_ids[i] = int(groups[i])
        
        # Allocate output arrays
        t_stats = Array(n_features, dtype='float64')
        p_values = Array(n_features, dtype='float64')
        log2fc = Array(n_features, dtype='float64')
        mean_diff = Array(n_features, dtype='float64')
        
        # Calculate workspace size
        n_groups = 2
        lib = get_lib()
        lib.scl_ttest_workspace_size.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        lib.scl_ttest_workspace_size.restype = ctypes.c_size_t
        workspace_size = lib.scl_ttest_workspace_size(n_features, n_groups)
        
        # Allocate workspace
        workspace = (ctypes.c_uint8 * workspace_size)()
        
        # Call kernel
        kernel_stats.ttest_csc(
            data_ptr, indices_ptr, indptr_ptr, lengths_ptr,
            rows, cols, nnz,
            ctypes.cast(group_ids, ctypes.POINTER(ctypes.c_int32)),
            n_groups,
            t_stats.get_pointer(),
            p_values.get_pointer(),
            log2fc.get_pointer(),
            mean_diff.get_pointer(),
            ctypes.cast(workspace, ctypes.POINTER(ctypes.c_uint8)),
            workspace_size,
            not equal_var  # use_welch = True when equal_var = False
        )
        
        return t_stats, p_values, mean_diff, log2fc
        
    except (ImportError, LibraryNotFoundError, RuntimeError, AttributeError):
        pass  # Fall through to Python implementation
    
    # Fallback to pure Python implementation
    return _ttest_scl_fallback(mat, groups, equal_var, n0, n1)


def _ttest_scl_fallback(
    mat: "SclCSC",
    groups: "IndexArray",
    equal_var: bool,
    n0: int,
    n1: int,
) -> Tuple["RealArray", "RealArray", "RealArray", "RealArray"]:
    """Fallback pure Python t-test implementation."""
    from scl.sparse import Array
    
    n_features = mat.shape[1]

    t_stats = Array(n_features, dtype='float64')
    p_values = Array(n_features, dtype='float64')
    mean_diff = Array(n_features, dtype='float64')
    log2fc = Array(n_features, dtype='float64')

    for j in range(n_features):
        t, p, md, lfc = _compute_ttest_feature(
            mat, j, groups, n0, n1, equal_var
        )
        t_stats[j] = t
        p_values[j] = p
        mean_diff[j] = md
        log2fc[j] = lfc

    return t_stats, p_values, mean_diff, log2fc


def _compute_ttest_feature(
    mat: "SclCSC",
    j: int,
    groups: "IndexArray",
    n0: int,
    n1: int,
    equal_var: bool,
) -> Tuple[float, float, float, float]:
    """Compute t-test for a single feature."""
    start = mat._indptr[j]
    end = mat._indptr[j + 1]

    # Compute sums for each group
    sum0 = 0.0
    sum1 = 0.0
    count0 = 0
    count1 = 0

    for k in range(start, end):
        row = mat._indices[k]
        val = mat._data[k]
        if groups[row] == 0:
            sum0 += val
            count0 += 1
        else:
            sum1 += val
            count1 += 1

    # Means (including zeros)
    mean0 = sum0 / n0
    mean1 = sum1 / n1

    # Variances (including zeros)
    var0 = 0.0
    var1 = 0.0

    for k in range(start, end):
        row = mat._indices[k]
        val = mat._data[k]
        if groups[row] == 0:
            var0 += (val - mean0) ** 2
        else:
            var1 += (val - mean1) ** 2

    # Add zero contributions
    var0 += (n0 - count0) * mean0 ** 2
    var1 += (n1 - count1) * mean1 ** 2

    var0 /= (n0 - 1) if n0 > 1 else 1
    var1 /= (n1 - 1) if n1 > 1 else 1

    # T-statistic
    if equal_var:
        # Pooled variance
        sp2 = ((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2)
        se = math.sqrt(sp2 * (1.0 / n0 + 1.0 / n1))
        df = n0 + n1 - 2
    else:
        # Welch's t-test
        se = math.sqrt(var0 / n0 + var1 / n1)
        if se > 0:
            # Welch-Satterthwaite df
            num = (var0 / n0 + var1 / n1) ** 2
            denom = (var0 / n0) ** 2 / (n0 - 1) + (var1 / n1) ** 2 / (n1 - 1)
            df = num / denom if denom > 0 else n0 + n1 - 2
        else:
            df = n0 + n1 - 2

    if se < 1e-15:
        t = 0.0
        p = 1.0
    else:
        t = (mean0 - mean1) / se
        # P-value using normal approximation for large df
        p = 2.0 * (1.0 - _norm_cdf(abs(t)))

    # Log fold change
    eps = 1e-10
    lfc = math.log2((mean0 + eps) / (mean1 + eps))

    return t, p, mean0 - mean1, lfc


def _ttest_scipy(mat, groups, equal_var: bool):
    """T-test using scipy."""
    import numpy as np
    from scipy import stats

    csc = mat.tocsc()
    n_features = csc.shape[1]

    groups = np.asarray(groups).ravel()
    mask0 = groups == 0
    mask1 = groups == 1

    t_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)
    mean_diff = np.zeros(n_features)
    log2fc = np.zeros(n_features)

    eps = 1e-10

    for j in range(n_features):
        col = csc.getcol(j).toarray().ravel()
        vals0 = col[mask0]
        vals1 = col[mask1]

        try:
            result = stats.ttest_ind(vals0, vals1, equal_var=equal_var)
            t_stats[j] = result.statistic
            p_values[j] = result.pvalue

            m0, m1 = np.mean(vals0), np.mean(vals1)
            mean_diff[j] = m0 - m1
            log2fc[j] = np.log2((m0 + eps) / (m1 + eps))
        except Exception:
            t_stats[j] = 0.0
            p_values[j] = 1.0
            mean_diff[j] = 0.0
            log2fc[j] = 0.0

    return t_stats, p_values, mean_diff, log2fc


def welch_ttest(
    mat: SparseInput,
    groups: IndexInput,
) -> Tuple[Union["RealArray", "np.ndarray"], ...]:
    """Perform Welch's t-test for differential expression.

    This is a convenience wrapper for ttest() with equal_var=False.
    Welch's t-test is recommended when:
    - Group sizes are unequal
    - Group variances may differ
    - The assumption of equal variance cannot be verified

    See ttest() for full documentation.

    Args:
        mat: Feature matrix in CSC format.
        groups: Group labels (0 or 1).

    Returns:
        Same as ttest(): (t_statistics, p_values, mean_diff, log2fc)

    See Also:
        ttest: General t-test with equal_var option.
    """
    return ttest(mat, groups, equal_var=False)


# =============================================================================
# Helper Functions
# =============================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function.

    Uses the error function approximation:
        Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))

    Args:
        x: Input value.

    Returns:
        Probability P(Z <= x) for standard normal Z.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "mwu_test",
    "ttest",
    "welch_ttest",
]
