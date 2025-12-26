"""
Quality Control Operations for Sparse Matrices.

This module provides quality control metrics and statistics for
evaluating and filtering single-cell data.

Implemented Metrics:
    - Per-cell QC: Gene counts, total UMIs
    - Per-gene QC: Detection rate, expression statistics
    - Moment statistics: Mean, variance, skewness, kurtosis

Quality control is essential for:
    - Removing low-quality cells
    - Filtering uninformative genes
    - Detecting technical artifacts
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union, overload

from scl._typing import (
    SparseInput,
    VectorInput,
    ensure_scl_csr,
    ensure_scl_csc,
    ensure_vector,
    get_format,
)

if TYPE_CHECKING:
    import numpy as np
    from scipy import sparse as sp
    from scl.sparse import SclCSR, SclCSC, Array

# Type alias for backward compatibility (use Array with dtype parameter)
RealArray = "Array"
IndexArray = "Array"


# =============================================================================
# Compute QC
# =============================================================================

@overload
def compute_qc(
    mat: "SclCSR",
) -> Tuple["IndexArray", "RealArray"]: ...

@overload
def compute_qc(
    mat: "sp.spmatrix",
) -> Tuple["np.ndarray", "np.ndarray"]: ...


def compute_qc(
    mat: SparseInput,
) -> Tuple[Union["IndexArray", "np.ndarray"], Union["RealArray", "np.ndarray"]]:
    """Compute per-observation quality control metrics.

    Calculates fundamental QC metrics for each observation (row), which
    are used for filtering low-quality cells in single-cell analysis.

    Computed Metrics:
        - n_genes_by_counts: Number of genes with non-zero count
        - total_counts: Total UMI/read count per cell

    These metrics help identify:
        - Empty droplets (low total_counts)
        - Doublets (abnormally high n_genes and total_counts)
        - Damaged cells (low n_genes relative to total_counts)

    Filtering Thresholds (typical for single-cell):
        - Minimum genes: 200-500 genes
        - Minimum UMIs: 1000-5000 counts
        - Maximum genes: 5000-10000 (doublet detection)

    Time Complexity:
        O(nnz) for counting non-zeros and summing values.

    Args:
        mat: Input sparse matrix (observations x features).
            Typically raw count data (cells x genes).

    Returns:
        Tuple of (n_genes_by_counts, total_counts):
            - n_genes_by_counts: Integer array of detected gene counts per cell
            - total_counts: Float array of total UMI counts per cell

    Examples:
        >>> import scl.feature as feat
        >>> from scl import SclCSR
        >>>
        >>> # Compute QC metrics
        >>> n_genes, total_counts = feat.compute_qc(raw_counts)
        >>>
        >>> # Filter cells
        >>> min_genes = 200
        >>> min_counts = 1000
        >>> max_genes = 5000
        >>>
        >>> good_cells = [
        ...     i for i in range(len(n_genes))
        ...     if min_genes <= n_genes[i] <= max_genes
        ...     and total_counts[i] >= min_counts
        ... ]

    Notes:
        - Input should be raw counts (not normalized or log-transformed).
        - For more detailed QC (mitochondrial fraction, etc.), compute
          additional metrics separately.
        - Both CSR and CSC formats are supported; CSR is more efficient
          for row-wise operations.

    See Also:
        detection_rate: Per-feature QC metric.
        standard_moments: Detailed statistical moments.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _compute_qc_scipy(mat)

    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()

    return _compute_qc_scl(csr)


def _compute_qc_scipy(mat):
    """QC using scipy/numpy."""
    import numpy as np

    csr = mat.tocsr()
    n_cells = csr.shape[0]

    # n_genes per cell = number of non-zeros per row
    n_genes = np.diff(csr.indptr)

    # total counts per cell
    total_counts = np.asarray(csr.sum(axis=1)).ravel()

    return n_genes, total_counts


def _compute_qc_scl(mat: "SclCSR") -> Tuple["IndexArray", "RealArray"]:
    """QC for SCL matrices."""
    from scl.sparse import Array

    n_cells = mat.shape[0]

    n_genes = Array(n_cells, dtype='int64')
    total_counts = Array(n_cells, dtype='float64')

    for i in range(n_cells):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]

        n_genes[i] = end - start

        total = 0.0
        for k in range(start, end):
            total += mat._data[k]
        total_counts[i] = total

    return n_genes, total_counts


# =============================================================================
# Standard Moments
# =============================================================================

@overload
def standard_moments(
    mat: "SclCSC",
    ddof: int = 0,
) -> Tuple["RealArray", "RealArray", "RealArray", "RealArray"]: ...

@overload
def standard_moments(
    mat: "sp.csc_matrix",
    ddof: int = 0,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]: ...


def standard_moments(
    mat: SparseInput,
    ddof: int = 0,
) -> Tuple[Union["RealArray", "np.ndarray"], ...]:
    """Compute standard statistical moments for each feature.

    Calculates mean, variance, skewness, and kurtosis for each column
    (feature) of the matrix. These moments characterize the distribution
    of expression values.

    Mathematical Definitions:
        For feature j with n values:

        Mean (1st moment):
            mu = (1/n) * sum(x_i)

        Variance (2nd central moment):
            sigma^2 = (1/(n-ddof)) * sum((x_i - mu)^2)

        Skewness (3rd standardized moment):
            gamma_1 = E[((x - mu) / sigma)^3]
                    = (1/n) * sum(((x_i - mu) / sigma)^3)

        Kurtosis (4th standardized moment, excess):
            gamma_2 = E[((x - mu) / sigma)^4] - 3
                    = (1/n) * sum(((x_i - mu) / sigma)^4) - 3

    Interpretation:
        - Skewness:
            gamma_1 > 0: Right-skewed (long right tail)
            gamma_1 < 0: Left-skewed (long left tail)
            gamma_1 = 0: Symmetric

        - Kurtosis (excess):
            gamma_2 > 0: Heavy tails (leptokurtic)
            gamma_2 < 0: Light tails (platykurtic)
            gamma_2 = 0: Normal distribution tails (mesokurtic)

    Time Complexity:
        O(nnz) for a single pass through the data.

    Args:
        mat: Input sparse matrix (observations x features) in CSC format.
        ddof: Delta degrees of freedom for variance. Default 0.

    Returns:
        Tuple of (means, variances, skewness, kurtosis):
            - means: Mean of each feature
            - variances: Variance of each feature
            - skewness: Skewness of each feature
            - kurtosis: Excess kurtosis of each feature

    Examples:
        >>> import scl.feature as feat
        >>>
        >>> # Get all moments
        >>> means, vars, skew, kurt = feat.standard_moments(mat)
        >>>
        >>> # Find highly skewed genes
        >>> skewed = [i for i, s in enumerate(skew) if abs(s) > 2]
        >>>
        >>> # Find heavy-tailed distributions
        >>> heavy_tail = [i for i, k in enumerate(kurt) if k > 3]

    Notes:
        - Includes implicit zeros in calculations (not just non-zeros).
        - Skewness and kurtosis may be undefined (set to 0) when
          variance is zero.
        - For sparse data with many zeros, distributions are typically
          right-skewed.

    See Also:
        clipped_moments: Moments with clipped values.
        compute_qc: Basic QC metrics.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _standard_moments_scipy(mat, ddof)

    # Native SCL
    csc = ensure_scl_csc(mat)
    csc.materialize()

    return _standard_moments_scl(csc, ddof)


def _standard_moments_scipy(mat, ddof: int):
    """Standard moments using scipy/numpy."""
    import numpy as np
    from scipy import stats

    csc = mat.tocsc()
    n = csc.shape[0]
    n_feat = csc.shape[1]

    means = np.zeros(n_feat)
    variances = np.zeros(n_feat)
    skewness = np.zeros(n_feat)
    kurtosis = np.zeros(n_feat)

    for j in range(n_feat):
        col = csc.getcol(j).toarray().ravel()
        means[j] = np.mean(col)
        variances[j] = np.var(col, ddof=ddof)

        if variances[j] > 0:
            std = np.sqrt(variances[j])
            z = (col - means[j]) / std
            skewness[j] = np.mean(z ** 3)
            kurtosis[j] = np.mean(z ** 4) - 3
        else:
            skewness[j] = 0.0
            kurtosis[j] = 0.0

    return means, variances, skewness, kurtosis


def _standard_moments_scl(mat: "SclCSC", ddof: int):
    """Standard moments for SCL matrices."""
    from scl.sparse import Array

    n = mat.shape[0]
    n_feat = mat.shape[1]

    means = Array(n_feat, dtype='float64')
    variances = Array(n_feat, dtype='float64')
    skewness = Array(n_feat, dtype='float64')
    kurtosis = Array(n_feat, dtype='float64')

    for j in range(n_feat):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        # First pass: sum
        total = 0.0
        for k in range(start, end):
            total += mat._data[k]

        mean = total / n
        means[j] = mean

        # Second pass: variance
        sq_sum = 0.0
        for k in range(start, end):
            sq_sum += (mat._data[k] - mean) ** 2

        # Zero contributions
        n_zeros = n - (end - start)
        sq_sum += n_zeros * mean ** 2

        var = sq_sum / (n - ddof) if n > ddof else 0.0
        variances[j] = var

        # Third/fourth pass: skewness and kurtosis
        if var > 0:
            std = math.sqrt(var)
            m3 = 0.0
            m4 = 0.0

            for k in range(start, end):
                z = (mat._data[k] - mean) / std
                m3 += z ** 3
                m4 += z ** 4

            # Zero contributions
            z_zero = (0 - mean) / std
            m3 += n_zeros * z_zero ** 3
            m4 += n_zeros * z_zero ** 4

            skewness[j] = m3 / n
            kurtosis[j] = m4 / n - 3
        else:
            skewness[j] = 0.0
            kurtosis[j] = 0.0

    return means, variances, skewness, kurtosis


# =============================================================================
# Clipped Moments
# =============================================================================

@overload
def clipped_moments(
    mat: "SclCSC",
    clip_max: Optional["RealArray"] = None,
) -> Tuple["RealArray", "RealArray"]: ...

@overload
def clipped_moments(
    mat: "sp.csc_matrix",
    clip_max: Optional["np.ndarray"] = None,
) -> Tuple["np.ndarray", "np.ndarray"]: ...


def clipped_moments(
    mat: SparseInput,
    clip_max: Optional[VectorInput] = None,
) -> Tuple[Union["RealArray", "np.ndarray"], Union["RealArray", "np.ndarray"]]:
    """Compute mean and variance with optional value clipping.

    Calculates moments after clipping values to a maximum threshold per
    feature. This is useful for robust statistics that are less sensitive
    to outliers.

    Algorithm:
        For each feature j:
            1. Clip values: x_clipped = min(x, clip_max[j])
            2. Compute mean of clipped values
            3. Compute variance of clipped values

    Applications:
        - Robust mean/variance estimation
        - Winsorized statistics
        - Reducing influence of highly expressed outlier cells

    Time Complexity:
        O(nnz) for clipping and computing statistics.

    Args:
        mat: Input sparse matrix (observations x features) in CSC format.
        clip_max: Maximum values per feature for clipping. If None,
            uses 10 * sqrt(mean(x)) as default threshold.
            Array of length n_features.

    Returns:
        Tuple of (clipped_means, clipped_variances):
            - clipped_means: Mean of each feature after clipping
            - clipped_variances: Variance of each feature after clipping

    Examples:
        >>> import scl.feature as feat
        >>> from scl.sparse import Array
        >>>
        >>> # Clip to 99th percentile equivalent
        >>> clip_vals = Array.from_list([10.0] * n_genes, dtype='float64')
        >>> means, vars = feat.clipped_moments(mat, clip_max=clip_vals)
        >>>
        >>> # Use default clipping (based on mean)
        >>> means, vars = feat.clipped_moments(mat)

    Notes:
        - Clipping only affects values above the threshold.
        - Zero values are not clipped.
        - Useful for highly variable gene selection robust to outliers.

    See Also:
        standard_moments: Moments without clipping.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _clipped_moments_scipy(mat, clip_max)

    # Native SCL
    csc = ensure_scl_csc(mat)
    csc.materialize()

    return _clipped_moments_scl(csc, clip_max)


def _clipped_moments_scipy(mat, clip_max):
    """Clipped moments using scipy/numpy."""
    import numpy as np

    csc = mat.tocsc()
    n = csc.shape[0]
    n_feat = csc.shape[1]

    # Default clip_max if not provided
    if clip_max is None:
        means_raw = np.asarray(csc.mean(axis=0)).ravel()
        clip_max = 10 * np.sqrt(np.maximum(means_raw, 1e-10))
    else:
        clip_max = np.asarray(clip_max).ravel()

    clipped_means = np.zeros(n_feat)
    clipped_vars = np.zeros(n_feat)

    for j in range(n_feat):
        col = csc.getcol(j).toarray().ravel()
        clipped = np.minimum(col, clip_max[j])
        clipped_means[j] = np.mean(clipped)
        clipped_vars[j] = np.var(clipped)

    return clipped_means, clipped_vars


def _clipped_moments_scl(mat: "SclCSC", clip_max):
    """Clipped moments for SCL matrices."""
    from scl.sparse import Array

    n = mat.shape[0]
    n_feat = mat.shape[1]

    # Compute default clip_max if not provided
    if clip_max is None:
        # First compute raw means
        raw_means = Array(n_feat, dtype='float64')
        for j in range(n_feat):
            start = mat._indptr[j]
            end = mat._indptr[j + 1]
            total = 0.0
            for k in range(start, end):
                total += mat._data[k]
            raw_means[j] = total / n

        # Set clip_max = 10 * sqrt(mean)
        clip_arr = Array(n_feat, dtype='float64')
        for j in range(n_feat):
            clip_arr[j] = 10 * math.sqrt(max(raw_means[j], 1e-10))
    else:
        clip_arr = ensure_vector(clip_max, size=n_feat)

    clipped_means = Array(n_feat, dtype='float64')
    clipped_vars = Array(n_feat, dtype='float64')

    for j in range(n_feat):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]
        clip_val = clip_arr[j]

        # First pass: clipped sum
        total = 0.0
        for k in range(start, end):
            val = min(mat._data[k], clip_val)
            total += val

        # Zeros are clipped to 0 (which is min(0, clip_val) = 0)
        mean = total / n
        clipped_means[j] = mean

        # Second pass: clipped variance
        sq_sum = 0.0
        for k in range(start, end):
            val = min(mat._data[k], clip_val)
            sq_sum += (val - mean) ** 2

        # Zero contributions
        n_zeros = n - (end - start)
        sq_sum += n_zeros * mean ** 2

        clipped_vars[j] = sq_sum / n

    return clipped_means, clipped_vars


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "compute_qc",
    "standard_moments",
    "clipped_moments",
]
