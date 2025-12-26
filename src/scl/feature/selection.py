"""
Feature Selection Operations for Sparse Matrices.

This module provides methods for identifying informative features in
high-dimensional data, particularly for single-cell RNA-seq analysis.

Implemented Methods:
    - Highly Variable Genes (HVG) selection
    - Dispersion-based ranking
    - Detection rate calculation

Feature selection helps:
    - Reduce dimensionality
    - Focus on biologically variable genes
    - Improve clustering and trajectory inference
    - Reduce computational burden
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union, overload

from scl._typing import (
    SparseInput,
    CSCInput,
    ensure_scl_csr,
    ensure_scl_csc,
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
# Highly Variable Genes
# =============================================================================

@overload
def highly_variable(
    mat: "SclCSR",
    n_top: int = 2000,
    *,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_dispersion: float = 0.5,
) -> "IndexArray": ...

@overload
def highly_variable(
    mat: "sp.spmatrix",
    n_top: int = 2000,
    *,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_dispersion: float = 0.5,
) -> "np.ndarray": ...


def highly_variable(
    mat: SparseInput,
    n_top: int = 2000,
    *,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_dispersion: float = 0.5,
) -> Union["IndexArray", "np.ndarray"]:
    """Select highly variable genes (features).

    Identifies genes with high variance relative to their mean expression,
    following the approach used in Seurat and Scanpy. This is the standard
    method for feature selection in single-cell RNA-seq analysis.

    Algorithm (Seurat v3 style):
        1. Compute mean and variance for each gene
        2. Fit a loess/local regression of log(variance) on log(mean)
        3. Compute standardized variance (normalized dispersion)
        4. Filter by mean and dispersion thresholds
        5. Select top genes by normalized dispersion

    Simplified Algorithm (used here):
        1. Compute mean (mu) and variance (sigma^2) per gene
        2. Compute coefficient of variation: CV = sqrt(var) / mean
        3. Compute dispersion: disp = var / mean (index of dispersion)
        4. Filter by mean and dispersion bounds
        5. Sort by dispersion and return top n_top indices

    Biological Rationale:
        - Highly variable genes capture biological variation
        - Lowly expressed genes have high technical noise
        - Very highly expressed genes (housekeeping) are often not informative
        - Dispersion normalizes variance by mean, identifying genes that vary
          more than expected given their expression level

    Time Complexity:
        O(nnz) for computing statistics, O(n_features * log(n_features))
        for sorting.

    Args:
        mat: Input sparse matrix (observations x features).
            Typically log-normalized expression data.
        n_top: Number of top variable genes to return. Default 2000.
        min_mean: Minimum mean expression threshold. Genes with mean
            below this are excluded. Default 0.0125.
        max_mean: Maximum mean expression threshold. Genes with mean
            above this are excluded. Default 3.0.
        min_dispersion: Minimum dispersion threshold. Default 0.5.

    Returns:
        Array of indices (column indices in mat) of highly variable genes,
        sorted by dispersion (most variable first).

    Examples:
        >>> import scl.feature as feat
        >>> from scl import SclCSR
        >>>
        >>> # Select top 2000 highly variable genes
        >>> hvg_idx = feat.highly_variable(log_normalized, n_top=2000)
        >>>
        >>> # Subset matrix to HVGs
        >>> # (requires indexing support in your sparse matrix implementation)
        >>> # hvg_data = log_normalized[:, hvg_idx]
        >>>
        >>> # Custom thresholds for different data
        >>> hvg_idx = feat.highly_variable(
        ...     mat, n_top=3000,
        ...     min_mean=0.01, max_mean=5.0,
        ...     min_dispersion=0.3
        ... )

    Notes:
        - Input should typically be log-normalized data (log1p(counts/total * scale))
        - The number of returned genes may be less than n_top if fewer genes
          pass the thresholds
        - Different tools use slightly different HVG methods (Seurat v2 vs v3,
          Scanpy cell_ranger flavor, etc.)
        - For raw counts, consider using a mean-variance model (e.g., negative
          binomial) instead

    References:
        Stuart, T., et al. (2019). "Comprehensive Integration of Single-Cell
        Data". Cell, 177(7), 1888-1902.

    See Also:
        dispersion: Compute dispersion values for all genes.
        detection_rate: Compute fraction of cells expressing each gene.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _highly_variable_scipy(mat, n_top, min_mean, max_mean, min_dispersion)

    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()

    return _highly_variable_scl(csr, n_top, min_mean, max_mean, min_dispersion)


def _highly_variable_scipy(mat, n_top: int, min_mean: float, max_mean: float,
                           min_dispersion: float):
    """HVG selection using scipy/numpy."""
    import numpy as np

    csc = mat.tocsc()
    n_cells = csc.shape[0]
    n_genes = csc.shape[1]

    # Compute per-gene statistics
    means = np.asarray(csc.mean(axis=0)).ravel()

    # Compute variance
    # var = E[X^2] - E[X]^2
    sq_mean = np.asarray(csc.multiply(csc).mean(axis=0)).ravel()
    variances = sq_mean - means ** 2

    # Dispersion = var / mean (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        dispersions = np.where(means > 0, variances / means, 0.0)

    # Apply filters
    mask = (
        (means >= min_mean) &
        (means <= max_mean) &
        (dispersions >= min_dispersion)
    )

    # Get passing indices
    passing_indices = np.where(mask)[0]

    if len(passing_indices) == 0:
        return np.array([], dtype=np.int64)

    # Sort by dispersion
    sorted_order = np.argsort(-dispersions[passing_indices])
    sorted_indices = passing_indices[sorted_order]

    # Return top n_top
    return sorted_indices[:n_top]


def _highly_variable_scl(mat: "SclCSR", n_top: int, min_mean: float,
                         max_mean: float, min_dispersion: float) -> "IndexArray":
    """HVG selection for SCL matrices using C++ kernel."""
    from scl.sparse import Array
    from scl._kernel import hvg as kernel_hvg, sparse as kernel_sparse

    csc = mat.to_csc()
    n_cells = csc.shape[0]
    n_genes = csc.shape[1]

    # First compute means and dispersions using kernel
    means = Array(n_genes, dtype='float64')

    data_ptr = csc.data.get_pointer()
    indices_ptr = csc.indices.get_pointer()
    indptr_ptr = csc.indptr.get_pointer()
    means_ptr = means.get_pointer()

    kernel_sparse.primary_means_csc(
        data_ptr, indices_ptr, indptr_ptr,
        n_cells, n_genes, means_ptr
    )

    # Compute dispersion
    dispersions = Array(n_genes, dtype='float64')

    for j in range(n_genes):
        start = csc._indptr[j]
        end = csc._indptr[j + 1]

        # Sum of squares
        sq_total = 0.0
        for k in range(start, end):
            val = csc._data[k]
            sq_total += val * val

        mean = means[j]
        if mean > 0:
            var = sq_total / n_cells - mean ** 2
            dispersions[j] = var / mean
        else:
            dispersions[j] = 0.0

    # Filter and collect candidates
    candidates = []  # (index, dispersion)

    for j in range(n_genes):
        m = means[j]
        d = dispersions[j]

        if m >= min_mean and m <= max_mean and d >= min_dispersion:
            candidates.append((j, d))

    # Sort by dispersion descending
    candidates.sort(key=lambda x: -x[1])

    # Return top n_top indices
    n_return = min(n_top, len(candidates))
    result = Array(n_return, dtype='int64')
    for i in range(n_return):
        result[i] = candidates[i][0]

    return result


# =============================================================================
# Dispersion
# =============================================================================

@overload
def dispersion(
    mat: "SclCSR",
    axis: int = 0,
) -> "RealArray": ...

@overload
def dispersion(
    mat: "sp.spmatrix",
    axis: int = 0,
) -> "np.ndarray": ...


def dispersion(
    mat: SparseInput,
    axis: int = 0,
) -> Union["RealArray", "np.ndarray"]:
    """Compute dispersion (index of dispersion) for each feature.

    The dispersion (also called index of dispersion or Fano factor) measures
    the variance relative to the mean. It's a key statistic for identifying
    highly variable features.

    Mathematical Definition:
        dispersion = variance / mean = sigma^2 / mu

    Interpretation:
        - dispersion = 1: Poisson distribution (variance equals mean)
        - dispersion > 1: Overdispersed (more variable than Poisson)
        - dispersion < 1: Underdispersed (less variable than Poisson)

        In single-cell data, genes with high dispersion are often
        biologically interesting.

    Time Complexity:
        O(nnz) for computing mean and variance.

    Args:
        mat: Input sparse matrix.
        axis: Axis along which to compute dispersion.
            - 0: Dispersion per column (feature)
            - 1: Dispersion per row (observation)

    Returns:
        Array of dispersion values.

    Examples:
        >>> import scl.feature as feat
        >>>
        >>> # Get dispersion for all genes
        >>> gene_disp = feat.dispersion(counts, axis=0)
        >>>
        >>> # Find overdispersed genes
        >>> overdispersed = [i for i, d in enumerate(gene_disp) if d > 2.0]

    Notes:
        - Dispersion is undefined (set to 0) when mean is zero.
        - For count data, Poisson assumption gives dispersion = 1.
        - Single-cell data typically shows overdispersion due to
          biological variation.

    See Also:
        highly_variable: Use dispersion for feature selection.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _dispersion_scipy(mat, axis)

    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()

    return _dispersion_scl(csr, axis)


def _dispersion_scipy(mat, axis: int):
    """Dispersion using scipy/numpy."""
    import numpy as np

    if axis == 0:
        csc = mat.tocsc()
        n = csc.shape[0]
        n_feat = csc.shape[1]
    else:
        csc = mat.tocsr()
        n = csc.shape[1]
        n_feat = csc.shape[0]

    means = np.asarray(csc.mean(axis=axis)).ravel()
    sq_means = np.asarray(csc.multiply(csc).mean(axis=axis)).ravel()
    variances = sq_means - means ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(means > 0, variances / means, 0.0)

    return result


def _dispersion_scl(mat: "SclCSR", axis: int) -> "RealArray":
    """Dispersion for SCL matrices using C++ kernel."""
    from scl.sparse import Array
    from scl._kernel import sparse as kernel_sparse, feature as kernel_feature

    if axis == 0:
        # Per column
        csc = mat.to_csc()
        n = csc.shape[0]
        n_feat = csc.shape[1]

        # Compute means and variances
        means = Array(n_feat, dtype='float64')
        variances = Array(n_feat, dtype='float64')

        data_ptr = csc.data.get_pointer()
        indices_ptr = csc.indices.get_pointer()
        indptr_ptr = csc.indptr.get_pointer()
        means_ptr = means.get_pointer()
        vars_ptr = variances.get_pointer()

        kernel_feature.standard_moments_csc(
            data_ptr, indices_ptr, indptr_ptr,
            n, n_feat, means_ptr, vars_ptr, 0
        )

        # Compute dispersion
        result = Array(n_feat, dtype='float64')
        for j in range(n_feat):
            mean = means[j]
            var = variances[j]
            result[j] = var / mean if mean > 0 else 0.0

        return result

    else:
        # Per row
        n = mat.shape[1]
        n_feat = mat.shape[0]

        # Compute means and variances for rows
        means = Array(n_feat, dtype='float64')
        variances = Array(n_feat, dtype='float64')

        data_ptr = mat.data.get_pointer()
        indices_ptr = mat.indices.get_pointer()
        indptr_ptr = mat.indptr.get_pointer()
        means_ptr = means.get_pointer()
        vars_ptr = variances.get_pointer()

        kernel_sparse.primary_means_csr(
            data_ptr, indices_ptr, indptr_ptr,
            n_feat, n, means_ptr
        )
        kernel_sparse.primary_variances_csr(
            data_ptr, indices_ptr, indptr_ptr,
            n_feat, n, 0, vars_ptr
        )

        # Compute dispersion
        result = Array(n_feat, dtype='float64')
        for i in range(n_feat):
            mean = means[i]
            var = variances[i]
            result[i] = var / mean if mean > 0 else 0.0

        return result


# =============================================================================
# Detection Rate
# =============================================================================

@overload
def detection_rate(
    mat: "SclCSC",
) -> "RealArray": ...

@overload
def detection_rate(
    mat: "sp.csc_matrix",
) -> "np.ndarray": ...


def detection_rate(
    mat: SparseInput,
) -> Union["RealArray", "np.ndarray"]:
    """Compute detection rate (fraction of cells expressing) each feature.

    The detection rate is the proportion of observations with non-zero values
    for each feature. It's a simple but important quality metric.

    Mathematical Definition:
        detection_rate[j] = (number of cells with X[*, j] > 0) / n_cells

    Interpretation:
        - High detection rate: Gene expressed in many cells
        - Low detection rate: Gene expressed in few cells (marker or lowly expressed)

    Applications:
        - Quality control: Filter genes by minimum detection rate
        - Marker gene identification: Low detection in one cluster, high in another
        - Dropout modeling: Estimate zero-inflation

    Time Complexity:
        O(n_features) for CSC format (just count non-zeros per column).

    Args:
        mat: Input sparse matrix (observations x features) in CSC format.

    Returns:
        Array of detection rates, one per feature (column).

    Examples:
        >>> import scl.feature as feat
        >>> from scl import SclCSC
        >>>
        >>> # Get detection rate for all genes
        >>> det_rate = feat.detection_rate(mat)
        >>>
        >>> # Filter genes detected in at least 1% of cells
        >>> filtered_genes = [i for i, dr in enumerate(det_rate) if dr >= 0.01]
        >>>
        >>> # Find highly specific marker genes
        >>> markers = [i for i, dr in enumerate(det_rate) if 0.05 < dr < 0.2]

    Notes:
        - Detection rate is always in [0, 1].
        - For CSC format, this is O(n_features) as we just count
          non-zeros per column from indptr.
        - Combined with mean expression, helps identify different gene types.

    See Also:
        compute_qc: Get both counts and sums per observation.
        dispersion: Variance-based feature metric.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _detection_rate_scipy(mat)

    # Native SCL
    csc = ensure_scl_csc(mat)
    csc.materialize()

    return _detection_rate_scl(csc)


def _detection_rate_scipy(mat):
    """Detection rate using scipy."""
    import numpy as np

    csc = mat.tocsc()
    n_cells = csc.shape[0]
    n_genes = csc.shape[1]

    nnz_per_col = np.diff(csc.indptr)
    return nnz_per_col / n_cells


def _detection_rate_scl(mat: "SclCSC") -> "RealArray":
    """Detection rate for SCL matrices using C++ kernel."""
    from scl.sparse import Array
    from scl._kernel import feature as kernel_feature

    n_cells = mat.shape[0]
    n_genes = mat.shape[1]

    result = Array(n_genes, dtype='float64')

    # Get C pointers
    indptr_ptr = mat.indptr.get_pointer()
    result_ptr = result.get_pointer()

    # Call C++ kernel
    kernel_feature.detection_rate_csc(
        indptr_ptr, n_cells, n_genes, result_ptr
    )

    return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "highly_variable",
    "dispersion",
    "detection_rate",
]
