"""
Spatial Autocorrelation Metrics.

This module provides spatial autocorrelation statistics for analyzing
spatial patterns in data, particularly useful for spatial transcriptomics.

Implemented Metrics:
    - Moran's I: Global spatial autocorrelation
    - MMD-RBF: Maximum Mean Discrepancy with RBF kernel

These metrics help identify:
    - Spatially variable genes
    - Clustering patterns
    - Distribution differences between regions
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Union, overload

from scl._typing import (
    SparseInput,
    CSCInput,
    ensure_scl_csc,
    get_format,
)

if TYPE_CHECKING:
    import numpy as np
    from scipy import sparse as sp
    from scl.sparse import SclCSC, Array

# Type alias for backward compatibility
RealArray = "Array"


# =============================================================================
# Moran's I
# =============================================================================

@overload
def morans_i(
    X: "SclCSC",
    W: "SclCSC",
) -> "RealArray": ...

@overload
def morans_i(
    X: "sp.csc_matrix",
    W: "sp.csc_matrix",
) -> "np.ndarray": ...


def morans_i(
    X: SparseInput,
    W: SparseInput,
) -> Union["RealArray", "np.ndarray"]:
    """Compute Moran's I spatial autocorrelation for each feature.

    Moran's I is a measure of spatial autocorrelation that indicates
    whether similar values tend to cluster together in space.

    Mathematical Definition:
        For feature j with values x_i across n locations:

            I_j = (n / S0) * (sum_i sum_k W_ik * z_i * z_k) / (sum_i z_i^2)

        where:
            z_i = x_i - mean(x)  (centered values)
            W_ik = spatial weight between locations i and k
            S0 = sum of all weights = sum_i sum_k W_ik

    Interpretation:
        - I > 0: Positive spatial autocorrelation (clustering)
            Similar values tend to be near each other
        - I < 0: Negative spatial autocorrelation (dispersion)
            Dissimilar values tend to be near each other
        - I approx 0: Random spatial pattern
            Values show no spatial structure

        Expected value under no spatial autocorrelation: -1/(n-1)

    Spatial Weights Matrix:
        W should be a spatial weights matrix where W_ik represents the
        spatial relationship between locations i and k:
        - Binary: W_ik = 1 if i and k are neighbors, else 0
        - Distance-based: W_ik = f(distance(i, k))
        - k-NN: W_ik = 1 if k is among i's k nearest neighbors

    Applications:
        - Identifying spatially variable genes in spatial transcriptomics
        - Detecting tissue domains/regions
        - Quality control for spatial data

    Time Complexity:
        O(n_features * nnz(W)) where nnz(W) is non-zeros in weight matrix.

    Args:
        X: Feature matrix in CSC format (observations x features).
            Each column is a feature, each row is a spatial location.
        W: Spatial weights matrix in CSC format (observations x observations).
            Should be row-normalized or symmetric for proper interpretation.

    Returns:
        Array of Moran's I values, one per feature (column of X).

    Raises:
        ValueError: If X rows != W rows or W rows != W cols.

    Examples:
        >>> import scl.spatial as spatial
        >>> from scl import SclCSC
        >>>
        >>> # Build k-NN spatial weights from coordinates
        >>> # (assuming W is pre-computed)
        >>> morans = spatial.morans_i(expression, spatial_weights)
        >>>
        >>> # Find spatially variable genes
        >>> threshold = 0.3  # Arbitrary threshold
        >>> sv_genes = [i for i, m in enumerate(morans) if m > threshold]

    Notes:
        - Moran's I ranges approximately from -1 to +1.
        - The weight matrix W should typically be row-normalized.
        - For significance testing, use permutation tests or analytical
          approximations (not provided here).
        - Large datasets may benefit from approximate methods.

    References:
        Moran, P. A. P. (1950). "Notes on Continuous Stochastic Phenomena".
        Biometrika, 37(1/2), 17-23.

    See Also:
        mmd_rbf: Maximum Mean Discrepancy for distribution comparison.
    """
    fmt_x = get_format(X)
    fmt_w = get_format(W)

    if fmt_x in ("scipy_csr", "scipy_csc", "scipy_other") or \
       fmt_w in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _morans_i_scipy(X, W)

    # Native SCL
    csc_x = ensure_scl_csc(X)
    csc_w = ensure_scl_csc(W)

    csc_x.materialize()
    csc_w.materialize()

    # Validate dimensions
    if csc_x.shape[0] != csc_w.shape[0]:
        raise ValueError(
            f"X rows {csc_x.shape[0]} != W rows {csc_w.shape[0]}"
        )
    if csc_w.shape[0] != csc_w.shape[1]:
        raise ValueError(
            f"W must be square, got {csc_w.shape}"
        )

    return _morans_i_scl(csc_x, csc_w)


def _morans_i_scipy(X, W):
    """Moran's I using scipy/numpy."""
    import numpy as np

    X_csc = X.tocsc()
    W_csc = W.tocsc()

    n = X_csc.shape[0]
    n_features = X_csc.shape[1]

    # Compute S0 (sum of all weights)
    S0 = W_csc.sum()

    if S0 == 0:
        return np.zeros(n_features)

    results = np.zeros(n_features)

    for j in range(n_features):
        col = X_csc.getcol(j).toarray().ravel()

        # Center the values
        mean_x = np.mean(col)
        z = col - mean_x

        # Compute denominator (sum of squared deviations)
        denom = np.sum(z ** 2)

        if denom == 0:
            results[j] = 0.0
            continue

        # Compute numerator (spatial lag correlation)
        # sum_i sum_k W_ik * z_i * z_k
        # = z^T @ W @ z
        Wz = W_csc.dot(z)
        numer = np.dot(z, Wz)

        results[j] = (n / S0) * (numer / denom)

    return results


def _morans_i_scl(X: "SclCSC", W: "SclCSC") -> "RealArray":
    """Moran's I for SCL matrices using C++ kernel."""
    from scl.sparse import Array
    from scl._kernel import spatial as kernel_spatial

    n = X.shape[0]
    n_features = X.shape[1]

    results = Array(n_features, dtype='float64')

    # Get C pointers - W needs to be CSR for graph format expected by kernel
    W_csr = W.to_csr()

    graph_data_ptr = W_csr.data.get_pointer()
    graph_indices_ptr = W_csr.indices.get_pointer()
    graph_indptr_ptr = W_csr.indptr.get_pointer()

    feat_data_ptr = X.data.get_pointer()
    feat_indices_ptr = X.indices.get_pointer()
    feat_indptr_ptr = X.indptr.get_pointer()

    output_ptr = results.get_pointer()

    # Call C++ kernel
    kernel_spatial.morans_i(
        graph_data_ptr, graph_indices_ptr, graph_indptr_ptr, n,
        feat_data_ptr, feat_indices_ptr, feat_indptr_ptr, n_features,
        output_ptr
    )

    return results


# =============================================================================
# Maximum Mean Discrepancy (MMD)
# =============================================================================

@overload
def mmd_rbf(
    X: "SclCSC",
    Y: "SclCSC",
    gamma: float = 1.0,
) -> float: ...

@overload
def mmd_rbf(
    X: "sp.csc_matrix",
    Y: "sp.csc_matrix",
    gamma: float = 1.0,
) -> float: ...


def mmd_rbf(
    X: SparseInput,
    Y: SparseInput,
    gamma: float = 1.0,
) -> float:
    """Compute Maximum Mean Discrepancy with RBF kernel.

    MMD is a kernel-based distance between two probability distributions.
    It measures how different two samples are in terms of their distributions.

    Mathematical Definition:
        MMD^2(X, Y) = E[k(x, x')] - 2*E[k(x, y)] + E[k(y, y')]

        where x, x' ~ P (distribution of X samples)
              y, y' ~ Q (distribution of Y samples)
              k(a, b) = exp(-gamma * ||a - b||^2)  (RBF kernel)

    Unbiased Estimator:
        MMD^2 = (1/(m(m-1))) * sum_{i!=j} k(x_i, x_j)
              - (2/(mn)) * sum_i sum_j k(x_i, y_j)
              + (1/(n(n-1))) * sum_{i!=j} k(y_i, y_j)

        where m = |X|, n = |Y|

    Interpretation:
        - MMD = 0: Distributions are identical (in the RKHS)
        - MMD > 0: Distributions differ
        - Larger MMD indicates greater distributional difference

    Applications:
        - Domain detection: Compare gene expression between regions
        - Batch effect quantification
        - Distribution shift detection
        - Two-sample hypothesis testing

    Kernel Parameter:
        gamma controls the bandwidth of the RBF kernel:
        - Large gamma: More sensitive to local differences
        - Small gamma: More sensitive to global differences
        - Rule of thumb: gamma = 1 / (2 * median(squared distances))

    Time Complexity:
        O(m^2 + mn + n^2) for computing all pairwise distances.

    Args:
        X: First sample matrix (n_samples_X x n_features).
        Y: Second sample matrix (n_samples_Y x n_features).
        gamma: RBF kernel bandwidth parameter. Default 1.0.

    Returns:
        MMD value (scalar). Non-negative, with 0 indicating identical
        distributions.

    Raises:
        ValueError: If X and Y have different number of features.

    Examples:
        >>> import scl.spatial as spatial
        >>> from scl import SclCSC
        >>>
        >>> # Compare expression between two tissue regions
        >>> region_A = expression[cells_in_A, :]
        >>> region_B = expression[cells_in_B, :]
        >>> mmd = spatial.mmd_rbf(region_A, region_B, gamma=0.5)
        >>> print(f"Distribution difference: {mmd:.4f}")

    Notes:
        - For large datasets, consider using random feature approximation
          or Nyström approximation.
        - The returned value is MMD (not MMD^2).
        - For significance testing, use permutation tests.

    References:
        Gretton, A., et al. (2012). "A Kernel Two-Sample Test".
        Journal of Machine Learning Research, 13, 723-773.

    See Also:
        morans_i: Spatial autocorrelation for single distribution.
    """
    fmt_x = get_format(X)
    fmt_y = get_format(Y)

    if fmt_x in ("scipy_csr", "scipy_csc", "scipy_other") or \
       fmt_y in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _mmd_rbf_scipy(X, Y, gamma)

    # Native SCL
    csc_x = ensure_scl_csc(X)
    csc_y = ensure_scl_csc(Y)

    csc_x.materialize()
    csc_y.materialize()

    if csc_x.shape[1] != csc_y.shape[1]:
        raise ValueError(
            f"X features {csc_x.shape[1]} != Y features {csc_y.shape[1]}"
        )

    return _mmd_rbf_scl(csc_x, csc_y, gamma)


def _mmd_rbf_scipy(X, Y, gamma: float) -> float:
    """MMD-RBF using scipy/numpy."""
    import numpy as np

    X_dense = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    Y_dense = Y.toarray() if hasattr(Y, 'toarray') else np.asarray(Y)

    m = X_dense.shape[0]
    n = Y_dense.shape[0]

    if m < 2 or n < 2:
        return 0.0

    # Compute kernel matrices
    def rbf_kernel(A, B, gamma):
        # ||a - b||^2 = ||a||^2 - 2*a.b + ||b||^2
        A_sq = np.sum(A ** 2, axis=1, keepdims=True)
        B_sq = np.sum(B ** 2, axis=1, keepdims=True)
        dists_sq = A_sq - 2 * A @ B.T + B_sq.T
        return np.exp(-gamma * dists_sq)

    K_XX = rbf_kernel(X_dense, X_dense, gamma)
    K_YY = rbf_kernel(Y_dense, Y_dense, gamma)
    K_XY = rbf_kernel(X_dense, Y_dense, gamma)

    # Unbiased estimator
    # XX term: exclude diagonal
    sum_XX = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
    # YY term: exclude diagonal
    sum_YY = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
    # XY term
    sum_XY = np.sum(K_XY) / (m * n)

    mmd_sq = sum_XX - 2 * sum_XY + sum_YY

    # Return MMD (take sqrt, clamp to 0 for numerical issues)
    return math.sqrt(max(0.0, mmd_sq))


def _mmd_rbf_scl(X: "SclCSC", Y: "SclCSC", gamma: float) -> float:
    """MMD-RBF for SCL matrices using C++ kernel."""
    from scl.sparse import Array
    from scl._kernel import mmd as kernel_mmd

    m = X.shape[0]
    n = Y.shape[0]
    d = X.shape[1]

    if m < 2 or n < 2:
        return 0.0

    output = Array(1, dtype='float64')

    # Get C pointers
    data_x_ptr = X.data.get_pointer()
    indices_x_ptr = X.indices.get_pointer()
    indptr_x_ptr = X.indptr.get_pointer()

    data_y_ptr = Y.data.get_pointer()
    indices_y_ptr = Y.indices.get_pointer()
    indptr_y_ptr = Y.indptr.get_pointer()

    output_ptr = output.get_pointer()

    # Call C++ kernel
    kernel_mmd.mmd_rbf_csc(
        data_x_ptr, indices_x_ptr, indptr_x_ptr, m, d,
        data_y_ptr, indices_y_ptr, indptr_y_ptr, n,
        output_ptr, gamma
    )

    return float(output[0])


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "morans_i",
    "mmd_rbf",
]
