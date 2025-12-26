"""
Group Statistics for Sparse Matrices.

This module provides functions for computing statistics within groups,
essential for differential expression analysis in single-cell genomics
where cells are grouped by cluster, condition, or other metadata.

Group-wise operations efficiently handle:
    - Multiple groups with varying sizes
    - Sparse data without densification
    - Parallel computation across features

Common use cases:
    - Computing mean expression per cell type
    - Calculating variance within clusters
    - Aggregating counts by sample/batch
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union, overload

from scl._typing import (
    SparseInput,
    CSRInput,
    CSCInput,
    IndexInput,
    ensure_scl_csr,
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
# Count Group Sizes
# =============================================================================

@overload
def count_group_sizes(
    groups: "IndexArray",
    n_groups: Optional[int] = None,
) -> "IndexArray": ...

@overload
def count_group_sizes(
    groups: "np.ndarray",
    n_groups: Optional[int] = None,
) -> "np.ndarray": ...

@overload
def count_group_sizes(
    groups: Sequence[int],
    n_groups: Optional[int] = None,
) -> "IndexArray": ...


def count_group_sizes(
    groups: IndexInput,
    n_groups: Optional[int] = None,
) -> Union["IndexArray", "np.ndarray"]:
    """Count the number of elements in each group.

    Computes a histogram of group labels, giving the size of each group.
    This is a fundamental operation for normalizing group statistics.

    Algorithm:
        For each label g in groups:
            counts[g] += 1

    Time Complexity:
        O(n) where n is the length of groups.

    Args:
        groups: Integer array of group labels. Values should be in
            range [0, n_groups) where n_groups is automatically
            determined if not provided.
        n_groups: Number of groups. If None, inferred from max(groups) + 1.

    Returns:
        Array of counts for each group. Length equals n_groups.

    Raises:
        ValueError: If groups contains negative values.
        ValueError: If any group label >= n_groups.

    Examples:
        >>> from scl.sparse import Array
        >>> groups = IndexArray.from_sequence([0, 1, 0, 2, 1, 1])
        >>> counts = count_group_sizes(groups)
        >>> counts.to_list()
        [2, 3, 1]  # 2 in group 0, 3 in group 1, 1 in group 2

    Notes:
        - This is equivalent to numpy.bincount for non-negative integers.
        - Useful for computing group means: mean_g = sum_g / count_g

    See Also:
        group_mean: Compute means per group.
        group_stats: Compute multiple statistics per group.
    """
    from scl._typing import is_numpy_array
    from scl.sparse import Array

    # Handle numpy input
    if is_numpy_array(groups):
        import numpy as np
        groups_arr = np.asarray(groups).ravel().astype(int)
        if n_groups is None:
            n_groups = int(np.max(groups_arr)) + 1 if len(groups_arr) > 0 else 0
        return np.bincount(groups_arr, minlength=n_groups)

    # Convert to IndexArray
    groups_arr = ensure_index_vector(groups)

    # Determine n_groups
    if n_groups is None:
        max_group = 0
        for i in range(groups_arr.size):
            if groups_arr[i] > max_group:
                max_group = groups_arr[i]
        n_groups = max_group + 1

    # Count
    counts = IndexArray(n_groups)
    for i in range(n_groups):
        counts[i] = 0

    for i in range(groups_arr.size):
        g = groups_arr[i]
        if g < 0:
            raise ValueError(f"Group labels must be non-negative, got {g}")
        if g >= n_groups:
            raise ValueError(f"Group label {g} >= n_groups {n_groups}")
        counts[g] += 1

    return counts


# =============================================================================
# Group Mean
# =============================================================================

@overload
def group_mean(
    mat: "SclCSC",
    groups: "IndexArray",
    n_groups: Optional[int] = None,
) -> "SclCSC": ...

@overload
def group_mean(
    mat: "sp.csc_matrix",
    groups: "np.ndarray",
    n_groups: Optional[int] = None,
) -> "np.ndarray": ...


def group_mean(
    mat: SparseInput,
    groups: IndexInput,
    n_groups: Optional[int] = None,
) -> Union["SclCSC", "np.ndarray"]:
    """Compute the mean of each feature within each group.

    For a feature matrix (features x cells) and group labels, computes
    the mean expression of each feature in each group.

    Mathematical Definition:
        For feature j and group g:

            mean[j, g] = (1/n_g) * sum(X[i, j] for i where groups[i] == g)

        where n_g is the number of cells in group g.

    Handling Zeros:
        The mean includes implicit zeros. For a sparse matrix:

            mean[j, g] = sum(non-zero values in group g) / n_g

        This gives the true mean, not the mean of non-zeros only.

    Time Complexity:
        O(nnz + n_features * n_groups) where nnz is the number of
        non-zero elements.

    Args:
        mat: Feature matrix in CSC format (features x cells).
            Each column is a feature, each row is a cell.
            Note: CSC is preferred as it allows efficient column access.
        groups: Integer array of group labels for each cell (row).
            Length must equal number of rows in mat.
        n_groups: Number of groups. If None, inferred from groups.

    Returns:
        Matrix of shape (n_groups, n_features) containing mean values.
        Return type matches input: SclCSC for SCL input, numpy array
        for scipy/numpy input.

    Raises:
        ValueError: If groups length doesn't match matrix rows.
        ValueError: If any group is empty (division by zero).

    Examples:
        >>> # Gene expression: genes x cells
        >>> # groups: cell type labels
        >>> mean_expr = group_mean(expr_matrix, cell_types, n_groups=5)
        >>> # mean_expr[g, j] = mean expression of gene j in cell type g

    Notes:
        - For very large matrices, consider using group_stats() which
          can compute multiple statistics in a single pass.
        - Empty groups will raise an error. Filter groups beforehand
          or handle the ValueError.

    See Also:
        group_var: Compute variance per group.
        group_stats: Compute multiple statistics per group.
        count_group_sizes: Count elements per group.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _group_mean_scipy(mat, groups, n_groups)

    # Native SCL
    csc = ensure_scl_csc(mat)
    groups_arr = ensure_index_vector(groups, size=csc.shape[0])

    csc.materialize()

    return _group_mean_scl(csc, groups_arr, n_groups)


def _group_mean_scl(
    mat: "SclCSC",
    groups: "IndexArray",
    n_groups: Optional[int],
) -> "SclCSC":
    """Group mean implementation for SCL matrices."""
    from scl.sparse import Array
    from scl.sparse import SclCSC

    n_cells = mat.shape[0]
    n_features = mat.shape[1]

    # Determine n_groups
    if n_groups is None:
        max_g = 0
        for i in range(n_cells):
            if groups[i] > max_g:
                max_g = groups[i]
        n_groups = max_g + 1

    # Count group sizes
    group_counts = [0] * n_groups
    for i in range(n_cells):
        group_counts[groups[i]] += 1

    # Check for empty groups
    for g in range(n_groups):
        if group_counts[g] == 0:
            raise ValueError(f"Group {g} is empty, cannot compute mean")

    # Compute sums per group per feature
    # Output shape: (n_groups, n_features) as dense for simplicity
    # We'll return as a dense-ish result
    result_data = []
    result_indices = []
    result_indptr = [0]

    for j in range(n_features):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        # Sum per group for this feature
        group_sums = [0.0] * n_groups

        for k in range(start, end):
            row = mat._indices[k]
            val = mat._data[k]
            g = groups[row]
            group_sums[g] += val

        # Compute means and store non-zeros
        for g in range(n_groups):
            mean_val = group_sums[g] / group_counts[g]
            if abs(mean_val) > 1e-15:
                result_data.append(mean_val)
                result_indices.append(g)

        result_indptr.append(len(result_data))

    # Create output CSC matrix (n_groups x n_features)
    return SclCSC.from_arrays(
        result_data, result_indices, result_indptr,
        (n_groups, n_features)
    )


def _group_mean_scipy(mat, groups, n_groups: Optional[int]):
    """Group mean using scipy/numpy."""
    import numpy as np

    csc = mat.tocsc()
    n_cells = csc.shape[0]
    n_features = csc.shape[1]

    groups = np.asarray(groups).ravel().astype(int)

    if n_groups is None:
        n_groups = int(np.max(groups)) + 1

    group_counts = np.bincount(groups, minlength=n_groups)

    if np.any(group_counts == 0):
        empty = np.where(group_counts == 0)[0]
        raise ValueError(f"Groups {empty.tolist()} are empty")

    # Result matrix (n_groups x n_features)
    result = np.zeros((n_groups, n_features))

    for j in range(n_features):
        col = csc.getcol(j).toarray().ravel()
        for g in range(n_groups):
            mask = groups == g
            result[g, j] = np.sum(col[mask]) / group_counts[g]

    return result


# =============================================================================
# Group Variance
# =============================================================================

@overload
def group_var(
    mat: "SclCSC",
    groups: "IndexArray",
    n_groups: Optional[int] = None,
    *,
    ddof: int = 0,
) -> "SclCSC": ...

@overload
def group_var(
    mat: "sp.csc_matrix",
    groups: "np.ndarray",
    n_groups: Optional[int] = None,
    *,
    ddof: int = 0,
) -> "np.ndarray": ...


def group_var(
    mat: SparseInput,
    groups: IndexInput,
    n_groups: Optional[int] = None,
    *,
    ddof: int = 0,
) -> Union["SclCSC", "np.ndarray"]:
    """Compute the variance of each feature within each group.

    For a feature matrix (features x cells) and group labels, computes
    the variance of each feature in each group.

    Mathematical Definition:
        For feature j and group g:

            var[j, g] = (1/(n_g - ddof)) * sum((X[i,j] - mean[j,g])^2
                                               for i where groups[i] == g)

    Algorithm (Two-Pass):
        1. First pass: Compute group means
        2. Second pass: Compute sum of squared deviations

        For sparse matrices, zeros contribute (0 - mean)^2 = mean^2 terms.

    Time Complexity:
        O(2 * nnz + n_features * n_groups)

    Args:
        mat: Feature matrix in CSC format (features x cells).
        groups: Integer array of group labels for each cell.
        n_groups: Number of groups. If None, inferred from groups.
        ddof: Delta degrees of freedom. Default is 0 (population variance).
            Use ddof=1 for sample variance (Bessel's correction).

    Returns:
        Matrix of shape (n_groups, n_features) containing variance values.

    Raises:
        ValueError: If any group has fewer than ddof + 1 elements.

    Examples:
        >>> # Population variance per cell type
        >>> var_pop = group_var(expr_matrix, cell_types)
        >>>
        >>> # Sample variance
        >>> var_sample = group_var(expr_matrix, cell_types, ddof=1)

    Notes:
        - Uses two-pass algorithm for numerical stability.
        - For single-cell data, high variance often indicates
          biologically variable genes (useful for feature selection).

    See Also:
        group_mean: Compute means per group.
        group_stats: Compute mean and variance in one pass.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _group_var_scipy(mat, groups, n_groups, ddof)

    # Native SCL
    csc = ensure_scl_csc(mat)
    groups_arr = ensure_index_vector(groups, size=csc.shape[0])

    csc.materialize()

    return _group_var_scl(csc, groups_arr, n_groups, ddof)


def _group_var_scl(
    mat: "SclCSC",
    groups: "IndexArray",
    n_groups: Optional[int],
    ddof: int,
) -> "SclCSC":
    """Group variance implementation for SCL matrices."""
    from scl.sparse import Array
    from scl.sparse import SclCSC

    n_cells = mat.shape[0]
    n_features = mat.shape[1]

    if n_groups is None:
        max_g = 0
        for i in range(n_cells):
            if groups[i] > max_g:
                max_g = groups[i]
        n_groups = max_g + 1

    # Count groups
    group_counts = [0] * n_groups
    for i in range(n_cells):
        group_counts[groups[i]] += 1

    # Check for insufficient samples
    for g in range(n_groups):
        if group_counts[g] <= ddof:
            raise ValueError(
                f"Group {g} has {group_counts[g]} samples, "
                f"need > {ddof} for ddof={ddof}"
            )

    result_data = []
    result_indices = []
    result_indptr = [0]

    for j in range(n_features):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        # First pass: means
        group_sums = [0.0] * n_groups
        group_nnz = [0] * n_groups

        for k in range(start, end):
            row = mat._indices[k]
            val = mat._data[k]
            g = groups[row]
            group_sums[g] += val
            group_nnz[g] += 1

        group_means = [s / c for s, c in zip(group_sums, group_counts)]

        # Second pass: squared deviations
        group_sq_sums = [0.0] * n_groups

        for k in range(start, end):
            row = mat._indices[k]
            val = mat._data[k]
            g = groups[row]
            group_sq_sums[g] += (val - group_means[g]) ** 2

        # Add zero contributions
        for g in range(n_groups):
            n_zeros = group_counts[g] - group_nnz[g]
            group_sq_sums[g] += n_zeros * group_means[g] ** 2

        # Compute variances
        for g in range(n_groups):
            var_val = group_sq_sums[g] / (group_counts[g] - ddof)
            if abs(var_val) > 1e-15:
                result_data.append(var_val)
                result_indices.append(g)

        result_indptr.append(len(result_data))

    return SclCSC.from_arrays(
        result_data, result_indices, result_indptr,
        (n_groups, n_features)
    )


def _group_var_scipy(mat, groups, n_groups: Optional[int], ddof: int):
    """Group variance using scipy/numpy."""
    import numpy as np

    csc = mat.tocsc()
    n_cells = csc.shape[0]
    n_features = csc.shape[1]

    groups = np.asarray(groups).ravel().astype(int)

    if n_groups is None:
        n_groups = int(np.max(groups)) + 1

    group_counts = np.bincount(groups, minlength=n_groups)

    if np.any(group_counts <= ddof):
        bad = np.where(group_counts <= ddof)[0]
        raise ValueError(f"Groups {bad.tolist()} have insufficient samples for ddof={ddof}")

    result = np.zeros((n_groups, n_features))

    for j in range(n_features):
        col = csc.getcol(j).toarray().ravel()
        for g in range(n_groups):
            mask = groups == g
            vals = col[mask]
            result[g, j] = np.var(vals, ddof=ddof)

    return result


# =============================================================================
# Group Stats (Combined Mean + Variance)
# =============================================================================

@overload
def group_stats(
    mat: "SclCSC",
    groups: "IndexArray",
    n_groups: Optional[int] = None,
    *,
    ddof: int = 0,
) -> Tuple["SclCSC", "SclCSC"]: ...

@overload
def group_stats(
    mat: "sp.csc_matrix",
    groups: "np.ndarray",
    n_groups: Optional[int] = None,
    *,
    ddof: int = 0,
) -> Tuple["np.ndarray", "np.ndarray"]: ...


def group_stats(
    mat: SparseInput,
    groups: IndexInput,
    n_groups: Optional[int] = None,
    *,
    ddof: int = 0,
) -> Tuple[Union["SclCSC", "np.ndarray"], Union["SclCSC", "np.ndarray"]]:
    """Compute mean and variance for each feature within each group.

    This function combines group_mean and group_var into a single
    efficient operation, computing both statistics in a single pass
    through the data (using Welford's online algorithm).

    Mathematical Definition:
        For each feature j and group g:

            mean[j, g] = (1/n_g) * sum(X[i, j] for i in group g)

            var[j, g] = (1/(n_g - ddof)) * sum((X[i,j] - mean[j,g])^2
                                               for i in group g)

    Algorithm (Welford's Online):
        For numerical stability, uses Welford's algorithm which
        computes mean and variance in a single pass:

        For each value x in group g:
            n += 1
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)

        After all values:
            variance = M2 / (n - ddof)

        This avoids the numerical instability of the naive two-pass
        algorithm when values are large or have high variance.

    Time Complexity:
        O(nnz + n_features * n_groups) - more efficient than calling
        group_mean and group_var separately.

    Args:
        mat: Feature matrix in CSC format (features x cells).
        groups: Integer array of group labels for each cell.
        n_groups: Number of groups. If None, inferred from groups.
        ddof: Delta degrees of freedom for variance. Default 0.

    Returns:
        Tuple of (means, variances):
            - means: Matrix (n_groups x n_features) of group means
            - variances: Matrix (n_groups x n_features) of group variances

    Examples:
        >>> # Compute mean and variance per cell type
        >>> means, variances = group_stats(expr_matrix, cell_types)
        >>>
        >>> # Find highly variable genes in each cluster
        >>> cv = np.sqrt(variances) / (means + 1e-10)  # Coefficient of variation

    Notes:
        - More efficient than separate calls to group_mean and group_var.
        - Uses numerically stable Welford's algorithm.
        - Essential for downstream analyses like highly variable gene
          selection and differential expression.

    See Also:
        group_mean: Compute means only.
        group_var: Compute variances only.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _group_stats_scipy(mat, groups, n_groups, ddof)

    # Native SCL
    csc = ensure_scl_csc(mat)
    groups_arr = ensure_index_vector(groups, size=csc.shape[0])

    csc.materialize()

    return _group_stats_scl(csc, groups_arr, n_groups, ddof)


def _group_stats_scl(
    mat: "SclCSC",
    groups: "IndexArray",
    n_groups: Optional[int],
    ddof: int,
) -> Tuple["SclCSC", "SclCSC"]:
    """Group stats implementation for SCL matrices using Welford's algorithm."""
    from scl.sparse import SclCSC

    n_cells = mat.shape[0]
    n_features = mat.shape[1]

    if n_groups is None:
        max_g = 0
        for i in range(n_cells):
            if groups[i] > max_g:
                max_g = groups[i]
        n_groups = max_g + 1

    # Count groups
    group_counts = [0] * n_groups
    for i in range(n_cells):
        group_counts[groups[i]] += 1

    # Check for empty groups
    for g in range(n_groups):
        if group_counts[g] <= ddof:
            raise ValueError(
                f"Group {g} has {group_counts[g]} samples, "
                f"need > {ddof} for ddof={ddof}"
            )

    # Build output arrays
    mean_data = []
    mean_indices = []
    mean_indptr = [0]

    var_data = []
    var_indices = []
    var_indptr = [0]

    for j in range(n_features):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        # Welford's algorithm state per group
        # n, mean, M2 for each group
        n_seen = [0] * n_groups
        means = [0.0] * n_groups
        m2s = [0.0] * n_groups

        # Process non-zeros
        for k in range(start, end):
            row = mat._indices[k]
            val = mat._data[k]
            g = groups[row]

            n_seen[g] += 1
            delta = val - means[g]
            means[g] += delta / n_seen[g]
            m2s[g] += delta * (val - means[g])

        # Process zeros (implicitly)
        # We need to incorporate (count - n_seen) zeros into each group
        for g in range(n_groups):
            n_zeros = group_counts[g] - n_seen[g]
            if n_zeros > 0:
                # Add zeros one "batch" at a time using parallel formula
                # For batch of k zeros with value 0:
                # combined_mean = (n*mean + k*0) / (n+k) = n*mean / (n+k)
                # delta = 0 - old_mean
                # M2 += k * delta * (0 - new_mean) + (n*k/(n+k)) * delta^2
                # Simplified: M2 += n * k / (n + k) * mean^2
                n_old = n_seen[g]
                n_new = n_old + n_zeros
                old_mean = means[g]

                if n_old > 0:
                    # New mean after adding zeros
                    new_mean = (n_old * old_mean) / n_new
                    # M2 update for adding zeros
                    m2s[g] += n_old * n_zeros / n_new * old_mean ** 2
                    means[g] = new_mean
                else:
                    # All values are zero
                    means[g] = 0.0
                    m2s[g] = 0.0

                n_seen[g] = n_new

        # Store results
        for g in range(n_groups):
            mean_val = means[g]
            if abs(mean_val) > 1e-15:
                mean_data.append(mean_val)
                mean_indices.append(g)

        mean_indptr.append(len(mean_data))

        for g in range(n_groups):
            var_val = m2s[g] / (group_counts[g] - ddof)
            if abs(var_val) > 1e-15:
                var_data.append(var_val)
                var_indices.append(g)

        var_indptr.append(len(var_data))

    means_mat = SclCSC.from_arrays(
        mean_data, mean_indices, mean_indptr, (n_groups, n_features)
    )
    vars_mat = SclCSC.from_arrays(
        var_data, var_indices, var_indptr, (n_groups, n_features)
    )

    return means_mat, vars_mat


def _group_stats_scipy(mat, groups, n_groups: Optional[int], ddof: int):
    """Group stats using scipy/numpy."""
    import numpy as np

    csc = mat.tocsc()
    n_cells = csc.shape[0]
    n_features = csc.shape[1]

    groups = np.asarray(groups).ravel().astype(int)

    if n_groups is None:
        n_groups = int(np.max(groups)) + 1

    group_counts = np.bincount(groups, minlength=n_groups)

    if np.any(group_counts <= ddof):
        bad = np.where(group_counts <= ddof)[0]
        raise ValueError(f"Groups {bad.tolist()} have insufficient samples")

    means = np.zeros((n_groups, n_features))
    variances = np.zeros((n_groups, n_features))

    for j in range(n_features):
        col = csc.getcol(j).toarray().ravel()
        for g in range(n_groups):
            mask = groups == g
            vals = col[mask]
            means[g, j] = np.mean(vals)
            variances[g, j] = np.var(vals, ddof=ddof)

    return means, variances


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "count_group_sizes",
    "group_mean",
    "group_var",
    "group_stats",
]
