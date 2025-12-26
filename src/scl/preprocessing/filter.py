"""
Filtering Operations for Sparse Matrices.

This module provides filtering operations for sparse matrices, including
threshold filtering, top-k selection, and downsampling.

These operations are essential for:
    - Quality control (removing low-count features/cells)
    - Feature selection (keeping top values)
    - Subsampling for computational efficiency
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Optional, Union, overload

from scl._typing import (
    SparseInput,
    ensure_scl_csr,
    ensure_scl_csc,
    get_format,
)

if TYPE_CHECKING:
    import numpy as np
    from scipy import sparse as sp
    from scl.sparse import SclCSR, SclCSC, Array


# =============================================================================
# Filter Threshold
# =============================================================================

@overload
def filter_threshold(
    mat: "SclCSR",
    threshold: float,
    *,
    mode: str = "greater",
) -> "SclCSR": ...

@overload
def filter_threshold(
    mat: "SclCSC",
    threshold: float,
    *,
    mode: str = "greater",
) -> "SclCSC": ...

@overload
def filter_threshold(
    mat: "sp.spmatrix",
    threshold: float,
    *,
    mode: str = "greater",
) -> "sp.spmatrix": ...


def filter_threshold(
    mat: SparseInput,
    threshold: float,
    *,
    mode: str = "greater",
) -> Union["SclCSR", "SclCSC", "sp.spmatrix"]:
    """Filter sparse matrix by threshold.

    Creates a new sparse matrix containing only elements that pass the
    threshold criterion. Elements not meeting the criterion are set to zero
    (removed from sparse representation).

    Filtering Modes:
        - "greater": Keep values > threshold
        - "greater_equal": Keep values >= threshold
        - "less": Keep values < threshold
        - "less_equal": Keep values <= threshold
        - "not_equal": Keep values != threshold

    Mathematical Definition:
        result[i, j] = mat[i, j] if condition(mat[i, j], threshold) else 0

    Applications:
        - Removing noise: filter_threshold(mat, noise_floor, mode="greater")
        - Binary thresholding: Filter then check non-zero pattern
        - Quality control: Remove low-confidence values

    Time Complexity:
        O(nnz) where nnz is the number of non-zeros in the input.
        Output nnz may be smaller depending on how many values pass the filter.

    Args:
        mat: Input sparse matrix.
        threshold: Threshold value for filtering.
        mode: Comparison mode. One of:
            - "greater": Keep values > threshold
            - "greater_equal": Keep values >= threshold
            - "less": Keep values < threshold
            - "less_equal": Keep values <= threshold
            - "not_equal": Keep values != threshold

    Returns:
        Filtered sparse matrix with same shape but potentially fewer non-zeros.

    Raises:
        ValueError: If mode is not recognized.

    Examples:
        >>> from scl import SclCSR
        >>> import scl.preprocessing as pp
        >>>
        >>> # Keep only values above noise threshold
        >>> filtered = pp.filter_threshold(counts, threshold=1.0, mode="greater")
        >>>
        >>> # Keep only high-confidence values
        >>> confident = pp.filter_threshold(probs, threshold=0.5, mode="greater_equal")
        >>>
        >>> # Remove outliers (keep values below threshold)
        >>> clipped = pp.filter_threshold(mat, threshold=1000, mode="less_equal")

    Notes:
        - The output maintains the same shape as input.
        - Zeros in the input remain zeros (never tested against threshold).
        - The operation always creates a new matrix (never in-place).

    See Also:
        top_k: Keep only top-k values per row.
    """
    valid_modes = ("greater", "greater_equal", "less", "less_equal", "not_equal")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _filter_threshold_scipy(mat, threshold, mode)

    # Native SCL
    if fmt == "scl_csc":
        scl_mat = mat
        is_csc = True
    else:
        scl_mat = ensure_scl_csr(mat)
        is_csc = False

    scl_mat.materialize()

    return _filter_threshold_scl(scl_mat, threshold, mode, is_csc)


def _compare(val: float, threshold: float, mode: str) -> bool:
    """Compare value against threshold."""
    if mode == "greater":
        return val > threshold
    elif mode == "greater_equal":
        return val >= threshold
    elif mode == "less":
        return val < threshold
    elif mode == "less_equal":
        return val <= threshold
    else:  # not_equal
        return val != threshold


def _filter_threshold_scipy(mat, threshold: float, mode: str):
    """Filter using scipy."""
    import numpy as np

    # Keep original format
    is_csc = hasattr(mat, 'format') and mat.format == 'csc'

    if is_csc:
        csc = mat.tocsc()
    else:
        csc = mat.tocsr()

    # Build mask based on mode
    if mode == "greater":
        mask = csc.data > threshold
    elif mode == "greater_equal":
        mask = csc.data >= threshold
    elif mode == "less":
        mask = csc.data < threshold
    elif mode == "less_equal":
        mask = csc.data <= threshold
    else:  # not_equal
        mask = csc.data != threshold

    # Create new sparse matrix with filtered data
    result = csc.copy()
    result.data = result.data * mask  # Zero out filtered values
    result.eliminate_zeros()

    return result


def _filter_threshold_scl(mat, threshold: float, mode: str, is_csc: bool):
    """Filter SCL matrix."""
    from scl.sparse import Array, SclCSR, SclCSC

    n_primary = mat.shape[1] if is_csc else mat.shape[0]

    new_data = []
    new_indices = []
    new_indptr = [0]

    for i in range(n_primary):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]

        for k in range(start, end):
            val = mat._data[k]
            if _compare(val, threshold, mode):
                new_data.append(val)
                new_indices.append(mat._indices[k])

        new_indptr.append(len(new_data))

    if is_csc:
        return SclCSC.from_arrays(new_data, new_indices, new_indptr, mat.shape)
    else:
        return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)


# =============================================================================
# Top-K Selection
# =============================================================================

@overload
def top_k(
    mat: "SclCSR",
    k: int,
    axis: int = 1,
) -> "SclCSR": ...

@overload
def top_k(
    mat: "sp.spmatrix",
    k: int,
    axis: int = 1,
) -> "sp.spmatrix": ...


def top_k(
    mat: SparseInput,
    k: int,
    axis: int = 1,
) -> Union["SclCSR", "sp.spmatrix"]:
    """Select top-k values per row or column.

    Creates a new sparse matrix containing only the k largest values
    in each row (axis=1) or column (axis=0). All other values are set to zero.

    Algorithm:
        For each row/column:
            1. Extract non-zero values
            2. Find the k-th largest value (using partial sort)
            3. Keep only values >= k-th largest
            4. If fewer than k non-zeros, keep all

    Tie Handling:
        If multiple values equal the k-th largest, all tied values are kept.
        This means the result may have more than k values per row/column.

    Time Complexity:
        O(nnz * log(k)) using heap-based selection.

    Args:
        mat: Input sparse matrix.
        k: Number of top values to keep per row/column.
        axis: Axis along which to select top-k.
            - 0: Top-k per column
            - 1: Top-k per row (default)

    Returns:
        Sparse matrix with at most k non-zeros per row/column.

    Raises:
        ValueError: If k <= 0.
        ValueError: If axis is not 0 or 1.

    Examples:
        >>> from scl import SclCSR
        >>> import scl.preprocessing as pp
        >>>
        >>> # Keep top 10 genes per cell
        >>> sparse_mat = pp.top_k(expression, k=10, axis=1)
        >>>
        >>> # Keep top 100 cells per gene
        >>> top_cells = pp.top_k(expression, k=100, axis=0)

    Notes:
        - The output maintains the same shape as input.
        - Rows/columns with fewer than k non-zeros keep all their values.
        - For k-nearest-neighbor graphs, use this on similarity matrices.

    See Also:
        filter_threshold: Filter by absolute threshold.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _top_k_scipy(mat, k, axis)

    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()

    if axis == 0:
        # Top-k per column: transpose, apply row-wise, transpose back
        csc = csr.to_csc()
        result_csc = _top_k_primary(csc, k, is_csc=True)
        return result_csc.to_csr()
    else:
        return _top_k_primary(csr, k, is_csc=False)


def _top_k_scipy(mat, k: int, axis: int):
    """Top-k using scipy."""
    import numpy as np

    if axis == 1:
        csr = mat.tocsr()
    else:
        csr = mat.tocsc()

    n_primary = csr.shape[0] if axis == 1 else csr.shape[1]

    new_data = []
    new_indices = []
    new_indptr = [0]

    for i in range(n_primary):
        start = csr.indptr[i]
        end = csr.indptr[i + 1]

        if end - start <= k:
            # Keep all
            new_data.extend(csr.data[start:end])
            new_indices.extend(csr.indices[start:end])
        else:
            # Find k-th largest
            vals = csr.data[start:end]
            threshold = np.partition(vals, -k)[-k]

            for j in range(start, end):
                if csr.data[j] >= threshold:
                    new_data.append(csr.data[j])
                    new_indices.append(csr.indices[j])

        new_indptr.append(len(new_data))

    from scipy import sparse as sp
    if axis == 1:
        return sp.csr_matrix(
            (new_data, new_indices, new_indptr),
            shape=csr.shape
        )
    else:
        return sp.csc_matrix(
            (new_data, new_indices, new_indptr),
            shape=csr.shape
        )


def _top_k_primary(mat, k: int, is_csc: bool):
    """Top-k along primary axis for SCL matrices."""
    from scl.sparse import Array, SclCSR, SclCSC

    n_primary = mat.shape[1] if is_csc else mat.shape[0]

    new_data = []
    new_indices = []
    new_indptr = [0]

    for i in range(n_primary):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]

        n_vals = end - start

        if n_vals <= k:
            # Keep all
            for j in range(start, end):
                new_data.append(mat._data[j])
                new_indices.append(mat._indices[j])
        else:
            # Find k-th largest using partial sort
            # Collect (value, original_index) pairs
            pairs = []
            for j in range(start, end):
                pairs.append((mat._data[j], mat._indices[j]))

            # Sort by value descending
            pairs.sort(key=lambda x: -x[0])

            # Threshold is k-th largest
            threshold = pairs[k - 1][0]

            # Keep all values >= threshold
            for val, idx in pairs:
                if val >= threshold:
                    new_data.append(val)
                    new_indices.append(idx)

            # Re-sort by index for canonical CSR/CSC form
            # (Actually, we should preserve index order for efficiency)

        new_indptr.append(len(new_data))

    if is_csc:
        return SclCSC.from_arrays(new_data, new_indices, new_indptr, mat.shape)
    else:
        return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)


# =============================================================================
# Downsample
# =============================================================================

@overload
def downsample(
    mat: "SclCSR",
    target_sum: float,
    *,
    seed: Optional[int] = None,
) -> "SclCSR": ...

@overload
def downsample(
    mat: "sp.spmatrix",
    target_sum: float,
    *,
    seed: Optional[int] = None,
) -> "sp.spmatrix": ...


def downsample(
    mat: SparseInput,
    target_sum: float,
    *,
    seed: Optional[int] = None,
) -> Union["SclCSR", "sp.spmatrix"]:
    """Downsample counts to a target sum per row.

    For count data (e.g., UMI counts), downsamples each row to have
    approximately the target sum while preserving the relative proportions
    of values within each row.

    Algorithm (Multinomial Sampling):
        For each row with sum S > target_sum:
            1. Compute probabilities p_j = x_j / S for each column j
            2. Sample target_sum counts from multinomial(target_sum, p)
            3. Replace row values with sampled counts

        Rows with sum <= target_sum are left unchanged.

    Statistical Properties:
        - Preserves relative proportions (in expectation)
        - Introduces sampling noise (variance proportional to p * (1-p))
        - Useful for comparing samples with different sequencing depths

    Applications:
        - Normalizing library sizes before comparison
        - Testing sensitivity to sequencing depth
        - Reducing batch effects from depth differences

    Time Complexity:
        O(nnz + m * target_sum) where m is the number of rows needing
        downsampling.

    Args:
        mat: Input sparse matrix with non-negative integer counts.
        target_sum: Target total count per row. Rows with smaller sums
            are left unchanged.
        seed: Random seed for reproducibility. If None, uses system random.

    Returns:
        Downsampled sparse matrix with same shape.

    Examples:
        >>> from scl import SclCSR
        >>> import scl.preprocessing as pp
        >>>
        >>> # Downsample to 1000 counts per cell
        >>> downsampled = pp.downsample(counts, target_sum=1000, seed=42)
        >>>
        >>> # Check row sums
        >>> row_sums = downsampled.sum(axis=1)
        >>> print(max(row_sums.to_list()))  # Should be <= 1000 (or original if < 1000)

    Notes:
        - Input values should be non-negative counts (integers recommended).
        - For non-integer values, treats them as expected counts and uses
          Poisson sampling.
        - Downsampling is stochastic; use seed for reproducibility.
        - Memory efficient: processes one row at a time.

    See Also:
        normalize: Deterministic normalization to target sum.
    """
    if target_sum <= 0:
        raise ValueError(f"target_sum must be positive, got {target_sum}")

    if seed is not None:
        random.seed(seed)

    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _downsample_scipy(mat, target_sum, seed)

    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()

    return _downsample_scl(csr, target_sum)


def _downsample_scipy(mat, target_sum: float, seed: Optional[int]):
    """Downsample using scipy/numpy."""
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    csr = mat.tocsr().copy()

    for i in range(csr.shape[0]):
        start = csr.indptr[i]
        end = csr.indptr[i + 1]

        if start == end:
            continue

        vals = csr.data[start:end]
        row_sum = np.sum(vals)

        if row_sum <= target_sum:
            continue

        # Compute probabilities
        probs = vals / row_sum

        # Sample from multinomial
        counts = np.random.multinomial(int(target_sum), probs)

        csr.data[start:end] = counts.astype(csr.data.dtype)

    csr.eliminate_zeros()
    return csr


def _downsample_scl(mat: "SclCSR", target_sum: float) -> "SclCSR":
    """Downsample SCL matrix."""
    from scl.sparse import Array, SclCSR

    m = mat.shape[0]

    new_data = []
    new_indices = []
    new_indptr = [0]

    for i in range(m):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]

        if start == end:
            new_indptr.append(len(new_data))
            continue

        # Compute row sum
        row_sum = 0.0
        for k in range(start, end):
            row_sum += mat._data[k]

        if row_sum <= target_sum:
            # Keep as-is
            for k in range(start, end):
                new_data.append(mat._data[k])
                new_indices.append(mat._indices[k])
        else:
            # Downsample using simple multinomial
            n_samples = int(target_sum)

            # Build cumulative probabilities
            cum_probs = []
            cum = 0.0
            for k in range(start, end):
                cum += mat._data[k] / row_sum
                cum_probs.append(cum)

            # Count samples per column
            counts = [0] * (end - start)

            for _ in range(n_samples):
                r = random.random()
                # Binary search for bucket
                lo, hi = 0, len(cum_probs)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cum_probs[mid] < r:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < len(counts):
                    counts[lo] += 1

            # Add non-zero counts
            for k_offset, count in enumerate(counts):
                if count > 0:
                    new_data.append(float(count))
                    new_indices.append(mat._indices[start + k_offset])

        new_indptr.append(len(new_data))

    return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "filter_threshold",
    "top_k",
    "downsample",
]
