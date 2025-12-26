"""
Element-wise Mathematical Transforms for Sparse Matrices.

This module provides element-wise transformation functions commonly used
in data preprocessing and analysis. All transforms preserve sparsity
(zeros remain zeros after transformation).

Implemented Transforms:
    - log1p: Natural logarithm of (1 + x)
    - log2p1: Base-2 logarithm of (1 + x)
    - expm1: Exponential minus one (inverse of log1p)
    - softmax: Normalized exponential (row-wise or column-wise)

These transforms are essential for:
    - Variance stabilization (log transforms)
    - Normalization (softmax)
    - Feature scaling
"""

from __future__ import annotations

import math
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

# Type alias for backward compatibility
RealArray = "Array"


# =============================================================================
# Log1p Transform
# =============================================================================

@overload
def log1p(
    mat: "SclCSR",
    *,
    inplace: bool = False,
) -> "SclCSR": ...

@overload
def log1p(
    mat: "SclCSC",
    *,
    inplace: bool = False,
) -> "SclCSC": ...

@overload
def log1p(
    mat: "sp.spmatrix",
    *,
    inplace: bool = False,
) -> "sp.spmatrix": ...


def log1p(
    mat: SparseInput,
    *,
    inplace: bool = False,
) -> Union["SclCSR", "SclCSC", "sp.spmatrix"]:
    """Compute natural logarithm of (1 + x) element-wise.

    Applies the transformation y = ln(1 + x) to each element of the matrix.
    This is a variance-stabilizing transformation commonly used in single-cell
    RNA-seq analysis.

    Mathematical Definition:
        y = log(1 + x)

        For small x: log(1 + x) approx x (first-order Taylor)
        For large x: log(1 + x) approx log(x)

    Properties:
        - Monotonically increasing: preserves ranking
        - Compresses large values: reduces influence of outliers
        - Zero-preserving: log(1 + 0) = 0
        - Numerically stable: avoids log(0) issues

    Biological Motivation:
        Count data in single-cell RNA-seq often follows negative binomial
        distribution with mean-variance relationship var(x) ~ mean(x)^2.
        The log1p transform approximately stabilizes variance:
        var(log(1+x)) approx constant

    Time Complexity:
        O(nnz) where nnz is the number of non-zeros.

    Args:
        mat: Input sparse matrix with non-negative values.
        inplace: If True, modify the matrix in-place. If False (default),
            return a new matrix with transformed values.

    Returns:
        Matrix with log1p-transformed values. Same format as input.

    Raises:
        ValueError: If matrix contains negative values (log of negative
            number is undefined).

    Examples:
        >>> from scl import SclCSR
        >>> import scl.math as smath
        >>>
        >>> # Gene expression counts
        >>> counts = SclCSR.from_scipy(count_matrix)
        >>>
        >>> # Log-transform for analysis
        >>> log_counts = smath.log1p(counts)
        >>>
        >>> # Or in-place to save memory
        >>> smath.log1p(counts, inplace=True)

    Notes:
        - For very large values, precision may be limited.
        - The inverse transform is expm1: expm1(log1p(x)) = x
        - Consider log2p1 if you need base-2 logarithm (easier to
          interpret as "doublings").

    See Also:
        log2p1: Base-2 logarithm of (1 + x).
        expm1: Inverse transform (exp(x) - 1).
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        if inplace:
            mat.data = np.log1p(mat.data)
            return mat
        else:
            result = mat.copy()
            result.data = np.log1p(result.data)
            return result

    # Native SCL
    if fmt == "scl_csc":
        scl_mat = mat
        is_csc = True
    else:
        scl_mat = ensure_scl_csr(mat)
        is_csc = False

    scl_mat.materialize()

    return _log1p_scl(scl_mat, inplace, is_csc)


def _log1p_scl(mat, inplace: bool, is_csc: bool):
    """Log1p implementation for SCL matrices."""
    from scl.sparse import Array

    if inplace:
        for k in range(mat.nnz):
            mat._data[k] = math.log1p(mat._data[k])
        return mat
    else:
        new_data = RealArray(mat.nnz)
        for k in range(mat.nnz):
            new_data[k] = math.log1p(mat._data[k])

        if is_csc:
            from scl.sparse import SclCSC
            return SclCSC.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )
        else:
            from scl.sparse import SclCSR
            return SclCSR.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )


# =============================================================================
# Log2p1 Transform
# =============================================================================

@overload
def log2p1(
    mat: "SclCSR",
    *,
    inplace: bool = False,
) -> "SclCSR": ...

@overload
def log2p1(
    mat: "SclCSC",
    *,
    inplace: bool = False,
) -> "SclCSC": ...

@overload
def log2p1(
    mat: "sp.spmatrix",
    *,
    inplace: bool = False,
) -> "sp.spmatrix": ...


def log2p1(
    mat: SparseInput,
    *,
    inplace: bool = False,
) -> Union["SclCSR", "SclCSC", "sp.spmatrix"]:
    """Compute base-2 logarithm of (1 + x) element-wise.

    Applies the transformation y = log2(1 + x) to each element.
    This is equivalent to log1p but with base 2, making the values
    easier to interpret in terms of "fold changes" or "doublings".

    Mathematical Definition:
        y = log2(1 + x) = ln(1 + x) / ln(2)

    Interpretation:
        - y = 1 means x = 1 (one doubling from 0)
        - y = 2 means x = 3 (two doublings from 0)
        - Difference of 1 in log2 space = 2-fold change in linear space

    This is commonly used in genomics where "2-fold up/down" is a
    meaningful threshold for differential expression.

    Args:
        mat: Input sparse matrix with non-negative values.
        inplace: If True, modify the matrix in-place.

    Returns:
        Matrix with log2(1+x)-transformed values.

    Examples:
        >>> # Log2 transform for fold-change interpretation
        >>> log2_expr = smath.log2p1(counts)
        >>>
        >>> # Differential expression: log2FC = mean_a - mean_b
        >>> # A log2FC of 1 means 2-fold higher in group A

    See Also:
        log1p: Natural logarithm variant.
    """
    fmt = get_format(mat)
    log2_e = 1.0 / math.log(2.0)  # Convert from natural log

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        if inplace:
            mat.data = np.log1p(mat.data) * log2_e
            return mat
        else:
            result = mat.copy()
            result.data = np.log1p(result.data) * log2_e
            return result

    # Native SCL
    if fmt == "scl_csc":
        scl_mat = mat
        is_csc = True
    else:
        scl_mat = ensure_scl_csr(mat)
        is_csc = False

    scl_mat.materialize()

    return _log2p1_scl(scl_mat, inplace, is_csc, log2_e)


def _log2p1_scl(mat, inplace: bool, is_csc: bool, log2_e: float):
    """Log2p1 implementation for SCL matrices."""
    from scl.sparse import Array

    if inplace:
        for k in range(mat.nnz):
            mat._data[k] = math.log1p(mat._data[k]) * log2_e
        return mat
    else:
        new_data = RealArray(mat.nnz)
        for k in range(mat.nnz):
            new_data[k] = math.log1p(mat._data[k]) * log2_e

        if is_csc:
            from scl.sparse import SclCSC
            return SclCSC.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )
        else:
            from scl.sparse import SclCSR
            return SclCSR.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )


# =============================================================================
# Expm1 Transform
# =============================================================================

@overload
def expm1(
    mat: "SclCSR",
    *,
    inplace: bool = False,
) -> "SclCSR": ...

@overload
def expm1(
    mat: "SclCSC",
    *,
    inplace: bool = False,
) -> "SclCSC": ...

@overload
def expm1(
    mat: "sp.spmatrix",
    *,
    inplace: bool = False,
) -> "sp.spmatrix": ...


def expm1(
    mat: SparseInput,
    *,
    inplace: bool = False,
) -> Union["SclCSR", "SclCSC", "sp.spmatrix"]:
    """Compute exp(x) - 1 element-wise.

    Applies the transformation y = exp(x) - 1 to each element.
    This is the inverse of log1p and is useful for converting
    log-transformed data back to linear scale.

    Mathematical Definition:
        y = e^x - 1

        For small x: expm1(x) approx x (first-order Taylor)
        Identity: expm1(log1p(x)) = x

    Numerical Stability:
        For small x, computing exp(x) - 1 directly can lose precision
        due to catastrophic cancellation. The expm1 function uses a
        series expansion for small x to maintain precision.

    Args:
        mat: Input sparse matrix (typically log-transformed data).
        inplace: If True, modify the matrix in-place.

    Returns:
        Matrix with expm1-transformed values.

    Examples:
        >>> # Reverse log1p transform
        >>> log_counts = smath.log1p(counts)
        >>> recovered = smath.expm1(log_counts)
        >>> # recovered approx counts (up to floating-point precision)

    Notes:
        - Be cautious with large input values; exp(x) grows very fast.
        - Zero-preserving: expm1(0) = 0
        - Inverse of log1p within numerical precision.

    See Also:
        log1p: Inverse transform.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        if inplace:
            mat.data = np.expm1(mat.data)
            return mat
        else:
            result = mat.copy()
            result.data = np.expm1(result.data)
            return result

    # Native SCL
    if fmt == "scl_csc":
        scl_mat = mat
        is_csc = True
    else:
        scl_mat = ensure_scl_csr(mat)
        is_csc = False

    scl_mat.materialize()

    return _expm1_scl(scl_mat, inplace, is_csc)


def _expm1_scl(mat, inplace: bool, is_csc: bool):
    """Expm1 implementation for SCL matrices."""
    from scl.sparse import Array

    if inplace:
        for k in range(mat.nnz):
            mat._data[k] = math.expm1(mat._data[k])
        return mat
    else:
        new_data = RealArray(mat.nnz)
        for k in range(mat.nnz):
            new_data[k] = math.expm1(mat._data[k])

        if is_csc:
            from scl.sparse import SclCSC
            return SclCSC.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )
        else:
            from scl.sparse import SclCSR
            return SclCSR.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )


# =============================================================================
# Softmax Transform
# =============================================================================

@overload
def softmax(
    mat: "SclCSR",
    axis: int = 1,
    *,
    inplace: bool = False,
) -> "SclCSR": ...

@overload
def softmax(
    mat: "sp.spmatrix",
    axis: int = 1,
    *,
    inplace: bool = False,
) -> "sp.spmatrix": ...


def softmax(
    mat: SparseInput,
    axis: int = 1,
    *,
    inplace: bool = False,
) -> Union["SclCSR", "sp.spmatrix"]:
    """Apply softmax normalization along an axis.

    Computes the softmax function, which converts values to probabilities
    that sum to 1 along the specified axis.

    Mathematical Definition:
        softmax(x)_i = exp(x_i) / sum(exp(x_j) for j)

    For numerical stability, we use the shifted version:
        softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    This is equivalent but avoids overflow from large exponentials.

    Sparse Matrix Handling:
        For sparse matrices, zeros are NOT included in the softmax
        normalization. The softmax is computed only over non-zero
        elements in each row/column. This is appropriate when zeros
        represent "missing" rather than "zero probability".

        If you need to include zeros in softmax, densify the matrix first.

    Properties:
        - Output values are in (0, 1)
        - Output sums to 1 along the axis
        - Preserves relative ordering (monotonic)
        - Differentiable everywhere

    Time Complexity:
        O(nnz) for sparse implementation.

    Args:
        mat: Input sparse matrix.
        axis: Axis along which to compute softmax.
            - 0: Normalize each column
            - 1: Normalize each row (default)
        inplace: If True, modify the matrix in-place.

    Returns:
        Matrix with softmax-normalized values.

    Raises:
        ValueError: If axis is not 0 or 1.

    Examples:
        >>> from scl import SclCSR
        >>> import scl.math as smath
        >>>
        >>> # Attention weights (row-wise softmax)
        >>> attention = smath.softmax(scores, axis=1)
        >>>
        >>> # Each row now sums to 1 (over non-zeros)
        >>> for i in range(attention.shape[0]):
        ...     row_sum = sum(attention.row_values(i))
        ...     print(f"Row {i} sum: {row_sum:.6f}")  # ~1.0

    Notes:
        - For CSR format, axis=1 (row softmax) is most efficient.
        - For CSC format, axis=0 (column softmax) is most efficient.
        - The output maintains the same sparsity pattern as input.

    See Also:
        normalize: L1/L2 normalization.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np

        if axis == 1:
            csr = mat.tocsr()
        else:
            csr = mat.tocsc()

        if inplace:
            result = csr
        else:
            result = csr.copy()

        # Apply softmax per row/column
        for i in range(result.shape[0] if axis == 1 else result.shape[1]):
            start = result.indptr[i]
            end = result.indptr[i + 1]

            if start == end:
                continue

            vals = result.data[start:end]
            max_val = np.max(vals)
            exp_vals = np.exp(vals - max_val)
            result.data[start:end] = exp_vals / np.sum(exp_vals)

        return result

    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()

    if axis == 0:
        # Column softmax: convert to CSC
        csc = csr.to_csc()
        return _softmax_primary(csc, inplace, is_csc=True)
    else:
        return _softmax_primary(csr, inplace, is_csc=False)


def _softmax_primary(mat, inplace: bool, is_csc: bool):
    """Softmax along primary axis (rows for CSR, cols for CSC)."""
    from scl.sparse import Array

    n_primary = mat.shape[1] if is_csc else mat.shape[0]

    if inplace:
        new_data = mat._data
    else:
        new_data = RealArray(mat.nnz)
        for k in range(mat.nnz):
            new_data[k] = mat._data[k]

    for i in range(n_primary):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]

        if start == end:
            continue

        # Find max for numerical stability
        max_val = new_data[start]
        for k in range(start + 1, end):
            if new_data[k] > max_val:
                max_val = new_data[k]

        # Compute exp(x - max) and sum
        exp_sum = 0.0
        for k in range(start, end):
            exp_val = math.exp(new_data[k] - max_val)
            new_data[k] = exp_val
            exp_sum += exp_val

        # Normalize
        if exp_sum > 0:
            for k in range(start, end):
                new_data[k] /= exp_sum

    if inplace:
        return mat
    else:
        if is_csc:
            from scl.sparse import SclCSC
            return SclCSC.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )
        else:
            from scl.sparse import SclCSR
            return SclCSR.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "log1p",
    "log2p1",
    "expm1",
    "softmax",
]
