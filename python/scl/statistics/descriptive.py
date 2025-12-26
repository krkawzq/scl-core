"""
Descriptive Statistics for Sparse Matrices.

This module provides fundamental statistical operations optimized for sparse
data structures. All functions handle the implicit zeros correctly and use
numerically stable algorithms where applicable.

Mathematical Background:
    For a sparse matrix X with shape (m, n), these operations compute
    statistics while properly accounting for both explicit non-zero values
    and implicit zero values in the sparse representation.

Supported Input Formats:
    - SclCSR / SclCSC: Native SCL sparse matrices
    - scipy.sparse.csr_matrix / csc_matrix: SciPy sparse matrices
    - numpy.ndarray: Dense arrays (converted to sparse internally)

Performance Notes:
    - Row operations (axis=1) are efficient for CSR format
    - Column operations (axis=0) are efficient for CSC format
    - The functions automatically choose optimal iteration order
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Union, overload

from scl._typing import (
    SparseInput,
    CSRInput,
    CSCInput,
    VectorInput,
    ensure_scl_csr,
    ensure_scl_csc,
    get_format,
    is_scipy_sparse,
    is_numpy_array,
)

if TYPE_CHECKING:
    import numpy as np
    from scipy import sparse as sp
    from scl.sparse import SclCSR, SclCSC
    from scl.array import RealArray, IndexArray


# =============================================================================
# Sum
# =============================================================================

@overload
def sum(
    mat: "SclCSR",
    axis: None = None,
) -> float: ...

@overload
def sum(
    mat: "SclCSR",
    axis: int,
) -> "RealArray": ...

@overload
def sum(
    mat: "SclCSC",
    axis: None = None,
) -> float: ...

@overload
def sum(
    mat: "SclCSC",
    axis: int,
) -> "RealArray": ...

@overload
def sum(
    mat: "sp.spmatrix",
    axis: None = None,
) -> float: ...

@overload
def sum(
    mat: "sp.spmatrix",
    axis: int,
) -> "np.ndarray": ...

@overload
def sum(
    mat: "np.ndarray",
    axis: Optional[int] = None,
) -> Union[float, "np.ndarray"]: ...


def sum(
    mat: SparseInput,
    axis: Optional[int] = None,
) -> Union[float, "RealArray", "np.ndarray"]:
    """Compute the sum of matrix elements.

    Calculates the sum of all elements, or the sum along a specified axis.
    For sparse matrices, only non-zero elements contribute to the sum,
    which is mathematically equivalent to summing all elements including
    implicit zeros.

    Algorithm:
        For global sum (axis=None):
            S = sum(x_ij for all stored (i,j))

        For row sums (axis=1, CSR):
            S_i = sum(data[indptr[i]:indptr[i+1]])

        For column sums (axis=0, CSC):
            S_j = sum(data[indptr[j]:indptr[j+1]])

    Time Complexity:
        O(nnz) for all cases, where nnz is the number of non-zeros.

    Args:
        mat: Input sparse matrix. Accepts SclCSR, SclCSC, scipy.sparse,
            or numpy.ndarray.
        axis: Axis along which to sum.
            - None: Sum all elements (returns scalar)
            - 0: Sum along columns (returns array of length cols)
            - 1: Sum along rows (returns array of length rows)

    Returns:
        - If axis is None: float, the total sum
        - If axis is 0: Array of column sums (length = cols)
        - If axis is 1: Array of row sums (length = rows)

        Return type matches input: RealArray for SCL types,
        numpy.ndarray for scipy/numpy inputs.

    Raises:
        ValueError: If axis is not None, 0, or 1.

    Examples:
        >>> from scl import SclCSR
        >>> mat = SclCSR.from_arrays([1, 2, 3, 4], [0, 1, 0, 1],
        ...                          [0, 2, 4], (2, 2))
        >>> # Matrix: [[1, 2], [3, 4]]
        >>>
        >>> sum(mat)  # Total sum
        10.0
        >>>
        >>> sum(mat, axis=0)  # Column sums
        RealArray([4.0, 6.0])
        >>>
        >>> sum(mat, axis=1)  # Row sums
        RealArray([3.0, 7.0])

    See Also:
        mean: Compute the mean of elements.
        var: Compute the variance of elements.
    """
    fmt = get_format(mat)

    # Handle scipy/numpy - delegate to their implementations
    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        result = mat.sum(axis=axis)
        if axis is not None:
            import numpy as np
            return np.asarray(result).ravel()
        return float(result)

    if fmt == "numpy":
        import numpy as np
        return np.sum(mat, axis=axis)

    # Native SCL implementation
    if fmt in ("scl_csr", "scl_csc"):
        scl_mat = mat
    else:
        scl_mat = ensure_scl_csr(mat)

    scl_mat.materialize()
    return scl_mat.sum(axis=axis)


# =============================================================================
# Mean
# =============================================================================

@overload
def mean(
    mat: "SclCSR",
    axis: None = None,
    *,
    skip_zeros: bool = False,
) -> float: ...

@overload
def mean(
    mat: "SclCSR",
    axis: int,
    *,
    skip_zeros: bool = False,
) -> "RealArray": ...

@overload
def mean(
    mat: "SclCSC",
    axis: None = None,
    *,
    skip_zeros: bool = False,
) -> float: ...

@overload
def mean(
    mat: "SclCSC",
    axis: int,
    *,
    skip_zeros: bool = False,
) -> "RealArray": ...

@overload
def mean(
    mat: "sp.spmatrix",
    axis: Optional[int] = None,
    *,
    skip_zeros: bool = False,
) -> Union[float, "np.ndarray"]: ...

@overload
def mean(
    mat: "np.ndarray",
    axis: Optional[int] = None,
    *,
    skip_zeros: bool = False,
) -> Union[float, "np.ndarray"]: ...


def mean(
    mat: SparseInput,
    axis: Optional[int] = None,
    *,
    skip_zeros: bool = False,
) -> Union[float, "RealArray", "np.ndarray"]:
    """Compute the arithmetic mean of matrix elements.

    Calculates the mean of all elements or along a specified axis.
    By default, includes implicit zeros in the calculation.

    Algorithm:
        For global mean (axis=None):
            mu = sum(X) / (m * n)

        For row means (axis=1):
            mu_i = sum(row_i) / n

        For column means (axis=0):
            mu_j = sum(col_j) / m

        When skip_zeros=True:
            mu = sum(X) / nnz
            (only non-zero elements are counted)

    Mathematical Properties:
        - The mean of a sparse matrix is typically close to zero
          due to the large number of implicit zeros.
        - Use skip_zeros=True for computing mean of non-zero values only,
          which is useful for log-transformed data.

    Time Complexity:
        O(nnz) for computation, O(m) or O(n) for output allocation.

    Args:
        mat: Input sparse matrix.
        axis: Axis along which to compute mean.
            - None: Mean of all elements
            - 0: Mean along columns (returns array of length cols)
            - 1: Mean along rows (returns array of length rows)
        skip_zeros: If True, only average non-zero elements.
            Useful for computing mean expression in single-cell data.

    Returns:
        - If axis is None: float, the global mean
        - If axis is 0: Array of column means
        - If axis is 1: Array of row means

    Examples:
        >>> mat = SclCSR.from_arrays([1, 2, 3], [0, 1, 0],
        ...                          [0, 2, 3], (2, 3))
        >>> # Matrix: [[1, 2, 0], [3, 0, 0]]
        >>>
        >>> mean(mat)  # (1+2+3+0+0+0) / 6 = 1.0
        1.0
        >>>
        >>> mean(mat, skip_zeros=True)  # (1+2+3) / 3 = 2.0
        2.0
        >>>
        >>> mean(mat, axis=1)  # Row means
        RealArray([1.0, 1.0])  # [3/3, 3/3]

    See Also:
        sum: Compute the sum of elements.
        var: Compute the variance of elements.
    """
    fmt = get_format(mat)

    # Handle scipy
    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        if skip_zeros:
            import numpy as np
            if axis is None:
                return float(mat.sum()) / mat.nnz if mat.nnz > 0 else 0.0
            elif axis == 0:
                # Column means of non-zeros
                sums = np.asarray(mat.sum(axis=0)).ravel()
                nnz_counts = np.diff(mat.tocsc().indptr)
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.where(nnz_counts > 0, sums / nnz_counts, 0.0)
                return result
            else:
                sums = np.asarray(mat.sum(axis=1)).ravel()
                nnz_counts = np.diff(mat.tocsr().indptr)
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.where(nnz_counts > 0, sums / nnz_counts, 0.0)
                return result
        else:
            result = mat.mean(axis=axis)
            if axis is not None:
                import numpy as np
                return np.asarray(result).ravel()
            return float(result)

    if fmt == "numpy":
        import numpy as np
        if skip_zeros:
            mask = mat != 0
            if axis is None:
                return float(np.sum(mat[mask]) / np.sum(mask)) if np.any(mask) else 0.0
            else:
                sums = np.sum(mat, axis=axis)
                counts = np.sum(mask, axis=axis)
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.where(counts > 0, sums / counts, 0.0)
        return np.mean(mat, axis=axis)

    # Native SCL implementation
    if fmt in ("scl_csr", "scl_csc"):
        scl_mat = mat
    else:
        scl_mat = ensure_scl_csr(mat)

    scl_mat.materialize()

    if skip_zeros:
        return _mean_skip_zeros(scl_mat, axis)
    else:
        return scl_mat.mean(axis=axis)


def _mean_skip_zeros(mat, axis: Optional[int]) -> Union[float, "RealArray"]:
    """Compute mean excluding implicit zeros."""
    from scl.array import RealArray

    mat.materialize()

    if axis is None:
        total = 0.0
        for k in range(mat.nnz):
            total += mat._data[k]
        return total / mat.nnz if mat.nnz > 0 else 0.0

    elif axis == 1:  # Row means
        result = RealArray(mat.shape[0])
        for i in range(mat.shape[0]):
            start = mat._indptr[i]
            end = mat._indptr[i + 1]
            nnz_row = end - start
            if nnz_row > 0:
                total = 0.0
                for k in range(start, end):
                    total += mat._data[k]
                result[i] = total / nnz_row
            else:
                result[i] = 0.0
        return result

    else:  # Column means (axis=0)
        # Convert to CSC for efficient column access
        from scl._typing import is_scl_csc
        if is_scl_csc(mat):
            csc = mat
        else:
            csc = mat.to_csc()

        result = RealArray(mat.shape[1])
        for j in range(mat.shape[1]):
            start = csc._indptr[j]
            end = csc._indptr[j + 1]
            nnz_col = end - start
            if nnz_col > 0:
                total = 0.0
                for k in range(start, end):
                    total += csc._data[k]
                result[j] = total / nnz_col
            else:
                result[j] = 0.0
        return result


# =============================================================================
# Variance
# =============================================================================

@overload
def var(
    mat: "SclCSR",
    axis: None = None,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> float: ...

@overload
def var(
    mat: "SclCSR",
    axis: int,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> "RealArray": ...

@overload
def var(
    mat: "SclCSC",
    axis: None = None,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> float: ...

@overload
def var(
    mat: "SclCSC",
    axis: int,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> "RealArray": ...

@overload
def var(
    mat: "sp.spmatrix",
    axis: Optional[int] = None,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> Union[float, "np.ndarray"]: ...


def var(
    mat: SparseInput,
    axis: Optional[int] = None,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> Union[float, "RealArray", "np.ndarray"]:
    """Compute the variance of matrix elements.

    Calculates the variance using the two-pass algorithm for numerical
    stability. Properly accounts for implicit zeros in sparse representation.

    Algorithm (Two-Pass):
        1. First pass: Compute mean mu
        2. Second pass: Compute sum of squared deviations

        Var(X) = (1/(n-ddof)) * sum((x_i - mu)^2)

        For sparse matrices with nnz non-zeros and N total elements:
        Var(X) = (1/(N-ddof)) * [sum((x_k - mu)^2 for non-zeros) +
                                  (N - nnz) * mu^2]

    Numerical Stability:
        The two-pass algorithm is used instead of the naive formula
        Var = E[X^2] - E[X]^2 to avoid catastrophic cancellation
        when the mean is large relative to the standard deviation.

    Time Complexity:
        O(2 * nnz) for the two passes through non-zero elements.

    Args:
        mat: Input sparse matrix.
        axis: Axis along which to compute variance.
            - None: Variance of all elements
            - 0: Variance along columns
            - 1: Variance along rows
        ddof: Delta degrees of freedom. The divisor is N - ddof,
            where N is the number of elements. Use ddof=1 for sample
            variance (Bessel's correction).
        skip_zeros: If True, compute variance of non-zero elements only.

    Returns:
        Variance value(s) matching input type.

    Raises:
        ValueError: If ddof >= number of elements.

    Examples:
        >>> mat = SclCSR.from_arrays([1, 3], [0, 0], [0, 1, 2], (2, 2))
        >>> # Matrix: [[1, 0], [3, 0]]
        >>>
        >>> var(mat)  # Population variance
        1.25  # Mean=1, var = ((1-1)^2 + (0-1)^2 + (3-1)^2 + (0-1)^2)/4
        >>>
        >>> var(mat, ddof=1)  # Sample variance
        1.6666...  # Same numerator / 3

    See Also:
        std: Compute standard deviation (sqrt of variance).
        mean: Compute the mean of elements.
    """
    fmt = get_format(mat)

    # Handle scipy
    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        dense = mat.toarray()
        if skip_zeros:
            mask = dense != 0
            if axis is None:
                vals = dense[mask]
                return float(np.var(vals, ddof=ddof)) if len(vals) > 0 else 0.0
            else:
                # More complex for axis-wise
                return np.var(dense, axis=axis, ddof=ddof)
        return np.var(dense, axis=axis, ddof=ddof)

    if fmt == "numpy":
        import numpy as np
        return np.var(mat, axis=axis, ddof=ddof)

    # Native SCL implementation
    if fmt in ("scl_csr", "scl_csc"):
        scl_mat = mat
    else:
        scl_mat = ensure_scl_csr(mat)

    scl_mat.materialize()

    if skip_zeros:
        return _var_skip_zeros(scl_mat, axis, ddof)
    else:
        return _var_full(scl_mat, axis, ddof)


def _var_full(mat, axis: Optional[int], ddof: int) -> Union[float, "RealArray"]:
    """Compute variance including implicit zeros."""
    from scl.array import RealArray
    from scl._typing import is_scl_csc

    mat.materialize()

    if axis is None:
        # Global variance
        n_total = mat.shape[0] * mat.shape[1]
        mean_val = mat.mean()

        sq_sum = 0.0
        for k in range(mat.nnz):
            sq_sum += (mat._data[k] - mean_val) ** 2

        # Add zero contributions
        n_zeros = n_total - mat.nnz
        sq_sum += n_zeros * mean_val ** 2

        denom = n_total - ddof
        return sq_sum / denom if denom > 0 else 0.0

    elif axis == 1:  # Row variance
        means = mat.mean(axis=1)
        result = RealArray(mat.shape[0])

        for i in range(mat.shape[0]):
            start = mat._indptr[i]
            end = mat._indptr[i + 1]
            mean_val = means[i]

            sq_sum = 0.0
            for k in range(start, end):
                sq_sum += (mat._data[k] - mean_val) ** 2

            n_zeros = mat.shape[1] - (end - start)
            sq_sum += n_zeros * mean_val ** 2

            denom = mat.shape[1] - ddof
            result[i] = sq_sum / denom if denom > 0 else 0.0

        return result

    else:  # Column variance (axis=0)
        if is_scl_csc(mat):
            csc = mat
        else:
            csc = mat.to_csc()

        means = csc.mean(axis=0)
        result = RealArray(mat.shape[1])

        for j in range(mat.shape[1]):
            start = csc._indptr[j]
            end = csc._indptr[j + 1]
            mean_val = means[j]

            sq_sum = 0.0
            for k in range(start, end):
                sq_sum += (csc._data[k] - mean_val) ** 2

            n_zeros = mat.shape[0] - (end - start)
            sq_sum += n_zeros * mean_val ** 2

            denom = mat.shape[0] - ddof
            result[j] = sq_sum / denom if denom > 0 else 0.0

        return result


def _var_skip_zeros(mat, axis: Optional[int], ddof: int) -> Union[float, "RealArray"]:
    """Compute variance of non-zero elements only."""
    from scl.array import RealArray
    from scl._typing import is_scl_csc

    mat.materialize()

    if axis is None:
        if mat.nnz <= ddof:
            return 0.0

        # First pass: mean
        total = 0.0
        for k in range(mat.nnz):
            total += mat._data[k]
        mean_val = total / mat.nnz

        # Second pass: variance
        sq_sum = 0.0
        for k in range(mat.nnz):
            sq_sum += (mat._data[k] - mean_val) ** 2

        return sq_sum / (mat.nnz - ddof)

    elif axis == 1:  # Row variance
        result = RealArray(mat.shape[0])

        for i in range(mat.shape[0]):
            start = mat._indptr[i]
            end = mat._indptr[i + 1]
            nnz_row = end - start

            if nnz_row <= ddof:
                result[i] = 0.0
                continue

            # Mean
            total = 0.0
            for k in range(start, end):
                total += mat._data[k]
            mean_val = total / nnz_row

            # Variance
            sq_sum = 0.0
            for k in range(start, end):
                sq_sum += (mat._data[k] - mean_val) ** 2

            result[i] = sq_sum / (nnz_row - ddof)

        return result

    else:  # Column variance
        if is_scl_csc(mat):
            csc = mat
        else:
            csc = mat.to_csc()

        result = RealArray(mat.shape[1])

        for j in range(mat.shape[1]):
            start = csc._indptr[j]
            end = csc._indptr[j + 1]
            nnz_col = end - start

            if nnz_col <= ddof:
                result[j] = 0.0
                continue

            total = 0.0
            for k in range(start, end):
                total += csc._data[k]
            mean_val = total / nnz_col

            sq_sum = 0.0
            for k in range(start, end):
                sq_sum += (csc._data[k] - mean_val) ** 2

            result[j] = sq_sum / (nnz_col - ddof)

        return result


# =============================================================================
# Standard Deviation
# =============================================================================

@overload
def std(
    mat: "SclCSR",
    axis: None = None,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> float: ...

@overload
def std(
    mat: "SclCSR",
    axis: int,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> "RealArray": ...

@overload
def std(
    mat: "SclCSC",
    axis: None = None,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> float: ...

@overload
def std(
    mat: "SclCSC",
    axis: int,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> "RealArray": ...

@overload
def std(
    mat: "sp.spmatrix",
    axis: Optional[int] = None,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> Union[float, "np.ndarray"]: ...


def std(
    mat: SparseInput,
    axis: Optional[int] = None,
    *,
    ddof: int = 0,
    skip_zeros: bool = False,
) -> Union[float, "RealArray", "np.ndarray"]:
    """Compute the standard deviation of matrix elements.

    Calculates the standard deviation as the square root of variance.
    See var() for algorithm details.

    Algorithm:
        std(X) = sqrt(var(X))

    Args:
        mat: Input sparse matrix.
        axis: Axis along which to compute std.
            - None: Std of all elements
            - 0: Std along columns
            - 1: Std along rows
        ddof: Delta degrees of freedom (see var()).
        skip_zeros: If True, compute std of non-zero elements only.

    Returns:
        Standard deviation value(s).

    Examples:
        >>> mat = SclCSR.from_arrays([1, 3], [0, 0], [0, 1, 2], (2, 2))
        >>> std(mat)  # sqrt(1.25) = 1.118...
        1.118033988749895

    See Also:
        var: Compute variance.
    """
    variance = var(mat, axis=axis, ddof=ddof, skip_zeros=skip_zeros)

    fmt = get_format(mat)
    if fmt in ("scipy_csr", "scipy_csc", "scipy_other", "numpy"):
        import numpy as np
        return np.sqrt(variance)

    # SCL output
    if isinstance(variance, float):
        return math.sqrt(variance)
    else:
        from scl.array import RealArray
        result = RealArray(variance.size)
        for i in range(variance.size):
            result[i] = math.sqrt(variance[i])
        return result


# =============================================================================
# Min / Max
# =============================================================================

@overload
def min(
    mat: "SclCSR",
    axis: None = None,
) -> float: ...

@overload
def min(
    mat: "SclCSR",
    axis: int,
) -> "RealArray": ...

@overload
def min(
    mat: "sp.spmatrix",
    axis: Optional[int] = None,
) -> Union[float, "np.ndarray"]: ...


def min(
    mat: SparseInput,
    axis: Optional[int] = None,
) -> Union[float, "RealArray", "np.ndarray"]:
    """Compute the minimum value of matrix elements.

    Finds the minimum value considering both non-zero elements and
    implicit zeros.

    Algorithm:
        min(X) = min(min(non-zeros), 0) if nnz < m*n
               = min(non-zeros) otherwise

    Note:
        For sparse matrices with fewer non-zeros than total elements,
        the minimum is at most 0 (unless all values are positive and
        there are no implicit zeros).

    Args:
        mat: Input sparse matrix.
        axis: Axis along which to compute minimum.
            - None: Global minimum
            - 0: Minimum along columns
            - 1: Minimum along rows

    Returns:
        Minimum value(s).

    Examples:
        >>> mat = SclCSR.from_arrays([1, 2, 3], [0, 1, 0],
        ...                          [0, 2, 3], (2, 3))
        >>> min(mat)  # Has implicit zeros
        0.0
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        dense = mat.toarray()
        return np.min(dense, axis=axis)

    if fmt == "numpy":
        import numpy as np
        return np.min(mat, axis=axis)

    # Native SCL
    if fmt in ("scl_csr", "scl_csc"):
        scl_mat = mat
    else:
        scl_mat = ensure_scl_csr(mat)

    scl_mat.materialize()
    return scl_mat.min(axis=axis)


@overload
def max(
    mat: "SclCSR",
    axis: None = None,
) -> float: ...

@overload
def max(
    mat: "SclCSR",
    axis: int,
) -> "RealArray": ...

@overload
def max(
    mat: "sp.spmatrix",
    axis: Optional[int] = None,
) -> Union[float, "np.ndarray"]: ...


def max(
    mat: SparseInput,
    axis: Optional[int] = None,
) -> Union[float, "RealArray", "np.ndarray"]:
    """Compute the maximum value of matrix elements.

    Finds the maximum value considering both non-zero elements and
    implicit zeros.

    Algorithm:
        max(X) = max(max(non-zeros), 0) if nnz < m*n
               = max(non-zeros) otherwise

    Args:
        mat: Input sparse matrix.
        axis: Axis along which to compute maximum.
            - None: Global maximum
            - 0: Maximum along columns
            - 1: Maximum along rows

    Returns:
        Maximum value(s).
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        dense = mat.toarray()
        return np.max(dense, axis=axis)

    if fmt == "numpy":
        import numpy as np
        return np.max(mat, axis=axis)

    # Native SCL
    if fmt in ("scl_csr", "scl_csc"):
        scl_mat = mat
    else:
        scl_mat = ensure_scl_csr(mat)

    scl_mat.materialize()
    return scl_mat.max(axis=axis)


# =============================================================================
# NNZ Count
# =============================================================================

@overload
def nnz_count(
    mat: "SclCSR",
    axis: None = None,
) -> int: ...

@overload
def nnz_count(
    mat: "SclCSR",
    axis: int,
) -> "IndexArray": ...

@overload
def nnz_count(
    mat: "sp.spmatrix",
    axis: None = None,
) -> int: ...

@overload
def nnz_count(
    mat: "sp.spmatrix",
    axis: int,
) -> "np.ndarray": ...


def nnz_count(
    mat: SparseInput,
    axis: Optional[int] = None,
) -> Union[int, "IndexArray", "np.ndarray"]:
    """Count the number of non-zero elements.

    Counts explicit non-zero values in the sparse representation.
    This is the sparsity complement: density = nnz / (m * n).

    Args:
        mat: Input sparse matrix.
        axis: Axis along which to count.
            - None: Total count
            - 0: Count per column
            - 1: Count per row

    Returns:
        - If axis is None: int, total non-zero count
        - If axis is 0: Array of per-column counts
        - If axis is 1: Array of per-row counts

    Examples:
        >>> mat = SclCSR.from_arrays([1, 2, 3], [0, 1, 0],
        ...                          [0, 2, 3], (2, 3))
        >>> nnz_count(mat)
        3
        >>> nnz_count(mat, axis=1)  # Per row
        IndexArray([2, 1])
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        if axis is None:
            return int(mat.nnz)
        elif axis == 0:
            return np.diff(mat.tocsc().indptr)
        else:
            return np.diff(mat.tocsr().indptr)

    if fmt == "numpy":
        import numpy as np
        if axis is None:
            return int(np.count_nonzero(mat))
        return np.count_nonzero(mat, axis=axis)

    # Native SCL
    from scl.array import IndexArray
    from scl._typing import is_scl_csc

    if fmt in ("scl_csr", "scl_csc"):
        scl_mat = mat
    else:
        scl_mat = ensure_scl_csr(mat)

    scl_mat.materialize()

    if axis is None:
        return scl_mat.nnz

    elif axis == 1:  # Per row
        result = IndexArray(scl_mat.shape[0])
        for i in range(scl_mat.shape[0]):
            result[i] = scl_mat._indptr[i + 1] - scl_mat._indptr[i]
        return result

    else:  # Per column (axis=0)
        if is_scl_csc(scl_mat):
            csc = scl_mat
        else:
            csc = scl_mat.to_csc()

        result = IndexArray(scl_mat.shape[1])
        for j in range(scl_mat.shape[1]):
            result[j] = csc._indptr[j + 1] - csc._indptr[j]
        return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "sum",
    "mean",
    "var",
    "std",
    "min",
    "max",
    "nnz_count",
]
