"""
Scaling Operations for Sparse Matrices.

This module provides scaling operations for sparse matrices, allowing
multiplication of rows or columns by specified factors.

Scaling is useful for:
    - Library size normalization (scaling to target sum)
    - Batch effect correction (scaling factors from reference)
    - Custom weighting of features/samples
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Union, overload

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


# =============================================================================
# Scale
# =============================================================================

@overload
def scale(
    mat: "SclCSR",
    *,
    row_factors: Optional["RealArray"] = None,
    col_factors: Optional["RealArray"] = None,
    inplace: bool = False,
) -> "SclCSR": ...

@overload
def scale(
    mat: "SclCSC",
    *,
    row_factors: Optional["RealArray"] = None,
    col_factors: Optional["RealArray"] = None,
    inplace: bool = False,
) -> "SclCSC": ...

@overload
def scale(
    mat: "sp.spmatrix",
    *,
    row_factors: Optional["np.ndarray"] = None,
    col_factors: Optional["np.ndarray"] = None,
    inplace: bool = False,
) -> "sp.spmatrix": ...


def scale(
    mat: SparseInput,
    *,
    row_factors: Optional[VectorInput] = None,
    col_factors: Optional[VectorInput] = None,
    inplace: bool = False,
) -> Union["SclCSR", "SclCSC", "sp.spmatrix"]:
    """Scale sparse matrix by row and/or column factors.

    Multiplies each row by its corresponding row factor and each column
    by its corresponding column factor. This is equivalent to:

        result[i, j] = mat[i, j] * row_factors[i] * col_factors[j]

    If only row_factors or col_factors is provided, the other is treated
    as all ones.

    Mathematical Definition:
        For diagonal matrices D_row and D_col:
            result = D_row @ mat @ D_col

        Where D_row[i, i] = row_factors[i] and D_col[j, j] = col_factors[j].

    Applications:
        - Library size normalization: row_factors = target_sum / row_sums
        - Feature weighting: col_factors = importance_weights
        - Batch correction: row_factors = batch_correction_factors
        - Inverse variance weighting: col_factors = 1 / std

    Time Complexity:
        O(nnz) where nnz is the number of non-zeros.

    Args:
        mat: Input sparse matrix of shape (m, n).
        row_factors: Array of length m, or None (treated as all ones).
            Each row i is multiplied by row_factors[i].
        col_factors: Array of length n, or None (treated as all ones).
            Each column j is multiplied by col_factors[j].
        inplace: If True, modify the matrix in-place.

    Returns:
        Scaled sparse matrix. Same format as input.

    Raises:
        ValueError: If row_factors length doesn't match number of rows.
        ValueError: If col_factors length doesn't match number of columns.

    Examples:
        >>> from scl import SclCSR
        >>> from scl.sparse import Array
        >>> import scl.preprocessing as pp
        >>>
        >>> # Scale rows to target sum (library size normalization)
        >>> row_sums = mat.sum(axis=1)
        >>> target = 10000
        >>> row_factors = Array.from_list([target / s for s in row_sums], dtype='float64')
        >>> scaled = pp.scale(mat, row_factors=row_factors)
        >>>
        >>> # Apply feature weights
        >>> weights = Array.from_list([1.0, 2.0, 0.5, 1.5], dtype='float64')
        >>> weighted = pp.scale(mat, col_factors=weights)
        >>>
        >>> # Both row and column scaling
        >>> result = pp.scale(mat, row_factors=row_scale, col_factors=col_scale)

    Notes:
        - Zero factors are allowed and will zero out corresponding rows/columns.
        - For CSR format, row scaling is more efficient.
        - For CSC format, column scaling is more efficient.
        - When both are provided, row scaling is applied first (order doesn't
          affect result due to element-wise multiplication).

    See Also:
        normalize: L1/L2/Max normalization.
        standardize: Zero-mean, unit-variance normalization.
    """
    if row_factors is None and col_factors is None:
        # No-op
        if inplace:
            return mat
        return mat.copy() if hasattr(mat, 'copy') else mat

    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _scale_scipy(mat, row_factors, col_factors, inplace)

    # Native SCL
    if fmt == "scl_csc":
        scl_mat = mat
        is_csc = True
    else:
        scl_mat = ensure_scl_csr(mat)
        is_csc = False

    scl_mat.materialize()

    return _scale_scl(scl_mat, row_factors, col_factors, inplace, is_csc)


def _scale_scipy(mat, row_factors, col_factors, inplace: bool):
    """Scale using scipy."""
    import numpy as np
    from scipy import sparse as sp

    if not inplace:
        mat = mat.copy()

    csr = mat.tocsr()

    # Apply row factors
    if row_factors is not None:
        row_factors = np.asarray(row_factors).ravel()
        if len(row_factors) != csr.shape[0]:
            raise ValueError(
                f"row_factors length {len(row_factors)} != rows {csr.shape[0]}"
            )

        for i in range(csr.shape[0]):
            start = csr.indptr[i]
            end = csr.indptr[i + 1]
            csr.data[start:end] *= row_factors[i]

    # Apply column factors
    if col_factors is not None:
        col_factors = np.asarray(col_factors).ravel()
        if len(col_factors) != csr.shape[1]:
            raise ValueError(
                f"col_factors length {len(col_factors)} != cols {csr.shape[1]}"
            )

        for i in range(csr.shape[0]):
            start = csr.indptr[i]
            end = csr.indptr[i + 1]
            for k in range(start, end):
                j = csr.indices[k]
                csr.data[k] *= col_factors[j]

    return csr


def _scale_scl(mat, row_factors, col_factors, inplace: bool, is_csc: bool):
    """Scale SCL matrix."""
    from scl.sparse import Array, SclCSR, SclCSC

    m, n = mat.shape

    # Validate and convert factors
    if row_factors is not None:
        rf = ensure_vector(row_factors, size=m)
    else:
        rf = None

    if col_factors is not None:
        cf = ensure_vector(col_factors, size=n)
    else:
        cf = None

    if inplace:
        new_data = mat._data
    else:
        new_data = Array(mat.nnz, dtype='float64')
        for k in range(mat.nnz):
            new_data[k] = mat._data[k]

    if is_csc:
        # CSC format: indptr indexes columns, indices are rows
        for j in range(n):
            start = mat._indptr[j]
            end = mat._indptr[j + 1]

            col_factor = cf[j] if cf is not None else 1.0

            for k in range(start, end):
                i = mat._indices[k]
                row_factor = rf[i] if rf is not None else 1.0
                new_data[k] *= row_factor * col_factor

        if inplace:
            return mat
        else:
            return SclCSC.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )
    else:
        # CSR format: indptr indexes rows, indices are columns
        for i in range(m):
            start = mat._indptr[i]
            end = mat._indptr[i + 1]

            row_factor = rf[i] if rf is not None else 1.0

            for k in range(start, end):
                j = mat._indices[k]
                col_factor = cf[j] if cf is not None else 1.0
                new_data[k] *= row_factor * col_factor

        if inplace:
            return mat
        else:
            return SclCSR.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "scale",
]
