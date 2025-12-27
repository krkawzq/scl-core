"""
Normalization Operations for Sparse Matrices.

This module provides various normalization methods for sparse matrices,
essential for removing technical biases in single-cell data.

Implemented Methods:
    - L1 normalization (sum to 1)
    - L2 normalization (unit Euclidean length)
    - Max normalization (max value = 1)
    - Standardization (zero mean, unit variance)

Normalization is crucial for:
    - Removing library size effects (total counts per cell)
    - Making cells/features comparable
    - Preparing data for machine learning algorithms
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
# Normalize
# =============================================================================

@overload
def normalize(
    mat: "SclCSR",
    norm: str = "l2",
    axis: int = 1,
    *,
    inplace: bool = False,
) -> "SclCSR": ...

@overload
def normalize(
    mat: "SclCSC",
    norm: str = "l2",
    axis: int = 0,
    *,
    inplace: bool = False,
) -> "SclCSC": ...

@overload
def normalize(
    mat: "sp.spmatrix",
    norm: str = "l2",
    axis: int = 1,
    *,
    inplace: bool = False,
) -> "sp.spmatrix": ...


def normalize(
    mat: SparseInput,
    norm: str = "l2",
    axis: int = 1,
    *,
    inplace: bool = False,
) -> Union["SclCSR", "SclCSC", "sp.spmatrix"]:
    """Normalize sparse matrix rows or columns.

    Scales each row (axis=1) or column (axis=0) to have a specified norm.
    This is essential for removing technical biases such as library size
    differences in single-cell data.

    Mathematical Definitions:
        For a vector v, the normalizations are:

        L1 (Manhattan):
            v_normalized = v / sum(|v_i|)
            Result: Elements sum to 1 (for non-negative data)

        L2 (Euclidean):
            v_normalized = v / sqrt(sum(v_i^2))
            Result: Unit Euclidean length

        Max (Infinity):
            v_normalized = v / max(|v_i|)
            Result: Maximum absolute value is 1

    Single-Cell Applications:
        - L1 (axis=1): Library size normalization
            Each cell's counts sum to 1 (or a target value)
            Removes sequencing depth differences

        - L2 (axis=1): Unit vector normalization
            Each cell becomes a unit vector
            Useful for cosine similarity computations

        - L1 (axis=0): Feature normalization
            Each gene's expression sums to 1 across cells
            Useful for comparing relative abundance

    Algorithm:
        For each row/column:
            1. Compute the norm (L1/L2/Max)
            2. Divide all elements by the norm
            3. Handle zero-norm vectors (leave unchanged)

    Time Complexity:
        O(nnz) where nnz is the number of non-zeros.

    Args:
        mat: Input sparse matrix.
        norm: Type of norm to use.
            - "l1": Sum of absolute values (Manhattan norm)
            - "l2": Square root of sum of squares (Euclidean norm)
            - "max": Maximum absolute value (Infinity norm)
        axis: Axis along which to normalize.
            - 0: Normalize each column
            - 1: Normalize each row (default)
        inplace: If True, modify the matrix in-place.

    Returns:
        Normalized sparse matrix. Same format as input.

    Raises:
        ValueError: If norm is not "l1", "l2", or "max".
        ValueError: If axis is not 0 or 1.

    Examples:
        >>> from scl import SclCSR
        >>> import scl.preprocessing as pp
        >>>
        >>> # L1 normalize rows (library size normalization)
        >>> normalized = pp.normalize(counts, norm="l1", axis=1)
        >>>
        >>> # L2 normalize for cosine similarity
        >>> unit_vectors = pp.normalize(mat, norm="l2", axis=1)
        >>>
        >>> # Normalize columns
        >>> col_normalized = pp.normalize(mat, norm="l1", axis=0)

    Notes:
        - Zero rows/columns remain zero (no division by zero).
        - For CSR format, axis=1 is most efficient.
        - For CSC format, axis=0 is most efficient.
        - The function automatically handles format conversion if needed.

    See Also:
        standardize: Zero-mean, unit-variance normalization.
        scale: Scale by arbitrary factors.
    """
    if norm not in ("l1", "l2", "max"):
        raise ValueError(f"norm must be 'l1', 'l2', or 'max', got '{norm}'")

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _normalize_scipy(mat, norm, axis, inplace)

    # Native SCL
    if axis == 1:
        # Row normalization - use CSR
        scl_mat = ensure_scl_csr(mat)
        scl_mat.materialize()
        return _normalize_csr_rows(scl_mat, norm, inplace)
    else:
        # Column normalization - use CSC
        scl_mat = ensure_scl_csc(mat)
        scl_mat.materialize()
        return _normalize_csc_cols(scl_mat, norm, inplace)


def _normalize_scipy(mat, norm: str, axis: int, inplace: bool):
    """Normalize using scipy."""
    import numpy as np
    from scipy import sparse as sp

    if axis == 1:
        csr = mat.tocsr()
    else:
        csr = mat.tocsc()

    if inplace:
        result = csr
    else:
        result = csr.copy()

    n_primary = result.shape[0] if axis == 1 else result.shape[1]

    for i in range(n_primary):
        start = result.indptr[i]
        end = result.indptr[i + 1]

        if start == end:
            continue

        vals = result.data[start:end]

        if norm == "l1":
            n = np.sum(np.abs(vals))
        elif norm == "l2":
            n = np.sqrt(np.sum(vals ** 2))
        else:  # max
            n = np.max(np.abs(vals))

        if n > 0:
            result.data[start:end] = vals / n

    return result


def _normalize_csr_rows(mat: "SclCSR", norm: str, inplace: bool) -> "SclCSR":
    """Normalize CSR rows using C++ kernel where available."""
    from scl.sparse import Array, SclCSR
    
    # Try to use C++ kernel for L1 normalization
    if norm == "l1":
        try:
            from scl._kernel import sparse as kernel_sparse
            from scl._kernel import normalize as kernel_normalize
            from scl._kernel.lib_loader import LibraryNotFoundError
            
            m = mat.shape[0]
            
            # Step 1: Compute row sums using kernel
            row_sums = Array(m, dtype='float64')
            data_ptr = mat.data.get_pointer()
            indices_ptr = mat.indices.get_pointer()
            indptr_ptr = mat.indptr.get_pointer()
            
            kernel_sparse.primary_sums_csr(
                data_ptr, indices_ptr, indptr_ptr,
                mat.shape[0], mat.shape[1],
                row_sums.get_pointer()
            )
            
            # Step 2: Compute inverse scales (1/sum), handle zeros
            scales = Array(m, dtype='float64')
            for i in range(m):
                scales[i] = 1.0 / row_sums[i] if row_sums[i] > 0 else 0.0
            
            # Step 3: Apply scaling using kernel
            if not inplace:
                result_mat = mat.copy()
                data_ptr = result_mat.data.get_pointer()
                indices_ptr = result_mat.indices.get_pointer()
                indptr_ptr = result_mat.indptr.get_pointer()
            
            kernel_normalize.scale_primary_csr(
                data_ptr, indices_ptr, indptr_ptr,
                mat.shape[0], mat.shape[1],
                scales.get_pointer()
            )
            
            return mat if inplace else result_mat
            
        except (ImportError, LibraryNotFoundError, RuntimeError, AttributeError):
            pass  # Fall through to Python implementation
    
    # Fallback to pure Python for L2/Max or if kernel not available
    return _normalize_csr_rows_fallback(mat, norm, inplace)


def _normalize_csr_rows_fallback(mat: "SclCSR", norm: str, inplace: bool) -> "SclCSR":
    """Fallback pure Python CSR row normalization."""
    from scl.sparse import Array, SclCSR

    m = mat.shape[0]

    if inplace:
        new_data = mat.data
    else:
        new_data = Array(mat.nnz, dtype='float64')
        for k in range(mat.nnz):
            new_data[k] = mat.data[k]

    for i in range(m):
        start = mat.indptr[i]
        end = mat.indptr[i + 1]

        if start == end:
            continue

        # Compute norm
        n = 0.0
        if norm == "l1":
            for k in range(start, end):
                n += abs(new_data[k])
        elif norm == "l2":
            for k in range(start, end):
                n += new_data[k] ** 2
            n = math.sqrt(n)
        else:  # max
            for k in range(start, end):
                n = max(n, abs(new_data[k]))

        # Normalize
        if n > 0:
            for k in range(start, end):
                new_data[k] /= n

    if inplace:
        return mat
    else:
        return SclCSR.from_arrays(
            new_data, mat.indices.copy(), mat.indptr.copy(), mat.shape
        )


def _normalize_csc_cols(mat: "SclCSC", norm: str, inplace: bool) -> "SclCSC":
    """Normalize CSC columns using C++ kernel where available."""
    from scl.sparse import Array, SclCSC
    
    # Try to use C++ kernel for L1 normalization
    if norm == "l1":
        try:
            from scl._kernel import sparse as kernel_sparse
            from scl._kernel import normalize as kernel_normalize
            from scl._kernel.lib_loader import LibraryNotFoundError
            
            n = mat.shape[1]
            
            # Step 1: Compute column sums using kernel
            col_sums = Array(n, dtype='float64')
            data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz = mat.get_c_pointers()
            
            kernel_sparse.primary_sums_csc(
                data_ptr, indices_ptr, indptr_ptr, lengths_ptr,
                rows, cols, nnz,
                col_sums.get_pointer()
            )
            
            # Step 2: Compute inverse scales (1/sum), handle zeros
            scales = Array(n, dtype='float64')
            for j in range(n):
                scales[j] = 1.0 / col_sums[j] if col_sums[j] > 0 else 0.0
            
            # Step 3: Apply scaling using kernel
            if not inplace:
                result_mat = mat.copy()
                data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz = result_mat.get_c_pointers()
            
            kernel_normalize.scale_primary_csc(
                data_ptr, indices_ptr, indptr_ptr, lengths_ptr,
                rows, cols, nnz,
                scales.get_pointer()
            )
            
            return mat if inplace else result_mat
            
        except (ImportError, LibraryNotFoundError, RuntimeError, AttributeError):
            pass  # Fall through to Python implementation
    
    # Fallback to pure Python for L2/Max or if kernel not available
    return _normalize_csc_cols_fallback(mat, norm, inplace)


def _normalize_csc_cols_fallback(mat: "SclCSC", norm: str, inplace: bool) -> "SclCSC":
    """Fallback pure Python CSC column normalization."""
    from scl.sparse import Array, SclCSC

    n = mat.shape[1]

    if inplace:
        new_data = mat.data
    else:
        new_data = Array(mat.nnz, dtype='float64')
        for k in range(mat.nnz):
            new_data[k] = mat.data[k]

    for j in range(n):
        start = mat.indptr[j]
        end = mat.indptr[j + 1]

        if start == end:
            continue

        # Compute norm
        norm_val = 0.0
        if norm == "l1":
            for k in range(start, end):
                norm_val += abs(new_data[k])
        elif norm == "l2":
            for k in range(start, end):
                norm_val += new_data[k] ** 2
            norm_val = math.sqrt(norm_val)
        else:  # max
            for k in range(start, end):
                norm_val = max(norm_val, abs(new_data[k]))

        # Normalize
        if norm_val > 0:
            for k in range(start, end):
                new_data[k] /= norm_val

    if inplace:
        return mat
    else:
        return SclCSC.from_arrays(
            new_data, mat.indices.copy(), mat.indptr.copy(), mat.shape
        )


# =============================================================================
# Standardize
# =============================================================================

@overload
def standardize(
    mat: "SclCSR",
    axis: int = 0,
    *,
    zero_center: bool = True,
    unit_variance: bool = True,
    inplace: bool = False,
) -> "SclCSR": ...

@overload
def standardize(
    mat: "sp.spmatrix",
    axis: int = 0,
    *,
    zero_center: bool = True,
    unit_variance: bool = True,
    inplace: bool = False,
) -> "sp.spmatrix": ...


def standardize(
    mat: SparseInput,
    axis: int = 0,
    *,
    zero_center: bool = True,
    unit_variance: bool = True,
    inplace: bool = False,
) -> Union["SclCSR", "sp.spmatrix"]:
    """Standardize features to zero mean and/or unit variance.

    Applies z-score normalization along the specified axis:
        z = (x - mean) / std

    This is commonly used before PCA and other algorithms that assume
    standardized features.

    Mathematical Definition:
        For each feature j (axis=0):
            mean_j = (1/m) * sum(X[i, j] for i)
            std_j = sqrt((1/m) * sum((X[i, j] - mean_j)^2 for i))
            X_standardized[i, j] = (X[i, j] - mean_j) / std_j

    Sparse Matrix Handling:
        Warning: Centering a sparse matrix (subtracting the mean) typically
        creates a dense matrix since zeros become non-zero. This function
        applies centering only to non-zero values to preserve sparsity.

        For true centering (which destroys sparsity), convert to dense first.

    Options:
        - zero_center=True, unit_variance=True: Full standardization (z-score)
        - zero_center=False, unit_variance=True: Scale to unit variance only
        - zero_center=True, unit_variance=False: Center only
        - zero_center=False, unit_variance=False: No-op

    Time Complexity:
        O(nnz) for computing statistics and transformation.

    Args:
        mat: Input sparse matrix.
        axis: Axis along which to standardize.
            - 0: Standardize each column (feature) across rows (samples)
            - 1: Standardize each row across columns
        zero_center: If True, subtract the mean (applied to non-zeros only
            to preserve sparsity).
        unit_variance: If True, divide by the standard deviation.
        inplace: If True, modify the matrix in-place.

    Returns:
        Standardized sparse matrix.

    Examples:
        >>> import scl.preprocessing as pp
        >>>
        >>> # Standard preprocessing before PCA
        >>> # Note: For sparse data, consider whether centering is appropriate
        >>> standardized = pp.standardize(mat, axis=0, zero_center=False)
        >>>
        >>> # Scale only (preserve sparse structure better)
        >>> scaled = pp.standardize(mat, axis=0, zero_center=False, unit_variance=True)

    Notes:
        - Features with zero variance are left unchanged to avoid division by zero.
        - For sparse single-cell data, centering destroys sparsity. Consider
          using unit_variance=True with zero_center=False.
        - Many single-cell workflows use log-transformed data which may not
          need centering.

    See Also:
        normalize: L1/L2/Max normalization.
        scale: Custom scaling factors.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    if not zero_center and not unit_variance:
        # No-op
        return mat if inplace else mat.copy() if hasattr(mat, 'copy') else mat

    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        return _standardize_scipy(mat, axis, zero_center, unit_variance, inplace)

    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()

    return _standardize_scl(csr, axis, zero_center, unit_variance, inplace)


def _standardize_scipy(mat, axis: int, zero_center: bool, unit_variance: bool, inplace: bool):
    """Standardize using scipy/numpy."""
    import numpy as np

    if axis == 0:
        csc = mat.tocsc()
        if not inplace:
            csc = csc.copy()

        n = csc.shape[1]
        m = csc.shape[0]

        for j in range(n):
            start = csc.indptr[j]
            end = csc.indptr[j + 1]

            if start == end:
                continue

            vals = csc.data[start:end]

            # Compute statistics including zeros
            nnz = end - start
            col_sum = np.sum(vals)
            mean = col_sum / m

            if unit_variance:
                sq_sum = np.sum(vals ** 2)
                var = sq_sum / m - mean ** 2
                # Add zero contributions to variance
                var = (sq_sum + (m - nnz) * 0) / m - mean ** 2
                std = np.sqrt(var) if var > 0 else 1.0
            else:
                std = 1.0

            # Apply transformation (only to non-zeros)
            if zero_center:
                csc.data[start:end] = (vals - mean) / std
            elif unit_variance:
                csc.data[start:end] = vals / std

        return csc

    else:
        csr = mat.tocsr()
        if not inplace:
            csr = csr.copy()

        m = csr.shape[0]
        n = csr.shape[1]

        for i in range(m):
            start = csr.indptr[i]
            end = csr.indptr[i + 1]

            if start == end:
                continue

            vals = csr.data[start:end]

            nnz = end - start
            row_sum = np.sum(vals)
            mean = row_sum / n

            if unit_variance:
                sq_sum = np.sum(vals ** 2)
                var = sq_sum / n - mean ** 2
                std = np.sqrt(var) if var > 0 else 1.0
            else:
                std = 1.0

            if zero_center:
                csr.data[start:end] = (vals - mean) / std
            elif unit_variance:
                csr.data[start:end] = vals / std

        return csr


def _standardize_scl(mat: "SclCSR", axis: int, zero_center: bool,
                     unit_variance: bool, inplace: bool) -> "SclCSR":
    """Standardize SCL matrix."""
    from scl.sparse import Array, SclCSR

    m, n = mat.shape

    if inplace:
        new_data = mat.data
    else:
        new_data = Array(mat.nnz, dtype='float64')
        for k in range(mat.nnz):
            new_data[k] = mat.data[k]

    if axis == 0:
        # Column-wise standardization
        csc = mat.to_csc()

        for j in range(n):
            start = csc.indptr[j]
            end = csc.indptr[j + 1]

            if start == end:
                continue

            nnz = end - start
            col_sum = 0.0
            for k in range(start, end):
                col_sum += csc.data[k]

            mean = col_sum / m

            if unit_variance:
                sq_sum = 0.0
                for k in range(start, end):
                    sq_sum += csc.data[k] ** 2

                var = sq_sum / m - mean ** 2
                std = math.sqrt(var) if var > 0 else 1.0
            else:
                std = 1.0

            # We need to map back to CSR positions
            # For simplicity, we work on CSC and convert back
            for k in range(start, end):
                if zero_center:
                    csc.data[k] = (csc.data[k] - mean) / std
                elif unit_variance:
                    csc.data[k] = csc.data[k] / std

        # Convert back to CSR
        return csc.to_csr()

    else:
        # Row-wise standardization (efficient for CSR)
        for i in range(m):
            start = mat.indptr[i]
            end = mat.indptr[i + 1]

            if start == end:
                continue

            nnz = end - start
            row_sum = 0.0
            for k in range(start, end):
                row_sum += new_data[k]

            mean = row_sum / n

            if unit_variance:
                sq_sum = 0.0
                for k in range(start, end):
                    sq_sum += new_data[k] ** 2

                var = sq_sum / n - mean ** 2
                std = math.sqrt(var) if var > 0 else 1.0
            else:
                std = 1.0

            for k in range(start, end):
                if zero_center:
                    new_data[k] = (new_data[k] - mean) / std
                elif unit_variance:
                    new_data[k] = new_data[k] / std

        if inplace:
            return mat
        else:
            return SclCSR.from_arrays(
                new_data, mat.indices.copy(), mat.indptr.copy(), mat.shape
            )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "normalize",
    "standardize",
]
