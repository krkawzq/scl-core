"""
Linear Algebra Operations for Sparse Matrices.

This module provides linear algebra operations optimized for sparse matrices,
including matrix-vector multiplication, matrix products, and correlation
computations.

These operations form the foundation for many machine learning and statistical
analysis algorithms used in single-cell genomics.

Implemented Operations:
    - Sparse matrix-dense vector multiplication (SpMV)
    - Sparse-sparse matrix multiplication
    - Gram matrix computation (X^T X)
    - Pearson correlation matrix
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union, overload

from scl._typing import (
    SparseInput,
    CSRInput,
    CSCInput,
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
# Sparse Matrix-Vector Multiplication
# =============================================================================

@overload
def spmv(
    mat: "SclCSR",
    x: "RealArray",
) -> "RealArray": ...

@overload
def spmv(
    mat: "sp.csr_matrix",
    x: "np.ndarray",
) -> "np.ndarray": ...


def spmv(
    mat: SparseInput,
    x: VectorInput,
) -> Union["RealArray", "np.ndarray"]:
    """Sparse matrix-vector multiplication (SpMV).

    Computes y = A * x where A is a sparse matrix and x is a dense vector.
    This is the fundamental operation for iterative solvers, power methods,
    and many machine learning algorithms.

    Mathematical Definition:
        y[i] = sum(A[i, j] * x[j] for j in range(n))

    Algorithm (CSR Format):
        For each row i:
            y[i] = sum(data[k] * x[indices[k]]
                       for k in range(indptr[i], indptr[i+1]))

        The CSR format enables efficient row-wise access, making SpMV
        O(nnz) with excellent cache locality.

    Performance Characteristics:
        - Time: O(nnz) where nnz is the number of non-zeros
        - Memory: O(m) for output where m is the number of rows
        - Cache: Good spatial locality due to sequential data access
        - Parallelization: Each row can be computed independently

    Args:
        mat: Sparse matrix of shape (m, n). CSR format preferred for
            optimal performance.
        x: Dense vector of length n.

    Returns:
        Dense vector y of length m containing the product A * x.

    Raises:
        ValueError: If x length doesn't match matrix columns.

    Examples:
        >>> from scl import SclCSR
        >>> from scl.sparse import Array
        >>> import scl.math as smath
        >>>
        >>> # Create sparse matrix [[1, 2], [3, 4]]
        >>> mat = SclCSR.from_arrays([1, 2, 3, 4], [0, 1, 0, 1],
        ...                          [0, 2, 4], (2, 2))
        >>> x = Array.from_list([1.0, 2.0], dtype='float64')
        >>> y = smath.spmv(mat, x)
        >>> y.to_list()
        [5.0, 11.0]  # [1*1 + 2*2, 3*1 + 4*2]

    Notes:
        - For CSC matrices, the operation is transposed (computing A^T * x).
        - For very sparse matrices (density < 1%), SpMV can be memory-bound
          rather than compute-bound.
        - Consider using batched SpMV (matrix-matrix product) if you need
          to multiply by multiple vectors.

    See Also:
        dot: Sparse-sparse matrix multiplication.
        gram: Compute X^T * X efficiently.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        csr = mat.tocsr()
        x_arr = np.asarray(x).ravel()
        return csr.dot(x_arr)

    # Native SCL
    csr = ensure_scl_csr(mat)
    x_arr = ensure_vector(x, size=csr.shape[1])

    csr.materialize()

    return _spmv_scl(csr, x_arr)


def _spmv_scl(mat: "SclCSR", x: "RealArray") -> "RealArray":
    """SpMV implementation for SCL matrices using C++ kernel."""
    from scl.sparse import Array
    
    try:
        from scl._kernel import algebra as kernel_algebra
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        m = mat.shape[0]
        result = Array(m, dtype='float64')
        
        # Get C pointers: (data, indices, indptr, lengths, rows, cols, nnz)
        data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz = mat.get_c_pointers()
        
        # Call kernel: y = 1.0 * A * x + 0.0 * y
        kernel_algebra.spmv_csr(
            data_ptr, indices_ptr, indptr_ptr, lengths_ptr,
            rows, cols, nnz,
            x.get_pointer(), result.get_pointer(),
            1.0, 0.0
        )
        
        return result
        
    except (ImportError, LibraryNotFoundError, RuntimeError, AttributeError):
        # Fallback to pure Python implementation
        return _spmv_scl_fallback(mat, x)


def _spmv_scl_fallback(mat: "SclCSR", x: "RealArray") -> "RealArray":
    """Fallback pure Python SpMV implementation."""
    from scl.sparse import Array

    m = mat.shape[0]
    result = Array(m, dtype='float64')

    for i in range(m):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]

        acc = 0.0
        for k in range(start, end):
            j = mat._indices[k]
            acc += mat._data[k] * x[j]

        result[i] = acc

    return result


# =============================================================================
# Sparse-Sparse Matrix Multiplication
# =============================================================================

@overload
def dot(
    a: "SclCSR",
    b: "SclCSR",
) -> "SclCSR": ...

@overload
def dot(
    a: "sp.spmatrix",
    b: "sp.spmatrix",
) -> "sp.csr_matrix": ...


def dot(
    a: SparseInput,
    b: SparseInput,
) -> Union["SclCSR", "sp.csr_matrix"]:
    """Sparse-sparse matrix multiplication.

    Computes C = A * B where both A and B are sparse matrices.
    The result is also sparse when the inputs are sparse.

    Mathematical Definition:
        C[i, k] = sum(A[i, j] * B[j, k] for j in range(n))

    Algorithm (Row-by-Row):
        For CSR input A and CSC input B (or converted):

        For each row i of A:
            Initialize sparse accumulator for row i of C
            For each non-zero A[i, j]:
                For each non-zero B[j, k]:
                    C[i, k] += A[i, j] * B[j, k]

        This approach is efficient when A has few non-zeros per row.

    Complexity Analysis:
        - Worst case: O(m * flops(A * B)) where flops is the number of
          scalar multiplications required
        - Best case: O(nnz(A) + nnz(B) + nnz(C)) when matrices have
          good sparsity structure
        - The output sparsity depends on the input patterns

    Args:
        a: Left sparse matrix of shape (m, n).
        b: Right sparse matrix of shape (n, p).

    Returns:
        Sparse matrix C of shape (m, p) containing the product A * B.

    Raises:
        ValueError: If inner dimensions don't match (a.cols != b.rows).

    Examples:
        >>> from scl import SclCSR
        >>> import scl.math as smath
        >>>
        >>> # A: 2x3, B: 3x2
        >>> A = SclCSR.from_arrays([1, 2, 3], [0, 1, 2],
        ...                        [0, 2, 3], (2, 3))
        >>> B = SclCSR.from_arrays([1, 1, 1], [0, 1, 0],
        ...                        [0, 1, 2, 3], (3, 2))
        >>> C = smath.dot(A, B)
        >>> print(C.shape)  # (2, 2)

    Notes:
        - For A * B where B is tall and thin, consider converting B to CSC.
        - The fill-in (new non-zeros in C) can be significant for some
          matrices, leading to dense results.
        - For repeated multiplications with the same sparsity pattern,
          consider using symbolic factorization.

    See Also:
        spmv: Sparse matrix-vector multiplication.
        gram: Efficient computation of X^T * X.
    """
    fmt_a = get_format(a)
    fmt_b = get_format(b)

    # Handle scipy
    if fmt_a in ("scipy_csr", "scipy_csc", "scipy_other") or \
       fmt_b in ("scipy_csr", "scipy_csc", "scipy_other"):
        from scipy import sparse as sp
        a_sp = a if sp.issparse(a) else sp.csr_matrix(a)
        b_sp = b if sp.issparse(b) else sp.csr_matrix(b)
        return a_sp.dot(b_sp).tocsr()

    # Native SCL
    csr_a = ensure_scl_csr(a)
    csr_b = ensure_scl_csr(b)

    csr_a.materialize()
    csr_b.materialize()

    if csr_a.shape[1] != csr_b.shape[0]:
        raise ValueError(
            f"Matrix dimensions incompatible: {csr_a.shape} x {csr_b.shape}"
        )

    return _dot_scl(csr_a, csr_b)


def _dot_scl(a: "SclCSR", b: "SclCSR") -> "SclCSR":
    """Sparse-sparse multiplication for SCL matrices."""
    from scl.sparse import Array, SclCSR

    m = a.shape[0]
    p = b.shape[1]

    # Convert b to CSC for efficient column access
    b_csc = b.to_csc()

    # Accumulator for result
    result_data = []
    result_indices = []
    result_indptr = [0]

    for i in range(m):
        # Sparse row i of A
        a_start = a._indptr[i]
        a_end = a._indptr[i + 1]

        # Accumulate row i of result
        row_acc = {}  # column -> value

        for a_k in range(a_start, a_end):
            j = a._indices[a_k]
            a_val = a._data[a_k]

            # Row j of B (column j of B^T)
            b_start = b_csc._indptr[j]
            b_end = b_csc._indptr[j + 1]

            for b_k in range(b_start, b_end):
                k = b_csc._indices[b_k]
                b_val = b_csc._data[b_k]

                if k in row_acc:
                    row_acc[k] += a_val * b_val
                else:
                    row_acc[k] = a_val * b_val

        # Sort by column and add to result
        for k in sorted(row_acc.keys()):
            val = row_acc[k]
            if abs(val) > 1e-15:
                result_data.append(val)
                result_indices.append(k)

        result_indptr.append(len(result_data))

    return SclCSR.from_arrays(
        result_data, result_indices, result_indptr, (m, p)
    )


# =============================================================================
# Gram Matrix
# =============================================================================

@overload
def gram(
    mat: "SclCSR",
) -> "RealArray": ...

@overload
def gram(
    mat: "sp.spmatrix",
) -> "np.ndarray": ...


def gram(
    mat: SparseInput,
) -> Union["RealArray", "np.ndarray"]:
    """Compute the Gram matrix X^T * X.

    The Gram matrix G = X^T * X contains the inner products between
    all pairs of columns. G[i, j] = <column_i, column_j>.

    Mathematical Definition:
        G[i, j] = sum(X[k, i] * X[k, j] for k in range(m))

    This is equivalent to computing the covariance matrix (up to centering
    and normalization) and is fundamental for PCA, kernel methods, and
    least-squares problems.

    Algorithm (Column-wise):
        For CSC format (or converted):

        For each pair of columns (i, j):
            G[i, j] = dot(col_i, col_j)

        Using sparse column intersection for efficiency.

    Symmetry Optimization:
        Since G is symmetric (G[i, j] = G[j, i]), we only compute the
        upper triangle and mirror the values.

    Time Complexity:
        O(n^2 * average_nnz_per_column) in the worst case, but typically
        much better due to sparse column intersections.

    Space Complexity:
        O(n^2) for the output Gram matrix (stored as dense).

    Args:
        mat: Input sparse matrix of shape (m, n).

    Returns:
        Dense array of shape (n * n) stored in row-major order,
        representing the n x n Gram matrix.

    Examples:
        >>> from scl import SclCSR
        >>> import scl.math as smath
        >>>
        >>> # Data matrix: samples x features
        >>> mat = SclCSR.from_scipy(expression_matrix)
        >>> G = smath.gram(mat)  # features x features inner products
        >>>
        >>> # Access G[i, j]
        >>> n = mat.shape[1]
        >>> inner_product_ij = G[i * n + j]

    Notes:
        - Output is a flattened 1D array; reshape to (n, n) if needed.
        - For very large n, consider using randomized methods or
          computing only a subset of the Gram matrix.
        - The Gram matrix is always positive semi-definite.

    See Also:
        pearson: Compute Pearson correlation matrix.
        dot: General sparse matrix multiplication.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np
        csc = mat.tocsc()
        # Compute X^T * X directly
        gram = (csc.T @ csc).toarray()
        return gram.ravel()

    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()

    return _gram_scl(csr)


def _gram_scl(mat: "SclCSR") -> "RealArray":
    """Gram matrix computation for SCL matrices using C++ kernel."""
    from scl.sparse import Array
    
    try:
        from scl._kernel import algebra as kernel_algebra
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        m, n = mat.shape
        
        # Convert to CSC for column access (X^T * X on columns)
        csc = mat.to_csc()
        
        # Output: n x n matrix stored as flat array
        result = Array(n * n, dtype='float64')
        
        # Get C pointers from CSC
        data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz = csc.get_c_pointers()
        
        # Call kernel
        kernel_algebra.gram_csc(
            data_ptr, indices_ptr, indptr_ptr, lengths_ptr,
            rows, cols, nnz,
            result.get_pointer()
        )
        
        return result
        
    except (ImportError, LibraryNotFoundError, RuntimeError, AttributeError):
        # Fallback to pure Python implementation
        return _gram_scl_fallback(mat)


def _gram_scl_fallback(mat: "SclCSR") -> "RealArray":
    """Fallback pure Python Gram matrix computation."""
    from scl.sparse import Array

    m, n = mat.shape

    # Convert to CSC for column access
    csc = mat.to_csc()

    # Output: n x n matrix stored as flat array
    result = Array(n * n, dtype='float64')
    for i in range(n * n):
        result[i] = 0.0

    # Compute upper triangle
    for i in range(n):
        # Column i
        i_start = csc._indptr[i]
        i_end = csc._indptr[i + 1]

        # Self inner product (diagonal)
        diag = 0.0
        for k in range(i_start, i_end):
            diag += csc._data[k] ** 2
        result[i * n + i] = diag

        # Cross products with columns j > i
        for j in range(i + 1, n):
            j_start = csc._indptr[j]
            j_end = csc._indptr[j + 1]

            # Sparse dot product
            inner = 0.0
            ki = i_start
            kj = j_start

            while ki < i_end and kj < j_end:
                row_i = csc._indices[ki]
                row_j = csc._indices[kj]

                if row_i == row_j:
                    inner += csc._data[ki] * csc._data[kj]
                    ki += 1
                    kj += 1
                elif row_i < row_j:
                    ki += 1
                else:
                    kj += 1

            result[i * n + j] = inner
            result[j * n + i] = inner  # Symmetric

    return result


# =============================================================================
# Pearson Correlation
# =============================================================================

@overload
def pearson(
    mat: "SclCSR",
) -> "RealArray": ...

@overload
def pearson(
    mat: "sp.spmatrix",
) -> "np.ndarray": ...


def pearson(
    mat: SparseInput,
) -> Union["RealArray", "np.ndarray"]:
    """Compute the Pearson correlation matrix between columns.

    Computes the pairwise Pearson correlation coefficients between all
    columns of the input matrix.

    Mathematical Definition:
        For columns X_i and X_j:

            r[i, j] = cov(X_i, X_j) / (std(X_i) * std(X_j))

        where:
            cov(X_i, X_j) = E[(X_i - mean_i)(X_j - mean_j)]
            std(X_i) = sqrt(E[(X_i - mean_i)^2])

    Equivalently:
        r[i, j] = (sum((X_i - mean_i)(X_j - mean_j))) /
                  sqrt(sum((X_i - mean_i)^2) * sum((X_j - mean_j)^2))

    Algorithm (Centered Gram):
        1. Compute column means
        2. Center the data (subtract means)
        3. Compute Gram matrix of centered data
        4. Normalize by standard deviations

        For sparse matrices, centering is handled implicitly to avoid
        creating a dense matrix.

    Properties:
        - r[i, i] = 1 (perfect self-correlation)
        - r[i, j] in [-1, 1]
        - r[i, j] = r[j, i] (symmetric)
        - r = 1: Perfect positive linear relationship
        - r = -1: Perfect negative linear relationship
        - r = 0: No linear relationship

    Time Complexity:
        O(n^2 * m) in the worst case, but better for sparse data.

    Args:
        mat: Input sparse matrix of shape (m, n).
            Rows are samples, columns are features.

    Returns:
        Dense array of shape (n * n) stored in row-major order,
        representing the n x n correlation matrix.

    Examples:
        >>> from scl import SclCSR
        >>> import scl.math as smath
        >>>
        >>> # Gene expression: cells x genes
        >>> corr = smath.pearson(expression_matrix)
        >>>
        >>> # Find highly correlated gene pairs
        >>> n_genes = expression_matrix.shape[1]
        >>> for i in range(n_genes):
        ...     for j in range(i + 1, n_genes):
        ...         r = corr[i * n_genes + j]
        ...         if abs(r) > 0.8:
        ...             print(f"Genes {i} and {j} correlated: r={r:.3f}")

    Notes:
        - For sparse single-cell data, many genes may have near-zero
          correlation due to dropout.
        - Consider using Spearman correlation for non-linear relationships.
        - Output is a flattened array; reshape to (n, n) if needed.

    See Also:
        gram: Compute unnormalized inner products.
    """
    fmt = get_format(mat)

    if fmt in ("scipy_csr", "scipy_csc", "scipy_other"):
        import numpy as np

        csc = mat.tocsc()
        n = csc.shape[1]

        # Compute means
        means = np.asarray(csc.mean(axis=0)).ravel()

        # Compute centered Gram matrix
        # For sparse, we compute: X^T X - n * mean_outer
        gram = (csc.T @ csc).toarray()

        # Subtract mean contributions
        m = csc.shape[0]
        for i in range(n):
            for j in range(n):
                gram[i, j] -= m * means[i] * means[j]

        # Compute standard deviations
        stds = np.sqrt(np.diag(gram) / m)
        stds[stds == 0] = 1.0  # Avoid division by zero

        # Normalize
        corr = gram / (m * np.outer(stds, stds))
        np.fill_diagonal(corr, 1.0)

        return corr.ravel()

    # Native SCL
    csr = ensure_scl_csr(mat)
    csr.materialize()

    return _pearson_scl(csr)


def _pearson_scl(mat: "SclCSR") -> "RealArray":
    """Pearson correlation for SCL matrices using C++ kernel."""
    from scl.sparse import Array
    
    try:
        from scl._kernel import algebra as kernel_algebra
        from scl._kernel.lib_loader import LibraryNotFoundError
        
        m, n = mat.shape
        csc = mat.to_csc()
        
        # Output: n x n correlation matrix
        result = Array(n * n, dtype='float64')
        
        # Workspace arrays for means and inverse stds
        workspace_means = Array(n, dtype='float64')
        workspace_inv_stds = Array(n, dtype='float64')
        
        # Get C pointers from CSC
        data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz = csc.get_c_pointers()
        
        # Call kernel
        kernel_algebra.pearson_csc(
            data_ptr, indices_ptr, indptr_ptr, lengths_ptr,
            rows, cols, nnz,
            result.get_pointer(),
            workspace_means.get_pointer(),
            workspace_inv_stds.get_pointer()
        )
        
        return result
        
    except (ImportError, LibraryNotFoundError, RuntimeError, AttributeError):
        # Fallback to pure Python implementation
        return _pearson_scl_fallback(mat)


def _pearson_scl_fallback(mat: "SclCSR") -> "RealArray":
    """Fallback pure Python Pearson correlation computation."""
    from scl.sparse import Array

    m, n = mat.shape
    csc = mat.to_csc()

    # Compute column means
    means = Array(n, dtype='float64')
    for j in range(n):
        start = csc._indptr[j]
        end = csc._indptr[j + 1]
        total = 0.0
        for k in range(start, end):
            total += csc._data[k]
        means[j] = total / m

    # Compute column standard deviations
    stds = Array(n, dtype='float64')
    for j in range(n):
        start = csc._indptr[j]
        end = csc._indptr[j + 1]
        mean_j = means[j]

        sq_sum = 0.0
        for k in range(start, end):
            sq_sum += (csc._data[k] - mean_j) ** 2

        # Add zero contributions
        n_zeros = m - (end - start)
        sq_sum += n_zeros * mean_j ** 2

        stds[j] = math.sqrt(sq_sum / m) if sq_sum > 0 else 1.0

    # Compute correlation matrix using identity: cov(X,Y) = E[XY] - E[X]E[Y]
    result = Array(n * n, dtype='float64')

    for i in range(n):
        # Diagonal is always 1
        result[i * n + i] = 1.0

        i_start = csc._indptr[i]
        i_end = csc._indptr[i + 1]
        mean_i = means[i]
        std_i = stds[i]

        for j in range(i + 1, n):
            j_start = csc._indptr[j]
            j_end = csc._indptr[j + 1]
            mean_j = means[j]
            std_j = stds[j]

            # Compute raw dot product
            raw_dot = 0.0
            ki = i_start
            kj = j_start

            while ki < i_end and kj < j_end:
                row_i = csc._indices[ki]
                row_j = csc._indices[kj]

                if row_i == row_j:
                    raw_dot += csc._data[ki] * csc._data[kj]
                    ki += 1
                    kj += 1
                elif row_i < row_j:
                    ki += 1
                else:
                    kj += 1

            cov = raw_dot / m - mean_i * mean_j
            corr = cov / (std_i * std_j) if std_i > 0 and std_j > 0 else 0.0

            # Clamp to [-1, 1]
            corr = max(-1.0, min(1.0, corr))

            result[i * n + j] = corr
            result[j * n + i] = corr

    return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "spmv",
    "dot",
    "gram",
    "pearson",
]
