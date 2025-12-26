"""
SCL Ops - Sparse Matrix Operations

Zero external dependencies. High-level operation interface
with automatic backend selection and configuration support.
"""

from __future__ import annotations

from ctypes import c_int64, c_double, c_void_p, byref
from typing import Optional, Union, Sequence, Tuple
import math

from scl._ffi import get_lib_with_signatures, check_error
from scl.array import RealArray, IndexArray, ByteArray
from scl.sparse import SclCSR, SclCSC, SparseBase
from scl._config import config, NormType
from scl._backend import Backend, MappedStorage


# =============================================================================
# Helper Functions
# =============================================================================

def _get_mapped_handle(mat: SparseBase) -> int:
    """Get mapped storage handle if available, else 0."""
    if isinstance(mat._storage, MappedStorage):
        return mat._storage.handle
    return 0


# =============================================================================
# Normalization
# =============================================================================

def normalize(mat: SclCSR, norm: str = "l2", axis: int = 1,
              inplace: bool = False) -> SclCSR:
    """
    Normalize sparse matrix.

    Args:
        mat: Input matrix
        norm: "l1", "l2", or "max"
        axis: 0=column, 1=row
        inplace: Modify in-place

    Returns:
        Normalized matrix
    """
    mat.materialize()

    if axis != 1:
        raise NotImplementedError("Only axis=1 (row normalization) supported")

    # Check if we have a mapped backend with handle
    has_handle = (isinstance(mat._storage, MappedStorage) and
                  mat._storage.handle != 0)

    if has_handle:
        lib = get_lib_with_signatures()
        handle = mat._storage.handle

        new_data = RealArray(mat.nnz)
        new_indices = IndexArray(mat.nnz)
        new_indptr = IndexArray(mat.shape[0] + 1)

        if norm == "l1":
            check_error(lib.scl_mmap_csr_normalize_l1(
                handle,
                c_void_p(new_data.data_ptr),
                c_void_p(new_indices.data_ptr),
                c_void_p(new_indptr.data_ptr)
            ))
        elif norm == "l2":
            check_error(lib.scl_mmap_csr_normalize_l2(
                handle,
                c_void_p(new_data.data_ptr),
                c_void_p(new_indices.data_ptr),
                c_void_p(new_indptr.data_ptr)
            ))
        else:
            raise ValueError(f"Unknown norm: {norm}")

        if inplace:
            mat._data = new_data
            mat._indices = new_indices
            mat._indptr = new_indptr
            return mat
        else:
            return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)
    else:
        # Pure Python implementation
        new_data = mat._data.copy()

        for i in range(mat.shape[0]):
            start = mat._indptr[i]
            end = mat._indptr[i + 1]
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
            elif norm == "max":
                for k in range(start, end):
                    n = max(n, abs(new_data[k]))
            else:
                raise ValueError(f"Unknown norm: {norm}")

            # Normalize
            if n > 0:
                for k in range(start, end):
                    new_data[k] /= n

        if inplace:
            mat._data = new_data
            return mat
        else:
            return SclCSR.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )


# =============================================================================
# Transforms
# =============================================================================

def log1p(mat: SclCSR, inplace: bool = False) -> SclCSR:
    """
    log1p transform: log(1 + x)

    Args:
        mat: Input matrix
        inplace: Modify in-place
    """
    mat.materialize()

    handle = _get_mapped_handle(mat)
    if handle:
        lib = get_lib_with_signatures()

        new_data = RealArray(mat.nnz)
        new_indices = IndexArray(mat.nnz)
        new_indptr = IndexArray(mat.shape[0] + 1)

        check_error(lib.scl_mmap_csr_log1p(
            handle,
            c_void_p(new_data.data_ptr),
            c_void_p(new_indices.data_ptr),
            c_void_p(new_indptr.data_ptr)
        ))

        if inplace:
            mat._data = new_data
            mat._indices = new_indices
            mat._indptr = new_indptr
            return mat
        else:
            return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)
    else:
        new_data = RealArray(mat.nnz)
        for k in range(mat.nnz):
            new_data[k] = math.log1p(mat._data[k])

        if inplace:
            mat._data = new_data
            return mat
        else:
            return SclCSR.from_arrays(
                new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
            )


def scale(mat: SclCSR, row_factors: Optional[Sequence[float]] = None,
          col_factors: Optional[Sequence[float]] = None,
          inplace: bool = False) -> SclCSR:
    """
    Scale matrix.

    Args:
        mat: Input matrix
        row_factors: Row scale factors
        col_factors: Column scale factors
        inplace: Modify in-place
    """
    mat.materialize()

    new_data = mat._data.copy() if not inplace else mat._data
    new_indices = mat._indices if inplace else mat._indices.copy()
    new_indptr = mat._indptr if inplace else mat._indptr.copy()

    handle = _get_mapped_handle(mat)
    if handle and row_factors is not None:
        lib = get_lib_with_signatures()
        row_arr = RealArray.from_sequence(row_factors)

        check_error(lib.scl_mmap_csr_scale_rows(
            handle,
            c_void_p(row_arr.data_ptr),
            c_void_p(new_data.data_ptr),
            c_void_p(new_indices.data_ptr),
            c_void_p(new_indptr.data_ptr)
        ))
    elif row_factors is not None:
        for i in range(mat.shape[0]):
            start = new_indptr[i]
            end = new_indptr[i + 1]
            factor = row_factors[i]
            for k in range(start, end):
                new_data[k] *= factor

    if handle and col_factors is not None:
        lib = get_lib_with_signatures()
        col_arr = RealArray.from_sequence(col_factors)

        check_error(lib.scl_mmap_csr_scale_cols(
            handle,
            c_void_p(col_arr.data_ptr),
            c_void_p(new_data.data_ptr),
            c_void_p(new_indices.data_ptr),
            c_void_p(new_indptr.data_ptr)
        ))
    elif col_factors is not None:
        for i in range(mat.shape[0]):
            start = new_indptr[i]
            end = new_indptr[i + 1]
            for k in range(start, end):
                new_data[k] *= col_factors[new_indices[k]]

    if inplace:
        mat._data = new_data
        return mat
    else:
        return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)


# =============================================================================
# Filtering
# =============================================================================

def filter_threshold(mat: SclCSR, threshold: float) -> SclCSR:
    """
    Filter elements below threshold (by absolute value).

    Args:
        mat: Input matrix
        threshold: Threshold value

    Returns:
        Filtered matrix
    """
    mat.materialize()

    handle = _get_mapped_handle(mat)
    if handle:
        lib = get_lib_with_signatures()

        # Allocate max possible space
        new_data = RealArray(mat.nnz)
        new_indices = IndexArray(mat.nnz)
        new_indptr = IndexArray(mat.shape[0] + 1)

        out_nnz = c_int64()
        check_error(lib.scl_mmap_csr_filter_threshold(
            handle,
            threshold,
            c_void_p(new_data.data_ptr),
            c_void_p(new_indices.data_ptr),
            c_void_p(new_indptr.data_ptr),
            byref(out_nnz)
        ))

        nnz = out_nnz.value
        # Create new arrays with correct size
        final_data = RealArray(nnz)
        final_indices = IndexArray(nnz)
        for k in range(nnz):
            final_data[k] = new_data[k]
            final_indices[k] = new_indices[k]

        return SclCSR.from_arrays(final_data, final_indices, new_indptr, mat.shape)
    else:
        new_data = []
        new_indices = []
        new_indptr = [0]

        for i in range(mat.shape[0]):
            start = mat._indptr[i]
            end = mat._indptr[i + 1]
            for k in range(start, end):
                if abs(mat._data[k]) >= threshold:
                    new_data.append(mat._data[k])
                    new_indices.append(mat._indices[k])
            new_indptr.append(len(new_data))

        return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)


def top_k(mat: SclCSR, k: int) -> SclCSR:
    """
    Keep top-k elements per row (by absolute value).

    Args:
        mat: Input matrix
        k: Number of elements to keep

    Returns:
        Filtered matrix
    """
    mat.materialize()

    handle = _get_mapped_handle(mat)
    if handle:
        lib = get_lib_with_signatures()

        # Allocate space (max k * rows)
        max_nnz = k * mat.shape[0]
        new_data = RealArray(max_nnz)
        new_indices = IndexArray(max_nnz)
        new_indptr = IndexArray(mat.shape[0] + 1)

        check_error(lib.scl_mmap_csr_top_k(
            handle,
            k,
            c_void_p(new_data.data_ptr),
            c_void_p(new_indices.data_ptr),
            c_void_p(new_indptr.data_ptr)
        ))

        nnz = new_indptr[mat.shape[0]]
        final_data = RealArray(nnz)
        final_indices = IndexArray(nnz)
        for i in range(nnz):
            final_data[i] = new_data[i]
            final_indices[i] = new_indices[i]

        return SclCSR.from_arrays(final_data, final_indices, new_indptr, mat.shape)
    else:
        new_data = []
        new_indices = []
        new_indptr = [0]

        for i in range(mat.shape[0]):
            start = mat._indptr[i]
            end = mat._indptr[i + 1]
            row_data = [(mat._data[idx], mat._indices[idx]) for idx in range(start, end)]

            # Sort by absolute value, take top-k
            row_data.sort(key=lambda x: abs(x[0]), reverse=True)
            row_data = row_data[:k]

            # Sort by index
            row_data.sort(key=lambda x: x[1])

            for val, idx in row_data:
                new_data.append(val)
                new_indices.append(idx)

            new_indptr.append(len(new_data))

        return SclCSR.from_arrays(new_data, new_indices, new_indptr, mat.shape)


# =============================================================================
# Matrix Operations
# =============================================================================

def spmv(mat: SclCSR, x: Sequence[float]) -> RealArray:
    """
    Sparse matrix-vector multiplication: y = A * x

    Args:
        mat: Sparse matrix
        x: Vector

    Returns:
        Result vector
    """
    mat.materialize()

    x_arr = RealArray.from_sequence(x) if not isinstance(x, RealArray) else x

    if x_arr.size != mat.shape[1]:
        raise ValueError(f"Vector length {x_arr.size} != matrix cols {mat.shape[1]}")

    y = RealArray(mat.shape[0])

    handle = _get_mapped_handle(mat)
    if handle:
        lib = get_lib_with_signatures()
        check_error(lib.scl_mmap_csr_spmv(
            handle,
            c_void_p(x_arr.data_ptr),
            c_void_p(y.data_ptr)
        ))
    else:
        for i in range(mat.shape[0]):
            start = mat._indptr[i]
            end = mat._indptr[i + 1]
            dot = 0.0
            for k in range(start, end):
                dot += mat._data[k] * x_arr[mat._indices[k]]
            y[i] = dot

    return y


def dot(a: SclCSR, b: SclCSR) -> SclCSR:
    """
    Sparse matrix multiplication: C = A * B

    Note: This is a pure Python implementation for now.
    For better performance, use scipy interop.

    Args:
        a: Left matrix
        b: Right matrix

    Returns:
        Result matrix
    """
    a.materialize()
    b.materialize()

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Matrix dimensions mismatch: {a.shape} @ {b.shape}")

    # Convert B to CSC for efficient column access
    b_csc = b.to_csc()

    # Result dimensions
    m, n = a.shape[0], b.shape[1]

    new_data = []
    new_indices = []
    new_indptr = [0]

    for i in range(m):
        a_start = a._indptr[i]
        a_end = a._indptr[i + 1]

        # Accumulate row i of result
        row_result = {}

        for a_k in range(a_start, a_end):
            a_col = a._indices[a_k]
            a_val = a._data[a_k]

            # Multiply with column a_col of B (row a_col in CSC)
            b_start = b_csc._indptr[a_col] if a_col < b_csc._cols else 0
            b_end = b_csc._indptr[a_col + 1] if a_col < b_csc._cols else 0

            for b_k in range(b_start, b_end):
                b_row = b_csc._indices[b_k]  # This is actually column in original B
                b_val = b_csc._data[b_k]

                if b_row in row_result:
                    row_result[b_row] += a_val * b_val
                else:
                    row_result[b_row] = a_val * b_val

        # Sort by column index
        for col in sorted(row_result.keys()):
            if row_result[col] != 0:
                new_data.append(row_result[col])
                new_indices.append(col)

        new_indptr.append(len(new_data))

    return SclCSR.from_arrays(new_data, new_indices, new_indptr, (m, n))


# =============================================================================
# Statistics
# =============================================================================

def var(mat: SclCSR, axis: Optional[int] = None,
        ddof: int = 0) -> Union[float, RealArray]:
    """
    Compute variance.

    Args:
        mat: Input matrix
        axis: None=global, 0=column, 1=row
        ddof: Degrees of freedom correction
    """
    mat.materialize()

    if axis == 1:
        # Row variance
        means = mat.mean(axis=1)
        result = RealArray(mat.shape[0])

        handle = _get_mapped_handle(mat)
        if handle:
            lib = get_lib_with_signatures()
            check_error(lib.scl_mmap_csr_row_var(
                handle,
                c_void_p(result.data_ptr),
                c_void_p(means.data_ptr),
                1  # count_zeros
            ))
        else:
            for i in range(mat.shape[0]):
                start = mat._indptr[i]
                end = mat._indptr[i + 1]
                mean = means[i]

                sum_sq = 0.0
                for k in range(start, end):
                    sum_sq += (mat._data[k] - mean) ** 2

                # Zero values contribution
                num_zeros = mat.shape[1] - (end - start)
                sum_sq += num_zeros * mean ** 2

                n = mat.shape[1] - ddof
                result[i] = sum_sq / n if n > 0 else 0

        return result

    elif axis == 0:
        # Column variance
        means = mat.mean(axis=0)
        result = RealArray(mat.shape[1])
        result.zero()

        # Sum of squared differences
        counts = [0] * mat.shape[1]
        sum_sq = [0.0] * mat.shape[1]

        for i in range(mat.shape[0]):
            start = mat._indptr[i]
            end = mat._indptr[i + 1]
            for k in range(start, end):
                j = mat._indices[k]
                sum_sq[j] += (mat._data[k] - means[j]) ** 2
                counts[j] += 1

        # Add contribution from zeros
        for j in range(mat.shape[1]):
            num_zeros = mat.shape[0] - counts[j]
            sum_sq[j] += num_zeros * means[j] ** 2
            n = mat.shape[0] - ddof
            result[j] = sum_sq[j] / n if n > 0 else 0

        return result

    else:
        # Global variance
        mean = mat.mean()
        total = 0.0
        for k in range(mat.nnz):
            total += (mat._data[k] - mean) ** 2
        num_zeros = mat.shape[0] * mat.shape[1] - mat.nnz
        total += num_zeros * mean ** 2
        n = mat.shape[0] * mat.shape[1] - ddof
        return total / n if n > 0 else 0


def std(mat: SclCSR, axis: Optional[int] = None,
        ddof: int = 0) -> Union[float, RealArray]:
    """Compute standard deviation."""
    v = var(mat, axis, ddof)
    if isinstance(v, RealArray):
        result = RealArray(v.size)
        for i in range(v.size):
            result[i] = math.sqrt(v[i])
        return result
    else:
        return math.sqrt(v)


# =============================================================================
# Utility Functions
# =============================================================================

def copy(mat: SparseBase) -> SparseBase:
    """Copy matrix."""
    return mat.copy()


def issparse(x) -> bool:
    """Check if x is a sparse matrix."""
    return isinstance(x, SparseBase)


# =============================================================================
# Additional Transforms
# =============================================================================

def log2p1(mat: SclCSR, inplace: bool = False) -> SclCSR:
    """
    log2(1 + x) transform.

    Args:
        mat: Input matrix
        inplace: Modify in-place
    """
    mat.materialize()

    if inplace:
        for k in range(mat.nnz):
            mat._data[k] = math.log2(1.0 + mat._data[k])
        return mat
    else:
        new_data = RealArray(mat.nnz)
        for k in range(mat.nnz):
            new_data[k] = math.log2(1.0 + mat._data[k])
        return SclCSR.from_arrays(
            new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
        )


def expm1(mat: SclCSR, inplace: bool = False) -> SclCSR:
    """
    exp(x) - 1 transform.

    Args:
        mat: Input matrix
        inplace: Modify in-place
    """
    mat.materialize()

    if inplace:
        for k in range(mat.nnz):
            mat._data[k] = math.expm1(mat._data[k])
        return mat
    else:
        new_data = RealArray(mat.nnz)
        for k in range(mat.nnz):
            new_data[k] = math.expm1(mat._data[k])
        return SclCSR.from_arrays(
            new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
        )


def softmax(mat: SclCSR, axis: int = 1, inplace: bool = False) -> SclCSR:
    """
    Softmax transform.

    Args:
        mat: Input matrix
        axis: 0=column, 1=row
        inplace: Modify in-place
    """
    mat.materialize()

    if axis != 1:
        raise NotImplementedError("Only axis=1 (row softmax) supported")

    new_data = mat._data if inplace else mat._data.copy()

    for i in range(mat.shape[0]):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]
        if start == end:
            continue

        # Find max for numerical stability
        max_val = new_data[start]
        for k in range(start + 1, end):
            if new_data[k] > max_val:
                max_val = new_data[k]

        # Compute exp and sum
        exp_sum = 0.0
        for k in range(start, end):
            new_data[k] = math.exp(new_data[k] - max_val)
            exp_sum += new_data[k]

        # Normalize
        if exp_sum > 0:
            for k in range(start, end):
                new_data[k] /= exp_sum

    if inplace:
        return mat
    else:
        return SclCSR.from_arrays(
            new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
        )


# =============================================================================
# Linear Algebra
# =============================================================================

def gram(mat: SclCSR) -> RealArray:
    """
    Compute Gram matrix (X @ X.T for CSR).

    Args:
        mat: Input matrix

    Returns:
        Dense Gram matrix (rows x rows)
    """
    mat.materialize()

    n = mat.shape[0]
    result = RealArray(n * n)
    result.zero()

    # Simple O(n^2 * avg_nnz) implementation
    for i in range(n):
        i_start = mat._indptr[i]
        i_end = mat._indptr[i + 1]

        for j in range(i, n):
            j_start = mat._indptr[j]
            j_end = mat._indptr[j + 1]

            # Dot product of row i and row j
            dot = 0.0
            ii, jj = i_start, j_start
            while ii < i_end and jj < j_end:
                if mat._indices[ii] == mat._indices[jj]:
                    dot += mat._data[ii] * mat._data[jj]
                    ii += 1
                    jj += 1
                elif mat._indices[ii] < mat._indices[jj]:
                    ii += 1
                else:
                    jj += 1

            result[i * n + j] = dot
            result[j * n + i] = dot

    return result


def pearson(mat: SclCSR) -> RealArray:
    """
    Compute Pearson correlation matrix.

    Args:
        mat: Input matrix

    Returns:
        Dense correlation matrix (rows x rows)
    """
    mat.materialize()

    n = mat.shape[0]
    m = mat.shape[1]

    # Compute means
    means = mat.mean(axis=1)

    # Compute standard deviations
    stds = std(mat, axis=1)

    result = RealArray(n * n)

    for i in range(n):
        result[i * n + i] = 1.0  # Diagonal is 1

        i_start = mat._indptr[i]
        i_end = mat._indptr[i + 1]

        for j in range(i + 1, n):
            j_start = mat._indptr[j]
            j_end = mat._indptr[j + 1]

            # Compute covariance
            cov = 0.0
            ii, jj = i_start, j_start

            # Terms for non-zero pairs
            while ii < i_end and jj < j_end:
                if mat._indices[ii] == mat._indices[jj]:
                    cov += (mat._data[ii] - means[i]) * (mat._data[jj] - means[j])
                    ii += 1
                    jj += 1
                elif mat._indices[ii] < mat._indices[jj]:
                    cov += (mat._data[ii] - means[i]) * (-means[j])
                    ii += 1
                else:
                    cov += (-means[i]) * (mat._data[jj] - means[j])
                    jj += 1

            # Remaining terms
            while ii < i_end:
                cov += (mat._data[ii] - means[i]) * (-means[j])
                ii += 1
            while jj < j_end:
                cov += (-means[i]) * (mat._data[jj] - means[j])
                jj += 1

            # Zero-zero pairs contribution
            nnz_i = i_end - i_start
            nnz_j = j_end - j_start
            # This is approximate - full calculation would require tracking indices
            cov += means[i] * means[j] * (m - max(nnz_i, nnz_j))
            cov /= m

            # Correlation
            if stds[i] > 0 and stds[j] > 0:
                corr = cov / (stds[i] * stds[j])
            else:
                corr = 0.0

            result[i * n + j] = corr
            result[j * n + i] = corr

    return result


# =============================================================================
# Quality Control
# =============================================================================

def compute_qc(mat: SclCSR) -> Tuple[IndexArray, RealArray]:
    """
    Compute basic QC metrics per row (cell).

    Args:
        mat: Input matrix (cells x genes)

    Returns:
        (n_genes, total_counts) - genes per cell and total counts per cell
    """
    mat.materialize()

    n_genes = IndexArray(mat.shape[0])
    total_counts = RealArray(mat.shape[0])

    for i in range(mat.shape[0]):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]
        n_genes[i] = end - start

        total = 0.0
        for k in range(start, end):
            total += mat._data[k]
        total_counts[i] = total

    return n_genes, total_counts


# =============================================================================
# Feature Selection
# =============================================================================

def dispersion(mat: SclCSR, axis: int = 0) -> RealArray:
    """
    Compute dispersion (variance / mean) per feature.

    Args:
        mat: Input matrix
        axis: 0=column (genes), 1=row (cells)

    Returns:
        Dispersion values
    """
    means = mat.mean(axis=axis)
    vars = var(mat, axis=axis)

    result = RealArray(vars.size)
    for i in range(vars.size):
        if means[i] > 0:
            result[i] = vars[i] / means[i]
        else:
            result[i] = 0.0

    return result


def highly_variable(mat: SclCSR, n_top: int = 2000,
                    method: str = "dispersion") -> Tuple[IndexArray, ByteArray]:
    """
    Select highly variable features.

    Args:
        mat: Input matrix (cells x genes, CSR)
        n_top: Number of top features to select
        method: "dispersion" or "variance"

    Returns:
        (indices, mask) - indices of HVG and boolean mask
    """
    mat.materialize()

    if method == "dispersion":
        scores = dispersion(mat, axis=0)
    elif method == "variance":
        scores = var(mat, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Sort by score
    n_features = mat.shape[1]
    sorted_indices = list(range(n_features))
    sorted_indices.sort(key=lambda i: -scores[i])

    # Take top n
    n_select = min(n_top, n_features)
    selected = set(sorted_indices[:n_select])

    indices = IndexArray(n_select)
    mask = ByteArray(n_features)

    for i, idx in enumerate(sorted(selected)):
        indices[i] = idx
        mask[idx] = 1

    return indices, mask


# =============================================================================
# Standardization
# =============================================================================

def standardize(mat: SclCSR, zero_center: bool = True,
                max_value: Optional[float] = None,
                inplace: bool = False) -> SclCSR:
    """
    Standardize matrix (z-score normalization).

    Args:
        mat: Input matrix
        zero_center: Whether to center data
        max_value: Clip values to this maximum
        inplace: Modify in-place

    Returns:
        Standardized matrix
    """
    mat.materialize()

    means = mat.mean(axis=0)
    stds = std(mat, axis=0)

    new_data = mat._data if inplace else mat._data.copy()

    for i in range(mat.shape[0]):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]
        for k in range(start, end):
            j = mat._indices[k]
            if stds[j] > 0:
                val = new_data[k]
                if zero_center:
                    val = (val - means[j]) / stds[j]
                else:
                    val = val / stds[j]
                if max_value is not None:
                    val = min(val, max_value)
                    val = max(val, -max_value)
                new_data[k] = val

    if inplace:
        return mat
    else:
        return SclCSR.from_arrays(
            new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
        )


# =============================================================================
# Resampling
# =============================================================================

def downsample(mat: SclCSR, target_sum: float,
               seed: Optional[int] = None,
               inplace: bool = False) -> SclCSR:
    """
    Downsample counts to target sum per cell.

    Args:
        mat: Input matrix (counts)
        target_sum: Target sum per cell
        seed: Random seed
        inplace: Modify in-place

    Returns:
        Downsampled matrix
    """
    mat.materialize()

    import random
    if seed is not None:
        random.seed(seed)

    new_data = mat._data if inplace else mat._data.copy()

    for i in range(mat.shape[0]):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]

        # Compute current sum
        current_sum = 0.0
        for k in range(start, end):
            current_sum += new_data[k]

        if current_sum <= target_sum:
            continue

        # Scale factor
        scale = target_sum / current_sum
        for k in range(start, end):
            new_data[k] *= scale

    if inplace:
        return mat
    else:
        return SclCSR.from_arrays(
            new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
        )


# =============================================================================
# Statistical Tests
# =============================================================================

def mwu_test(mat: SclCSC, groups: Sequence[int],
             ) -> Tuple[RealArray, RealArray, RealArray]:
    """
    Mann-Whitney U test for differential expression.

    Args:
        mat: Input matrix (genes x cells, CSC format)
        groups: Group labels per cell (0 or 1)

    Returns:
        (u_stats, p_values, effect_sizes) for each gene
    """
    mat.materialize()

    n_features = mat.shape[0]

    u_stats = RealArray(n_features)
    p_values = RealArray(n_features)
    effect_sizes = RealArray(n_features)

    # Pure Python implementation
    from scl.array import IndexArray as IA

    group_arr = IA.from_sequence(groups)

    # Count groups
    n0 = sum(1 for g in groups if g == 0)
    n1 = sum(1 for g in groups if g == 1)

    if n0 == 0 or n1 == 0:
        u_stats.zero()
        p_values.fill(1.0)
        effect_sizes.zero()
        return u_stats, p_values, effect_sizes

    for j in range(mat.shape[1]):  # For each gene (column in CSC)
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        # Collect values for each group
        vals0 = []
        vals1 = []

        for k in range(start, end):
            row = mat._indices[k]
            val = mat._data[k]
            if groups[row] == 0:
                vals0.append(val)
            else:
                vals1.append(val)

        # Add zeros for missing cells
        vals0.extend([0.0] * (n0 - len(vals0)))
        vals1.extend([0.0] * (n1 - len(vals1)))

        # Compute U statistic
        u = 0.0
        for v0 in vals0:
            for v1 in vals1:
                if v0 > v1:
                    u += 1.0
                elif v0 == v1:
                    u += 0.5

        u_stats[j] = u

        # Effect size (rank-biserial correlation)
        effect_sizes[j] = 2 * u / (n0 * n1) - 1.0

        # P-value approximation (normal approximation for large samples)
        mu = n0 * n1 / 2.0
        sigma = math.sqrt(n0 * n1 * (n0 + n1 + 1) / 12.0)
        if sigma > 0:
            z = (u - mu) / sigma
            # Two-tailed p-value using normal approximation
            p_values[j] = 2.0 * (1.0 - _norm_cdf(abs(z)))
        else:
            p_values[j] = 1.0

    return u_stats, p_values, effect_sizes


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    # Approximation using error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ttest(mat: SclCSC, groups: Sequence[int],
          ) -> Tuple[RealArray, RealArray, RealArray, RealArray]:
    """
    T-test for differential expression.

    Args:
        mat: Input matrix (genes x cells, CSC format)
        groups: Group labels per cell (0 or 1)

    Returns:
        (t_stats, p_values, mean_diff, log_fc) for each gene
    """
    mat.materialize()

    n_features = mat.shape[1]  # Columns in CSC

    t_stats = RealArray(n_features)
    p_values = RealArray(n_features)
    mean_diff = RealArray(n_features)
    log_fc = RealArray(n_features)

    # Count groups
    n0 = sum(1 for g in groups if g == 0)
    n1 = sum(1 for g in groups if g == 1)

    if n0 < 2 or n1 < 2:
        t_stats.zero()
        p_values.fill(1.0)
        mean_diff.zero()
        log_fc.zero()
        return t_stats, p_values, mean_diff, log_fc

    for j in range(n_features):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        # Compute means and variances for each group
        sum0, sum1 = 0.0, 0.0
        count0, count1 = 0, 0

        for k in range(start, end):
            row = mat._indices[k]
            val = mat._data[k]
            if groups[row] == 0:
                sum0 += val
                count0 += 1
            else:
                sum1 += val
                count1 += 1

        mean0 = sum0 / n0
        mean1 = sum1 / n1

        # Variance
        var0, var1 = 0.0, 0.0
        for k in range(start, end):
            row = mat._indices[k]
            val = mat._data[k]
            if groups[row] == 0:
                var0 += (val - mean0) ** 2
            else:
                var1 += (val - mean1) ** 2

        # Add contribution from zeros
        var0 += (n0 - count0) * mean0 ** 2
        var1 += (n1 - count1) * mean1 ** 2

        var0 /= (n0 - 1) if n0 > 1 else 1
        var1 /= (n1 - 1) if n1 > 1 else 1

        # Welch's t-test
        se = math.sqrt(var0 / n0 + var1 / n1) if (var0 / n0 + var1 / n1) > 0 else 1e-10
        t = (mean0 - mean1) / se

        t_stats[j] = t
        mean_diff[j] = mean0 - mean1

        # Log fold change
        eps = 1e-10
        log_fc[j] = math.log2((mean0 + eps) / (mean1 + eps))

        # P-value (approximate using normal for large df)
        p_values[j] = 2.0 * (1.0 - _norm_cdf(abs(t)))

    return t_stats, p_values, mean_diff, log_fc


# =============================================================================
# Group Statistics
# =============================================================================

def group_stats(mat: SclCSC, groups: Sequence[int], n_groups: int,
                ddof: int = 0, count_zeros: bool = True
                ) -> Tuple[RealArray, RealArray]:
    """
    Compute mean and variance per group for each feature.

    Args:
        mat: Input matrix (cells x genes, CSC)
        groups: Group labels per cell
        n_groups: Number of groups
        ddof: Degrees of freedom correction
        count_zeros: Include zeros in statistics

    Returns:
        (means, variances) - shape (n_features, n_groups) flattened
    """
    mat.materialize()

    n_features = mat.shape[1]
    output_size = n_features * n_groups

    means = RealArray(output_size)
    variances = RealArray(output_size)
    means.zero()
    variances.zero()

    # Count group sizes
    group_sizes = count_group_sizes(groups, n_groups)

    # Compute per-group sums
    for j in range(n_features):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        group_sums = [0.0] * n_groups
        group_counts = [0] * n_groups

        for k in range(start, end):
            row = mat._indices[k]
            val = mat._data[k]
            g = groups[row]
            if 0 <= g < n_groups:
                group_sums[g] += val
                group_counts[g] += 1

        # Compute means
        for g in range(n_groups):
            size = group_sizes[g] if count_zeros else group_counts[g]
            if size > 0:
                means[j * n_groups + g] = group_sums[g] / size

        # Compute variances
        for k in range(start, end):
            row = mat._indices[k]
            val = mat._data[k]
            g = groups[row]
            if 0 <= g < n_groups:
                mean = means[j * n_groups + g]
                variances[j * n_groups + g] += (val - mean) ** 2

        # Add zero contributions if count_zeros
        if count_zeros:
            for g in range(n_groups):
                n_zeros = group_sizes[g] - group_counts[g]
                mean = means[j * n_groups + g]
                variances[j * n_groups + g] += n_zeros * mean ** 2

        # Normalize variances
        for g in range(n_groups):
            size = group_sizes[g] if count_zeros else group_counts[g]
            denom = size - ddof
            if denom > 0:
                variances[j * n_groups + g] /= denom

    return means, variances


def count_group_sizes(groups: Sequence[int], n_groups: int) -> IndexArray:
    """
    Count elements per group.

    Args:
        groups: Group labels
        n_groups: Number of groups

    Returns:
        Array of group sizes
    """
    sizes = IndexArray(n_groups)
    sizes.zero()

    for g in groups:
        if 0 <= g < n_groups:
            sizes[g] += 1

    return sizes


# =============================================================================
# Feature Statistics
# =============================================================================

def detection_rate(mat: SclCSC) -> RealArray:
    """
    Compute detection rate (fraction of non-zeros) per feature.

    Args:
        mat: Input matrix (cells x genes, CSC)

    Returns:
        Detection rate per gene
    """
    mat.materialize()

    n_features = mat.shape[1]
    n_cells = mat.shape[0]
    result = RealArray(n_features)

    for j in range(n_features):
        nnz = mat._indptr[j + 1] - mat._indptr[j]
        result[j] = nnz / n_cells if n_cells > 0 else 0.0

    return result


def standard_moments(mat: SclCSC, ddof: int = 0
                     ) -> Tuple[RealArray, RealArray]:
    """
    Compute mean and variance per feature.

    Args:
        mat: Input matrix (cells x genes, CSC)
        ddof: Degrees of freedom correction

    Returns:
        (means, variances) per gene
    """
    mat.materialize()

    n_features = mat.shape[1]
    n_cells = mat.shape[0]

    means = RealArray(n_features)
    variances = RealArray(n_features)

    for j in range(n_features):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        # Sum
        total = 0.0
        for k in range(start, end):
            total += mat._data[k]

        mean = total / n_cells if n_cells > 0 else 0.0
        means[j] = mean

        # Variance
        var_sum = 0.0
        for k in range(start, end):
            var_sum += (mat._data[k] - mean) ** 2

        # Zero contributions
        n_zeros = n_cells - (end - start)
        var_sum += n_zeros * mean ** 2

        denom = n_cells - ddof
        variances[j] = var_sum / denom if denom > 0 else 0.0

    return means, variances


def clipped_moments(mat: SclCSC, clip_max: Optional[Sequence[float]] = None
                    ) -> Tuple[RealArray, RealArray]:
    """
    Compute mean and variance with optional clipping.

    Args:
        mat: Input matrix (cells x genes, CSC)
        clip_max: Optional max values per gene for clipping

    Returns:
        (means, variances) per gene
    """
    mat.materialize()

    n_features = mat.shape[1]
    n_cells = mat.shape[0]

    means = RealArray(n_features)
    variances = RealArray(n_features)

    for j in range(n_features):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        clip_val = clip_max[j] if clip_max is not None else float('inf')

        # Sum with clipping
        total = 0.0
        for k in range(start, end):
            val = min(mat._data[k], clip_val)
            total += val

        mean = total / n_cells if n_cells > 0 else 0.0
        means[j] = mean

        # Variance with clipping
        var_sum = 0.0
        for k in range(start, end):
            val = min(mat._data[k], clip_val)
            var_sum += (val - mean) ** 2

        n_zeros = n_cells - (end - start)
        var_sum += n_zeros * mean ** 2

        variances[j] = var_sum / n_cells if n_cells > 0 else 0.0

    return means, variances


# =============================================================================
# Spatial Statistics
# =============================================================================

def mmd_rbf(X: SclCSC, Y: SclCSC, gamma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel.

    Args:
        X: First matrix (cells x genes, CSC)
        Y: Second matrix (cells x genes, CSC)
        gamma: RBF kernel parameter

    Returns:
        MMD value
    """
    X.materialize()
    Y.materialize()

    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Feature dimension mismatch: {X.shape[1]} vs {Y.shape[1]}")

    n_x = X.shape[0]
    n_y = Y.shape[0]

    # Convert to dense for simplicity (could be optimized)
    X_dense = X.to_dense()
    Y_dense = Y.to_dense()
    n_features = X.shape[1]

    def rbf_kernel(a_start: int, b_start: int, a_dense: RealArray, b_dense: RealArray) -> float:
        sq_dist = 0.0
        for f in range(n_features):
            diff = a_dense[a_start + f] - b_dense[b_start + f]
            sq_dist += diff * diff
        return math.exp(-gamma * sq_dist)

    # K(X, X)
    kxx = 0.0
    for i in range(n_x):
        for j in range(n_x):
            if i != j:
                kxx += rbf_kernel(i * n_features, j * n_features, X_dense, X_dense)
    kxx /= (n_x * (n_x - 1)) if n_x > 1 else 1

    # K(Y, Y)
    kyy = 0.0
    for i in range(n_y):
        for j in range(n_y):
            if i != j:
                kyy += rbf_kernel(i * n_features, j * n_features, Y_dense, Y_dense)
    kyy /= (n_y * (n_y - 1)) if n_y > 1 else 1

    # K(X, Y)
    kxy = 0.0
    for i in range(n_x):
        for j in range(n_y):
            kxy += rbf_kernel(i * n_features, j * n_features, X_dense, Y_dense)
    kxy /= (n_x * n_y) if (n_x > 0 and n_y > 0) else 1

    return kxx + kyy - 2 * kxy


def morans_i(X: SclCSC, W: SclCSC) -> RealArray:
    """
    Compute Moran's I spatial autocorrelation.

    Args:
        X: Feature matrix (cells x genes, CSC)
        W: Spatial weight matrix (cells x cells, CSC)

    Returns:
        Moran's I per gene
    """
    X.materialize()
    W.materialize()

    n_cells = X.shape[0]
    n_features = X.shape[1]

    if W.shape[0] != n_cells or W.shape[1] != n_cells:
        raise ValueError("Weight matrix dimensions must match number of cells")

    result = RealArray(n_features)

    # Convert weight matrix to list of (i, j, w) tuples for easier iteration
    W_list = []
    W_sum = 0.0
    for j in range(W.shape[1]):
        start = W._indptr[j]
        end = W._indptr[j + 1]
        for k in range(start, end):
            i = W._indices[k]
            w = W._data[k]
            W_list.append((i, j, w))
            W_sum += w

    if W_sum == 0:
        result.zero()
        return result

    # For each feature
    for f in range(n_features):
        # Get feature values
        x = [0.0] * n_cells
        start = X._indptr[f]
        end = X._indptr[f + 1]
        for k in range(start, end):
            x[X._indices[k]] = X._data[k]

        # Mean
        x_mean = sum(x) / n_cells

        # Deviation from mean
        z = [v - x_mean for v in x]

        # Sum of squared deviations
        ss = sum(zi * zi for zi in z)

        if ss == 0:
            result[f] = 0.0
            continue

        # Compute numerator
        numerator = 0.0
        for i, j, w in W_list:
            numerator += w * z[i] * z[j]

        # Moran's I
        result[f] = (n_cells / W_sum) * (numerator / ss)

    return result


# =============================================================================
# Row/Column NNZ Counts
# =============================================================================

def nnz_per_row(mat: SclCSR) -> IndexArray:
    """
    Count non-zeros per row.

    Args:
        mat: Input CSR matrix

    Returns:
        NNZ count per row
    """
    mat.materialize()

    result = IndexArray(mat.shape[0])
    for i in range(mat.shape[0]):
        result[i] = mat._indptr[i + 1] - mat._indptr[i]

    return result


def nnz_per_col(mat: SclCSC) -> IndexArray:
    """
    Count non-zeros per column.

    Args:
        mat: Input CSC matrix

    Returns:
        NNZ count per column
    """
    mat.materialize()

    result = IndexArray(mat.shape[1])
    for j in range(mat.shape[1]):
        result[j] = mat._indptr[j + 1] - mat._indptr[j]

    return result


# =============================================================================
# Reordering
# =============================================================================

def align_secondary(mat: SclCSC, old_indices: Sequence[int],
                    new_indices: Sequence[int]) -> SclCSC:
    """
    Align secondary indices (rows for CSC).

    Args:
        mat: Input CSC matrix
        old_indices: Original row indices
        new_indices: Target row indices

    Returns:
        Realigned matrix
    """
    mat.materialize()

    # Create mapping from old to new
    remap = {}
    for old, new in zip(old_indices, new_indices):
        remap[old] = new

    new_n_rows = max(new_indices) + 1 if new_indices else mat.shape[0]

    new_data = []
    new_indices_arr = []
    new_indptr = [0]

    for j in range(mat.shape[1]):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        col_entries = []
        for k in range(start, end):
            old_row = mat._indices[k]
            if old_row in remap:
                new_row = remap[old_row]
                col_entries.append((new_row, mat._data[k]))

        # Sort by new row index
        col_entries.sort(key=lambda x: x[0])

        for new_row, val in col_entries:
            new_indices_arr.append(new_row)
            new_data.append(val)

        new_indptr.append(len(new_data))

    return SclCSC.from_arrays(new_data, new_indices_arr, new_indptr,
                               (new_n_rows, mat.shape[1]))


# =============================================================================
# CSC Statistics
# =============================================================================

def sum_csc(mat: SclCSC, axis: Optional[int] = None) -> Union[float, RealArray]:
    """Sum for CSC matrix."""
    mat.materialize()
    return mat.sum(axis)


def mean_csc(mat: SclCSC, axis: Optional[int] = None) -> Union[float, RealArray]:
    """Mean for CSC matrix."""
    mat.materialize()
    return mat.mean(axis)


def var_csc(mat: SclCSC, axis: Optional[int] = None,
            ddof: int = 0) -> Union[float, RealArray]:
    """
    Compute variance for CSC matrix.

    Args:
        mat: Input CSC matrix
        axis: None=global, 0=column, 1=row
        ddof: Degrees of freedom correction
    """
    mat.materialize()

    if axis == 0:  # Column variance (efficient for CSC)
        result = RealArray(mat.shape[1])

        for j in range(mat.shape[1]):
            start = mat._indptr[j]
            end = mat._indptr[j + 1]

            # Mean
            total = 0.0
            for k in range(start, end):
                total += mat._data[k]
            mean = total / mat.shape[0]

            # Variance
            var_sum = 0.0
            for k in range(start, end):
                var_sum += (mat._data[k] - mean) ** 2

            # Zero contributions
            n_zeros = mat.shape[0] - (end - start)
            var_sum += n_zeros * mean ** 2

            denom = mat.shape[0] - ddof
            result[j] = var_sum / denom if denom > 0 else 0.0

        return result

    elif axis == 1:  # Row variance
        means = mat.mean(axis=1)
        result = RealArray(mat.shape[0])
        result.zero()

        counts = [0] * mat.shape[0]
        sum_sq = [0.0] * mat.shape[0]

        for j in range(mat.shape[1]):
            start = mat._indptr[j]
            end = mat._indptr[j + 1]
            for k in range(start, end):
                i = mat._indices[k]
                sum_sq[i] += (mat._data[k] - means[i]) ** 2
                counts[i] += 1

        for i in range(mat.shape[0]):
            n_zeros = mat.shape[1] - counts[i]
            sum_sq[i] += n_zeros * means[i] ** 2
            denom = mat.shape[1] - ddof
            result[i] = sum_sq[i] / denom if denom > 0 else 0.0

        return result

    else:  # Global variance
        mean = mat.mean()
        total = 0.0
        for k in range(mat.nnz):
            total += (mat._data[k] - mean) ** 2
        n_zeros = mat.shape[0] * mat.shape[1] - mat.nnz
        total += n_zeros * mean ** 2
        denom = mat.shape[0] * mat.shape[1] - ddof
        return total / denom if denom > 0 else 0.0


def std_csc(mat: SclCSC, axis: Optional[int] = None,
            ddof: int = 0) -> Union[float, RealArray]:
    """Standard deviation for CSC matrix."""
    v = var_csc(mat, axis, ddof)
    if isinstance(v, RealArray):
        result = RealArray(v.size)
        for i in range(v.size):
            result[i] = math.sqrt(v[i])
        return result
    else:
        return math.sqrt(v)


# =============================================================================
# CSC Transforms
# =============================================================================

def log1p_csc(mat: SclCSC, inplace: bool = False) -> SclCSC:
    """log1p transform for CSC matrix."""
    mat.materialize()

    new_data = RealArray(mat.nnz)
    for k in range(mat.nnz):
        new_data[k] = math.log1p(mat._data[k])

    if inplace:
        mat._data = new_data
        return mat
    else:
        return SclCSC.from_arrays(
            new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
        )


def scale_csc(mat: SclCSC, col_factors: Optional[Sequence[float]] = None,
              row_factors: Optional[Sequence[float]] = None,
              inplace: bool = False) -> SclCSC:
    """
    Scale CSC matrix.

    Args:
        mat: Input CSC matrix
        col_factors: Column scale factors (efficient for CSC)
        row_factors: Row scale factors
        inplace: Modify in-place
    """
    mat.materialize()

    new_data = mat._data if inplace else mat._data.copy()

    if col_factors is not None:
        for j in range(mat.shape[1]):
            start = mat._indptr[j]
            end = mat._indptr[j + 1]
            factor = col_factors[j]
            for k in range(start, end):
                new_data[k] *= factor

    if row_factors is not None:
        for j in range(mat.shape[1]):
            start = mat._indptr[j]
            end = mat._indptr[j + 1]
            for k in range(start, end):
                new_data[k] *= row_factors[mat._indices[k]]

    if inplace:
        return mat
    else:
        return SclCSC.from_arrays(
            new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
        )


def normalize_csc(mat: SclCSC, norm: str = "l2", axis: int = 0,
                  inplace: bool = False) -> SclCSC:
    """
    Normalize CSC matrix.

    Args:
        mat: Input CSC matrix
        norm: "l1", "l2", or "max"
        axis: 0=column (efficient), 1=row
        inplace: Modify in-place
    """
    mat.materialize()

    if axis != 0:
        raise NotImplementedError("Only axis=0 (column normalization) supported for CSC")

    new_data = mat._data if inplace else mat._data.copy()

    for j in range(mat.shape[1]):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]
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
        elif norm == "max":
            for k in range(start, end):
                n = max(n, abs(new_data[k]))
        else:
            raise ValueError(f"Unknown norm: {norm}")

        # Normalize
        if n > 0:
            for k in range(start, end):
                new_data[k] /= n

    if inplace:
        return mat
    else:
        return SclCSC.from_arrays(
            new_data, mat._indices.copy(), mat._indptr.copy(), mat.shape
        )


# =============================================================================
# Primary Axis Statistics (row for CSR, column for CSC)
# =============================================================================

def row_sums(mat: SclCSR) -> RealArray:
    """
    Compute row sums (efficient for CSR).

    Args:
        mat: Input CSR matrix

    Returns:
        Sum per row
    """
    return mat.sum(axis=1)


def col_sums(mat: SclCSC) -> RealArray:
    """
    Compute column sums (efficient for CSC).

    Args:
        mat: Input CSC matrix

    Returns:
        Sum per column
    """
    return mat.sum(axis=0)


def row_means(mat: SclCSR, count_zeros: bool = True) -> RealArray:
    """
    Compute row means.

    Args:
        mat: Input CSR matrix
        count_zeros: Include zeros in mean calculation

    Returns:
        Mean per row
    """
    mat.materialize()

    result = RealArray(mat.shape[0])
    for i in range(mat.shape[0]):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]

        total = 0.0
        for k in range(start, end):
            total += mat._data[k]

        if count_zeros:
            result[i] = total / mat.shape[1] if mat.shape[1] > 0 else 0.0
        else:
            nnz = end - start
            result[i] = total / nnz if nnz > 0 else 0.0

    return result


def col_means(mat: SclCSC, count_zeros: bool = True) -> RealArray:
    """
    Compute column means.

    Args:
        mat: Input CSC matrix
        count_zeros: Include zeros in mean calculation

    Returns:
        Mean per column
    """
    mat.materialize()

    result = RealArray(mat.shape[1])
    for j in range(mat.shape[1]):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]

        total = 0.0
        for k in range(start, end):
            total += mat._data[k]

        if count_zeros:
            result[j] = total / mat.shape[0] if mat.shape[0] > 0 else 0.0
        else:
            nnz = end - start
            result[j] = total / nnz if nnz > 0 else 0.0

    return result


def row_variances(mat: SclCSR, ddof: int = 0, count_zeros: bool = True) -> RealArray:
    """
    Compute row variances.

    Args:
        mat: Input CSR matrix
        ddof: Degrees of freedom correction
        count_zeros: Include zeros in variance calculation

    Returns:
        Variance per row
    """
    mat.materialize()

    means = row_means(mat, count_zeros)
    result = RealArray(mat.shape[0])

    for i in range(mat.shape[0]):
        start = mat._indptr[i]
        end = mat._indptr[i + 1]
        mean = means[i]

        var_sum = 0.0
        for k in range(start, end):
            var_sum += (mat._data[k] - mean) ** 2

        if count_zeros:
            n_zeros = mat.shape[1] - (end - start)
            var_sum += n_zeros * mean ** 2
            denom = mat.shape[1] - ddof
        else:
            denom = (end - start) - ddof

        result[i] = var_sum / denom if denom > 0 else 0.0

    return result


def col_variances(mat: SclCSC, ddof: int = 0, count_zeros: bool = True) -> RealArray:
    """
    Compute column variances.

    Args:
        mat: Input CSC matrix
        ddof: Degrees of freedom correction
        count_zeros: Include zeros in variance calculation

    Returns:
        Variance per column
    """
    mat.materialize()

    means = col_means(mat, count_zeros)
    result = RealArray(mat.shape[1])

    for j in range(mat.shape[1]):
        start = mat._indptr[j]
        end = mat._indptr[j + 1]
        mean = means[j]

        var_sum = 0.0
        for k in range(start, end):
            var_sum += (mat._data[k] - mean) ** 2

        if count_zeros:
            n_zeros = mat.shape[0] - (end - start)
            var_sum += n_zeros * mean ** 2
            denom = mat.shape[0] - ddof
        else:
            denom = (end - start) - ddof

        result[j] = var_sum / denom if denom > 0 else 0.0

    return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Normalization
    "normalize",
    "normalize_csc",
    # Transforms
    "log1p",
    "log1p_csc",
    "log2p1",
    "expm1",
    "softmax",
    "scale",
    "scale_csc",
    # Filtering
    "filter_threshold",
    "top_k",
    # Matrix operations
    "spmv",
    "dot",
    "gram",
    "pearson",
    # Statistics
    "var",
    "var_csc",
    "std",
    "std_csc",
    "sum_csc",
    "mean_csc",
    "dispersion",
    "nnz_per_row",
    "nnz_per_col",
    # Primary axis statistics
    "row_sums",
    "col_sums",
    "row_means",
    "col_means",
    "row_variances",
    "col_variances",
    # Statistical tests
    "mwu_test",
    "ttest",
    # Group statistics
    "group_stats",
    "count_group_sizes",
    # Feature statistics
    "detection_rate",
    "standard_moments",
    "clipped_moments",
    # Quality Control
    "compute_qc",
    # Feature Selection
    "highly_variable",
    # Standardization
    "standardize",
    # Resampling
    "downsample",
    # Spatial
    "mmd_rbf",
    "morans_i",
    # Reordering
    "align_secondary",
    # Utility
    "copy",
    "issparse",
]
