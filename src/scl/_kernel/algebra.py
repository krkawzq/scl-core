"""
Linear Algebra Kernels

Low-level C bindings for sparse matrix-vector operations.
"""

import ctypes
import numpy as np
from .lib_loader import get_lib
from .types import c_real, c_index, check_error, as_c_ptr

__all__ = [
    'spmv_csr',
    'spmv_trans_csc',
    'gram_csr',
    'gram_csc',
]

# =============================================================================
# Function Signatures
# =============================================================================

def _init_signatures():
    """Initialize C function signatures."""
    lib = get_lib()
    
    # spmv_csr
    lib.scl_spmv_csr.argtypes = [
        ctypes.POINTER(c_real),   # A_data
        ctypes.POINTER(c_index),  # A_indices
        ctypes.POINTER(c_index),  # A_indptr
        ctypes.POINTER(c_index),  # A_row_lengths
        c_index,                   # A_rows
        c_index,                   # A_cols
        c_index,                   # A_nnz
        ctypes.POINTER(c_real),   # x
        ctypes.POINTER(c_real),   # y
        c_real,                    # alpha
        c_real,                    # beta
    ]
    lib.scl_spmv_csr.restype = ctypes.c_int
    
    # spmv_trans_csc
    lib.scl_spmv_trans_csc.argtypes = [
        ctypes.POINTER(c_real),   # A_data
        ctypes.POINTER(c_index),  # A_indices
        ctypes.POINTER(c_index),  # A_indptr
        ctypes.POINTER(c_index),  # A_col_lengths
        c_index,                   # A_rows
        c_index,                   # A_cols
        c_index,                   # A_nnz
        ctypes.POINTER(c_real),   # x
        ctypes.POINTER(c_real),   # y
        c_real,                    # alpha
        c_real,                    # beta
    ]
    lib.scl_spmv_trans_csc.restype = ctypes.c_int
    
    # gram_csr
    lib.scl_gram_csr.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # row_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_real),   # output
    ]
    lib.scl_gram_csr.restype = ctypes.c_int
    
    # gram_csc
    lib.scl_gram_csc.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # col_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_real),   # output
    ]
    lib.scl_gram_csc.restype = ctypes.c_int


# Initialize signatures lazily
try:
    _init_signatures()
except Exception as e:
    import warnings
    warnings.warn(f"SCL library not ready: {e}")

# =============================================================================
# Python Wrappers
# =============================================================================

def spmv_csr(
    A_data: np.ndarray,
    A_indices: np.ndarray,
    A_indptr: np.ndarray,
    A_row_lengths: np.ndarray,
    A_rows: int,
    A_cols: int,
    A_nnz: int,
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0
) -> None:
    """
    Sparse matrix-vector multiplication: y = alpha * A * x + beta * y
    
    Args:
        A_data: CSR data array
        A_indices: CSR column indices
        A_indptr: CSR row pointers
        A_row_lengths: Explicit row lengths or None
        A_rows: Number of rows
        A_cols: Number of columns
        A_nnz: Number of non-zeros
        x: Input vector, shape (A_cols,)
        y: Output vector, shape (A_rows,) - modified in-place
        alpha: Scalar multiplier for A*x
        beta: Scalar multiplier for y
    """
    lib = get_lib()
    
    status = lib.scl_spmv_csr(
        as_c_ptr(A_data, c_real),
        as_c_ptr(A_indices, c_index),
        as_c_ptr(A_indptr, c_index),
        as_c_ptr(A_row_lengths, c_index) if A_row_lengths is not None else None,
        A_rows, A_cols, A_nnz,
        as_c_ptr(x, c_real),
        as_c_ptr(y, c_real),
        alpha, beta
    )
    
    check_error(status, "spmv_csr")


def spmv_trans_csc(
    A_data: np.ndarray,
    A_indices: np.ndarray,
    A_indptr: np.ndarray,
    A_col_lengths: np.ndarray,
    A_rows: int,
    A_cols: int,
    A_nnz: int,
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0
) -> None:
    """
    Transposed sparse matrix-vector multiplication: y = alpha * A^T * x + beta * y
    
    Args:
        A_data: CSC data array
        A_indices: CSC row indices
        A_indptr: CSC column pointers
        A_col_lengths: Explicit column lengths or None
        A_rows: Number of rows
        A_cols: Number of columns
        A_nnz: Number of non-zeros
        x: Input vector, shape (A_rows,)
        y: Output vector, shape (A_cols,) - modified in-place
        alpha: Scalar multiplier for A^T*x
        beta: Scalar multiplier for y
    """
    lib = get_lib()
    
    status = lib.scl_spmv_trans_csc(
        as_c_ptr(A_data, c_real),
        as_c_ptr(A_indices, c_index),
        as_c_ptr(A_indptr, c_index),
        as_c_ptr(A_col_lengths, c_index) if A_col_lengths is not None else None,
        A_rows, A_cols, A_nnz,
        as_c_ptr(x, c_real),
        as_c_ptr(y, c_real),
        alpha, beta
    )
    
    check_error(status, "spmv_trans_csc")


def gram_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    row_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    output: np.ndarray
) -> None:
    """
    Compute Gram matrix A * A^T (sample similarity).
    
    Args:
        data: CSR data array
        indices: CSR column indices
        indptr: CSR row pointers
        row_lengths: Explicit row lengths or None
        rows: Number of rows
        cols: Number of columns
        nnz: Number of non-zeros
        output: Output dense matrix, shape (rows, rows) - row-major
    """
    lib = get_lib()
    
    status = lib.scl_gram_csr(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(row_lengths, c_index) if row_lengths is not None else None,
        rows, cols, nnz,
        as_c_ptr(output, c_real)
    )
    
    check_error(status, "gram_csr")


def gram_csc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    col_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    output: np.ndarray
) -> None:
    """
    Compute Gram matrix A^T * A (feature correlation).
    
    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        col_lengths: Explicit column lengths or None
        rows: Number of rows
        cols: Number of columns
        nnz: Number of non-zeros
        output: Output dense matrix, shape (cols, cols) - row-major
    """
    lib = get_lib()
    
    status = lib.scl_gram_csc(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(col_lengths, c_index) if col_lengths is not None else None,
        rows, cols, nnz,
        as_c_ptr(output, c_real)
    )
    
    check_error(status, "gram_csc")

