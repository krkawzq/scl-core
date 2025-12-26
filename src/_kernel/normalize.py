"""
Normalization Kernels

Low-level C bindings for normalization operations.
"""

import ctypes
import numpy as np
from .lib_loader import get_lib
from .types import c_real, c_index, check_error, as_c_ptr

__all__ = [
    'scale_rows_csr',
    'scale_cols_csc',
]

# =============================================================================
# Function Signatures
# =============================================================================

def _init_signatures():
    """Initialize C function signatures."""
    lib = get_lib()
    
    # scale_rows_csr
    lib.scl_scale_rows_csr.argtypes = [
        ctypes.POINTER(c_real),   # data (mutable)
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # row_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_real),   # scales
    ]
    lib.scl_scale_rows_csr.restype = ctypes.c_int
    
    # scale_cols_csc
    lib.scl_scale_cols_csc.argtypes = [
        ctypes.POINTER(c_real),   # data (mutable)
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # col_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_real),   # scales
    ]
    lib.scl_scale_cols_csc.restype = ctypes.c_int


_init_signatures()

# =============================================================================
# Python Wrappers
# =============================================================================

def scale_rows_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    row_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    scales: np.ndarray
) -> None:
    """
    Scale each row by a factor (CSR matrix, in-place).
    
    Args:
        data: CSR data array (modified in-place)
        indices: CSR column indices
        indptr: CSR row pointers
        row_lengths: Explicit row lengths or None
        rows: Number of rows
        cols: Number of columns
        nnz: Number of non-zeros
        scales: Scale factors per row, shape (rows,)
    """
    lib = get_lib()
    
    status = lib.scl_scale_rows_csr(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(row_lengths, c_index) if row_lengths is not None else None,
        rows, cols, nnz,
        as_c_ptr(scales, c_real)
    )
    
    check_error(status, "scale_rows_csr")


def scale_cols_csc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    col_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    scales: np.ndarray
) -> None:
    """
    Scale each column by a factor (CSC matrix, in-place).
    
    Args:
        data: CSC data array (modified in-place)
        indices: CSC row indices
        indptr: CSC column pointers
        col_lengths: Explicit column lengths or None
        rows: Number of rows
        cols: Number of columns
        nnz: Number of non-zeros
        scales: Scale factors per column, shape (cols,)
    """
    lib = get_lib()
    
    status = lib.scl_scale_cols_csc(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(col_lengths, c_index) if col_lengths is not None else None,
        rows, cols, nnz,
        as_c_ptr(scales, c_real)
    )
    
    check_error(status, "scale_cols_csc")

