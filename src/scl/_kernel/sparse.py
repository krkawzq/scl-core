"""
Sparse Matrix Statistics Kernels

Low-level C bindings for sparse matrix aggregation operations.
"""

import ctypes
import numpy as np
from .lib_loader import get_lib
from .types import c_real, c_index, c_size, np_real, np_index, check_error, as_c_ptr

__all__ = [
    'row_sums_csr',
    'col_sums_csc',
    'row_statistics_csr',
    'col_statistics_csc',
]

# =============================================================================
# Function Signatures
# =============================================================================

def _init_signatures():
    """Initialize C function signatures."""
    lib = get_lib()
    
    # row_sums_csr
    lib.scl_row_sums_csr.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # row_lengths (can be NULL)
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_real),   # output
    ]
    lib.scl_row_sums_csr.restype = ctypes.c_int
    
    # col_sums_csc
    lib.scl_col_sums_csc.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # col_lengths (can be NULL)
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_real),   # output
    ]
    lib.scl_col_sums_csc.restype = ctypes.c_int
    
    # row_statistics_csr
    lib.scl_row_statistics_csr.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # row_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_real),   # out_means
        ctypes.POINTER(c_real),   # out_vars
    ]
    lib.scl_row_statistics_csr.restype = ctypes.c_int
    
    # col_statistics_csc
    lib.scl_col_statistics_csc.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # col_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_real),   # out_means
        ctypes.POINTER(c_real),   # out_vars
    ]
    lib.scl_col_statistics_csc.restype = ctypes.c_int


# Try to initialize signatures on import, but don't fail
try:
    _init_signatures()
except Exception as e:
    # Library not available yet - will fail on actual function calls
    import warnings
    warnings.warn(f"SCL C++ library not available: {e}. Functions will fail until library is built.")

# =============================================================================
# Python Wrappers
# =============================================================================

def row_sums_csr(
    data,  # ctypes pointer or Array
    indices,  # ctypes pointer or Array  
    indptr,  # ctypes pointer or Array
    row_lengths,  # ctypes pointer or Array or None
    rows: int,
    cols: int,
    output  # ctypes pointer or Array
) -> None:
    """
    Compute row sums for CSR matrix.
    
    Args:
        data: CSR data array pointer
        indices: CSR column indices pointer
        indptr: CSR row pointers pointer
        row_lengths: Explicit row lengths pointer or None
        rows: Number of rows
        cols: Number of columns
        output: Output array pointer - modified in-place
        
    Raises:
        RuntimeError: If C function fails
    """
    lib = get_lib()
    
    # Convert to ctypes pointers if needed
    data_ptr = data if isinstance(data, (int, type(None))) or hasattr(data, 'value') else data
    indices_ptr = indices if isinstance(indices, (int, type(None))) or hasattr(indices, 'value') else indices
    indptr_ptr = indptr if isinstance(indptr, (int, type(None))) or hasattr(indptr, 'value') else indptr
    row_lengths_ptr = row_lengths if isinstance(row_lengths, (int, type(None))) or hasattr(row_lengths, 'value') else row_lengths
    output_ptr = output if isinstance(output, (int, type(None))) or hasattr(output, 'value') else output
    
    status = lib.scl_row_sums_csr(
        data_ptr,
        indices_ptr,
        indptr_ptr,
        row_lengths_ptr,
        rows,
        cols,
        output_ptr
    )
    
    check_error(status, "row_sums_csr")


def col_sums_csc(
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
    Compute column sums for CSC matrix.
    
    Args:
        data: CSC data array (nnz,)
        indices: CSC row indices (nnz,)
        indptr: CSC column pointers (cols+1,)
        col_lengths: Explicit column lengths or None
        rows: Number of rows
        cols: Number of columns
        nnz: Number of non-zeros
        output: Output array (cols,) - modified in-place
        
    Raises:
        RuntimeError: If C function fails
    """
    lib = get_lib()
    
    status = lib.scl_col_sums_csc(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(col_lengths, c_index) if col_lengths is not None else None,
        rows,
        cols,
        nnz,
        as_c_ptr(output, c_real)
    )
    
    check_error(status, "col_sums_csc")


def row_statistics_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    row_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    out_means: np.ndarray,
    out_vars: np.ndarray
) -> None:
    """
    Compute row statistics (mean + variance) for CSR matrix.
    
    Args:
        data: CSR data array (nnz,)
        indices: CSR column indices (nnz,)
        indptr: CSR row pointers (rows+1,)
        row_lengths: Explicit row lengths or None
        rows: Number of rows
        cols: Number of columns
        nnz: Number of non-zeros
        out_means: Output means array (rows,) - modified in-place
        out_vars: Output variances array (rows,) - modified in-place
        
    Raises:
        RuntimeError: If C function fails
    """
    lib = get_lib()
    
    status = lib.scl_row_statistics_csr(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(row_lengths, c_index) if row_lengths is not None else None,
        rows,
        cols,
        nnz,
        as_c_ptr(out_means, c_real),
        as_c_ptr(out_vars, c_real)
    )
    
    check_error(status, "row_statistics_csr")


def col_statistics_csc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    col_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    out_means: np.ndarray,
    out_vars: np.ndarray
) -> None:
    """
    Compute column statistics (mean + variance) for CSC matrix.
    
    Args:
        data: CSC data array (nnz,)
        indices: CSC row indices (nnz,)
        indptr: CSC column pointers (cols+1,)
        col_lengths: Explicit column lengths or None
        rows: Number of rows
        cols: Number of columns
        nnz: Number of non-zeros
        out_means: Output means array (cols,) - modified in-place
        out_vars: Output variances array (cols,) - modified in-place
        
    Raises:
        RuntimeError: If C function fails
    """
    lib = get_lib()
    
    status = lib.scl_col_statistics_csc(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(col_lengths, c_index) if col_lengths is not None else None,
        rows,
        cols,
        nnz,
        as_c_ptr(out_means, c_real),
        as_c_ptr(out_vars, c_real)
    )
    
    check_error(status, "col_statistics_csc")

