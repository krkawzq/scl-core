"""Sparse matrix statistics kernels.

Low-level C bindings for sparse matrix aggregation operations.
All functions work with raw pointers - numpy arrays are converted by caller.
"""

import ctypes
from typing import Any, Optional

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = [
    'primary_sums_csr',
    'primary_sums_csc',
    'primary_means_csr',
    'primary_means_csc',
    'primary_variances_csr',
    'primary_variances_csc',
    'primary_nnz_counts_csr',
    'primary_nnz_counts_csc',
]


def primary_sums_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    row_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    output: Any
) -> None:
    """Compute row sums for CSR matrix.
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        row_lengths: Explicit row lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_sums_csr.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index, ctypes.POINTER(c_real)
    ]
    lib.scl_primary_sums_csr.restype = ctypes.c_int
    
    status = lib.scl_primary_sums_csr(data, indices, indptr, row_lengths, rows, cols, nnz, output)
    check_error(status, "primary_sums_csr")


def primary_sums_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    col_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    output: Any
) -> None:
    """Compute column sums for CSC matrix.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        col_lengths: Explicit column lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_sums_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index, ctypes.POINTER(c_real)
    ]
    lib.scl_primary_sums_csc.restype = ctypes.c_int
    
    status = lib.scl_primary_sums_csc(data, indices, indptr, col_lengths, rows, cols, nnz, output)
    check_error(status, "primary_sums_csc")


def primary_means_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    row_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    output: Any
) -> None:
    """Compute row means for CSR matrix.
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        row_lengths: Explicit row lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_means_csr.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index, ctypes.POINTER(c_real)
    ]
    lib.scl_primary_means_csr.restype = ctypes.c_int
    
    status = lib.scl_primary_means_csr(data, indices, indptr, row_lengths, rows, cols, nnz, output)
    check_error(status, "primary_means_csr")


def primary_means_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    col_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    output: Any
) -> None:
    """Compute column means for CSC matrix.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        col_lengths: Explicit column lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_means_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index, ctypes.POINTER(c_real)
    ]
    lib.scl_primary_means_csc.restype = ctypes.c_int
    
    status = lib.scl_primary_means_csc(data, indices, indptr, col_lengths, rows, cols, nnz, output)
    check_error(status, "primary_means_csc")


def primary_variances_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    row_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    ddof: int,
    output: Any
) -> None:
    """Compute row variances for CSR matrix.
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        row_lengths: Explicit row lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        ddof: Delta degrees of freedom (typically 0 or 1).
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_variances_csr.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index, ctypes.c_int, ctypes.POINTER(c_real)
    ]
    lib.scl_primary_variances_csr.restype = ctypes.c_int
    
    status = lib.scl_primary_variances_csr(data, indices, indptr, row_lengths, rows, cols, nnz, ddof, output)
    check_error(status, "primary_variances_csr")


def primary_variances_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    col_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    ddof: int,
    output: Any
) -> None:
    """Compute column variances for CSC matrix.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        col_lengths: Explicit column lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        ddof: Delta degrees of freedom (typically 0 or 1).
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_variances_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index, ctypes.c_int, ctypes.POINTER(c_real)
    ]
    lib.scl_primary_variances_csc.restype = ctypes.c_int
    
    status = lib.scl_primary_variances_csc(data, indices, indptr, col_lengths, rows, cols, nnz, ddof, output)
    check_error(status, "primary_variances_csc")


def primary_nnz_counts_csr(
    indptr: Any,
    rows: int,
    output: Any
) -> None:
    """Count non-zero elements per row (CSR).
    
    Args:
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        output: Output counts pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_nnz_counts_csr.argtypes = [
        ctypes.POINTER(c_index), c_index, ctypes.POINTER(c_index)
    ]
    lib.scl_primary_nnz_counts_csr.restype = ctypes.c_int
    
    status = lib.scl_primary_nnz_counts_csr(indptr, rows, output)
    check_error(status, "primary_nnz_counts_csr")


def primary_nnz_counts_csc(
    indptr: Any,
    cols: int,
    output: Any
) -> None:
    """Count non-zero elements per column (CSC).
    
    Args:
        indptr: CSC column pointers pointer.
        cols: Number of columns.
        output: Output counts pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_nnz_counts_csc.argtypes = [
        ctypes.POINTER(c_index), c_index, ctypes.POINTER(c_index)
    ]
    lib.scl_primary_nnz_counts_csc.restype = ctypes.c_int
    
    status = lib.scl_primary_nnz_counts_csc(indptr, cols, output)
    check_error(status, "primary_nnz_counts_csc")
