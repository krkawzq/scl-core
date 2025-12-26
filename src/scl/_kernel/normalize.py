"""Normalization operation kernels.

Low-level C bindings for normalization operations.
"""

import ctypes
from typing import Any, Optional

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = ['scale_primary_csr', 'scale_primary_csc']


def scale_primary_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    row_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    scales: Any
) -> None:
    """Scale each row by a factor (CSR matrix, in-place).
    
    Args:
        data: CSR data array pointer (modified in-place).
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        row_lengths: Explicit row lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zeros.
        scales: Scale factors per row pointer.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_scale_primary_csr.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index, ctypes.POINTER(c_real)
    ]
    lib.scl_scale_primary_csr.restype = ctypes.c_int
    
    status = lib.scl_scale_primary_csr(data, indices, indptr, row_lengths, rows, cols, nnz, scales)
    check_error(status, "scale_primary_csr")


def scale_primary_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    col_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    scales: Any
) -> None:
    """Scale each column by a factor (CSC matrix, in-place).
    
    Args:
        data: CSC data array pointer (modified in-place).
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        col_lengths: Explicit column lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zeros.
        scales: Scale factors per column pointer.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_scale_primary_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index, ctypes.POINTER(c_real)
    ]
    lib.scl_scale_primary_csc.restype = ctypes.c_int
    
    status = lib.scl_scale_primary_csc(data, indices, indptr, col_lengths, rows, cols, nnz, scales)
    check_error(status, "scale_primary_csc")
