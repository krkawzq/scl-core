"""Normalization operation kernels.

Low-level C bindings for normalization operations.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = ['scale_primary_csr', 'scale_primary_csc']


def scale_primary_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    scales: Any
) -> None:
    """Scale each row by a factor (CSR matrix, in-place).
    
    Multiplies each element in row i by scales[i].
    
    Args:
        data: CSR data array pointer (modified in-place).
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        scales: Scale factors per row pointer [rows].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_scale_primary_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p
    ]
    lib.scl_scale_primary_csr.restype = ctypes.c_int
    
    status = lib.scl_scale_primary_csr(data, indices, indptr, rows, cols, scales)
    check_error(status, "scale_primary_csr")


def scale_primary_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    scales: Any
) -> None:
    """Scale each column by a factor (CSC matrix, in-place).
    
    Multiplies each element in column j by scales[j].
    
    Args:
        data: CSC data array pointer (modified in-place).
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        scales: Scale factors per column pointer [cols].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_scale_primary_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p
    ]
    lib.scl_scale_primary_csc.restype = ctypes.c_int
    
    status = lib.scl_scale_primary_csc(data, indices, indptr, rows, cols, scales)
    check_error(status, "scale_primary_csc")
