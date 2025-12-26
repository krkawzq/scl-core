"""Data transformation kernels.

Low-level C bindings for log transforms, softmax, etc.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, c_size, check_error


__all__ = [
    'log1p_inplace_array',
    'log1p_inplace_csr',
    'log2p1_inplace_array',
    'expm1_inplace_array',
    'softmax_inplace_csr',
]


def log1p_inplace_array(data: Any, size: int) -> None:
    """Apply ln(1 + x) transformation to array in-place.
    
    Args:
        data: Data array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_log1p_inplace_array.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_log1p_inplace_array.restype = ctypes.c_int
    
    status = lib.scl_log1p_inplace_array(data, size)
    check_error(status, "log1p_inplace_array")


def log1p_inplace_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int
) -> None:
    """Apply ln(1 + x) transformation to CSR matrix in-place.
    
    Args:
        data: CSR data array pointer (modified in-place).
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_log1p_inplace_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index
    ]
    lib.scl_log1p_inplace_csr.restype = ctypes.c_int
    
    status = lib.scl_log1p_inplace_csr(data, indices, indptr, rows, cols)
    check_error(status, "log1p_inplace_csr")


def log2p1_inplace_array(data: Any, size: int) -> None:
    """Apply log2(1 + x) transformation to array in-place.
    
    Args:
        data: Data array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_log2p1_inplace_array.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_log2p1_inplace_array.restype = ctypes.c_int
    
    status = lib.scl_log2p1_inplace_array(data, size)
    check_error(status, "log2p1_inplace_array")


def expm1_inplace_array(data: Any, size: int) -> None:
    """Apply exp(x) - 1 transformation to array in-place.
    
    Args:
        data: Data array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_expm1_inplace_array.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_expm1_inplace_array.restype = ctypes.c_int
    
    status = lib.scl_expm1_inplace_array(data, size)
    check_error(status, "expm1_inplace_array")


def softmax_inplace_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int
) -> None:
    """Apply softmax to rows of CSR matrix in-place.
    
    Args:
        data: CSR data array pointer (modified in-place).
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_softmax_inplace_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index
    ]
    lib.scl_softmax_inplace_csr.restype = ctypes.c_int
    
    status = lib.scl_softmax_inplace_csr(data, indices, indptr, rows, cols)
    check_error(status, "softmax_inplace_csr")
