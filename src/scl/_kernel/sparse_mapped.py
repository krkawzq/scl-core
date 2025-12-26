"""Mapped sparse matrix statistics kernels.

Low-level C bindings for sparse matrix aggregation operations on memory-mapped data.
These functions are designed for out-of-core computation where the matrix is stored
on disk and accessed via memory mapping.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = [
    'primary_sums_mapped_csr',
    'primary_sums_mapped_csc',
    'primary_means_mapped_csr',
    'primary_means_mapped_csc',
    'primary_variances_mapped_csr',
    'primary_variances_mapped_csc',
    'primary_nnz_counts_mapped_csr',
    'primary_nnz_counts_mapped_csc',
]


def primary_sums_mapped_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    nnz: int,
    output: Any
) -> None:
    """Compute row sums for mapped CSR matrix.
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_sums_mapped_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_index, ctypes.c_void_p
    ]
    lib.scl_primary_sums_mapped_csr.restype = ctypes.c_int
    
    status = lib.scl_primary_sums_mapped_csr(data, indices, indptr, rows, cols, nnz, output)
    check_error(status, "primary_sums_mapped_csr")


def primary_sums_mapped_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    nnz: int,
    output: Any
) -> None:
    """Compute column sums for mapped CSC matrix.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_sums_mapped_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_index, ctypes.c_void_p
    ]
    lib.scl_primary_sums_mapped_csc.restype = ctypes.c_int
    
    status = lib.scl_primary_sums_mapped_csc(data, indices, indptr, rows, cols, nnz, output)
    check_error(status, "primary_sums_mapped_csc")


def primary_means_mapped_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    nnz: int,
    output: Any
) -> None:
    """Compute row means for mapped CSR matrix.
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_means_mapped_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_index, ctypes.c_void_p
    ]
    lib.scl_primary_means_mapped_csr.restype = ctypes.c_int
    
    status = lib.scl_primary_means_mapped_csr(data, indices, indptr, rows, cols, nnz, output)
    check_error(status, "primary_means_mapped_csr")


def primary_means_mapped_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    nnz: int,
    output: Any
) -> None:
    """Compute column means for mapped CSC matrix.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_means_mapped_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_index, ctypes.c_void_p
    ]
    lib.scl_primary_means_mapped_csc.restype = ctypes.c_int
    
    status = lib.scl_primary_means_mapped_csc(data, indices, indptr, rows, cols, nnz, output)
    check_error(status, "primary_means_mapped_csc")


def primary_variances_mapped_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    nnz: int,
    ddof: int,
    output: Any
) -> None:
    """Compute row variances for mapped CSR matrix.
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        ddof: Delta degrees of freedom (typically 0 or 1).
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_variances_mapped_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_index, ctypes.c_int, ctypes.c_void_p
    ]
    lib.scl_primary_variances_mapped_csr.restype = ctypes.c_int
    
    status = lib.scl_primary_variances_mapped_csr(data, indices, indptr, rows, cols, nnz, ddof, output)
    check_error(status, "primary_variances_mapped_csr")


def primary_variances_mapped_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    nnz: int,
    ddof: int,
    output: Any
) -> None:
    """Compute column variances for mapped CSC matrix.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        ddof: Delta degrees of freedom (typically 0 or 1).
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_variances_mapped_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_index, ctypes.c_int, ctypes.c_void_p
    ]
    lib.scl_primary_variances_mapped_csc.restype = ctypes.c_int
    
    status = lib.scl_primary_variances_mapped_csc(data, indices, indptr, rows, cols, nnz, ddof, output)
    check_error(status, "primary_variances_mapped_csc")


def primary_nnz_counts_mapped_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    nnz: int,
    output: Any
) -> None:
    """Count non-zero elements per row for mapped CSR matrix.
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_nnz_counts_mapped_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_index, ctypes.c_void_p
    ]
    lib.scl_primary_nnz_counts_mapped_csr.restype = ctypes.c_int
    
    status = lib.scl_primary_nnz_counts_mapped_csr(data, indices, indptr, rows, cols, nnz, output)
    check_error(status, "primary_nnz_counts_mapped_csr")


def primary_nnz_counts_mapped_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    nnz: int,
    output: Any
) -> None:
    """Count non-zero elements per column for mapped CSC matrix.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zero elements.
        output: Output array pointer (modified in-place).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_primary_nnz_counts_mapped_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_index, ctypes.c_void_p
    ]
    lib.scl_primary_nnz_counts_mapped_csc.restype = ctypes.c_int
    
    status = lib.scl_primary_nnz_counts_mapped_csc(data, indices, indptr, rows, cols, nnz, output)
    check_error(status, "primary_nnz_counts_mapped_csc")

