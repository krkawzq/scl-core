"""Linear algebra kernels.

Low-level C bindings for sparse matrix-vector operations and Gram matrix.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = ['gram_csc', 'gram_csr', 'pearson_csc', 'pearson_csr', 'spmv_csr', 'spmv_trans_csc']


def gram_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    output: Any
) -> None:
    """Compute Gram matrix (X^T X) for CSC matrix.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        output: Output Gram matrix pointer [cols * cols] (row-major).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_gram_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p
    ]
    lib.scl_gram_csc.restype = ctypes.c_int
    
    status = lib.scl_gram_csc(data, indices, indptr, rows, cols, output)
    check_error(status, "gram_csc")


def gram_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    output: Any
) -> None:
    """Compute Gram matrix (X X^T) for CSR matrix.
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        output: Output Gram matrix pointer [rows * rows] (row-major).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_gram_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p
    ]
    lib.scl_gram_csr.restype = ctypes.c_int
    
    status = lib.scl_gram_csr(data, indices, indptr, rows, cols, output)
    check_error(status, "gram_csr")


def pearson_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    output: Any,
    workspace_means: Any,
    workspace_inv_stds: Any
) -> None:
    """Compute Pearson correlation matrix for CSC matrix.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        output: Output correlation matrix pointer [cols * cols] (row-major).
        workspace_means: Workspace for means pointer [cols].
        workspace_inv_stds: Workspace for inverse stds pointer [cols].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_pearson_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_pearson_csc.restype = ctypes.c_int
    
    status = lib.scl_pearson_csc(
        data, indices, indptr, rows, cols,
        output, workspace_means, workspace_inv_stds
    )
    check_error(status, "pearson_csc")


def pearson_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    output: Any,
    workspace_means: Any,
    workspace_inv_stds: Any
) -> None:
    """Compute Pearson correlation matrix for CSR matrix.
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        output: Output correlation matrix pointer [rows * rows] (row-major).
        workspace_means: Workspace for means pointer [rows].
        workspace_inv_stds: Workspace for inverse stds pointer [rows].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_pearson_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_pearson_csr.restype = ctypes.c_int
    
    status = lib.scl_pearson_csr(
        data, indices, indptr, rows, cols,
        output, workspace_means, workspace_inv_stds
    )
    check_error(status, "pearson_csr")


def spmv_csr(
    A_data: Any,
    A_indices: Any,
    A_indptr: Any,
    A_rows: int,
    A_cols: int,
    x: Any,
    y: Any,
    alpha: float,
    beta: float
) -> None:
    """Sparse matrix-vector multiplication (y = alpha * A * x + beta * y) for CSR.
    
    Args:
        A_data: CSR data array pointer.
        A_indices: CSR column indices pointer.
        A_indptr: CSR row pointers pointer.
        A_rows: Number of rows in A.
        A_cols: Number of columns in A.
        x: Input vector pointer [A_cols].
        y: Output vector pointer [A_rows] (modified in-place).
        alpha: Scaling factor for A*x.
        beta: Scaling factor for y.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_spmv_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index,
        ctypes.c_void_p, ctypes.c_void_p, c_real, c_real
    ]
    lib.scl_spmv_csr.restype = ctypes.c_int
    
    status = lib.scl_spmv_csr(A_data, A_indices, A_indptr, A_rows, A_cols, x, y, alpha, beta)
    check_error(status, "spmv_csr")


def spmv_trans_csc(
    A_data: Any,
    A_indices: Any,
    A_indptr: Any,
    A_rows: int,
    A_cols: int,
    x: Any,
    y: Any,
    alpha: float,
    beta: float
) -> None:
    """Sparse matrix-vector multiplication with transpose (y = alpha * A^T * x + beta * y) for CSC.
    
    Args:
        A_data: CSC data array pointer.
        A_indices: CSC row indices pointer.
        A_indptr: CSC column pointers pointer.
        A_rows: Number of rows in A.
        A_cols: Number of columns in A.
        x: Input vector pointer [A_rows].
        y: Output vector pointer [A_cols] (modified in-place).
        alpha: Scaling factor for A^T*x.
        beta: Scaling factor for y.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_spmv_trans_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index,
        ctypes.c_void_p, ctypes.c_void_p, c_real, c_real
    ]
    lib.scl_spmv_trans_csc.restype = ctypes.c_int
    
    status = lib.scl_spmv_trans_csc(A_data, A_indices, A_indptr, A_rows, A_cols, x, y, alpha, beta)
    check_error(status, "spmv_trans_csc")
