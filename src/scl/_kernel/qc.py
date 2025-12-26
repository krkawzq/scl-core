"""Quality control metrics kernels.

Low-level C bindings for QC metric computation.
"""

import ctypes
from typing import Any, Optional

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = ['compute_basic_qc_csr', 'compute_basic_qc_csc']


def compute_basic_qc_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    row_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    out_n_genes: Any,
    out_total_counts: Any
) -> None:
    """Compute basic QC metrics for cells (CSR matrix).
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        row_lengths: Explicit row lengths pointer or None.
        rows: Number of cells.
        cols: Number of genes.
        nnz: Number of non-zeros.
        out_n_genes: Output n_genes per cell pointer.
        out_total_counts: Output total counts per cell pointer.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_compute_basic_qc_csr.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index,
        ctypes.POINTER(c_index), ctypes.POINTER(c_real)
    ]
    lib.scl_compute_basic_qc_csr.restype = ctypes.c_int
    
    status = lib.scl_compute_basic_qc_csr(
        data, indices, indptr, row_lengths, rows, cols, nnz,
        out_n_genes, out_total_counts
    )
    check_error(status, "compute_basic_qc_csr")


def compute_basic_qc_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    col_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    out_n_cells: Any,
    out_total_counts: Any
) -> None:
    """Compute basic QC metrics for genes (CSC matrix).
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        col_lengths: Explicit column lengths pointer or None.
        rows: Number of cells.
        cols: Number of genes.
        nnz: Number of non-zeros.
        out_n_cells: Output n_cells per gene pointer.
        out_total_counts: Output total counts per gene pointer.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_compute_basic_qc_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index,
        ctypes.POINTER(c_index), ctypes.POINTER(c_real)
    ]
    lib.scl_compute_basic_qc_csc.restype = ctypes.c_int
    
    status = lib.scl_compute_basic_qc_csc(
        data, indices, indptr, col_lengths, rows, cols, nnz,
        out_n_cells, out_total_counts
    )
    check_error(status, "compute_basic_qc_csc")

