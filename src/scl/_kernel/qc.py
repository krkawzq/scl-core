"""Quality control kernels.

Low-level C bindings for quality control metrics computation.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = [
    'compute_basic_qc_csr',
    'compute_basic_qc_csc',
]


def compute_basic_qc_csr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    out_n_genes: Any,
    out_total_counts: Any
) -> None:
    """Compute basic QC metrics for CSR matrix (cells x genes).
    
    Computes per-cell metrics:
    - n_genes: Number of expressed genes per cell
    - total_counts: Total UMI counts per cell
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows (cells).
        cols: Number of columns (genes).
        out_n_genes: Output array for gene counts per cell [rows].
        out_total_counts: Output array for total counts per cell [rows].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_compute_basic_qc_csr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_compute_basic_qc_csr.restype = ctypes.c_int
    
    status = lib.scl_compute_basic_qc_csr(
        data, indices, indptr, rows, cols, out_n_genes, out_total_counts
    )
    check_error(status, "compute_basic_qc_csr")


def compute_basic_qc_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    out_n_cells: Any,
    out_total_counts: Any
) -> None:
    """Compute basic QC metrics for CSC matrix (cells x genes).
    
    Computes per-gene metrics:
    - n_cells: Number of cells expressing each gene
    - total_counts: Total UMI counts per gene
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows (cells).
        cols: Number of columns (genes).
        out_n_cells: Output array for cell counts per gene [cols].
        out_total_counts: Output array for total counts per gene [cols].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_compute_basic_qc_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_compute_basic_qc_csc.restype = ctypes.c_int
    
    status = lib.scl_compute_basic_qc_csc(
        data, indices, indptr, rows, cols, out_n_cells, out_total_counts
    )
    check_error(status, "compute_basic_qc_csc")
