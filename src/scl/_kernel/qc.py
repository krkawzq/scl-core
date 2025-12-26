"""
Quality Control Kernels

Low-level C bindings for QC metric computation.
"""

import ctypes
import numpy as np
from .lib_loader import get_lib
from .types import c_real, c_index, np_real, np_index, check_error, as_c_ptr

__all__ = [
    'compute_basic_qc_csr',
    'compute_basic_gene_qc_csc',
]

# =============================================================================
# Function Signatures
# =============================================================================

def _init_signatures():
    """Initialize C function signatures."""
    lib = get_lib()
    
    # compute_basic_qc_csr
    lib.scl_compute_basic_qc_csr.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # row_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_index),  # out_n_genes
        ctypes.POINTER(c_real),   # out_total_counts
    ]
    lib.scl_compute_basic_qc_csr.restype = ctypes.c_int
    
    # compute_basic_gene_qc_csc
    lib.scl_compute_basic_gene_qc_csc.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # col_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_index),  # out_n_cells
        ctypes.POINTER(c_real),   # out_total_counts
    ]
    lib.scl_compute_basic_gene_qc_csc.restype = ctypes.c_int


# Initialize signatures lazily
try:
    _init_signatures()
except Exception as e:
    import warnings
    warnings.warn(f"SCL library not ready: {e}")

# =============================================================================
# Python Wrappers
# =============================================================================

def compute_basic_qc_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    row_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    out_n_genes: np.ndarray,
    out_total_counts: np.ndarray
) -> None:
    """
    Compute basic QC metrics for cells (CSR matrix).
    
    Args:
        data: CSR data array
        indices: CSR column indices
        indptr: CSR row pointers
        row_lengths: Explicit row lengths or None
        rows: Number of cells
        cols: Number of genes
        nnz: Number of non-zeros
        out_n_genes: Output n_genes per cell, shape (rows,)
        out_total_counts: Output total counts per cell, shape (rows,)
    """
    lib = get_lib()
    
    status = lib.scl_compute_basic_qc_csr(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(row_lengths, c_index) if row_lengths is not None else None,
        rows, cols, nnz,
        as_c_ptr(out_n_genes, c_index),
        as_c_ptr(out_total_counts, c_real)
    )
    
    check_error(status, "compute_basic_qc_csr")


def compute_basic_gene_qc_csc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    col_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    out_n_cells: np.ndarray,
    out_total_counts: np.ndarray
) -> None:
    """
    Compute basic QC metrics for genes (CSC matrix).
    
    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        col_lengths: Explicit column lengths or None
        rows: Number of cells
        cols: Number of genes
        nnz: Number of non-zeros
        out_n_cells: Output n_cells per gene, shape (cols,)
        out_total_counts: Output total counts per gene, shape (cols,)
    """
    lib = get_lib()
    
    status = lib.scl_compute_basic_gene_qc_csc(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(col_lengths, c_index) if col_lengths is not None else None,
        rows, cols, nnz,
        as_c_ptr(out_n_cells, c_index),
        as_c_ptr(out_total_counts, c_real)
    )
    
    check_error(status, "compute_basic_gene_qc_csc")

