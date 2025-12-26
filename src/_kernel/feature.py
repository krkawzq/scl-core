"""
Feature Selection Kernels

Low-level C bindings for HVG selection and feature statistics.
"""

import ctypes
import numpy as np
from .lib_loader import get_lib
from .types import c_real, c_index, c_size, check_error, as_c_ptr

__all__ = [
    'standard_moments_csc',
    'dispersion',
    'hvg_by_dispersion_csc',
    'hvg_by_variance_csc',
]

# =============================================================================
# Function Signatures
# =============================================================================

def _init_signatures():
    """Initialize C function signatures."""
    lib = get_lib()
    
    # standard_moments_csc
    lib.scl_standard_moments_csc.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # col_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        ctypes.POINTER(c_real),   # out_means
        ctypes.POINTER(c_real),   # out_vars
        ctypes.c_int,             # ddof
    ]
    lib.scl_standard_moments_csc.restype = ctypes.c_int
    
    # dispersion
    lib.scl_dispersion.argtypes = [
        ctypes.POINTER(c_real),   # means
        ctypes.POINTER(c_real),   # vars
        c_size,                    # size
        ctypes.POINTER(c_real),   # out_dispersion
    ]
    lib.scl_dispersion.restype = ctypes.c_int
    
    # hvg_by_dispersion_csc
    lib.scl_hvg_by_dispersion_csc.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # col_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        c_size,                    # n_top
        ctypes.POINTER(c_index),  # out_indices
        ctypes.POINTER(ctypes.c_uint8),  # out_mask
        ctypes.POINTER(c_real),   # out_dispersions
    ]
    lib.scl_hvg_by_dispersion_csc.restype = ctypes.c_int
    
    # hvg_by_variance_csc
    lib.scl_hvg_by_variance_csc.argtypes = [
        ctypes.POINTER(c_real),   # data
        ctypes.POINTER(c_index),  # indices
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # col_lengths
        c_index,                   # rows
        c_index,                   # cols
        c_index,                   # nnz
        c_size,                    # n_top
        ctypes.POINTER(c_index),  # out_indices
        ctypes.POINTER(ctypes.c_uint8),  # out_mask
    ]
    lib.scl_hvg_by_variance_csc.restype = ctypes.c_int


_init_signatures()

# =============================================================================
# Python Wrappers
# =============================================================================

def standard_moments_csc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    col_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    out_means: np.ndarray,
    out_vars: np.ndarray,
    ddof: int = 1
) -> None:
    """
    Compute mean and variance for each gene (CSC matrix).
    
    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        col_lengths: Explicit column lengths or None
        rows: Number of cells
        cols: Number of genes
        nnz: Number of non-zeros
        out_means: Output means, shape (cols,)
        out_vars: Output variances, shape (cols,)
        ddof: Delta degrees of freedom (0=population, 1=sample)
    """
    lib = get_lib()
    
    status = lib.scl_standard_moments_csc(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(col_lengths, c_index) if col_lengths is not None else None,
        rows, cols, nnz,
        as_c_ptr(out_means, c_real),
        as_c_ptr(out_vars, c_real),
        ddof
    )
    
    check_error(status, "standard_moments_csc")


def dispersion(
    means: np.ndarray,
    vars: np.ndarray,
    out_dispersion: np.ndarray
) -> None:
    """
    Compute dispersion (variance / mean).
    
    Args:
        means: Mean values
        vars: Variance values
        out_dispersion: Output dispersion values (same shape as means)
    """
    lib = get_lib()
    
    status = lib.scl_dispersion(
        as_c_ptr(means, c_real),
        as_c_ptr(vars, c_real),
        means.size,
        as_c_ptr(out_dispersion, c_real)
    )
    
    check_error(status, "dispersion")


def hvg_by_dispersion_csc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    col_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    n_top: int,
    out_indices: np.ndarray,
    out_mask: np.ndarray,
    out_dispersions: np.ndarray
) -> None:
    """
    Select highly variable genes by dispersion (CSC matrix).
    
    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        col_lengths: Explicit column lengths or None
        rows: Number of cells
        cols: Number of genes
        nnz: Number of non-zeros
        n_top: Number of top genes to select
        out_indices: Output indices of selected genes, shape (n_top,)
        out_mask: Output binary mask, shape (cols,)
        out_dispersions: Output dispersion values, shape (cols,)
    """
    lib = get_lib()
    
    status = lib.scl_hvg_by_dispersion_csc(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(col_lengths, c_index) if col_lengths is not None else None,
        rows, cols, nnz,
        n_top,
        as_c_ptr(out_indices, c_index),
        as_c_ptr(out_mask, ctypes.c_uint8),
        as_c_ptr(out_dispersions, c_real)
    )
    
    check_error(status, "hvg_by_dispersion_csc")


def hvg_by_variance_csc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    col_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    n_top: int,
    out_indices: np.ndarray,
    out_mask: np.ndarray
) -> None:
    """
    Select highly variable genes by variance (CSC matrix).
    
    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        col_lengths: Explicit column lengths or None
        rows: Number of cells
        cols: Number of genes
        nnz: Number of non-zeros
        n_top: Number of top genes to select
        out_indices: Output indices of selected genes, shape (n_top,)
        out_mask: Output binary mask, shape (cols,)
    """
    lib = get_lib()
    
    status = lib.scl_hvg_by_variance_csc(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(col_lengths, c_index) if col_lengths is not None else None,
        rows, cols, nnz,
        n_top,
        as_c_ptr(out_indices, c_index),
        as_c_ptr(out_mask, ctypes.c_uint8)
    )
    
    check_error(status, "hvg_by_variance_csc")

