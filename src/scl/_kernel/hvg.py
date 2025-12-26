"""Highly Variable Gene (HVG) selection kernels.

Low-level C bindings for HVG identification algorithms.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, c_size, c_byte, check_error


__all__ = [
    'hvg_by_dispersion_csc',
    'hvg_by_variance_csc',
]


def hvg_by_dispersion_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    cols: int,
    n_top: int,
    out_indices: Any,
    out_mask: Any,
    out_dispersions: Any
) -> None:
    """Select highly variable genes by normalized dispersion.
    
    Uses the Seurat v3 method: genes are binned by mean expression,
    then ranked by normalized dispersion within each bin.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        cols: Number of columns (genes).
        n_top: Number of top HVGs to select.
        out_indices: Output array for top HVG indices [n_top].
        out_mask: Output boolean mask [cols].
        out_dispersions: Output normalized dispersions [cols].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_hvg_by_dispersion_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_size, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_hvg_by_dispersion_csc.restype = ctypes.c_int
    
    status = lib.scl_hvg_by_dispersion_csc(
        data, indices, indptr, cols, n_top, out_indices, out_mask, out_dispersions
    )
    check_error(status, "hvg_by_dispersion_csc")


def hvg_by_variance_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    n_top: int,
    out_indices: Any,
    out_mask: Any
) -> None:
    """Select highly variable genes by raw variance.
    
    Simple variance-based selection: returns genes with highest variance.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows (cells).
        cols: Number of columns (genes).
        n_top: Number of top HVGs to select.
        out_indices: Output array for top HVG indices [n_top].
        out_mask: Output boolean mask [cols].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_hvg_by_variance_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_size, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_hvg_by_variance_csc.restype = ctypes.c_int
    
    status = lib.scl_hvg_by_variance_csc(
        data, indices, indptr, rows, cols, n_top, out_indices, out_mask
    )
    check_error(status, "hvg_by_variance_csc")
