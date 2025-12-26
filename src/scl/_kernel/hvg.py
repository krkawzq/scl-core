"""Highly variable gene selection kernels."""
import ctypes
from typing import Any, Optional
from .lib_loader import get_lib
from .types import c_real, c_index, c_size, check_error

__all__ = ['hvg_by_dispersion_csc', 'hvg_by_variance_csc']

def hvg_by_dispersion_csc(data: Any, indices: Any, indptr: Any, col_lengths: Optional[Any],
                           rows: int, cols: int, nnz: int, n_top: int,
                           out_indices: Any, out_mask: Any, out_dispersions: Any) -> None:
    """Select highly variable genes by dispersion for CSC matrix."""
    lib = get_lib()
    lib.scl_hvg_by_dispersion_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, c_index, c_index, c_size,
        ctypes.POINTER(c_index), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(c_real)
    ]
    lib.scl_hvg_by_dispersion_csc.restype = ctypes.c_int
    status = lib.scl_hvg_by_dispersion_csc(data, indices, indptr, col_lengths, rows, cols, nnz, n_top,
                                            out_indices, out_mask, out_dispersions)
    check_error(status, "hvg_by_dispersion_csc")

def hvg_by_variance_csc(data: Any, indices: Any, indptr: Any, col_lengths: Optional[Any],
                        rows: int, cols: int, nnz: int, n_top: int,
                        out_indices: Any, out_mask: Any) -> None:
    """Select highly variable genes by variance for CSC matrix."""
    lib = get_lib()
    lib.scl_hvg_by_variance_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, c_index, c_index, c_size, ctypes.POINTER(c_index), ctypes.POINTER(ctypes.c_uint8)
    ]
    lib.scl_hvg_by_variance_csc.restype = ctypes.c_int
    status = lib.scl_hvg_by_variance_csc(data, indices, indptr, col_lengths, rows, cols, nnz, n_top, out_indices, out_mask)
    check_error(status, "hvg_by_variance_csc")
