"""Resampling kernels.

Low-level C bindings for count resampling operations.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = ['downsample_counts_csc']


def downsample_counts_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    target_sum: float,
    seed: int
) -> None:
    """Downsample counts to target sum for CSC matrix.
    
    Randomly subsamples counts so that each column sums to at most
    target_sum. Used for normalization by downsampling in scRNA-seq.
    
    Args:
        data: CSC data array pointer (modified in-place).
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows (cells).
        cols: Number of columns (genes).
        target_sum: Target sum per cell.
        seed: Random seed for reproducibility.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_downsample_counts_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_real, ctypes.c_uint64
    ]
    lib.scl_downsample_counts_csc.restype = ctypes.c_int
    
    status = lib.scl_downsample_counts_csc(
        data, indices, indptr, rows, cols, target_sum, seed
    )
    check_error(status, "downsample_counts_csc")
