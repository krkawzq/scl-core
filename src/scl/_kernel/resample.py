"""Resampling kernels."""
import ctypes
from typing import Any, Optional
from .lib_loader import get_lib
from .types import c_real, c_index, check_error

__all__ = ['downsample_counts_csc']

def downsample_counts_csc(data: Any, indices: Any, indptr: Any, col_lengths: Optional[Any],
                          rows: int, cols: int, nnz: int, target_sum: float, seed: int) -> None:
    """Downsample counts to target sum for CSC matrix."""
    lib = get_lib()
    lib.scl_downsample_counts_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, c_index, c_index, c_real, ctypes.c_uint64
    ]
    lib.scl_downsample_counts_csc.restype = ctypes.c_int
    status = lib.scl_downsample_counts_csc(data, indices, indptr, col_lengths, rows, cols, nnz, target_sum, seed)
    check_error(status, "downsample_counts_csc")
