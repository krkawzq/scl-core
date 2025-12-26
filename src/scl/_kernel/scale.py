"""Standardization kernels."""
import ctypes
from typing import Any, Optional
from .lib_loader import get_lib
from .types import c_real, c_index, check_error

__all__ = ['standardize_csc']

def standardize_csc(data: Any, indices: Any, indptr: Any, col_lengths: Optional[Any],
                    rows: int, cols: int, nnz: int, means: Any, stds: Any, max_value: float, zero_center: bool) -> None:
    """Standardize features (z-score normalization) for CSC matrix."""
    lib = get_lib()
    lib.scl_standardize_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, c_index, c_index, ctypes.POINTER(c_real), ctypes.POINTER(c_real), c_real, ctypes.c_bool
    ]
    lib.scl_standardize_csc.restype = ctypes.c_int
    status = lib.scl_standardize_csc(data, indices, indptr, col_lengths, rows, cols, nnz, means, stds, max_value, zero_center)
    check_error(status, "standardize_csc")
