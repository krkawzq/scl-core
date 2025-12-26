"""Reordering kernels."""
import ctypes
from typing import Any, Optional
from .lib_loader import get_lib
from .types import c_real, c_index, check_error

__all__ = ['align_secondary_csc']

def align_secondary_csc(data: Any, indices: Any, indptr: Any, col_lengths: Optional[Any],
                        rows: int, cols: int, nnz: int, index_map: Any, out_lengths: Any, new_cols: int) -> None:
    """Align secondary dimension indices for CSC matrix."""
    lib = get_lib()
    lib.scl_align_secondary_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, c_index, c_index, ctypes.POINTER(c_index), ctypes.POINTER(c_index), c_index
    ]
    lib.scl_align_secondary_csc.restype = ctypes.c_int
    status = lib.scl_align_secondary_csc(data, indices, indptr, col_lengths, rows, cols, nnz, index_map, out_lengths, new_cols)
    check_error(status, "align_secondary_csc")
