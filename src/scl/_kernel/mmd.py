"""Maximum Mean Discrepancy kernels."""
import ctypes
from typing import Any, Optional
from .lib_loader import get_lib
from .types import c_real, c_index, check_error

__all__ = ['mmd_rbf_csc']

def mmd_rbf_csc(data_x: Any, indices_x: Any, indptr_x: Any, col_lengths_x: Optional[Any], rows_x: int, cols: int, nnz_x: int,
                data_y: Any, indices_y: Any, indptr_y: Any, col_lengths_y: Optional[Any], rows_y: int, nnz_y: int,
                output: Any, gamma: float) -> None:
    """Compute Maximum Mean Discrepancy with RBF kernel for CSC matrices."""
    lib = get_lib()
    lib.scl_mmd_rbf_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, c_index, c_index,
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, c_index, ctypes.POINTER(c_real), c_real
    ]
    lib.scl_mmd_rbf_csc.restype = ctypes.c_int
    status = lib.scl_mmd_rbf_csc(data_x, indices_x, indptr_x, col_lengths_x, rows_x, cols, nnz_x,
                                  data_y, indices_y, indptr_y, col_lengths_y, rows_y, nnz_y, output, gamma)
    check_error(status, "mmd_rbf_csc")
