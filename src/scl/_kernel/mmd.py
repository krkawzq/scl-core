"""Maximum Mean Discrepancy kernels.

Low-level C bindings for MMD computation.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = ['mmd_rbf_csc']


def mmd_rbf_csc(
    data_x: Any,
    indices_x: Any,
    indptr_x: Any,
    rows_x: int,
    cols: int,
    data_y: Any,
    indices_y: Any,
    indptr_y: Any,
    rows_y: int,
    output: Any,
    gamma: float
) -> None:
    """Compute Maximum Mean Discrepancy with RBF kernel for CSC matrices.
    
    MMD measures the distance between two distributions represented by
    sample matrices X and Y using a Gaussian (RBF) kernel.
    
    Args:
        data_x: CSC data array pointer for X.
        indices_x: CSC row indices pointer for X.
        indptr_x: CSC column pointers pointer for X.
        rows_x: Number of rows in X.
        cols: Number of columns (shared by X and Y).
        data_y: CSC data array pointer for Y.
        indices_y: CSC row indices pointer for Y.
        indptr_y: CSC column pointers pointer for Y.
        rows_y: Number of rows in Y.
        output: Output MMD value pointer (single value).
        gamma: RBF kernel parameter (1 / (2 * sigma^2)).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_mmd_rbf_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, ctypes.c_void_p, c_real
    ]
    lib.scl_mmd_rbf_csc.restype = ctypes.c_int
    
    status = lib.scl_mmd_rbf_csc(
        data_x, indices_x, indptr_x, rows_x, cols,
        data_y, indices_y, indptr_y, rows_y,
        output, gamma
    )
    check_error(status, "mmd_rbf_csc")
