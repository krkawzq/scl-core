"""Standardization kernels.

Low-level C bindings for data standardization operations.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = [
    'standardize_csc',
]


def standardize_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    means: Any,
    stds: Any,
    max_value: float,
    zero_center: bool
) -> None:
    """Standardize CSC matrix (in-place).
    
    Transforms data using: (x - mean) / std, optionally clipped to max_value.
    If zero_center is False, only scales by std without centering.
    
    Args:
        data: CSC data array pointer (modified in-place).
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        means: Pre-computed means per column [cols].
        stds: Pre-computed standard deviations per column [cols].
        max_value: Maximum value to clip to (use large value to disable).
        zero_center: Whether to center by subtracting mean.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_standardize_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p, ctypes.c_void_p,
        c_real, ctypes.c_bool
    ]
    lib.scl_standardize_csc.restype = ctypes.c_int
    
    status = lib.scl_standardize_csc(
        data, indices, indptr, rows, cols, means, stds, max_value, zero_center
    )
    check_error(status, "standardize_csc")
