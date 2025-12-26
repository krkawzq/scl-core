"""Feature statistics kernels.

Low-level C bindings for feature statistics computation.
"""

import ctypes
from typing import Any, Optional

from .lib_loader import get_lib
from .types import c_real, c_index, c_size, check_error


__all__ = [
    'standard_moments_csc',
    'clipped_moments_csc',
    'detection_rate_csc',
    'dispersion',
]


def standard_moments_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    col_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    out_means: Any,
    out_vars: Any,
    ddof: int
) -> None:
    """Compute standard moments (mean and variance) for features (CSC).
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        col_lengths: Explicit column lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zeros.
        out_means: Output means pointer.
        out_vars: Output variances pointer.
        ddof: Delta degrees of freedom.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_standard_moments_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index,
        ctypes.POINTER(c_real), ctypes.POINTER(c_real), ctypes.c_int
    ]
    lib.scl_standard_moments_csc.restype = ctypes.c_int
    
    status = lib.scl_standard_moments_csc(
        data, indices, indptr, col_lengths, rows, cols, nnz,
        out_means, out_vars, ddof
    )
    check_error(status, "standard_moments_csc")


def clipped_moments_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    col_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    clip_vals: Any,
    out_means: Any,
    out_vars: Any
) -> None:
    """Compute clipped moments (mean and variance with clipping) for features (CSC).
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        col_lengths: Explicit column lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zeros.
        clip_vals: Clipping values pointer.
        out_means: Output means pointer.
        out_vars: Output variances pointer.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_clipped_moments_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index,
        ctypes.POINTER(c_real), ctypes.POINTER(c_real), ctypes.POINTER(c_real)
    ]
    lib.scl_clipped_moments_csc.restype = ctypes.c_int
    
    status = lib.scl_clipped_moments_csc(
        data, indices, indptr, col_lengths, rows, cols, nnz,
        clip_vals, out_means, out_vars
    )
    check_error(status, "clipped_moments_csc")


def detection_rate_csc(
    indptr: Any,
    rows: int,
    cols: int,
    out_rates: Any
) -> None:
    """Compute detection rate (fraction of non-zero cells) for features (CSC).
    
    Args:
        indptr: CSC column pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        out_rates: Output detection rates pointer.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_detection_rate_csc.argtypes = [
        ctypes.POINTER(c_index), c_index, c_index, ctypes.POINTER(c_real)
    ]
    lib.scl_detection_rate_csc.restype = ctypes.c_int
    
    status = lib.scl_detection_rate_csc(indptr, rows, cols, out_rates)
    check_error(status, "detection_rate_csc")


def dispersion(
    means: Any,
    vars: Any,
    size: int,
    out_dispersion: Any
) -> None:
    """Compute dispersion (variance / mean) for features.
    
    Args:
        means: Mean values pointer.
        vars: Variance values pointer.
        size: Number of features.
        out_dispersion: Output dispersion values pointer.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_dispersion.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_real), c_size, ctypes.POINTER(c_real)
    ]
    lib.scl_dispersion.restype = ctypes.c_int
    
    status = lib.scl_dispersion(means, vars, size, out_dispersion)
    check_error(status, "dispersion")

