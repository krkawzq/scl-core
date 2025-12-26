"""Feature statistics kernels.

Low-level C bindings for feature-level statistics computation.
"""

import ctypes
from typing import Any

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
    rows: int,
    cols: int,
    out_means: Any,
    out_vars: Any,
    ddof: int
) -> None:
    """Compute standard moments (mean, variance) for each feature.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows (cells).
        cols: Number of columns (genes).
        out_means: Output array for means [cols].
        out_vars: Output array for variances [cols].
        ddof: Delta degrees of freedom.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_standard_moments_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    ]
    lib.scl_standard_moments_csc.restype = ctypes.c_int
    
    status = lib.scl_standard_moments_csc(
        data, indices, indptr, rows, cols, out_means, out_vars, ddof
    )
    check_error(status, "standard_moments_csc")


def clipped_moments_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    clip_vals: Any,
    out_means: Any,
    out_vars: Any
) -> None:
    """Compute clipped moments (mean, variance) for each feature.
    
    Values are clipped to [0, clip_val] before computing moments.
    Used for robust dispersion estimation.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows (cells).
        cols: Number of columns (genes).
        clip_vals: Clipping values per column [cols].
        out_means: Output array for means [cols].
        out_vars: Output array for variances [cols].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_clipped_moments_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_clipped_moments_csc.restype = ctypes.c_int
    
    status = lib.scl_clipped_moments_csc(
        data, indices, indptr, rows, cols, clip_vals, out_means, out_vars
    )
    check_error(status, "clipped_moments_csc")


def detection_rate_csc(
    indptr: Any,
    rows: int,
    cols: int,
    out_rates: Any
) -> None:
    """Compute detection rate (fraction of cells expressing) for each feature.
    
    Args:
        indptr: CSC column pointers pointer.
        rows: Number of rows (cells).
        cols: Number of columns (genes).
        out_rates: Output array for detection rates [cols].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_detection_rate_csc.argtypes = [
        ctypes.c_void_p, c_index, c_index, ctypes.c_void_p
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
    """Compute dispersion (coefficient of variation squared) from means and variances.
    
    dispersion = variance / mean^2
    
    Args:
        means: Input means array pointer.
        vars: Input variances array pointer.
        size: Number of elements.
        out_dispersion: Output dispersion array pointer.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_dispersion.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, c_size, ctypes.c_void_p
    ]
    lib.scl_dispersion.restype = ctypes.c_int
    
    status = lib.scl_dispersion(means, vars, size, out_dispersion)
    check_error(status, "dispersion")
