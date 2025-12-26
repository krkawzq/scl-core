"""
Data Transformation Kernels

Low-level C bindings for log transforms, softmax, etc.
"""

import ctypes
import numpy as np
from .lib_loader import get_lib
from .types import c_real, c_index, c_size, np_real, check_error, as_c_ptr

__all__ = [
    'log1p_inplace',
    'log2p1_inplace',
    'expm1_inplace',
]

# =============================================================================
# Function Signatures
# =============================================================================

def _init_signatures():
    """Initialize C function signatures."""
    lib = get_lib()
    
    # log1p_inplace
    lib.scl_log1p_inplace.argtypes = [
        ctypes.POINTER(c_real),  # data
        c_size,                   # size
    ]
    lib.scl_log1p_inplace.restype = ctypes.c_int
    
    # log2p1_inplace
    lib.scl_log2p1_inplace.argtypes = [
        ctypes.POINTER(c_real),  # data
        c_size,                   # size
    ]
    lib.scl_log2p1_inplace.restype = ctypes.c_int
    
    # expm1_inplace
    lib.scl_expm1_inplace.argtypes = [
        ctypes.POINTER(c_real),  # data
        c_size,                   # size
    ]
    lib.scl_expm1_inplace.restype = ctypes.c_int


_init_signatures()

# =============================================================================
# Python Wrappers
# =============================================================================

def log1p_inplace(data: np.ndarray) -> None:
    """
    Apply ln(1 + x) transformation in-place.
    
    Args:
        data: Array to transform (modified in-place)
    """
    lib = get_lib()
    status = lib.scl_log1p_inplace(
        as_c_ptr(data, c_real),
        data.size
    )
    check_error(status, "log1p_inplace")


def log2p1_inplace(data: np.ndarray) -> None:
    """
    Apply log2(1 + x) transformation in-place.
    
    Args:
        data: Array to transform (modified in-place)
    """
    lib = get_lib()
    status = lib.scl_log2p1_inplace(
        as_c_ptr(data, c_real),
        data.size
    )
    check_error(status, "log2p1_inplace")


def expm1_inplace(data: np.ndarray) -> None:
    """
    Apply exp(x) - 1 transformation in-place.
    
    Args:
        data: Array to transform (modified in-place)
    """
    lib = get_lib()
    status = lib.scl_expm1_inplace(
        as_c_ptr(data, c_real),
        data.size
    )
    check_error(status, "expm1_inplace")

