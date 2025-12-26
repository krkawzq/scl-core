"""
C Type Definitions and Conversions

Maps Python/NumPy types to C types for ctypes bindings.
"""

import ctypes
import numpy as np
from typing import Any

__all__ = [
    'c_real', 'c_index', 'c_size', 'c_byte',
    'np_real', 'np_index',
    'check_error', 'get_last_error',
]

# =============================================================================
# C Type Aliases
# =============================================================================

# SCL types (configured at compile time)
c_real = ctypes.c_float   # or c_double (depends on SCL_PRECISION)
c_index = ctypes.c_int64
c_size = ctypes.c_size_t
c_byte = ctypes.c_uint8

# NumPy dtype equivalents
np_real = np.float32      # or np.float64
np_index = np.int64

# =============================================================================
# Precision Detection
# =============================================================================

def detect_precision():
    """
    Detect SCL library precision at runtime.
    
    Updates global c_real and np_real based on library configuration.
    """
    global c_real, np_real
    
    from .lib_loader import get_lib
    
    lib = get_lib()
    lib.scl_precision_type.restype = ctypes.c_int
    precision_code = lib.scl_precision_type()
    
    if precision_code == 0:  # float32
        c_real = ctypes.c_float
        np_real = np.float32
    elif precision_code == 1:  # float64
        c_real = ctypes.c_double
        np_real = np.float64
    else:
        raise RuntimeError(f"Unknown SCL precision code: {precision_code}")


# Auto-detect on import
try:
    detect_precision()
except:
    # Fallback to float32 if detection fails
    pass

# =============================================================================
# Error Handling
# =============================================================================

def get_last_error() -> str:
    """
    Get last error message from SCL library.
    
    Returns:
        Error message string
    """
    from .lib_loader import get_lib
    
    lib = get_lib()
    lib.scl_get_last_error.restype = ctypes.c_char_p
    error_ptr = lib.scl_get_last_error()
    
    if error_ptr:
        return error_ptr.decode('utf-8')
    return ""


def check_error(status: int, operation: str = "SCL operation"):
    """
    Check return status and raise exception if error occurred.
    
    Args:
        status: Return code from C function (0 = success)
        operation: Operation name for error message
        
    Raises:
        RuntimeError: If status != 0
    """
    if status != 0:
        error_msg = get_last_error()
        if error_msg:
            raise RuntimeError(f"{operation} failed: {error_msg}")
        else:
            raise RuntimeError(f"{operation} failed with code {status}")


# =============================================================================
# Array Conversion Helpers
# =============================================================================

def as_c_ptr(arr: np.ndarray, dtype) -> Any:
    """
    Convert NumPy array to ctypes pointer.
    
    Args:
        arr: NumPy array
        dtype: Target ctypes type
        
    Returns:
        ctypes pointer
    """
    if arr is None:
        return None
    return arr.ctypes.data_as(ctypes.POINTER(dtype))


def ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is C-contiguous.
    
    Args:
        arr: Input array
        
    Returns:
        C-contiguous array (may be a copy)
    """
    if not arr.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(arr)
    return arr


def validate_csr_matrix(data, indices, indptr, shape):
    """
    Validate CSR matrix arrays.
    
    Args:
        data: Value array
        indices: Column indices
        indptr: Row pointers
        shape: (rows, cols)
        
    Raises:
        ValueError: If arrays are invalid
    """
    rows, cols = shape
    
    if len(indptr) != rows + 1:
        raise ValueError(f"CSR: indptr length must be rows+1 ({rows+1}), got {len(indptr)}")
    
    if len(data) != len(indices):
        raise ValueError(f"CSR: data and indices must have same length")
    
    if indptr[0] != 0:
        raise ValueError(f"CSR: indptr[0] must be 0, got {indptr[0]}")
    
    if indptr[-1] != len(data):
        raise ValueError(f"CSR: indptr[-1] must equal nnz ({len(data)}), got {indptr[-1]}")


def validate_csc_matrix(data, indices, indptr, shape):
    """
    Validate CSC matrix arrays.
    
    Args:
        data: Value array
        indices: Row indices
        indptr: Column pointers
        shape: (rows, cols)
        
    Raises:
        ValueError: If arrays are invalid
    """
    rows, cols = shape
    
    if len(indptr) != cols + 1:
        raise ValueError(f"CSC: indptr length must be cols+1 ({cols+1}), got {len(indptr)}")
    
    if len(data) != len(indices):
        raise ValueError(f"CSC: data and indices must have same length")
    
    if indptr[0] != 0:
        raise ValueError(f"CSC: indptr[0] must be 0, got {indptr[0]}")
    
    if indptr[-1] != len(data):
        raise ValueError(f"CSC: indptr[-1] must equal nnz ({len(data)}), got {indptr[-1]}")

