"""C type definitions and error handling for SCL bindings.

Maps Python types to C types for ctypes bindings. No external dependencies.
"""

import ctypes
import os
from typing import Any, Optional


__all__ = [
    'c_real', 'c_index', 'c_size', 'c_byte',
    'check_error', 'get_last_error', 'clear_error',
]


# =============================================================================
# C Type Aliases
# =============================================================================

# SCL types (configured at compile time)
c_real = ctypes.c_float   # or c_double (depends on SCL_PRECISION)
c_index = ctypes.c_int64  # or c_int32/c_int16 (depends on SCL_INDEX_PRECISION)
c_size = ctypes.c_size_t
c_byte = ctypes.c_uint8


# =============================================================================
# Precision Detection
# =============================================================================

def detect_precision(precision: Optional[str] = None) -> None:
    """Detect SCL library precision at runtime.
    
    Updates global c_real and c_index based on library configuration.
    
    Args:
        precision: Optional precision to use ('f32' or 'f64').
                   If None, uses SCL_PRECISION environment variable or 'f32'.
    """
    global c_real, c_index
    
    from .lib_loader import get_lib
    
    # Determine which library to use
    if precision is None:
        precision = os.environ.get('SCL_PRECISION', 'f32')
    
    lib = get_lib(precision)
    
    # Detect real type
    lib.scl_precision_type.restype = ctypes.c_int
    precision_code = lib.scl_precision_type()
    
    if precision_code == 0:  # float32
        c_real = ctypes.c_float
    elif precision_code == 1:  # float64
        c_real = ctypes.c_double
    else:
        raise RuntimeError(f"Unknown SCL precision code: {precision_code}")
    
    # Detect index type
    lib.scl_index_type.restype = ctypes.c_int
    index_code = lib.scl_index_type()
    
    if index_code == 0:  # int16
        c_index = ctypes.c_int16
    elif index_code == 1:  # int32
        c_index = ctypes.c_int32
    elif index_code == 2:  # int64
        c_index = ctypes.c_int64
    else:
        raise RuntimeError(f"Unknown SCL index code: {index_code}")


# Auto-detect on import
try:
    detect_precision()
except:
    # Fallback to defaults if detection fails
    pass


# =============================================================================
# Error Handling
# =============================================================================

def get_last_error() -> str:
    """Get last error message from SCL library.
    
    Returns:
        Error message string, or empty string if no error.
    """
    from .lib_loader import get_lib
    
    lib = get_lib()
    lib.scl_get_last_error.restype = ctypes.c_char_p
    error_ptr = lib.scl_get_last_error()
    
    if error_ptr:
        return error_ptr.decode('utf-8')
    return ""


def clear_error() -> None:
    """Clear error state in SCL library."""
    from .lib_loader import get_lib
    
    lib = get_lib()
    lib.scl_clear_error()


def check_error(status: int, operation: str = "SCL operation") -> None:
    """Check return status and raise exception if error occurred.
    
    Args:
        status: Return code from C function (0 = success).
        operation: Operation name for error message.
        
    Raises:
        RuntimeError: If status != 0.
    """
    if status != 0:
        error_msg = get_last_error()
        if error_msg:
            raise RuntimeError(f"{operation} failed: {error_msg}")
        else:
            raise RuntimeError(f"{operation} failed with code {status}")

