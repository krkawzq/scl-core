"""
Error handling for SCL.

Error codes are aligned with C API (scl/binding/c_api/core/core.h).
"""

from __future__ import annotations

from typing import Optional, Callable
import ctypes


# =============================================================================
# Error Codes (Aligned with C API)
# =============================================================================

# Success
SCL_OK = 0

# General errors (1-9)
SCL_ERROR_UNKNOWN = 1
SCL_ERROR_INTERNAL = 2
SCL_ERROR_OUT_OF_MEMORY = 3
SCL_ERROR_NULL_POINTER = 4

# Argument errors (10-19)
SCL_ERROR_INVALID_ARGUMENT = 10
SCL_ERROR_DIMENSION_MISMATCH = 11
SCL_ERROR_DOMAIN_ERROR = 12
SCL_ERROR_RANGE_ERROR = 13
SCL_ERROR_INDEX_OUT_OF_BOUNDS = 14

# Type errors (20-29)
SCL_ERROR_TYPE_ERROR = 20
SCL_ERROR_TYPE_MISMATCH = 21

# I/O errors (30-39)
SCL_ERROR_IO_ERROR = 30
SCL_ERROR_FILE_NOT_FOUND = 31
SCL_ERROR_PERMISSION_DENIED = 32
SCL_ERROR_READ_ERROR = 33
SCL_ERROR_WRITE_ERROR = 34
SCL_ERROR_UNREGISTERED_POINTER = 35
SCL_ERROR_BUFFER_NOT_FOUND = 36

# Feature errors (40-49)
SCL_ERROR_NOT_IMPLEMENTED = 40
SCL_ERROR_FEATURE_UNAVAILABLE = 41

# Numerical errors (50-59)
SCL_ERROR_NUMERICAL_ERROR = 50
SCL_ERROR_DIVISION_BY_ZERO = 51
SCL_ERROR_OVERFLOW = 52
SCL_ERROR_UNDERFLOW = 53
SCL_ERROR_CONVERGENCE_ERROR = 54


# Error code to message mapping
_ERROR_MESSAGES = {
    SCL_OK: "Success",
    SCL_ERROR_UNKNOWN: "Unknown error",
    SCL_ERROR_INTERNAL: "Internal error",
    SCL_ERROR_OUT_OF_MEMORY: "Out of memory",
    SCL_ERROR_NULL_POINTER: "Null pointer",
    SCL_ERROR_INVALID_ARGUMENT: "Invalid argument",
    SCL_ERROR_DIMENSION_MISMATCH: "Dimension mismatch",
    SCL_ERROR_DOMAIN_ERROR: "Domain error",
    SCL_ERROR_RANGE_ERROR: "Range error",
    SCL_ERROR_INDEX_OUT_OF_BOUNDS: "Index out of bounds",
    SCL_ERROR_TYPE_ERROR: "Type error",
    SCL_ERROR_TYPE_MISMATCH: "Type mismatch",
    SCL_ERROR_IO_ERROR: "I/O error",
    SCL_ERROR_FILE_NOT_FOUND: "File not found",
    SCL_ERROR_PERMISSION_DENIED: "Permission denied",
    SCL_ERROR_READ_ERROR: "Read error",
    SCL_ERROR_WRITE_ERROR: "Write error",
    SCL_ERROR_UNREGISTERED_POINTER: "Unregistered pointer",
    SCL_ERROR_BUFFER_NOT_FOUND: "Buffer not found",
    SCL_ERROR_NOT_IMPLEMENTED: "Not implemented",
    SCL_ERROR_FEATURE_UNAVAILABLE: "Feature unavailable",
    SCL_ERROR_NUMERICAL_ERROR: "Numerical error",
    SCL_ERROR_DIVISION_BY_ZERO: "Division by zero",
    SCL_ERROR_OVERFLOW: "Overflow",
    SCL_ERROR_UNDERFLOW: "Underflow",
    SCL_ERROR_CONVERGENCE_ERROR: "Convergence error",
}


# =============================================================================
# Exception Class
# =============================================================================

class SCLError(Exception):
    """
    Base exception for all SCL errors.
    
    Error codes are aligned with C API (core.h).
    """
    
    # Re-export error codes as class attributes for convenience
    OK = SCL_OK
    ERROR_UNKNOWN = SCL_ERROR_UNKNOWN
    ERROR_INTERNAL = SCL_ERROR_INTERNAL
    ERROR_OUT_OF_MEMORY = SCL_ERROR_OUT_OF_MEMORY
    ERROR_NULL_POINTER = SCL_ERROR_NULL_POINTER
    ERROR_INVALID_ARGUMENT = SCL_ERROR_INVALID_ARGUMENT
    ERROR_DIMENSION_MISMATCH = SCL_ERROR_DIMENSION_MISMATCH
    ERROR_DOMAIN_ERROR = SCL_ERROR_DOMAIN_ERROR
    ERROR_RANGE_ERROR = SCL_ERROR_RANGE_ERROR
    ERROR_INDEX_OUT_OF_BOUNDS = SCL_ERROR_INDEX_OUT_OF_BOUNDS
    ERROR_TYPE_ERROR = SCL_ERROR_TYPE_ERROR
    ERROR_TYPE_MISMATCH = SCL_ERROR_TYPE_MISMATCH
    ERROR_IO_ERROR = SCL_ERROR_IO_ERROR
    ERROR_FILE_NOT_FOUND = SCL_ERROR_FILE_NOT_FOUND
    ERROR_PERMISSION_DENIED = SCL_ERROR_PERMISSION_DENIED
    ERROR_READ_ERROR = SCL_ERROR_READ_ERROR
    ERROR_WRITE_ERROR = SCL_ERROR_WRITE_ERROR
    ERROR_UNREGISTERED_POINTER = SCL_ERROR_UNREGISTERED_POINTER
    ERROR_BUFFER_NOT_FOUND = SCL_ERROR_BUFFER_NOT_FOUND
    ERROR_NOT_IMPLEMENTED = SCL_ERROR_NOT_IMPLEMENTED
    ERROR_FEATURE_UNAVAILABLE = SCL_ERROR_FEATURE_UNAVAILABLE
    ERROR_NUMERICAL_ERROR = SCL_ERROR_NUMERICAL_ERROR
    ERROR_DIVISION_BY_ZERO = SCL_ERROR_DIVISION_BY_ZERO
    ERROR_OVERFLOW = SCL_ERROR_OVERFLOW
    ERROR_UNDERFLOW = SCL_ERROR_UNDERFLOW
    ERROR_CONVERGENCE_ERROR = SCL_ERROR_CONVERGENCE_ERROR
    
    def __init__(self, code: int, message: Optional[str] = None):
        """
        Create SCL exception.
        
        Args:
            code: Error code from C API
            message: Optional detailed message (fetched from C API if not provided)
        """
        self.code = code
        if message is None:
            message = _ERROR_MESSAGES.get(code, f"Unknown error (code={code})")
        self.message = message
        super().__init__(f"SCL Error {code}: {message}")
    
    @classmethod
    def from_code(cls, code: int, context: str = "") -> "SCLError":
        """Create exception from error code with optional context."""
        base_msg = _ERROR_MESSAGES.get(code, f"Unknown error")
        msg = f"{context}: {base_msg}" if context else base_msg
        return cls(code, msg)


# =============================================================================
# Error Checking Functions
# =============================================================================

# Global reference to library's error functions (set during initialization)
_get_last_error: Optional[Callable[[], bytes]] = None
_get_last_error_code: Optional[Callable[[], int]] = None
_clear_error: Optional[Callable[[], None]] = None


def _init_error_functions(lib) -> None:
    """
    Initialize error functions from library.
    
    Called by Library during initialization.
    """
    global _get_last_error, _get_last_error_code, _clear_error
    
    # scl_get_last_error returns const char*
    lib.scl_get_last_error.restype = ctypes.c_char_p
    lib.scl_get_last_error.argtypes = []
    
    # scl_get_last_error_code returns int32_t
    lib.scl_get_last_error_code.restype = ctypes.c_int32
    lib.scl_get_last_error_code.argtypes = []
    
    # scl_clear_error returns void
    lib.scl_clear_error.restype = None
    lib.scl_clear_error.argtypes = []
    
    _get_last_error = lib.scl_get_last_error
    _get_last_error_code = lib.scl_get_last_error_code
    _clear_error = lib.scl_clear_error


def get_last_error() -> tuple[int, str]:
    """
    Get the last error from the native library (thread-local).
    
    Returns:
        Tuple of (error_code, error_message)
    """
    if _get_last_error_code is None:
        return SCL_OK, "No error (library not initialized)"
    
    code = _get_last_error_code()
    
    if _get_last_error is not None:
        msg_ptr = _get_last_error()
        if msg_ptr:
            # c_char_p returns bytes or None
            if isinstance(msg_ptr, bytes):
                return code, msg_ptr.decode('utf-8')
            # If it's already a string, return directly
            if isinstance(msg_ptr, str):
                return code, msg_ptr
    
    return code, _ERROR_MESSAGES.get(code, f"Unknown error (code={code})")


def clear_error() -> None:
    """Clear the last error (thread-local)."""
    if _clear_error is not None:
        _clear_error()


def check_error(code: int, context: str = "") -> None:
    """
    Check error code and raise exception if not OK.
    
    This function fetches the detailed error message from the C library
    using thread-local storage.
    
    Args:
        code: Error code from SCL function
        context: Optional context message for better error reporting
        
    Raises:
        SCLError: If code indicates an error
    """
    if code == SCL_OK:
        return
    
    # Get detailed message from C library
    _, c_msg = get_last_error()
    
    # Build final message
    if context:
        msg = f"{context}: {c_msg}"
    else:
        msg = c_msg
    
    raise SCLError(code, msg)


# =============================================================================
# Convenience Aliases (for backward compatibility)
# =============================================================================

# Deprecated: Use SCL_OK, SCL_ERROR_* directly
ErrorCode = SCLError  # Alias for backward compatibility
