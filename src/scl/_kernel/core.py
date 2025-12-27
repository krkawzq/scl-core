"""Core utility kernels.

Low-level C bindings for basic array operations and math reductions.
"""

import ctypes
from typing import Any, Tuple

from .lib_loader import get_lib
from .types import c_real, c_index, c_size, check_error


__all__ = [
    # Memory operations
    'fill_real',
    'fill_index',
    'zero_real',
    'zero_index',
    'copy_fast_real',
    'copy_safe_real',
    'stream_copy_real',
    # Array utilities
    'iota_index',
    'reverse_real',
    'reverse_index',
    'unique_real',
    # Math reductions
    'sum_real',
    'mean_real',
    'variance_real',
    'min_real',
    'max_real',
    'minmax_real',
    'dot_real',
    'norm_real',
    # Element-wise operations
    'add_scalar_real',
    'mul_scalar_real',
    'add_arrays_real',
    'mul_arrays_real',
    'clip_real',
]


# =============================================================================
# Memory Operations
# =============================================================================

def fill_real(data: Any, size: int, value: float) -> None:
    """Fill array with a constant value.
    
    Args:
        data: Data array pointer (modified in-place).
        size: Number of elements.
        value: Fill value.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_fill_real.argtypes = [ctypes.c_void_p, c_size, c_real]
    lib.scl_fill_real.restype = ctypes.c_int
    
    status = lib.scl_fill_real(data, size, value)
    check_error(status, "fill_real")


def fill_index(data: Any, size: int, value: int) -> None:
    """Fill index array with a constant value.
    
    Args:
        data: Index array pointer (modified in-place).
        size: Number of elements.
        value: Fill value.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_fill_index.argtypes = [ctypes.c_void_p, c_size, c_index]
    lib.scl_fill_index.restype = ctypes.c_int
    
    status = lib.scl_fill_index(data, size, value)
    check_error(status, "fill_index")


def zero_real(data: Any, size: int) -> None:
    """Zero out real array.
    
    Args:
        data: Data array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_zero_real.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_zero_real.restype = ctypes.c_int
    
    status = lib.scl_zero_real(data, size)
    check_error(status, "zero_real")


def zero_index(data: Any, size: int) -> None:
    """Zero out index array.
    
    Args:
        data: Index array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_zero_index.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_zero_index.restype = ctypes.c_int
    
    status = lib.scl_zero_index(data, size)
    check_error(status, "zero_index")


def copy_fast_real(src: Any, dst: Any, size: int) -> None:
    """Fast copy (assumes non-overlapping).
    
    Args:
        src: Source array pointer.
        dst: Destination array pointer.
        size: Number of elements.
        
    Raises:
        RuntimeError: If copy fails.
    """
    lib = get_lib()
    lib.scl_copy_fast_real.argtypes = [ctypes.c_void_p, ctypes.c_void_p, c_size]
    lib.scl_copy_fast_real.restype = ctypes.c_int
    
    status = lib.scl_copy_fast_real(src, dst, size)
    check_error(status, "copy_fast_real")


def copy_safe_real(src: Any, dst: Any, size: int) -> None:
    """Safe copy (handles overlapping).
    
    Args:
        src: Source array pointer.
        dst: Destination array pointer.
        size: Number of elements.
        
    Raises:
        RuntimeError: If copy fails.
    """
    lib = get_lib()
    lib.scl_copy_safe_real.argtypes = [ctypes.c_void_p, ctypes.c_void_p, c_size]
    lib.scl_copy_safe_real.restype = ctypes.c_int
    
    status = lib.scl_copy_safe_real(src, dst, size)
    check_error(status, "copy_safe_real")


def stream_copy_real(src: Any, dst: Any, size: int) -> None:
    """Streaming copy (non-temporal, bypasses cache).
    
    Args:
        src: Source array pointer.
        dst: Destination array pointer.
        size: Number of elements.
        
    Raises:
        RuntimeError: If copy fails.
    """
    lib = get_lib()
    lib.scl_stream_copy_real.argtypes = [ctypes.c_void_p, ctypes.c_void_p, c_size]
    lib.scl_stream_copy_real.restype = ctypes.c_int
    
    status = lib.scl_stream_copy_real(src, dst, size)
    check_error(status, "stream_copy_real")


# =============================================================================
# Array Utilities
# =============================================================================

def iota_index(data: Any, size: int) -> None:
    """Fill array with sequence 0, 1, 2, ..., size-1.
    
    Args:
        data: Index array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_iota_index.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_iota_index.restype = ctypes.c_int
    
    status = lib.scl_iota_index(data, size)
    check_error(status, "iota_index")


def reverse_real(data: Any, size: int) -> None:
    """Reverse array in-place.
    
    Args:
        data: Data array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_reverse_real.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_reverse_real.restype = ctypes.c_int
    
    status = lib.scl_reverse_real(data, size)
    check_error(status, "reverse_real")


def reverse_index(data: Any, size: int) -> None:
    """Reverse index array in-place.
    
    Args:
        data: Index array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_reverse_index.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_reverse_index.restype = ctypes.c_int
    
    status = lib.scl_reverse_index(data, size)
    check_error(status, "reverse_index")


def unique_real(data: Any, size: int) -> int:
    """Remove duplicates from sorted array (in-place).
    
    Args:
        data: Sorted data array pointer (modified in-place).
        size: Number of elements.
        
    Returns:
        New size after removing duplicates.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_unique_real.argtypes = [
        ctypes.c_void_p, c_size, ctypes.POINTER(c_size)
    ]
    lib.scl_unique_real.restype = ctypes.c_int
    
    new_size = c_size()
    status = lib.scl_unique_real(data, size, ctypes.byref(new_size))
    check_error(status, "unique_real")
    return new_size.value


# =============================================================================
# Math Reductions
# =============================================================================

def sum_real(data: Any, size: int) -> float:
    """Compute sum of array.
    
    Args:
        data: Data array pointer.
        size: Number of elements.
        
    Returns:
        Sum of elements.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_sum_real.argtypes = [ctypes.c_void_p, c_size, ctypes.POINTER(c_real)]
    lib.scl_sum_real.restype = ctypes.c_int
    
    result = c_real()
    status = lib.scl_sum_real(data, size, ctypes.byref(result))
    check_error(status, "sum_real")
    return result.value


def mean_real(data: Any, size: int) -> float:
    """Compute mean of array.
    
    Args:
        data: Data array pointer.
        size: Number of elements.
        
    Returns:
        Mean of elements.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_mean_real.argtypes = [ctypes.c_void_p, c_size, ctypes.POINTER(c_real)]
    lib.scl_mean_real.restype = ctypes.c_int
    
    result = c_real()
    status = lib.scl_mean_real(data, size, ctypes.byref(result))
    check_error(status, "mean_real")
    return result.value


def variance_real(data: Any, size: int, mean: float, ddof: int) -> float:
    """Compute variance of array.
    
    Args:
        data: Data array pointer.
        size: Number of elements.
        mean: Pre-computed mean.
        ddof: Delta degrees of freedom.
        
    Returns:
        Variance.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_variance_real.argtypes = [
        ctypes.c_void_p, c_size, c_real, ctypes.c_int, ctypes.POINTER(c_real)
    ]
    lib.scl_variance_real.restype = ctypes.c_int
    
    result = c_real()
    status = lib.scl_variance_real(data, size, mean, ddof, ctypes.byref(result))
    check_error(status, "variance_real")
    return result.value


def min_real(data: Any, size: int) -> float:
    """Find minimum value in array.
    
    Args:
        data: Data array pointer.
        size: Number of elements.
        
    Returns:
        Minimum value.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_min_real.argtypes = [ctypes.c_void_p, c_size, ctypes.POINTER(c_real)]
    lib.scl_min_real.restype = ctypes.c_int
    
    result = c_real()
    status = lib.scl_min_real(data, size, ctypes.byref(result))
    check_error(status, "min_real")
    return result.value


def max_real(data: Any, size: int) -> float:
    """Find maximum value in array.
    
    Args:
        data: Data array pointer.
        size: Number of elements.
        
    Returns:
        Maximum value.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_max_real.argtypes = [ctypes.c_void_p, c_size, ctypes.POINTER(c_real)]
    lib.scl_max_real.restype = ctypes.c_int
    
    result = c_real()
    status = lib.scl_max_real(data, size, ctypes.byref(result))
    check_error(status, "max_real")
    return result.value


def minmax_real(data: Any, size: int) -> Tuple[float, float]:
    """Find both minimum and maximum values.
    
    Args:
        data: Data array pointer.
        size: Number of elements.
        
    Returns:
        Tuple of (min, max).
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_minmax_real.argtypes = [
        ctypes.c_void_p, c_size, ctypes.POINTER(c_real), ctypes.POINTER(c_real)
    ]
    lib.scl_minmax_real.restype = ctypes.c_int
    
    min_val = c_real()
    max_val = c_real()
    status = lib.scl_minmax_real(data, size, ctypes.byref(min_val), ctypes.byref(max_val))
    check_error(status, "minmax_real")
    return (min_val.value, max_val.value)


def dot_real(x: Any, y: Any, size: int) -> float:
    """Compute dot product of two arrays.
    
    Args:
        x: First array pointer.
        y: Second array pointer.
        size: Number of elements.
        
    Returns:
        Dot product.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_dot_real.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, c_size, ctypes.POINTER(c_real)
    ]
    lib.scl_dot_real.restype = ctypes.c_int
    
    result = c_real()
    status = lib.scl_dot_real(x, y, size, ctypes.byref(result))
    check_error(status, "dot_real")
    return result.value


def norm_real(data: Any, size: int) -> float:
    """Compute L2 norm of array.
    
    Args:
        data: Data array pointer.
        size: Number of elements.
        
    Returns:
        L2 norm.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_norm_real.argtypes = [ctypes.c_void_p, c_size, ctypes.POINTER(c_real)]
    lib.scl_norm_real.restype = ctypes.c_int
    
    result = c_real()
    status = lib.scl_norm_real(data, size, ctypes.byref(result))
    check_error(status, "norm_real")
    return result.value


# =============================================================================
# Element-wise Operations
# =============================================================================

def add_scalar_real(x: Any, size: int, scalar: float, y: Any) -> None:
    """Add scalar to array: y = x + scalar.
    
    Args:
        x: Input array pointer.
        size: Number of elements.
        scalar: Scalar to add.
        y: Output array pointer.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_add_scalar_real.argtypes = [
        ctypes.c_void_p, c_size, c_real, ctypes.c_void_p
    ]
    lib.scl_add_scalar_real.restype = ctypes.c_int
    
    status = lib.scl_add_scalar_real(x, size, scalar, y)
    check_error(status, "add_scalar_real")


def mul_scalar_real(x: Any, size: int, scalar: float, y: Any) -> None:
    """Multiply array by scalar: y = x * scalar.
    
    Args:
        x: Input array pointer.
        size: Number of elements.
        scalar: Scalar to multiply.
        y: Output array pointer.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_mul_scalar_real.argtypes = [
        ctypes.c_void_p, c_size, c_real, ctypes.c_void_p
    ]
    lib.scl_mul_scalar_real.restype = ctypes.c_int
    
    status = lib.scl_mul_scalar_real(x, size, scalar, y)
    check_error(status, "mul_scalar_real")


def add_arrays_real(x: Any, y: Any, size: int, z: Any) -> None:
    """Add two arrays: z = x + y.
    
    Args:
        x: First input array pointer.
        y: Second input array pointer.
        size: Number of elements.
        z: Output array pointer.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_add_arrays_real.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, c_size, ctypes.c_void_p
    ]
    lib.scl_add_arrays_real.restype = ctypes.c_int
    
    status = lib.scl_add_arrays_real(x, y, size, z)
    check_error(status, "add_arrays_real")


def mul_arrays_real(x: Any, y: Any, size: int, z: Any) -> None:
    """Multiply two arrays element-wise: z = x * y.
    
    Args:
        x: First input array pointer.
        y: Second input array pointer.
        size: Number of elements.
        z: Output array pointer.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_mul_arrays_real.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, c_size, ctypes.c_void_p
    ]
    lib.scl_mul_arrays_real.restype = ctypes.c_int
    
    status = lib.scl_mul_arrays_real(x, y, size, z)
    check_error(status, "mul_arrays_real")


def clip_real(data: Any, size: int, min_val: float, max_val: float) -> None:
    """Clip array values to [min_val, max_val] (in-place).
    
    Args:
        data: Data array pointer (modified in-place).
        size: Number of elements.
        min_val: Minimum value.
        max_val: Maximum value.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_clip_real.argtypes = [ctypes.c_void_p, c_size, c_real, c_real]
    lib.scl_clip_real.restype = ctypes.c_int
    
    status = lib.scl_clip_real(data, size, min_val, max_val)
    check_error(status, "clip_real")

