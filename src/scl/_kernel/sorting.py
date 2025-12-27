"""Sorting operation kernels.

Low-level C bindings for SIMD-optimized sorting operations.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, c_size, c_byte, check_error


__all__ = [
    # VQSort (SIMD sorting)
    'vqsort_real_ascending',
    'vqsort_real_descending',
    'vqsort_index_ascending',
    'vqsort_index_descending',
    'vqsort_int32_ascending',
    'vqsort_int32_descending',
    # Argsort
    'argsort_real_ascending',
    'argsort_real_descending',
    'argsort_real_buffered',
    'argsort_real_buffered_descending',
    # Pair sorting
    'sort_pairs_real_real_ascending',
    'sort_pairs_real_real_descending',
    'sort_pairs_real_index_ascending',
    'sort_pairs_real_index_descending',
    'sort_pairs_index_real_ascending',
    'sort_pairs_index_index_ascending',
    # Top-K
    'topk_real',
    'topk_real_with_indices',
    # Utility
    'is_sorted_real_ascending',
    'is_sorted_real_descending',
    'argsort_buffer_size',
]


# =============================================================================
# VQSort (SIMD-Optimized Sorting)
# =============================================================================

def vqsort_real_ascending(data: Any, size: int) -> None:
    """Sort real array in ascending order (in-place, SIMD-optimized).
    
    Args:
        data: Data array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_vqsort_real_ascending.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_vqsort_real_ascending.restype = ctypes.c_int
    
    status = lib.scl_vqsort_real_ascending(data, size)
    check_error(status, "vqsort_real_ascending")


def vqsort_real_descending(data: Any, size: int) -> None:
    """Sort real array in descending order (in-place, SIMD-optimized).
    
    Args:
        data: Data array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_vqsort_real_descending.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_vqsort_real_descending.restype = ctypes.c_int
    
    status = lib.scl_vqsort_real_descending(data, size)
    check_error(status, "vqsort_real_descending")


def vqsort_index_ascending(data: Any, size: int) -> None:
    """Sort index array in ascending order (in-place, SIMD-optimized).
    
    Args:
        data: Index array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_vqsort_index_ascending.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_vqsort_index_ascending.restype = ctypes.c_int
    
    status = lib.scl_vqsort_index_ascending(data, size)
    check_error(status, "vqsort_index_ascending")


def vqsort_index_descending(data: Any, size: int) -> None:
    """Sort index array in descending order (in-place, SIMD-optimized).
    
    Args:
        data: Index array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_vqsort_index_descending.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_vqsort_index_descending.restype = ctypes.c_int
    
    status = lib.scl_vqsort_index_descending(data, size)
    check_error(status, "vqsort_index_descending")


def vqsort_int32_ascending(data: Any, size: int) -> None:
    """Sort int32 array in ascending order (in-place, SIMD-optimized).
    
    Args:
        data: Int32 array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_vqsort_int32_ascending.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_vqsort_int32_ascending.restype = ctypes.c_int
    
    status = lib.scl_vqsort_int32_ascending(data, size)
    check_error(status, "vqsort_int32_ascending")


def vqsort_int32_descending(data: Any, size: int) -> None:
    """Sort int32 array in descending order (in-place, SIMD-optimized).
    
    Args:
        data: Int32 array pointer (modified in-place).
        size: Number of elements.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_vqsort_int32_descending.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_vqsort_int32_descending.restype = ctypes.c_int
    
    status = lib.scl_vqsort_int32_descending(data, size)
    check_error(status, "vqsort_int32_descending")


# =============================================================================
# Argsort (Indirect Sorting)
# =============================================================================

def argsort_real_ascending(keys: Any, size: int, indices: Any) -> None:
    """Compute argsort indices for ascending order.
    
    Args:
        keys: Key array pointer.
        size: Number of elements.
        indices: Output indices pointer (modified in-place).
        
    Raises:
        RuntimeError: If argsort fails.
    """
    lib = get_lib()
    lib.scl_argsort_real_ascending.argtypes = [
        ctypes.c_void_p, c_size, ctypes.c_void_p
    ]
    lib.scl_argsort_real_ascending.restype = ctypes.c_int
    
    status = lib.scl_argsort_real_ascending(keys, size, indices)
    check_error(status, "argsort_real_ascending")


def argsort_real_descending(keys: Any, size: int, indices: Any) -> None:
    """Compute argsort indices for descending order.
    
    Args:
        keys: Key array pointer.
        size: Number of elements.
        indices: Output indices pointer (modified in-place).
        
    Raises:
        RuntimeError: If argsort fails.
    """
    lib = get_lib()
    lib.scl_argsort_real_descending.argtypes = [
        ctypes.c_void_p, c_size, ctypes.c_void_p
    ]
    lib.scl_argsort_real_descending.restype = ctypes.c_int
    
    status = lib.scl_argsort_real_descending(keys, size, indices)
    check_error(status, "argsort_real_descending")


def argsort_real_buffered(
    keys: Any,
    size: int,
    indices: Any,
    buffer: Any
) -> None:
    """Compute argsort with user-provided buffer.
    
    Args:
        keys: Key array pointer.
        size: Number of elements.
        indices: Output indices pointer (modified in-place).
        buffer: Workspace buffer pointer.
        
    Raises:
        RuntimeError: If argsort fails.
    """
    lib = get_lib()
    lib.scl_argsort_real_buffered.argtypes = [
        ctypes.c_void_p, c_size, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_argsort_real_buffered.restype = ctypes.c_int
    
    status = lib.scl_argsort_real_buffered(keys, size, indices, buffer)
    check_error(status, "argsort_real_buffered")


def argsort_real_buffered_descending(
    keys: Any,
    size: int,
    indices: Any,
    buffer: Any
) -> None:
    """Compute descending argsort with user-provided buffer.
    
    Args:
        keys: Key array pointer.
        size: Number of elements.
        indices: Output indices pointer (modified in-place).
        buffer: Workspace buffer pointer.
        
    Raises:
        RuntimeError: If argsort fails.
    """
    lib = get_lib()
    lib.scl_argsort_real_buffered_descending.argtypes = [
        ctypes.c_void_p, c_size, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_argsort_real_buffered_descending.restype = ctypes.c_int
    
    status = lib.scl_argsort_real_buffered_descending(keys, size, indices, buffer)
    check_error(status, "argsort_real_buffered_descending")


# =============================================================================
# Pair Sorting
# =============================================================================

def sort_pairs_real_real_ascending(keys: Any, values: Any, size: int) -> None:
    """Sort (key, value) pairs by keys in ascending order.
    
    Args:
        keys: Key array pointer (modified in-place).
        values: Value array pointer (modified in-place).
        size: Number of pairs.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_sort_pairs_real_real_ascending.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, c_size
    ]
    lib.scl_sort_pairs_real_real_ascending.restype = ctypes.c_int
    
    status = lib.scl_sort_pairs_real_real_ascending(keys, values, size)
    check_error(status, "sort_pairs_real_real_ascending")


def sort_pairs_real_real_descending(keys: Any, values: Any, size: int) -> None:
    """Sort (key, value) pairs by keys in descending order.
    
    Args:
        keys: Key array pointer (modified in-place).
        values: Value array pointer (modified in-place).
        size: Number of pairs.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_sort_pairs_real_real_descending.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, c_size
    ]
    lib.scl_sort_pairs_real_real_descending.restype = ctypes.c_int
    
    status = lib.scl_sort_pairs_real_real_descending(keys, values, size)
    check_error(status, "sort_pairs_real_real_descending")


def sort_pairs_real_index_ascending(keys: Any, values: Any, size: int) -> None:
    """Sort (real key, index value) pairs in ascending order.
    
    Args:
        keys: Real key array pointer (modified in-place).
        values: Index value array pointer (modified in-place).
        size: Number of pairs.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_sort_pairs_real_index_ascending.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, c_size
    ]
    lib.scl_sort_pairs_real_index_ascending.restype = ctypes.c_int
    
    status = lib.scl_sort_pairs_real_index_ascending(keys, values, size)
    check_error(status, "sort_pairs_real_index_ascending")


def sort_pairs_real_index_descending(keys: Any, values: Any, size: int) -> None:
    """Sort (real key, index value) pairs in descending order.
    
    Args:
        keys: Real key array pointer (modified in-place).
        values: Index value array pointer (modified in-place).
        size: Number of pairs.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_sort_pairs_real_index_descending.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, c_size
    ]
    lib.scl_sort_pairs_real_index_descending.restype = ctypes.c_int
    
    status = lib.scl_sort_pairs_real_index_descending(keys, values, size)
    check_error(status, "sort_pairs_real_index_descending")


def sort_pairs_index_real_ascending(keys: Any, values: Any, size: int) -> None:
    """Sort (index key, real value) pairs in ascending order.
    
    Args:
        keys: Index key array pointer (modified in-place).
        values: Real value array pointer (modified in-place).
        size: Number of pairs.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_sort_pairs_index_real_ascending.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, c_size
    ]
    lib.scl_sort_pairs_index_real_ascending.restype = ctypes.c_int
    
    status = lib.scl_sort_pairs_index_real_ascending(keys, values, size)
    check_error(status, "sort_pairs_index_real_ascending")


def sort_pairs_index_index_ascending(keys: Any, values: Any, size: int) -> None:
    """Sort (index key, index value) pairs in ascending order.
    
    Args:
        keys: Index key array pointer (modified in-place).
        values: Index value array pointer (modified in-place).
        size: Number of pairs.
        
    Raises:
        RuntimeError: If sort fails.
    """
    lib = get_lib()
    lib.scl_sort_pairs_index_index_ascending.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, c_size
    ]
    lib.scl_sort_pairs_index_index_ascending.restype = ctypes.c_int
    
    status = lib.scl_sort_pairs_index_index_ascending(keys, values, size)
    check_error(status, "sort_pairs_index_index_ascending")


# =============================================================================
# Top-K Selection
# =============================================================================

def topk_real(data: Any, size: int, k: int) -> None:
    """Partially sort to get top-k elements (in-place).
    
    After this operation, the first k elements are the top-k (unsorted),
    and the rest are unspecified.
    
    Args:
        data: Data array pointer (modified in-place).
        size: Total number of elements.
        k: Number of top elements to select.
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_topk_real.argtypes = [ctypes.c_void_p, c_size, c_size]
    lib.scl_topk_real.restype = ctypes.c_int
    
    status = lib.scl_topk_real(data, size, k)
    check_error(status, "topk_real")


def topk_real_with_indices(
    keys: Any,
    size: int,
    k: int,
    out_values: Any,
    out_indices: Any
) -> None:
    """Get top-k elements with their original indices.
    
    Args:
        keys: Input key array pointer.
        size: Total number of elements.
        k: Number of top elements to select.
        out_values: Output top-k values [k].
        out_indices: Output original indices [k].
        
    Raises:
        RuntimeError: If operation fails.
    """
    lib = get_lib()
    lib.scl_topk_real_with_indices.argtypes = [
        ctypes.c_void_p, c_size, c_size, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_topk_real_with_indices.restype = ctypes.c_int
    
    status = lib.scl_topk_real_with_indices(keys, size, k, out_values, out_indices)
    check_error(status, "topk_real_with_indices")


# =============================================================================
# Utility
# =============================================================================

def is_sorted_real_ascending(data: Any, size: int) -> bool:
    """Check if array is sorted in ascending order.
    
    Args:
        data: Data array pointer.
        size: Number of elements.
        
    Returns:
        True if sorted, False otherwise.
        
    Raises:
        RuntimeError: If check fails.
    """
    lib = get_lib()
    lib.scl_is_sorted_real_ascending.argtypes = [
        ctypes.c_void_p, c_size, ctypes.POINTER(ctypes.c_int)
    ]
    lib.scl_is_sorted_real_ascending.restype = ctypes.c_int
    
    is_sorted = ctypes.c_int()
    status = lib.scl_is_sorted_real_ascending(data, size, ctypes.byref(is_sorted))
    check_error(status, "is_sorted_real_ascending")
    return bool(is_sorted.value)


def is_sorted_real_descending(data: Any, size: int) -> bool:
    """Check if array is sorted in descending order.
    
    Args:
        data: Data array pointer.
        size: Number of elements.
        
    Returns:
        True if sorted, False otherwise.
        
    Raises:
        RuntimeError: If check fails.
    """
    lib = get_lib()
    lib.scl_is_sorted_real_descending.argtypes = [
        ctypes.c_void_p, c_size, ctypes.POINTER(ctypes.c_int)
    ]
    lib.scl_is_sorted_real_descending.restype = ctypes.c_int
    
    is_sorted = ctypes.c_int()
    status = lib.scl_is_sorted_real_descending(data, size, ctypes.byref(is_sorted))
    check_error(status, "is_sorted_real_descending")
    return bool(is_sorted.value)


def argsort_buffer_size(size: int) -> int:
    """Get required buffer size for buffered argsort.
    
    Args:
        size: Number of elements to sort.
        
    Returns:
        Required buffer size in bytes.
    """
    lib = get_lib()
    lib.scl_argsort_buffer_size.argtypes = [c_size]
    lib.scl_argsort_buffer_size.restype = c_size
    
    return lib.scl_argsort_buffer_size(size)

