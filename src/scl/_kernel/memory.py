"""Memory management kernels."""
import ctypes
from .lib_loader import get_lib
from .types import c_size, check_error

__all__ = ['malloc', 'calloc', 'malloc_aligned', 'free', 'free_aligned', 'memzero', 'memcpy',
           'is_valid_value', 'sizeof_real', 'sizeof_index', 'alignment',
           'ttest_workspace_size', 'diff_expr_output_size', 'group_stats_output_size',
           'gram_output_size', 'correlation_workspace_size']

def malloc(bytes: int) -> int:
    """Allocate memory."""
    lib = get_lib()
    lib.scl_malloc.argtypes = [c_size, ctypes.POINTER(ctypes.c_void_p)]
    lib.scl_malloc.restype = ctypes.c_int
    ptr = ctypes.c_void_p()
    status = lib.scl_malloc(bytes, ctypes.byref(ptr))
    check_error(status, "malloc")
    return ptr.value

def calloc(bytes: int) -> int:
    """Allocate zero-initialized memory."""
    lib = get_lib()
    lib.scl_calloc.argtypes = [c_size, ctypes.POINTER(ctypes.c_void_p)]
    lib.scl_calloc.restype = ctypes.c_int
    ptr = ctypes.c_void_p()
    status = lib.scl_calloc(bytes, ctypes.byref(ptr))
    check_error(status, "calloc")
    return ptr.value

def malloc_aligned(bytes: int, alignment: int) -> int:
    """Allocate aligned memory."""
    lib = get_lib()
    lib.scl_malloc_aligned.argtypes = [c_size, c_size, ctypes.POINTER(ctypes.c_void_p)]
    lib.scl_malloc_aligned.restype = ctypes.c_int
    ptr = ctypes.c_void_p()
    status = lib.scl_malloc_aligned(bytes, alignment, ctypes.byref(ptr))
    check_error(status, "malloc_aligned")
    return ptr.value

def free(ptr: int) -> None:
    """Free memory."""
    lib = get_lib()
    lib.scl_free.argtypes = [ctypes.c_void_p]
    lib.scl_free(ptr)

def free_aligned(ptr: int) -> None:
    """Free aligned memory."""
    lib = get_lib()
    lib.scl_free_aligned.argtypes = [ctypes.c_void_p]
    lib.scl_free_aligned(ptr)

def memzero(ptr: int, bytes: int) -> None:
    """Zero out memory."""
    lib = get_lib()
    lib.scl_memzero.argtypes = [ctypes.c_void_p, c_size]
    lib.scl_memzero(ptr, bytes)

def memcpy(src: int, dst: int, bytes: int) -> None:
    """Copy memory."""
    lib = get_lib()
    lib.scl_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, c_size]
    lib.scl_memcpy.restype = ctypes.c_int
    status = lib.scl_memcpy(src, dst, bytes)
    check_error(status, "memcpy")

def is_valid_value(value: float) -> bool:
    """Check if value is finite."""
    lib = get_lib()
    lib.scl_is_valid_value.restype = ctypes.c_bool
    return lib.scl_is_valid_value(value)

def sizeof_real() -> int:
    """Get size of Real type in bytes."""
    lib = get_lib()
    lib.scl_sizeof_real.restype = c_size
    return lib.scl_sizeof_real()

def sizeof_index() -> int:
    """Get size of Index type in bytes."""
    lib = get_lib()
    lib.scl_sizeof_index.restype = c_size
    return lib.scl_sizeof_index()

def alignment() -> int:
    """Get recommended memory alignment in bytes."""
    lib = get_lib()
    lib.scl_alignment.restype = c_size
    return lib.scl_alignment()

def ttest_workspace_size(n_features: int, n_groups: int) -> int:
    """Calculate workspace size for T-test."""
    lib = get_lib()
    lib.scl_ttest_workspace_size.restype = c_size
    return lib.scl_ttest_workspace_size(n_features, n_groups)

def diff_expr_output_size(n_features: int, n_groups: int) -> int:
    """Calculate output size for differential expression tests."""
    lib = get_lib()
    lib.scl_diff_expr_output_size.restype = c_size
    return lib.scl_diff_expr_output_size(n_features, n_groups)

def group_stats_output_size(n_features: int, n_groups: int) -> int:
    """Calculate output size for group statistics."""
    lib = get_lib()
    lib.scl_group_stats_output_size.restype = c_size
    return lib.scl_group_stats_output_size(n_features, n_groups)

def gram_output_size(n: int) -> int:
    """Calculate output size for Gram matrix."""
    lib = get_lib()
    lib.scl_gram_output_size.restype = c_size
    return lib.scl_gram_output_size(n)

def correlation_workspace_size(n: int) -> int:
    """Calculate workspace size for correlation."""
    lib = get_lib()
    lib.scl_correlation_workspace_size.restype = c_size
    return lib.scl_correlation_workspace_size(n)
