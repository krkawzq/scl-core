"""Group aggregation kernels."""
import ctypes
from typing import Any, Optional
from .lib_loader import get_lib
from .types import c_real, c_index, c_size, check_error

__all__ = ['group_stats_csc', 'count_group_sizes']

def group_stats_csc(data: Any, indices: Any, indptr: Any, col_lengths: Optional[Any],
                    rows: int, cols: int, nnz: int, group_ids: Any, n_groups: int,
                    group_sizes: Any, out_means: Any, out_vars: Any, ddof: int, include_zeros: bool) -> None:
    """Compute group statistics (mean and variance) for CSC matrix."""
    lib = get_lib()
    lib.scl_group_stats_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, c_index, c_index, ctypes.POINTER(ctypes.c_int32), c_size, ctypes.POINTER(c_size),
        ctypes.POINTER(c_real), ctypes.POINTER(c_real), ctypes.c_int, ctypes.c_bool
    ]
    lib.scl_group_stats_csc.restype = ctypes.c_int
    status = lib.scl_group_stats_csc(data, indices, indptr, col_lengths, rows, cols, nnz,
                                      group_ids, n_groups, group_sizes, out_means, out_vars, ddof, include_zeros)
    check_error(status, "group_stats_csc")

def count_group_sizes(group_ids: Any, n_elements: int, n_groups: int, out_sizes: Any) -> None:
    """Count group sizes from group labels."""
    lib = get_lib()
    lib.scl_count_group_sizes.argtypes = [ctypes.POINTER(ctypes.c_int32), c_size, c_size, ctypes.POINTER(c_size)]
    lib.scl_count_group_sizes.restype = ctypes.c_int
    status = lib.scl_count_group_sizes(group_ids, n_elements, n_groups, out_sizes)
    check_error(status, "count_group_sizes")
