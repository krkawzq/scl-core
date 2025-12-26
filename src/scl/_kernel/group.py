"""Group aggregation kernels.

Low-level C bindings for group-wise statistics computation.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, c_size, check_error


__all__ = [
    'group_stats_csc',
    'count_group_sizes',
]


def group_stats_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    group_ids: Any,
    n_groups: int,
    group_sizes: Any,
    out_means: Any,
    out_vars: Any,
    ddof: int,
    include_zeros: bool
) -> None:
    """Compute per-group statistics for each feature.
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows (cells).
        cols: Number of columns (genes).
        group_ids: Group IDs for each cell (0 to n_groups-1) [rows].
        n_groups: Number of groups.
        group_sizes: Pre-computed group sizes [n_groups].
        out_means: Output means array [cols * n_groups].
        out_vars: Output variances array [cols * n_groups].
        ddof: Delta degrees of freedom.
        include_zeros: Whether to include zeros in mean/var computation.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_group_stats_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, ctypes.c_void_p, c_size, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool
    ]
    lib.scl_group_stats_csc.restype = ctypes.c_int
    
    status = lib.scl_group_stats_csc(
        data, indices, indptr, rows, cols, group_ids, n_groups, group_sizes,
        out_means, out_vars, ddof, include_zeros
    )
    check_error(status, "group_stats_csc")


def count_group_sizes(
    group_ids: Any,
    n_elements: int,
    n_groups: int,
    out_sizes: Any
) -> None:
    """Count the number of elements in each group.
    
    Args:
        group_ids: Group IDs array (0 to n_groups-1) [n_elements].
        n_elements: Number of elements.
        n_groups: Number of groups.
        out_sizes: Output array for group sizes [n_groups].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_count_group_sizes.argtypes = [
        ctypes.c_void_p, c_size, c_size, ctypes.c_void_p
    ]
    lib.scl_count_group_sizes.restype = ctypes.c_int
    
    status = lib.scl_count_group_sizes(group_ids, n_elements, n_groups, out_sizes)
    check_error(status, "count_group_sizes")
