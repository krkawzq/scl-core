"""Statistical test kernels.

Low-level C bindings for Mann-Whitney U test, T-test, etc.
"""

import ctypes
from typing import Any

from .lib_loader import get_lib
from .types import c_real, c_index, c_size, check_error


__all__ = ['mwu_test_csc', 'ttest_csc']


def mwu_test_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    group_ids: Any,
    out_u_stats: Any,
    out_p_values: Any,
    out_log2_fc: Any
) -> None:
    """Mann-Whitney U test for differential expression (CSC).
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows (cells).
        cols: Number of columns (genes).
        group_ids: Group labels pointer (0 or 1) [rows].
        out_u_stats: Output U statistics pointer [cols].
        out_p_values: Output P-values pointer [cols].
        out_log2_fc: Output log2 fold changes pointer [cols].
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_mwu_test_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mwu_test_csc.restype = ctypes.c_int
    
    status = lib.scl_mwu_test_csc(
        data, indices, indptr, rows, cols,
        group_ids, out_u_stats, out_p_values, out_log2_fc
    )
    check_error(status, "mwu_test_csc")


def ttest_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    group_ids: Any,
    n_groups: int,
    out_t_stats: Any,
    out_p_values: Any,
    out_log2_fc: Any,
    use_welch: bool
) -> None:
    """T-test for differential expression with multiple groups (CSC).
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        rows: Number of rows (cells).
        cols: Number of columns (genes).
        group_ids: Group labels pointer (0 to n_groups-1) [rows].
        n_groups: Number of groups.
        out_t_stats: Output T statistics pointer [cols].
        out_p_values: Output P-values pointer [cols].
        out_log2_fc: Output log2 fold changes pointer [cols].
        use_welch: Use Welch's t-test (unequal variances).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_ttest_csc.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index,
        ctypes.c_void_p, c_size,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_bool
    ]
    lib.scl_ttest_csc.restype = ctypes.c_int
    
    status = lib.scl_ttest_csc(
        data, indices, indptr, rows, cols,
        group_ids, n_groups,
        out_t_stats, out_p_values, out_log2_fc,
        use_welch
    )
    check_error(status, "ttest_csc")
