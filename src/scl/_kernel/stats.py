"""Statistical test kernels.

Low-level C bindings for Mann-Whitney U test, T-test, etc.
"""

import ctypes
from typing import Any, Optional

from .lib_loader import get_lib
from .types import c_real, c_index, c_size, c_byte, check_error


__all__ = ['mwu_test_csc', 'ttest_csc']


def mwu_test_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    col_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
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
        col_lengths: Explicit column lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zeros.
        group_ids: Group labels pointer (0 or 1).
        out_u_stats: Output U statistics pointer.
        out_p_values: Output P-values pointer.
        out_log2_fc: Output log2 fold changes pointer.
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_mwu_test_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(c_real), ctypes.POINTER(c_real), ctypes.POINTER(c_real)
    ]
    lib.scl_mwu_test_csc.restype = ctypes.c_int
    
    status = lib.scl_mwu_test_csc(
        data, indices, indptr, col_lengths, rows, cols, nnz,
        group_ids, out_u_stats, out_p_values, out_log2_fc
    )
    check_error(status, "mwu_test_csc")


def ttest_csc(
    data: Any,
    indices: Any,
    indptr: Any,
    col_lengths: Optional[Any],
    rows: int,
    cols: int,
    nnz: int,
    group_ids: Any,
    n_groups: int,
    out_t_stats: Any,
    out_p_values: Any,
    out_log2_fc: Any,
    out_mean_diff: Any,
    workspace: Any,
    workspace_size: int,
    use_welch: bool
) -> None:
    """T-test for differential expression with multiple groups (CSC).
    
    Args:
        data: CSC data array pointer.
        indices: CSC row indices pointer.
        indptr: CSC column pointers pointer.
        col_lengths: Explicit column lengths pointer or None.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zeros.
        group_ids: Group labels pointer (0 to n_groups-1).
        n_groups: Number of groups.
        out_t_stats: Output T statistics pointer.
        out_p_values: Output P-values pointer.
        out_log2_fc: Output log2 fold changes pointer.
        out_mean_diff: Output mean differences pointer.
        workspace: Workspace buffer pointer.
        workspace_size: Size of workspace buffer.
        use_welch: Use Welch's t-test (unequal variances).
        
    Raises:
        RuntimeError: If C function fails.
    """
    lib = get_lib()
    lib.scl_ttest_csc.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index, c_index, c_index,
        ctypes.POINTER(ctypes.c_int32), c_size,
        ctypes.POINTER(c_real), ctypes.POINTER(c_real), ctypes.POINTER(c_real), ctypes.POINTER(c_real),
        ctypes.POINTER(c_byte), c_size, ctypes.c_bool
    ]
    lib.scl_ttest_csc.restype = ctypes.c_int
    
    status = lib.scl_ttest_csc(
        data, indices, indptr, col_lengths, rows, cols, nnz,
        group_ids, n_groups,
        out_t_stats, out_p_values, out_log2_fc, out_mean_diff,
        workspace, workspace_size, use_welch
    )
    check_error(status, "ttest_csc")
