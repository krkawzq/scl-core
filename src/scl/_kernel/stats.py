"""
Statistical Test Kernels

Low-level C bindings for Mann-Whitney U test, T-test, etc.
"""

import ctypes
import numpy as np
from .lib_loader import get_lib
from .types import c_real, c_index, c_size, c_byte, np_real, check_error, as_c_ptr

__all__ = [
    'mwu_test_csc',
    'mwu_test_csr',
    'ttest_csc',
]

# =============================================================================
# Function Signatures
# =============================================================================

def _init_signatures():
    """Initialize C function signatures."""
    lib = get_lib()
    
    # mwu_test_csc
    lib.scl_mwu_test_csc.argtypes = [
        ctypes.POINTER(c_real),    # data
        ctypes.POINTER(c_index),   # indices
        ctypes.POINTER(c_index),   # indptr
        ctypes.POINTER(c_index),   # col_lengths
        c_index,                    # rows
        c_index,                    # cols
        c_index,                    # nnz
        ctypes.POINTER(ctypes.c_int32),  # group_ids
        ctypes.POINTER(c_real),    # out_u_stats
        ctypes.POINTER(c_real),    # out_p_values
        ctypes.POINTER(c_real),    # out_log2_fc
    ]
    lib.scl_mwu_test_csc.restype = ctypes.c_int
    
    # mwu_test_csr
    lib.scl_mwu_test_csr.argtypes = [
        ctypes.POINTER(c_real),    # data
        ctypes.POINTER(c_index),   # indices
        ctypes.POINTER(c_index),   # indptr
        ctypes.POINTER(c_index),   # row_lengths
        c_index,                    # rows
        c_index,                    # cols
        c_index,                    # nnz
        ctypes.POINTER(ctypes.c_int32),  # group_ids
        ctypes.POINTER(c_real),    # out_u_stats
        ctypes.POINTER(c_real),    # out_p_values
        ctypes.POINTER(c_real),    # out_log2_fc
    ]
    lib.scl_mwu_test_csr.restype = ctypes.c_int
    
    # ttest_csc
    lib.scl_ttest_csc.argtypes = [
        ctypes.POINTER(c_real),    # data
        ctypes.POINTER(c_index),   # indices
        ctypes.POINTER(c_index),   # indptr
        ctypes.POINTER(c_index),   # col_lengths
        c_index,                    # rows
        c_index,                    # cols
        c_index,                    # nnz
        ctypes.POINTER(ctypes.c_int32),  # group_ids
        c_size,                     # n_groups
        ctypes.POINTER(c_real),    # out_t_stats
        ctypes.POINTER(c_real),    # out_p_values
        ctypes.POINTER(c_real),    # out_log2_fc
        ctypes.POINTER(c_real),    # out_mean_diff
        ctypes.POINTER(c_byte),    # workspace
        c_size,                     # workspace_size
        ctypes.c_bool,             # use_welch
    ]
    lib.scl_ttest_csc.restype = ctypes.c_int


# Initialize signatures lazily
try:
    _init_signatures()
except Exception as e:
    import warnings
    warnings.warn(f"SCL library not ready: {e}")

# =============================================================================
# Python Wrappers
# =============================================================================

def mwu_test_csc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    col_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    group_ids: np.ndarray,
    out_u_stats: np.ndarray,
    out_p_values: np.ndarray,
    out_log2_fc: np.ndarray
) -> None:
    """
    Mann-Whitney U test for each gene (CSC matrix).
    
    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        col_lengths: Explicit column lengths or None
        rows: Number of cells
        cols: Number of genes
        nnz: Number of non-zeros
        group_ids: Cell group labels (0 or 1), shape (rows,)
        out_u_stats: Output U statistics, shape (cols,)
        out_p_values: Output P-values, shape (cols,)
        out_log2_fc: Output log2 fold changes, shape (cols,)
    """
    lib = get_lib()
    
    status = lib.scl_mwu_test_csc(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(col_lengths, c_index) if col_lengths is not None else None,
        rows, cols, nnz,
        as_c_ptr(group_ids, ctypes.c_int32),
        as_c_ptr(out_u_stats, c_real),
        as_c_ptr(out_p_values, c_real),
        as_c_ptr(out_log2_fc, c_real)
    )
    
    check_error(status, "mwu_test_csc")


def mwu_test_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    row_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    group_ids: np.ndarray,
    out_u_stats: np.ndarray,
    out_p_values: np.ndarray,
    out_log2_fc: np.ndarray
) -> None:
    """
    Mann-Whitney U test for each sample (CSR matrix).
    
    Args:
        data: CSR data array
        indices: CSR column indices
        indptr: CSR row pointers
        row_lengths: Explicit row lengths or None
        rows: Number of samples
        cols: Number of features
        nnz: Number of non-zeros
        group_ids: Feature group labels (0 or 1), shape (cols,)
        out_u_stats: Output U statistics, shape (rows,)
        out_p_values: Output P-values, shape (rows,)
        out_log2_fc: Output log2 fold changes, shape (rows,)
    """
    lib = get_lib()
    
    status = lib.scl_mwu_test_csr(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(row_lengths, c_index) if row_lengths is not None else None,
        rows, cols, nnz,
        as_c_ptr(group_ids, ctypes.c_int32),
        as_c_ptr(out_u_stats, c_real),
        as_c_ptr(out_p_values, c_real),
        as_c_ptr(out_log2_fc, c_real)
    )
    
    check_error(status, "mwu_test_csr")


def ttest_csc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    col_lengths: np.ndarray,
    rows: int,
    cols: int,
    nnz: int,
    group_ids: np.ndarray,
    n_groups: int,
    out_t_stats: np.ndarray,
    out_p_values: np.ndarray,
    out_log2_fc: np.ndarray,
    out_mean_diff: np.ndarray,
    workspace: np.ndarray,
    use_welch: bool = True
) -> None:
    """
    T-test for differential expression (CSC matrix).
    
    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        col_lengths: Explicit column lengths or None
        rows: Number of cells
        cols: Number of genes
        nnz: Number of non-zeros
        group_ids: Cell group labels, shape (rows,)
        n_groups: Total number of groups
        out_t_stats: Output T statistics, shape (cols, n_groups-1)
        out_p_values: Output P-values, shape (cols, n_groups-1)
        out_log2_fc: Output log2 fold changes, shape (cols, n_groups-1)
        out_mean_diff: Output mean differences, shape (cols, n_groups-1)
        workspace: Temporary workspace buffer
        use_welch: Use Welch's t-test (unequal variance)
    """
    lib = get_lib()
    
    status = lib.scl_ttest_csc(
        as_c_ptr(data, c_real),
        as_c_ptr(indices, c_index),
        as_c_ptr(indptr, c_index),
        as_c_ptr(col_lengths, c_index) if col_lengths is not None else None,
        rows, cols, nnz,
        as_c_ptr(group_ids, ctypes.c_int32),
        n_groups,
        as_c_ptr(out_t_stats, c_real),
        as_c_ptr(out_p_values, c_real),
        as_c_ptr(out_log2_fc, c_real),
        as_c_ptr(out_mean_diff, c_real),
        as_c_ptr(workspace, c_byte),
        workspace.nbytes,
        use_welch
    )
    
    check_error(status, "ttest_csc")

