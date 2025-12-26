"""
Matrix Utilities

Low-level C bindings for high-performance matrix manipulation operations.
"""

import ctypes
from .lib_loader import get_lib
from .types import c_real, c_index, c_size, c_byte, check_error, as_c_ptr

__all__ = [
    'compute_lengths',
    'inspect_slice_rows',
    'materialize_slice_rows',
    'inspect_filter_cols',
    'materialize_filter_cols',
    'align_rows',
]

# =============================================================================
# Function Signatures
# =============================================================================

def _init_signatures():
    """Initialize C function signatures."""
    lib = get_lib()
    
    # compute_lengths
    lib.scl_compute_lengths.argtypes = [
        ctypes.POINTER(c_index),  # indptr
        c_index,                   # n
        ctypes.POINTER(c_index),  # lengths
    ]
    lib.scl_compute_lengths.restype = ctypes.c_int
    
    # inspect_slice_rows
    lib.scl_inspect_slice_rows.argtypes = [
        ctypes.POINTER(c_index),  # indptr
        ctypes.POINTER(c_index),  # row_indices
        c_index,                   # n_keep
        ctypes.POINTER(c_index),  # out_nnz
    ]
    lib.scl_inspect_slice_rows.restype = ctypes.c_int
    
    # materialize_slice_rows
    lib.scl_materialize_slice_rows.argtypes = [
        ctypes.POINTER(c_real),   # src_data
        ctypes.POINTER(c_index),  # src_indices
        ctypes.POINTER(c_index),  # src_indptr
        ctypes.POINTER(c_index),  # row_indices
        c_index,                   # n_keep
        ctypes.POINTER(c_real),   # dst_data
        ctypes.POINTER(c_index),  # dst_indices
        ctypes.POINTER(c_index),  # dst_indptr
    ]
    lib.scl_materialize_slice_rows.restype = ctypes.c_int
    
    # inspect_filter_cols
    lib.scl_inspect_filter_cols.argtypes = [
        ctypes.POINTER(c_index),  # src_indices
        ctypes.POINTER(c_index),  # src_indptr
        c_index,                   # rows
        ctypes.POINTER(c_byte),   # col_mask
        ctypes.POINTER(c_index),  # out_nnz
    ]
    lib.scl_inspect_filter_cols.restype = ctypes.c_int
    
    # materialize_filter_cols
    lib.scl_materialize_filter_cols.argtypes = [
        ctypes.POINTER(c_real),   # src_data
        ctypes.POINTER(c_index),  # src_indices
        ctypes.POINTER(c_index),  # src_indptr
        c_index,                   # rows
        ctypes.POINTER(c_byte),   # col_mask
        ctypes.POINTER(c_index),  # col_mapping
        ctypes.POINTER(c_real),   # dst_data
        ctypes.POINTER(c_index),  # dst_indices
        ctypes.POINTER(c_index),  # dst_indptr
    ]
    lib.scl_materialize_filter_cols.restype = ctypes.c_int
    
    # align_rows
    lib.scl_align_rows.argtypes = [
        ctypes.POINTER(c_real),   # src_data
        ctypes.POINTER(c_index),  # src_indices
        ctypes.POINTER(c_index),  # src_indptr
        c_index,                   # src_rows
        ctypes.POINTER(c_index),  # old_to_new
        c_index,                   # new_rows
        ctypes.POINTER(c_real),   # dst_data
        ctypes.POINTER(c_index),  # dst_indices
        ctypes.POINTER(c_index),  # dst_indptr
        ctypes.POINTER(c_index),  # dst_row_lengths
    ]
    lib.scl_align_rows.restype = ctypes.c_int

# Initialize on module load
_init_signatures()

# =============================================================================
# Python Functions
# =============================================================================

def compute_lengths(indptr, n, lengths):
    """
    Compute row/column lengths from indptr (parallel diff).
    
    Args:
        indptr: ctypes pointer to indptr array [n+1]
        n: Number of rows/columns
        lengths: ctypes pointer to output array [n]
    """
    lib = get_lib()
    status = lib.scl_compute_lengths(indptr, n, lengths)
    check_error(status)


def inspect_slice_rows(indptr, row_indices, n_keep):
    """
    Inspect CSR row slice: compute output nnz.
    
    Args:
        indptr: ctypes pointer to indptr array
        row_indices: ctypes pointer to row indices to keep
        n_keep: Number of rows to keep
        
    Returns:
        Output nnz
    """
    lib = get_lib()
    out_nnz = c_index()
    status = lib.scl_inspect_slice_rows(indptr, row_indices, n_keep, ctypes.byref(out_nnz))
    check_error(status)
    return out_nnz.value


def materialize_slice_rows(src_data, src_indices, src_indptr, row_indices, n_keep,
                           dst_data, dst_indices, dst_indptr):
    """
    Materialize CSR row slice: copy selected rows.
    
    Args:
        src_data: ctypes pointer to source data
        src_indices: ctypes pointer to source indices
        src_indptr: ctypes pointer to source indptr
        row_indices: ctypes pointer to rows to keep
        n_keep: Number of rows to keep
        dst_data: ctypes pointer to destination data
        dst_indices: ctypes pointer to destination indices
        dst_indptr: ctypes pointer to destination indptr
    """
    lib = get_lib()
    status = lib.scl_materialize_slice_rows(
        src_data, src_indices, src_indptr, row_indices, n_keep,
        dst_data, dst_indices, dst_indptr
    )
    check_error(status)


def inspect_filter_cols(src_indices, src_indptr, rows, col_mask):
    """
    Inspect CSR column filter: compute output nnz.
    
    Args:
        src_indices: ctypes pointer to source indices
        src_indptr: ctypes pointer to source indptr
        rows: Number of rows
        col_mask: ctypes pointer to column mask (uint8)
        
    Returns:
        Output nnz
    """
    lib = get_lib()
    out_nnz = c_index()
    status = lib.scl_inspect_filter_cols(src_indices, src_indptr, rows, col_mask, ctypes.byref(out_nnz))
    check_error(status)
    return out_nnz.value


def materialize_filter_cols(src_data, src_indices, src_indptr, rows, col_mask, col_mapping,
                            dst_data, dst_indices, dst_indptr):
    """
    Materialize CSR column filter: copy filtered columns.
    
    Args:
        src_data: ctypes pointer to source data
        src_indices: ctypes pointer to source indices
        src_indptr: ctypes pointer to source indptr
        rows: Number of rows
        col_mask: ctypes pointer to column mask
        col_mapping: ctypes pointer to column mapping
        dst_data: ctypes pointer to destination data
        dst_indices: ctypes pointer to destination indices
        dst_indptr: ctypes pointer to destination indptr
    """
    lib = get_lib()
    status = lib.scl_materialize_filter_cols(
        src_data, src_indices, src_indptr, rows, col_mask, col_mapping,
        dst_data, dst_indices, dst_indptr
    )
    check_error(status)


def align_rows(src_data, src_indices, src_indptr, src_rows, old_to_new, new_rows,
               dst_data, dst_indices, dst_indptr, dst_row_lengths):
    """
    Align rows (with drop and pad support).
    
    Args:
        src_data: ctypes pointer to source data
        src_indices: ctypes pointer to source indices
        src_indptr: ctypes pointer to source indptr
        src_rows: Number of source rows
        old_to_new: ctypes pointer to mapping array, -1=drop
        new_rows: Target number of rows
        dst_data: ctypes pointer to destination data
        dst_indices: ctypes pointer to destination indices
        dst_indptr: ctypes pointer to destination indptr
        dst_row_lengths: ctypes pointer to destination row lengths
    """
    lib = get_lib()
    status = lib.scl_align_rows(
        src_data, src_indices, src_indptr, src_rows, old_to_new, new_rows,
        dst_data, dst_indices, dst_indptr, dst_row_lengths
    )
    check_error(status)

