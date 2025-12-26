"""Matrix utility functions.

Low-level C bindings for matrix manipulation operations.

Note: These functions are not yet in the main C API (c_api.cpp).
      They may be implemented in separate utility modules or need to be added.
      
This module provides placeholder implementations that will be connected
to C functions once they are added to the C API.
"""

import ctypes
from typing import Any, Optional

from .lib_loader import get_lib
from .types import c_real, c_index, check_error


__all__ = [
    'compute_lengths',
    'inspect_slice_rows',
    'materialize_slice_rows',
    'inspect_filter_cols',
    'materialize_filter_cols',
    'align_rows',
]


def compute_lengths(indptr: Any, n: int, lengths: Any) -> None:
    """Compute row/column lengths from indptr (parallel diff).
    
    Note: This function is not yet in c_api.cpp and needs to be added.
    
    Args:
        indptr: Indptr array pointer [n+1].
        n: Number of rows/columns.
        lengths: Output lengths pointer [n].
        
    Raises:
        RuntimeError: If C function fails.
        NotImplementedError: If C function not available.
    """
    lib = get_lib()
    
    if not hasattr(lib, 'scl_compute_lengths'):
        raise NotImplementedError(
            "scl_compute_lengths not found in C API. "
            "This function needs to be added to scl/binding/c_api.cpp"
        )
    
    lib.scl_compute_lengths.argtypes = [
        ctypes.POINTER(c_index), c_index, ctypes.POINTER(c_index)
    ]
    lib.scl_compute_lengths.restype = ctypes.c_int
    
    status = lib.scl_compute_lengths(indptr, n, lengths)
    check_error(status, "compute_lengths")


def inspect_slice_rows(indptr: Any, row_indices: Any, n_keep: int) -> int:
    """Inspect CSR row slice: compute output nnz.
    
    Note: This function is not yet in c_api.cpp and needs to be added.
    
    Args:
        indptr: Indptr array pointer.
        row_indices: Row indices to keep pointer.
        n_keep: Number of rows to keep.
        
    Returns:
        Output nnz.
        
    Raises:
        RuntimeError: If C function fails.
        NotImplementedError: If C function not available.
    """
    lib = get_lib()
    
    if not hasattr(lib, 'scl_inspect_slice_rows'):
        raise NotImplementedError(
            "scl_inspect_slice_rows not found in C API. "
            "This function needs to be added to scl/binding/c_api.cpp"
        )
    
    lib.scl_inspect_slice_rows.argtypes = [
        ctypes.POINTER(c_index), ctypes.POINTER(c_index), c_index, ctypes.POINTER(c_index)
    ]
    lib.scl_inspect_slice_rows.restype = ctypes.c_int
    
    out_nnz = c_index()
    status = lib.scl_inspect_slice_rows(indptr, row_indices, n_keep, ctypes.byref(out_nnz))
    check_error(status, "inspect_slice_rows")
    return out_nnz.value


def materialize_slice_rows(
    src_data: Any,
    src_indices: Any,
    src_indptr: Any,
    row_indices: Any,
    n_keep: int,
    dst_data: Any,
    dst_indices: Any,
    dst_indptr: Any
) -> None:
    """Materialize CSR row slice: copy selected rows.
    
    Note: This function is not yet in c_api.cpp and needs to be added.
    
    Args:
        src_data: Source data pointer.
        src_indices: Source indices pointer.
        src_indptr: Source indptr pointer.
        row_indices: Rows to keep pointer.
        n_keep: Number of rows to keep.
        dst_data: Destination data pointer.
        dst_indices: Destination indices pointer.
        dst_indptr: Destination indptr pointer.
        
    Raises:
        RuntimeError: If C function fails.
        NotImplementedError: If C function not available.
    """
    lib = get_lib()
    
    if not hasattr(lib, 'scl_materialize_slice_rows'):
        raise NotImplementedError(
            "scl_materialize_slice_rows not found in C API. "
            "This function needs to be added to scl/binding/c_api.cpp"
        )
    
    lib.scl_materialize_slice_rows.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index), c_index,
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index)
    ]
    lib.scl_materialize_slice_rows.restype = ctypes.c_int
    
    status = lib.scl_materialize_slice_rows(
        src_data, src_indices, src_indptr, row_indices, n_keep,
        dst_data, dst_indices, dst_indptr
    )
    check_error(status, "materialize_slice_rows")


def inspect_filter_cols(
    src_indices: Any,
    src_indptr: Any,
    rows: int,
    col_mask: Any
) -> int:
    """Inspect CSR column filter: compute output nnz.
    
    Note: This function is not yet in c_api.cpp and needs to be added.
    
    Args:
        src_indices: Source indices pointer.
        src_indptr: Source indptr pointer.
        rows: Number of rows.
        col_mask: Column mask pointer (uint8).
        
    Returns:
        Output nnz.
        
    Raises:
        RuntimeError: If C function fails.
        NotImplementedError: If C function not available.
    """
    lib = get_lib()
    
    if not hasattr(lib, 'scl_inspect_filter_cols'):
        raise NotImplementedError(
            "scl_inspect_filter_cols not found in C API. "
            "This function needs to be added to scl/binding/c_api.cpp"
        )
    
    lib.scl_inspect_filter_cols.argtypes = [
        ctypes.POINTER(c_index), ctypes.POINTER(c_index), c_index,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(c_index)
    ]
    lib.scl_inspect_filter_cols.restype = ctypes.c_int
    
    out_nnz = c_index()
    status = lib.scl_inspect_filter_cols(src_indices, src_indptr, rows, col_mask, ctypes.byref(out_nnz))
    check_error(status, "inspect_filter_cols")
    return out_nnz.value


def materialize_filter_cols(
    src_data: Any,
    src_indices: Any,
    src_indptr: Any,
    rows: int,
    col_mask: Any,
    col_mapping: Any,
    dst_data: Any,
    dst_indices: Any,
    dst_indptr: Any
) -> None:
    """Materialize CSR column filter: copy filtered columns.
    
    Note: This function is not yet in c_api.cpp and needs to be added.
    
    Args:
        src_data: Source data pointer.
        src_indices: Source indices pointer.
        src_indptr: Source indptr pointer.
        rows: Number of rows.
        col_mask: Column mask pointer.
        col_mapping: Column mapping pointer.
        dst_data: Destination data pointer.
        dst_indices: Destination indices pointer.
        dst_indptr: Destination indptr pointer.
        
    Raises:
        RuntimeError: If C function fails.
        NotImplementedError: If C function not available.
    """
    lib = get_lib()
    
    if not hasattr(lib, 'scl_materialize_filter_cols'):
        raise NotImplementedError(
            "scl_materialize_filter_cols not found in C API. "
            "This function needs to be added to scl/binding/c_api.cpp"
        )
    
    lib.scl_materialize_filter_cols.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(c_index),
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index)
    ]
    lib.scl_materialize_filter_cols.restype = ctypes.c_int
    
    status = lib.scl_materialize_filter_cols(
        src_data, src_indices, src_indptr, rows, col_mask, col_mapping,
        dst_data, dst_indices, dst_indptr
    )
    check_error(status, "materialize_filter_cols")


def align_rows(
    src_data: Any,
    src_indices: Any,
    src_indptr: Any,
    src_rows: int,
    old_to_new: Any,
    new_rows: int,
    dst_data: Any,
    dst_indices: Any,
    dst_indptr: Any,
    dst_row_lengths: Any
) -> None:
    """Align rows (with drop and pad support).
    
    Note: This function is not yet in c_api.cpp and needs to be added.
    
    Args:
        src_data: Source data pointer.
        src_indices: Source indices pointer.
        src_indptr: Source indptr pointer.
        src_rows: Number of source rows.
        old_to_new: Mapping array pointer, -1=drop.
        new_rows: Target number of rows.
        dst_data: Destination data pointer.
        dst_indices: Destination indices pointer.
        dst_indptr: Destination indptr pointer.
        dst_row_lengths: Destination row lengths pointer.
        
    Raises:
        RuntimeError: If C function fails.
        NotImplementedError: If C function not available.
    """
    lib = get_lib()
    
    if not hasattr(lib, 'scl_align_rows'):
        raise NotImplementedError(
            "scl_align_rows not found in C API. "
            "This function needs to be added to scl/binding/c_api.cpp"
        )
    
    lib.scl_align_rows.argtypes = [
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        c_index, ctypes.POINTER(c_index), c_index,
        ctypes.POINTER(c_real), ctypes.POINTER(c_index), ctypes.POINTER(c_index),
        ctypes.POINTER(c_index)
    ]
    lib.scl_align_rows.restype = ctypes.c_int
    
    status = lib.scl_align_rows(
        src_data, src_indices, src_indptr, src_rows, old_to_new, new_rows,
        dst_data, dst_indices, dst_indptr, dst_row_lengths
    )
    check_error(status, "align_rows")

