"""Memory-mapped sparse matrix kernels.

Low-level C bindings for memory-mapped (out-of-core) sparse matrix operations.
Enables efficient processing of matrices larger than available RAM.
"""

import ctypes
from typing import Any, Optional, Tuple

from .lib_loader import get_lib
from .types import c_real, c_index, c_size, c_byte, check_error


__all__ = [
    # Lifecycle
    'mmap_create_csr_from_ptr',
    'mmap_open_csr_file',
    'mmap_release',
    'mmap_type',
    # Properties
    'mmap_csr_shape',
    # Load Operations
    'mmap_csr_load_full',
    'mmap_csr_load_masked',
    'mmap_csr_compute_masked_nnz',
    'mmap_csr_load_indexed',
    # View Operations
    'mmap_csr_create_view',
    'mmap_view_shape',
    # Reorder Operations
    'mmap_csr_reorder_rows',
    'mmap_csr_reorder_cols',
    # Format Conversion
    'mmap_csr_to_csc',
    'mmap_csr_to_dense',
    # Statistics
    'mmap_csr_row_sum',
    'mmap_csr_row_mean',
    'mmap_csr_row_var',
    'mmap_csr_col_sum',
    'mmap_csr_global_sum',
    # Normalization
    'mmap_csr_normalize_l1',
    'mmap_csr_normalize_l2',
    # Transforms
    'mmap_csr_log1p',
    'mmap_csr_scale_rows',
    'mmap_csr_scale_cols',
    # SpMV
    'mmap_csr_spmv',
    # Filtering
    'mmap_csr_filter_threshold',
    'mmap_csr_top_k',
    # Utility
    'mmap_get_config',
    'mmap_estimate_memory',
    'mmap_suggest_backend',
]


# =============================================================================
# Lifecycle Functions
# =============================================================================

def mmap_create_csr_from_ptr(
    data: Any,
    indices: Any,
    indptr: Any,
    rows: int,
    cols: int,
    nnz: int,
    max_pages: int
) -> int:
    """Create a memory-mapped CSR matrix from existing pointers.
    
    Args:
        data: CSR data array pointer.
        indices: CSR column indices pointer.
        indptr: CSR row pointers pointer.
        rows: Number of rows.
        cols: Number of columns.
        nnz: Number of non-zeros.
        max_pages: Maximum number of pages to cache.
        
    Returns:
        Handle to the created mmap object.
        
    Raises:
        RuntimeError: If creation fails.
    """
    lib = get_lib()
    lib.scl_mmap_create_csr_from_ptr.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        c_index, c_index, c_index, c_index, ctypes.POINTER(ctypes.c_int64)
    ]
    lib.scl_mmap_create_csr_from_ptr.restype = ctypes.c_int
    
    handle = ctypes.c_int64()
    status = lib.scl_mmap_create_csr_from_ptr(
        data, indices, indptr, rows, cols, nnz, max_pages, ctypes.byref(handle)
    )
    check_error(status, "mmap_create_csr_from_ptr")
    return handle.value


def mmap_open_csr_file(filepath: str, max_pages: int) -> int:
    """Open a memory-mapped CSR matrix from file.
    
    Args:
        filepath: Path to the matrix file.
        max_pages: Maximum number of pages to cache.
        
    Returns:
        Handle to the opened mmap object.
        
    Raises:
        RuntimeError: If opening fails.
    """
    lib = get_lib()
    lib.scl_mmap_open_csr_file.argtypes = [
        ctypes.c_char_p, c_index, ctypes.POINTER(ctypes.c_int64)
    ]
    lib.scl_mmap_open_csr_file.restype = ctypes.c_int
    
    handle = ctypes.c_int64()
    status = lib.scl_mmap_open_csr_file(
        filepath.encode('utf-8'), max_pages, ctypes.byref(handle)
    )
    check_error(status, "mmap_open_csr_file")
    return handle.value


def mmap_release(handle: int) -> None:
    """Release a memory-mapped matrix handle.
    
    Args:
        handle: Handle to release.
        
    Raises:
        RuntimeError: If release fails.
    """
    lib = get_lib()
    lib.scl_mmap_release.argtypes = [ctypes.c_int64]
    lib.scl_mmap_release.restype = ctypes.c_int
    
    status = lib.scl_mmap_release(handle)
    check_error(status, "mmap_release")


def mmap_type(handle: int) -> str:
    """Get the type of a memory-mapped matrix.
    
    Args:
        handle: Matrix handle.
        
    Returns:
        Type string (e.g., "csr", "csc").
    """
    lib = get_lib()
    lib.scl_mmap_type.argtypes = [ctypes.c_int64]
    lib.scl_mmap_type.restype = ctypes.c_char_p
    
    result = lib.scl_mmap_type(handle)
    return result.decode('utf-8') if result else ""


# =============================================================================
# Properties
# =============================================================================

def mmap_csr_shape(handle: int) -> Tuple[int, int, int]:
    """Get shape of a memory-mapped CSR matrix.
    
    Args:
        handle: Matrix handle.
        
    Returns:
        Tuple of (rows, cols, nnz).
        
    Raises:
        RuntimeError: If query fails.
    """
    lib = get_lib()
    lib.scl_mmap_csr_shape.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index)
    ]
    lib.scl_mmap_csr_shape.restype = ctypes.c_int
    
    rows = c_index()
    cols = c_index()
    nnz = c_index()
    status = lib.scl_mmap_csr_shape(handle, ctypes.byref(rows), ctypes.byref(cols), ctypes.byref(nnz))
    check_error(status, "mmap_csr_shape")
    return (rows.value, cols.value, nnz.value)


# =============================================================================
# Load Operations
# =============================================================================

def mmap_csr_load_full(
    handle: int,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> None:
    """Load the full matrix into memory.
    
    Args:
        handle: Matrix handle.
        data_out: Output data array.
        indices_out: Output indices array.
        indptr_out: Output indptr array.
        
    Raises:
        RuntimeError: If load fails.
    """
    lib = get_lib()
    lib.scl_mmap_csr_load_full.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mmap_csr_load_full.restype = ctypes.c_int
    
    status = lib.scl_mmap_csr_load_full(handle, data_out, indices_out, indptr_out)
    check_error(status, "mmap_csr_load_full")


def mmap_csr_load_masked(
    handle: int,
    row_mask: Any,
    col_mask: Any,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> Tuple[int, int, int]:
    """Load a masked subset of the matrix.
    
    Args:
        handle: Matrix handle.
        row_mask: Boolean mask for rows.
        col_mask: Boolean mask for columns.
        data_out: Output data array.
        indices_out: Output indices array.
        indptr_out: Output indptr array.
        
    Returns:
        Tuple of (out_rows, out_cols, out_nnz).
        
    Raises:
        RuntimeError: If load fails.
    """
    lib = get_lib()
    lib.scl_mmap_csr_load_masked.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index)
    ]
    lib.scl_mmap_csr_load_masked.restype = ctypes.c_int
    
    out_rows = c_index()
    out_cols = c_index()
    out_nnz = c_index()
    status = lib.scl_mmap_csr_load_masked(
        handle, row_mask, col_mask, data_out, indices_out, indptr_out,
        ctypes.byref(out_rows), ctypes.byref(out_cols), ctypes.byref(out_nnz)
    )
    check_error(status, "mmap_csr_load_masked")
    return (out_rows.value, out_cols.value, out_nnz.value)


def mmap_csr_compute_masked_nnz(
    handle: int,
    row_mask: Any,
    col_mask: Any
) -> int:
    """Compute the nnz of a masked subset without loading.
    
    Args:
        handle: Matrix handle.
        row_mask: Boolean mask for rows.
        col_mask: Boolean mask for columns.
        
    Returns:
        Number of non-zeros in the masked subset.
        
    Raises:
        RuntimeError: If computation fails.
    """
    lib = get_lib()
    lib.scl_mmap_csr_compute_masked_nnz.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(c_index)
    ]
    lib.scl_mmap_csr_compute_masked_nnz.restype = ctypes.c_int
    
    out_nnz = c_index()
    status = lib.scl_mmap_csr_compute_masked_nnz(handle, row_mask, col_mask, ctypes.byref(out_nnz))
    check_error(status, "mmap_csr_compute_masked_nnz")
    return out_nnz.value


def mmap_csr_load_indexed(
    handle: int,
    row_indices: Any,
    num_rows: int,
    col_indices: Any,
    num_cols: int,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> int:
    """Load a subset of rows and columns by index.
    
    Args:
        handle: Matrix handle.
        row_indices: Indices of rows to load.
        num_rows: Number of row indices.
        col_indices: Indices of columns to load.
        num_cols: Number of column indices.
        data_out: Output data array.
        indices_out: Output indices array.
        indptr_out: Output indptr array.
        
    Returns:
        Number of non-zeros loaded.
        
    Raises:
        RuntimeError: If load fails.
    """
    lib = get_lib()
    lib.scl_mmap_csr_load_indexed.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, c_index, ctypes.c_void_p, c_index,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(c_index)
    ]
    lib.scl_mmap_csr_load_indexed.restype = ctypes.c_int
    
    out_nnz = c_index()
    status = lib.scl_mmap_csr_load_indexed(
        handle, row_indices, num_rows, col_indices, num_cols,
        data_out, indices_out, indptr_out, ctypes.byref(out_nnz)
    )
    check_error(status, "mmap_csr_load_indexed")
    return out_nnz.value


# =============================================================================
# View Operations
# =============================================================================

def mmap_csr_create_view(
    handle: int,
    row_mask: Any,
    col_mask: Any
) -> int:
    """Create a view of a subset of the matrix.
    
    Args:
        handle: Original matrix handle.
        row_mask: Boolean mask for rows.
        col_mask: Boolean mask for columns.
        
    Returns:
        Handle to the view.
        
    Raises:
        RuntimeError: If view creation fails.
    """
    lib = get_lib()
    lib.scl_mmap_csr_create_view.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)
    ]
    lib.scl_mmap_csr_create_view.restype = ctypes.c_int
    
    out_handle = ctypes.c_int64()
    status = lib.scl_mmap_csr_create_view(handle, row_mask, col_mask, ctypes.byref(out_handle))
    check_error(status, "mmap_csr_create_view")
    return out_handle.value


def mmap_view_shape(handle: int) -> Tuple[int, int, int]:
    """Get shape of a matrix view.
    
    Args:
        handle: View handle.
        
    Returns:
        Tuple of (rows, cols, nnz).
        
    Raises:
        RuntimeError: If query fails.
    """
    lib = get_lib()
    lib.scl_mmap_view_shape.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(c_index), ctypes.POINTER(c_index), ctypes.POINTER(c_index)
    ]
    lib.scl_mmap_view_shape.restype = ctypes.c_int
    
    rows = c_index()
    cols = c_index()
    nnz = c_index()
    status = lib.scl_mmap_view_shape(handle, ctypes.byref(rows), ctypes.byref(cols), ctypes.byref(nnz))
    check_error(status, "mmap_view_shape")
    return (rows.value, cols.value, nnz.value)


# =============================================================================
# Reorder Operations
# =============================================================================

def mmap_csr_reorder_rows(
    handle: int,
    order: Any,
    count: int,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> None:
    """Reorder rows of a memory-mapped matrix.
    
    Args:
        handle: Matrix handle.
        order: New row order indices.
        count: Number of rows in new order.
        data_out: Output data array.
        indices_out: Output indices array.
        indptr_out: Output indptr array.
        
    Raises:
        RuntimeError: If reorder fails.
    """
    lib = get_lib()
    lib.scl_mmap_csr_reorder_rows.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, c_index,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mmap_csr_reorder_rows.restype = ctypes.c_int
    
    status = lib.scl_mmap_csr_reorder_rows(
        handle, order, count, data_out, indices_out, indptr_out
    )
    check_error(status, "mmap_csr_reorder_rows")


def mmap_csr_reorder_cols(
    handle: int,
    col_order: Any,
    num_cols: int,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> None:
    """Reorder columns of a memory-mapped matrix.
    
    Args:
        handle: Matrix handle.
        col_order: New column order indices.
        num_cols: Number of columns in new order.
        data_out: Output data array.
        indices_out: Output indices array.
        indptr_out: Output indptr array.
        
    Raises:
        RuntimeError: If reorder fails.
    """
    lib = get_lib()
    lib.scl_mmap_csr_reorder_cols.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, c_index,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mmap_csr_reorder_cols.restype = ctypes.c_int
    
    status = lib.scl_mmap_csr_reorder_cols(
        handle, col_order, num_cols, data_out, indices_out, indptr_out
    )
    check_error(status, "mmap_csr_reorder_cols")


# =============================================================================
# Format Conversion
# =============================================================================

def mmap_csr_to_csc(
    handle: int,
    csc_data: Any,
    csc_indices: Any,
    csc_indptr: Any
) -> None:
    """Convert memory-mapped CSR to CSC format.
    
    Args:
        handle: CSR matrix handle.
        csc_data: Output CSC data array.
        csc_indices: Output CSC row indices array.
        csc_indptr: Output CSC column pointers array.
        
    Raises:
        RuntimeError: If conversion fails.
    """
    lib = get_lib()
    lib.scl_mmap_csr_to_csc.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mmap_csr_to_csc.restype = ctypes.c_int
    
    status = lib.scl_mmap_csr_to_csc(handle, csc_data, csc_indices, csc_indptr)
    check_error(status, "mmap_csr_to_csc")


def mmap_csr_to_dense(handle: int, dense_out: Any) -> None:
    """Convert memory-mapped CSR to dense format.
    
    Args:
        handle: CSR matrix handle.
        dense_out: Output dense array [rows * cols].
        
    Raises:
        RuntimeError: If conversion fails.
    """
    lib = get_lib()
    lib.scl_mmap_csr_to_dense.argtypes = [ctypes.c_int64, ctypes.c_void_p]
    lib.scl_mmap_csr_to_dense.restype = ctypes.c_int
    
    status = lib.scl_mmap_csr_to_dense(handle, dense_out)
    check_error(status, "mmap_csr_to_dense")


# =============================================================================
# Statistics
# =============================================================================

def mmap_csr_row_sum(handle: int, out: Any) -> None:
    """Compute row sums of memory-mapped CSR matrix."""
    lib = get_lib()
    lib.scl_mmap_csr_row_sum.argtypes = [ctypes.c_int64, ctypes.c_void_p]
    lib.scl_mmap_csr_row_sum.restype = ctypes.c_int
    status = lib.scl_mmap_csr_row_sum(handle, out)
    check_error(status, "mmap_csr_row_sum")


def mmap_csr_row_mean(handle: int, out: Any, count_zeros: bool) -> None:
    """Compute row means of memory-mapped CSR matrix."""
    lib = get_lib()
    lib.scl_mmap_csr_row_mean.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
    lib.scl_mmap_csr_row_mean.restype = ctypes.c_int
    status = lib.scl_mmap_csr_row_mean(handle, out, 1 if count_zeros else 0)
    check_error(status, "mmap_csr_row_mean")


def mmap_csr_row_var(handle: int, out: Any, means: Any, count_zeros: bool) -> None:
    """Compute row variances of memory-mapped CSR matrix."""
    lib = get_lib()
    lib.scl_mmap_csr_row_var.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.scl_mmap_csr_row_var.restype = ctypes.c_int
    status = lib.scl_mmap_csr_row_var(handle, out, means, 1 if count_zeros else 0)
    check_error(status, "mmap_csr_row_var")


def mmap_csr_col_sum(handle: int, out: Any) -> None:
    """Compute column sums of memory-mapped CSR matrix."""
    lib = get_lib()
    lib.scl_mmap_csr_col_sum.argtypes = [ctypes.c_int64, ctypes.c_void_p]
    lib.scl_mmap_csr_col_sum.restype = ctypes.c_int
    status = lib.scl_mmap_csr_col_sum(handle, out)
    check_error(status, "mmap_csr_col_sum")


def mmap_csr_global_sum(handle: int) -> float:
    """Compute global sum of memory-mapped CSR matrix."""
    lib = get_lib()
    lib.scl_mmap_csr_global_sum.argtypes = [ctypes.c_int64, ctypes.POINTER(c_real)]
    lib.scl_mmap_csr_global_sum.restype = ctypes.c_int
    out = c_real()
    status = lib.scl_mmap_csr_global_sum(handle, ctypes.byref(out))
    check_error(status, "mmap_csr_global_sum")
    return out.value


# =============================================================================
# Normalization
# =============================================================================

def mmap_csr_normalize_l1(
    handle: int,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> None:
    """L1 normalize rows of memory-mapped CSR matrix."""
    lib = get_lib()
    lib.scl_mmap_csr_normalize_l1.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mmap_csr_normalize_l1.restype = ctypes.c_int
    status = lib.scl_mmap_csr_normalize_l1(handle, data_out, indices_out, indptr_out)
    check_error(status, "mmap_csr_normalize_l1")


def mmap_csr_normalize_l2(
    handle: int,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> None:
    """L2 normalize rows of memory-mapped CSR matrix."""
    lib = get_lib()
    lib.scl_mmap_csr_normalize_l2.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mmap_csr_normalize_l2.restype = ctypes.c_int
    status = lib.scl_mmap_csr_normalize_l2(handle, data_out, indices_out, indptr_out)
    check_error(status, "mmap_csr_normalize_l2")


# =============================================================================
# Transforms
# =============================================================================

def mmap_csr_log1p(
    handle: int,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> None:
    """Apply log1p transform to memory-mapped CSR matrix."""
    lib = get_lib()
    lib.scl_mmap_csr_log1p.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mmap_csr_log1p.restype = ctypes.c_int
    status = lib.scl_mmap_csr_log1p(handle, data_out, indices_out, indptr_out)
    check_error(status, "mmap_csr_log1p")


def mmap_csr_scale_rows(
    handle: int,
    row_factors: Any,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> None:
    """Scale rows of memory-mapped CSR matrix."""
    lib = get_lib()
    lib.scl_mmap_csr_scale_rows.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mmap_csr_scale_rows.restype = ctypes.c_int
    status = lib.scl_mmap_csr_scale_rows(handle, row_factors, data_out, indices_out, indptr_out)
    check_error(status, "mmap_csr_scale_rows")


def mmap_csr_scale_cols(
    handle: int,
    col_factors: Any,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> None:
    """Scale columns of memory-mapped CSR matrix."""
    lib = get_lib()
    lib.scl_mmap_csr_scale_cols.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mmap_csr_scale_cols.restype = ctypes.c_int
    status = lib.scl_mmap_csr_scale_cols(handle, col_factors, data_out, indices_out, indptr_out)
    check_error(status, "mmap_csr_scale_cols")


# =============================================================================
# SpMV
# =============================================================================

def mmap_csr_spmv(handle: int, x: Any, y: Any) -> None:
    """Sparse matrix-vector multiply on memory-mapped CSR."""
    lib = get_lib()
    lib.scl_mmap_csr_spmv.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    lib.scl_mmap_csr_spmv.restype = ctypes.c_int
    status = lib.scl_mmap_csr_spmv(handle, x, y)
    check_error(status, "mmap_csr_spmv")


# =============================================================================
# Filtering
# =============================================================================

def mmap_csr_filter_threshold(
    handle: int,
    threshold: float,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> int:
    """Filter values below threshold from memory-mapped CSR."""
    lib = get_lib()
    lib.scl_mmap_csr_filter_threshold.argtypes = [
        ctypes.c_int64, c_real, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(c_index)
    ]
    lib.scl_mmap_csr_filter_threshold.restype = ctypes.c_int
    out_nnz = c_index()
    status = lib.scl_mmap_csr_filter_threshold(
        handle, threshold, data_out, indices_out, indptr_out, ctypes.byref(out_nnz)
    )
    check_error(status, "mmap_csr_filter_threshold")
    return out_nnz.value


def mmap_csr_top_k(
    handle: int,
    k: int,
    data_out: Any,
    indices_out: Any,
    indptr_out: Any
) -> None:
    """Keep top-k values per row from memory-mapped CSR."""
    lib = get_lib()
    lib.scl_mmap_csr_top_k.argtypes = [
        ctypes.c_int64, c_index, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.scl_mmap_csr_top_k.restype = ctypes.c_int
    status = lib.scl_mmap_csr_top_k(handle, k, data_out, indices_out, indptr_out)
    check_error(status, "mmap_csr_top_k")


# =============================================================================
# Utility
# =============================================================================

def mmap_get_config() -> Tuple[int, int]:
    """Get mmap configuration (page_size, default_pool_size)."""
    lib = get_lib()
    lib.scl_mmap_get_config.argtypes = [ctypes.POINTER(c_index), ctypes.POINTER(c_index)]
    lib.scl_mmap_get_config.restype = ctypes.c_int
    page_size = c_index()
    pool_size = c_index()
    status = lib.scl_mmap_get_config(ctypes.byref(page_size), ctypes.byref(pool_size))
    check_error(status, "mmap_get_config")
    return (page_size.value, pool_size.value)


def mmap_estimate_memory(rows: int, nnz: int) -> int:
    """Estimate memory required for a matrix."""
    lib = get_lib()
    lib.scl_mmap_estimate_memory.argtypes = [c_index, c_index, ctypes.POINTER(c_index)]
    lib.scl_mmap_estimate_memory.restype = ctypes.c_int
    out_bytes = c_index()
    status = lib.scl_mmap_estimate_memory(rows, nnz, ctypes.byref(out_bytes))
    check_error(status, "mmap_estimate_memory")
    return out_bytes.value


def mmap_suggest_backend(data_bytes: int, available_mb: int) -> int:
    """Suggest backend (0=in-memory, 1=mmap) based on resources."""
    lib = get_lib()
    lib.scl_mmap_suggest_backend.argtypes = [c_index, c_index, ctypes.POINTER(ctypes.c_int)]
    lib.scl_mmap_suggest_backend.restype = ctypes.c_int
    out_backend = ctypes.c_int()
    status = lib.scl_mmap_suggest_backend(data_bytes, available_mb, ctypes.byref(out_backend))
    check_error(status, "mmap_suggest_backend")
    return out_backend.value

