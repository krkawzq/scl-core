"""
Callback-based Sparse Matrix Kernel Bindings

Low-level ctypes bindings for callback-based sparse matrices.
These allow Python to implement custom data access patterns that integrate
with C++ SCL operators.

Usage Pattern:

    1. Define callback functions using CFUNCTYPE decorators
    2. Create a VTable structure with the callbacks
    3. Call create_callback_csr/csc to get a handle
    4. Use the handle with operator functions
    5. Call destroy_callback_csr/csc to release

Warning:

    - Performance: Each callback crosses Python-C++ boundary
    - GIL: Callbacks hold the GIL, preventing parallel execution
    - Best for I/O-bound scenarios, not compute-intensive loops

Example:

    from scl._kernel.callback import (
        CallbackVTable, GetRowsFunc, GetPrimaryValuesFunc,
        create_callback_csr, callback_csr_row_sums, destroy_callback_csr
    )
    
    @GetRowsFunc
    def get_rows(ctx):
        return 100
    
    @GetPrimaryValuesFunc
    def get_values(ctx, i, out_data, out_len):
        # Fill out_data and out_len
        return 0
    
    vtable = CallbackVTable(get_rows=get_rows, ...)
    handle = create_callback_csr(id(self), vtable)
    
    output = Array.zeros(100, dtype='float64')
    callback_csr_row_sums(handle, output.get_pointer())
    
    destroy_callback_csr(handle)
"""

import ctypes
from ctypes import (
    CFUNCTYPE, POINTER, Structure,
    c_void_p, c_int, c_int64, c_double, c_float
)
from typing import Optional, Callable

from .lib_loader import get_library
from .types import c_real, c_index, c_size

# =============================================================================
# Callback Function Type Definitions
# =============================================================================

# Index (*get_rows)(void* context)
GetRowsFunc = CFUNCTYPE(c_index, c_void_p)

# Index (*get_cols)(void* context)
GetColsFunc = CFUNCTYPE(c_index, c_void_p)

# Index (*get_nnz)(void* context)
GetNnzFunc = CFUNCTYPE(c_index, c_void_p)

# int (*get_primary_values)(void* context, Index i, Real** out_data, Index* out_len)
GetPrimaryValuesFunc = CFUNCTYPE(
    c_int,                      # return: error code
    c_void_p,                   # context
    c_index,                    # i (row/col index)
    POINTER(POINTER(c_real)),   # out_data
    POINTER(c_index)            # out_len
)

# int (*get_primary_indices)(void* context, Index i, Index** out_indices, Index* out_len)
GetPrimaryIndicesFunc = CFUNCTYPE(
    c_int,                      # return: error code
    c_void_p,                   # context
    c_index,                    # i (row/col index)
    POINTER(POINTER(c_index)),  # out_indices
    POINTER(c_index)            # out_len
)

# Index (*get_primary_length)(void* context, Index i)  [optional]
GetPrimaryLengthFunc = CFUNCTYPE(c_index, c_void_p, c_index)

# int (*prefetch_range)(void* context, Index start, Index end)  [optional]
PrefetchRangeFunc = CFUNCTYPE(c_int, c_void_p, c_index, c_index)

# int (*release_primary)(void* context, Index i)  [optional]
ReleasePrimaryFunc = CFUNCTYPE(c_int, c_void_p, c_index)

# =============================================================================
# VTable Structure
# =============================================================================

class CallbackVTable(Structure):
    """
    Virtual function table for callback-based sparse matrices.
    
    Required fields (must be set):
        get_rows: Returns number of rows
        get_cols: Returns number of columns
        get_nnz: Returns total non-zero count
        get_primary_values: Returns values for row/col i
        get_primary_indices: Returns indices for row/col i
    
    Optional fields (can be None):
        get_primary_length: Fast length query
        prefetch_range: Batch prefetch for performance
        release_primary: Release resources after use
    
    Example:
    
        @GetRowsFunc
        def get_rows(ctx):
            return 100
        
        vtable = CallbackVTable(
            get_rows=get_rows,
            get_cols=get_cols,
            get_nnz=get_nnz,
            get_primary_values=get_values,
            get_primary_indices=get_indices,
        )
    """
    _fields_ = [
        ("get_rows", GetRowsFunc),
        ("get_cols", GetColsFunc),
        ("get_nnz", GetNnzFunc),
        ("get_primary_values", GetPrimaryValuesFunc),
        ("get_primary_indices", GetPrimaryIndicesFunc),
        ("get_primary_length", GetPrimaryLengthFunc),     # Optional
        ("prefetch_range", PrefetchRangeFunc),            # Optional
        ("release_primary", ReleasePrimaryFunc),          # Optional
    ]

# =============================================================================
# C API Bindings
# =============================================================================

_lib = get_library()

# Handle type
CallbackHandle = c_int64

# --- Lifecycle ---

# int scl_create_callback_csr(void* context, const scl_callback_vtable_t* vtable, 
#                              scl_callback_handle_t* out_handle)
create_callback_csr = _lib.scl_create_callback_csr
create_callback_csr.argtypes = [c_void_p, POINTER(CallbackVTable), POINTER(CallbackHandle)]
create_callback_csr.restype = c_int

create_callback_csc = _lib.scl_create_callback_csc
create_callback_csc.argtypes = [c_void_p, POINTER(CallbackVTable), POINTER(CallbackHandle)]
create_callback_csc.restype = c_int

destroy_callback_csr = _lib.scl_destroy_callback_csr
destroy_callback_csr.argtypes = [CallbackHandle]
destroy_callback_csr.restype = c_int

destroy_callback_csc = _lib.scl_destroy_callback_csc
destroy_callback_csc.argtypes = [CallbackHandle]
destroy_callback_csc.restype = c_int

# --- Properties ---

callback_csr_shape = _lib.scl_callback_csr_shape
callback_csr_shape.argtypes = [CallbackHandle, POINTER(c_index), POINTER(c_index), POINTER(c_index)]
callback_csr_shape.restype = c_int

callback_csc_shape = _lib.scl_callback_csc_shape
callback_csc_shape.argtypes = [CallbackHandle, POINTER(c_index), POINTER(c_index), POINTER(c_index)]
callback_csc_shape.restype = c_int

# --- Statistics (CSR) ---

callback_csr_row_sums = _lib.scl_callback_csr_row_sums
callback_csr_row_sums.argtypes = [CallbackHandle, c_void_p]
callback_csr_row_sums.restype = c_int

callback_csr_row_means = _lib.scl_callback_csr_row_means
callback_csr_row_means.argtypes = [CallbackHandle, c_void_p]
callback_csr_row_means.restype = c_int

callback_csr_row_variances = _lib.scl_callback_csr_row_variances
callback_csr_row_variances.argtypes = [CallbackHandle, c_void_p, c_int]
callback_csr_row_variances.restype = c_int

callback_csr_row_nnz = _lib.scl_callback_csr_row_nnz
callback_csr_row_nnz.argtypes = [CallbackHandle, c_void_p]
callback_csr_row_nnz.restype = c_int

# --- Statistics (CSC) ---

callback_csc_col_sums = _lib.scl_callback_csc_col_sums
callback_csc_col_sums.argtypes = [CallbackHandle, c_void_p]
callback_csc_col_sums.restype = c_int

callback_csc_col_means = _lib.scl_callback_csc_col_means
callback_csc_col_means.argtypes = [CallbackHandle, c_void_p]
callback_csc_col_means.restype = c_int

callback_csc_col_variances = _lib.scl_callback_csc_col_variances
callback_csc_col_variances.argtypes = [CallbackHandle, c_void_p, c_int]
callback_csc_col_variances.restype = c_int

callback_csc_col_nnz = _lib.scl_callback_csc_col_nnz
callback_csc_col_nnz.argtypes = [CallbackHandle, c_void_p]
callback_csc_col_nnz.restype = c_int

# --- Utility ---

callback_csr_prefetch = _lib.scl_callback_csr_prefetch
callback_csr_prefetch.argtypes = [CallbackHandle, c_index, c_index]
callback_csr_prefetch.restype = c_int

callback_csc_prefetch = _lib.scl_callback_csc_prefetch
callback_csc_prefetch.argtypes = [CallbackHandle, c_index, c_index]
callback_csc_prefetch.restype = c_int

callback_csr_invalidate_cache = _lib.scl_callback_csr_invalidate_cache
callback_csr_invalidate_cache.argtypes = [CallbackHandle]
callback_csr_invalidate_cache.restype = c_int

callback_csc_invalidate_cache = _lib.scl_callback_csc_invalidate_cache
callback_csc_invalidate_cache.argtypes = [CallbackHandle]
callback_csc_invalidate_cache.restype = c_int

# --- Direct Access ---

callback_csr_get_row = _lib.scl_callback_csr_get_row
callback_csr_get_row.argtypes = [
    CallbackHandle, c_index,
    POINTER(POINTER(c_real)), POINTER(POINTER(c_index)), POINTER(c_index)
]
callback_csr_get_row.restype = c_int

callback_csc_get_col = _lib.scl_callback_csc_get_col
callback_csc_get_col.argtypes = [
    CallbackHandle, c_index,
    POINTER(POINTER(c_real)), POINTER(POINTER(c_index)), POINTER(c_index)
]
callback_csc_get_col.restype = c_int

# =============================================================================
# Helper Functions
# =============================================================================

def create_vtable(
    get_rows: Callable[[int], int],
    get_cols: Callable[[int], int],
    get_nnz: Callable[[int], int],
    get_primary_values: Callable,
    get_primary_indices: Callable,
    get_primary_length: Optional[Callable] = None,
    prefetch_range: Optional[Callable] = None,
    release_primary: Optional[Callable] = None
) -> CallbackVTable:
    """
    Create a CallbackVTable from Python functions.
    
    This helper wraps Python functions into ctypes callbacks and returns
    a properly initialized VTable structure.
    
    Args:
        get_rows: Function returning number of rows
        get_cols: Function returning number of columns
        get_nnz: Function returning total nnz
        get_primary_values: Function to get values for row/col i
        get_primary_indices: Function to get indices for row/col i
        get_primary_length: Optional function for fast length query
        prefetch_range: Optional function for batch prefetch
        release_primary: Optional function to release resources
    
    Returns:
        CallbackVTable structure ready for use with create_callback_csr/csc
    
    Note:
        The returned VTable holds references to the wrapped callbacks.
        You must keep the VTable alive as long as the callback handle exists.
    """
    vtable = CallbackVTable()
    
    # Wrap required callbacks
    vtable.get_rows = GetRowsFunc(get_rows)
    vtable.get_cols = GetColsFunc(get_cols)
    vtable.get_nnz = GetNnzFunc(get_nnz)
    vtable.get_primary_values = GetPrimaryValuesFunc(get_primary_values)
    vtable.get_primary_indices = GetPrimaryIndicesFunc(get_primary_indices)
    
    # Wrap optional callbacks
    if get_primary_length is not None:
        vtable.get_primary_length = GetPrimaryLengthFunc(get_primary_length)
    if prefetch_range is not None:
        vtable.prefetch_range = PrefetchRangeFunc(prefetch_range)
    if release_primary is not None:
        vtable.release_primary = ReleasePrimaryFunc(release_primary)
    
    return vtable


__all__ = [
    # Callback function types
    'GetRowsFunc',
    'GetColsFunc',
    'GetNnzFunc',
    'GetPrimaryValuesFunc',
    'GetPrimaryIndicesFunc',
    'GetPrimaryLengthFunc',
    'PrefetchRangeFunc',
    'ReleasePrimaryFunc',
    
    # VTable structure
    'CallbackVTable',
    'CallbackHandle',
    
    # Lifecycle functions
    'create_callback_csr',
    'create_callback_csc',
    'destroy_callback_csr',
    'destroy_callback_csc',
    
    # Property functions
    'callback_csr_shape',
    'callback_csc_shape',
    
    # Statistics (CSR)
    'callback_csr_row_sums',
    'callback_csr_row_means',
    'callback_csr_row_variances',
    'callback_csr_row_nnz',
    
    # Statistics (CSC)
    'callback_csc_col_sums',
    'callback_csc_col_means',
    'callback_csc_col_variances',
    'callback_csc_col_nnz',
    
    # Utility
    'callback_csr_prefetch',
    'callback_csc_prefetch',
    'callback_csr_invalidate_cache',
    'callback_csc_invalidate_cache',
    
    # Direct access
    'callback_csr_get_row',
    'callback_csc_get_col',
    
    # Helper
    'create_vtable',
]

