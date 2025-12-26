"""
SCL FFI - ctypes Binding Layer

Provides low-level interface to the C++ library.
All C API functions are exposed here for use by higher-level modules.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    c_int32, c_int64, c_uint8, c_uint64, c_double, c_bool, c_size_t,
    c_void_p, c_char_p, POINTER, byref, Structure
)
from functools import lru_cache
from typing import Optional, Tuple

# =============================================================================
# Error Codes
# =============================================================================

OK = 0
ERR_INVALID_HANDLE = -1
ERR_INVALID_PARAM = -2
ERR_OUT_OF_MEMORY = -3
ERR_IO = -4
ERR_TYPE_MISMATCH = -5
ERR_NOT_IMPLEMENTED = -6
ERR_DIMENSION_MISMATCH = -7
ERR_UNKNOWN = -99

_ERROR_MESSAGES = {
    OK: "Success",
    ERR_INVALID_HANDLE: "Invalid handle",
    ERR_INVALID_PARAM: "Invalid parameter",
    ERR_OUT_OF_MEMORY: "Out of memory",
    ERR_IO: "I/O error",
    ERR_TYPE_MISMATCH: "Type mismatch",
    ERR_NOT_IMPLEMENTED: "Not implemented",
    ERR_DIMENSION_MISMATCH: "Dimension mismatch",
    ERR_UNKNOWN: "Unknown error",
}


class SclError(Exception):
    """SCL library error."""
    def __init__(self, code: int, message: str = ""):
        self.code = code
        self.message = message or _ERROR_MESSAGES.get(code, f"Error code {code}")
        super().__init__(self.message)


def check_error(code: int) -> None:
    """Check error code and raise exception if needed."""
    if code != OK:
        raise SclError(code)


# =============================================================================
# Library Loading
# =============================================================================

@lru_cache(maxsize=1)
def get_lib():
    """Get SCL shared library."""
    lib_names = []

    if sys.platform == "linux":
        lib_names = ["libscl.so", "libscl_mmap.so"]
    elif sys.platform == "darwin":
        lib_names = ["libscl.dylib", "libscl_mmap.dylib"]
    elif sys.platform == "win32":
        lib_names = ["scl.dll", "scl_mmap.dll"]

    search_paths = [
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), ".."),
        os.path.join(os.path.dirname(__file__), "..", "lib"),
        os.path.join(os.path.dirname(__file__), "..", "..", "build"),
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "Release"),
        "/usr/local/lib",
        "/usr/lib",
    ]

    if "SCL_LIB_PATH" in os.environ:
        search_paths.insert(0, os.environ["SCL_LIB_PATH"])

    for path in search_paths:
        for name in lib_names:
            lib_path = os.path.join(path, name)
            if os.path.exists(lib_path):
                return ctypes.CDLL(lib_path)

    raise RuntimeError(
        f"Cannot find SCL library. Searched: {search_paths}. "
        "Set SCL_LIB_PATH environment variable or install the library."
    )


# =============================================================================
# Function Signatures Setup
# =============================================================================

def _setup_functions(lib):
    """Set up all C API function signatures."""

    # =========================================================================
    # Memory Management (Section 20, 21)
    # =========================================================================

    lib.scl_malloc.argtypes = [c_int64, POINTER(c_void_p)]
    lib.scl_malloc.restype = c_int32

    lib.scl_calloc.argtypes = [c_int64, POINTER(c_void_p)]
    lib.scl_calloc.restype = c_int32

    lib.scl_malloc_aligned.argtypes = [c_int64, c_int64, POINTER(c_void_p)]
    lib.scl_malloc_aligned.restype = c_int32

    lib.scl_free.argtypes = [c_void_p]
    lib.scl_free.restype = None

    lib.scl_free_aligned.argtypes = [c_void_p]
    lib.scl_free_aligned.restype = None

    lib.scl_memzero.argtypes = [c_void_p, c_int64]
    lib.scl_memzero.restype = None

    lib.scl_memcpy.argtypes = [c_void_p, c_void_p, c_int64]
    lib.scl_memcpy.restype = c_int32

    # Helper functions
    lib.scl_sizeof_real.argtypes = []
    lib.scl_sizeof_real.restype = c_int64

    lib.scl_sizeof_index.argtypes = []
    lib.scl_sizeof_index.restype = c_int64

    lib.scl_alignment.argtypes = []
    lib.scl_alignment.restype = c_int64

    # =========================================================================
    # Sparse Matrix Statistics (Section 3)
    # =========================================================================

    # Row/Column sums
    lib.scl_primary_sums_csr.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64, POINTER(c_double)
    ]
    lib.scl_primary_sums_csr.restype = c_int32

    lib.scl_primary_sums_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64, POINTER(c_double)
    ]
    lib.scl_primary_sums_csc.restype = c_int32

    # Row/Column means
    lib.scl_primary_means_csr.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64, POINTER(c_double)
    ]
    lib.scl_primary_means_csr.restype = c_int32

    lib.scl_primary_means_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64, POINTER(c_double)
    ]
    lib.scl_primary_means_csc.restype = c_int32

    # Row/Column variances
    lib.scl_primary_variances_csr.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64, c_int32, POINTER(c_double)
    ]
    lib.scl_primary_variances_csr.restype = c_int32

    lib.scl_primary_variances_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64, c_int32, POINTER(c_double)
    ]
    lib.scl_primary_variances_csc.restype = c_int32

    # NNZ counts
    lib.scl_primary_nnz_counts_csr.argtypes = [POINTER(c_int64), c_int64, POINTER(c_int64)]
    lib.scl_primary_nnz_counts_csr.restype = c_int32

    lib.scl_primary_nnz_counts_csc.argtypes = [POINTER(c_int64), c_int64, POINTER(c_int64)]
    lib.scl_primary_nnz_counts_csc.restype = c_int32

    # =========================================================================
    # Quality Control (Section 4)
    # =========================================================================

    lib.scl_compute_basic_qc_csr.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64,
        POINTER(c_int64), POINTER(c_double)
    ]
    lib.scl_compute_basic_qc_csr.restype = c_int32

    lib.scl_compute_basic_qc_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64,
        POINTER(c_int64), POINTER(c_double)
    ]
    lib.scl_compute_basic_qc_csc.restype = c_int32

    # =========================================================================
    # Normalization (Section 5)
    # =========================================================================

    lib.scl_scale_primary_csr.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64, POINTER(c_double)
    ]
    lib.scl_scale_primary_csr.restype = c_int32

    lib.scl_scale_primary_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64, POINTER(c_double)
    ]
    lib.scl_scale_primary_csc.restype = c_int32

    # =========================================================================
    # Feature Statistics (Section 6)
    # =========================================================================

    lib.scl_standard_moments_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64,
        POINTER(c_double), POINTER(c_double), c_int32
    ]
    lib.scl_standard_moments_csc.restype = c_int32

    lib.scl_clipped_moments_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64,
        POINTER(c_double), POINTER(c_double), POINTER(c_double)
    ]
    lib.scl_clipped_moments_csc.restype = c_int32

    lib.scl_detection_rate_csc.argtypes = [
        POINTER(c_int64), c_int64, c_int64, POINTER(c_double)
    ]
    lib.scl_detection_rate_csc.restype = c_int32

    lib.scl_dispersion.argtypes = [
        POINTER(c_double), POINTER(c_double), c_int64, POINTER(c_double)
    ]
    lib.scl_dispersion.restype = c_int32

    # =========================================================================
    # Statistical Tests (Section 7)
    # =========================================================================

    lib.scl_mwu_test_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64,
        POINTER(c_int32),
        POINTER(c_double), POINTER(c_double), POINTER(c_double)
    ]
    lib.scl_mwu_test_csc.restype = c_int32

    lib.scl_ttest_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64,
        POINTER(c_int32), c_int64,
        POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
        POINTER(c_uint8), c_int64, c_bool
    ]
    lib.scl_ttest_csc.restype = c_int32

    # =========================================================================
    # Data Transformations (Section 8)
    # =========================================================================

    lib.scl_log1p_inplace_array.argtypes = [POINTER(c_double), c_int64]
    lib.scl_log1p_inplace_array.restype = c_int32

    lib.scl_log1p_inplace_csr.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64
    ]
    lib.scl_log1p_inplace_csr.restype = c_int32

    lib.scl_log2p1_inplace_array.argtypes = [POINTER(c_double), c_int64]
    lib.scl_log2p1_inplace_array.restype = c_int32

    lib.scl_expm1_inplace_array.argtypes = [POINTER(c_double), c_int64]
    lib.scl_expm1_inplace_array.restype = c_int32

    lib.scl_softmax_inplace_csr.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64
    ]
    lib.scl_softmax_inplace_csr.restype = c_int32

    # =========================================================================
    # Linear Algebra (Sections 9, 10, 16)
    # =========================================================================

    # Gram matrix
    lib.scl_gram_csr.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64, POINTER(c_double)
    ]
    lib.scl_gram_csr.restype = c_int32

    lib.scl_gram_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64, POINTER(c_double)
    ]
    lib.scl_gram_csc.restype = c_int32

    # Pearson correlation
    lib.scl_pearson_csr.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64,
        POINTER(c_double), POINTER(c_double), POINTER(c_double)
    ]
    lib.scl_pearson_csr.restype = c_int32

    lib.scl_pearson_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64,
        POINTER(c_double), POINTER(c_double), POINTER(c_double)
    ]
    lib.scl_pearson_csc.restype = c_int32

    # SpMV
    lib.scl_spmv_csr.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64,
        POINTER(c_double), POINTER(c_double), c_double, c_double
    ]
    lib.scl_spmv_csr.restype = c_int32

    lib.scl_spmv_trans_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64, c_int64, c_int64,
        POINTER(c_double), POINTER(c_double), c_double, c_double
    ]
    lib.scl_spmv_trans_csc.restype = c_int32

    # =========================================================================
    # Group Aggregations (Section 11)
    # =========================================================================

    lib.scl_group_stats_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64),
        c_int64, c_int64, c_int64, POINTER(c_int32), c_int64, POINTER(c_int64),
        POINTER(c_double), POINTER(c_double), c_int32, c_bool
    ]
    lib.scl_group_stats_csc.restype = c_int32

    lib.scl_count_group_sizes.argtypes = [
        POINTER(c_int32), c_int64, c_int64, POINTER(c_int64)
    ]
    lib.scl_count_group_sizes.restype = c_int32

    # =========================================================================
    # Standardization (Section 12)
    # =========================================================================

    lib.scl_standardize_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64),
        c_int64, c_int64, c_int64, POINTER(c_double), POINTER(c_double), c_double, c_bool
    ]
    lib.scl_standardize_csc.restype = c_int32

    # =========================================================================
    # MMD (Section 14)
    # =========================================================================

    lib.scl_mmd_rbf_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64),
        c_int64, c_int64, c_int64,
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64),
        c_int64, c_int64, POINTER(c_double), c_double
    ]
    lib.scl_mmd_rbf_csc.restype = c_int32

    # =========================================================================
    # Spatial Statistics (Section 15)
    # =========================================================================

    lib.scl_morans_i.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64),
        c_int64, c_int64,
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64),
        c_int64, c_int64, POINTER(c_double)
    ]
    lib.scl_morans_i.restype = c_int32

    # =========================================================================
    # HVG Selection (Section 17)
    # =========================================================================

    lib.scl_hvg_by_dispersion_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64),
        c_int64, c_int64, c_int64, c_int64,
        POINTER(c_int64), POINTER(c_uint8), POINTER(c_double)
    ]
    lib.scl_hvg_by_dispersion_csc.restype = c_int32

    lib.scl_hvg_by_variance_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64),
        c_int64, c_int64, c_int64, c_int64,
        POINTER(c_int64), POINTER(c_uint8)
    ]
    lib.scl_hvg_by_variance_csc.restype = c_int32

    # =========================================================================
    # Reordering (Section 18)
    # =========================================================================

    lib.scl_align_secondary_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64),
        c_int64, c_int64, c_int64, POINTER(c_int64), POINTER(c_int64), c_int64
    ]
    lib.scl_align_secondary_csc.restype = c_int32

    # =========================================================================
    # Resampling (Section 19)
    # =========================================================================

    lib.scl_downsample_counts_csc.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64),
        c_int64, c_int64, c_int64, c_double, c_uint64
    ]
    lib.scl_downsample_counts_csc.restype = c_int32

    # =========================================================================
    # Utility Functions
    # =========================================================================

    lib.scl_compute_lengths.argtypes = [POINTER(c_int64), c_int64, POINTER(c_int64)]
    lib.scl_compute_lengths.restype = c_int32

    lib.scl_inspect_slice_rows.argtypes = [
        POINTER(c_int64), POINTER(c_int64), c_int64, POINTER(c_int64)
    ]
    lib.scl_inspect_slice_rows.restype = c_int32

    lib.scl_materialize_slice_rows.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64), c_int64,
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64)
    ]
    lib.scl_materialize_slice_rows.restype = c_int32

    lib.scl_inspect_filter_cols.argtypes = [
        POINTER(c_int64), POINTER(c_int64), c_int64,
        POINTER(c_uint8), POINTER(c_int64)
    ]
    lib.scl_inspect_filter_cols.restype = c_int32

    lib.scl_materialize_filter_cols.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        c_int64, POINTER(c_uint8), POINTER(c_int64),
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64)
    ]
    lib.scl_materialize_filter_cols.restype = c_int32

    lib.scl_align_rows.argtypes = [
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        c_int64, POINTER(c_int64), c_int64,
        POINTER(c_double), POINTER(c_int64), POINTER(c_int64),
        POINTER(c_int64)
    ]
    lib.scl_align_rows.restype = c_int32

    # Workspace size calculators
    lib.scl_ttest_workspace_size.argtypes = [c_int64, c_int64]
    lib.scl_ttest_workspace_size.restype = c_int64

    lib.scl_diff_expr_output_size.argtypes = [c_int64, c_int64]
    lib.scl_diff_expr_output_size.restype = c_int64

    lib.scl_group_stats_output_size.argtypes = [c_int64, c_int64]
    lib.scl_group_stats_output_size.restype = c_int64

    lib.scl_gram_output_size.argtypes = [c_int64]
    lib.scl_gram_output_size.restype = c_int64

    lib.scl_correlation_workspace_size.argtypes = [c_int64]
    lib.scl_correlation_workspace_size.restype = c_int64

    # =========================================================================
    # MMap Operations (for mapped backend)
    # =========================================================================

    lib.scl_mmap_create_csr_from_ptr.argtypes = [
        c_void_p, c_void_p, c_void_p,
        c_int64, c_int64, c_int64,
        c_int64,
        POINTER(c_int64)
    ]
    lib.scl_mmap_create_csr_from_ptr.restype = c_int32

    lib.scl_mmap_open_csr_file.argtypes = [c_char_p, c_int64, POINTER(c_int64)]
    lib.scl_mmap_open_csr_file.restype = c_int32

    lib.scl_mmap_release.argtypes = [c_int64]
    lib.scl_mmap_release.restype = c_int32

    lib.scl_mmap_type.argtypes = [c_int64]
    lib.scl_mmap_type.restype = c_char_p

    lib.scl_mmap_csr_shape.argtypes = [
        c_int64, POINTER(c_int64), POINTER(c_int64), POINTER(c_int64)
    ]
    lib.scl_mmap_csr_shape.restype = c_int32

    lib.scl_mmap_csr_load_full.argtypes = [c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_load_full.restype = c_int32

    lib.scl_mmap_csr_load_masked.argtypes = [
        c_int64,
        POINTER(c_uint8), POINTER(c_uint8),
        c_void_p, c_void_p, c_void_p,
        POINTER(c_int64), POINTER(c_int64), POINTER(c_int64)
    ]
    lib.scl_mmap_csr_load_masked.restype = c_int32

    lib.scl_mmap_csr_compute_masked_nnz.argtypes = [
        c_int64, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_int64)
    ]
    lib.scl_mmap_csr_compute_masked_nnz.restype = c_int32

    lib.scl_mmap_csr_load_indexed.argtypes = [
        c_int64,
        c_void_p, c_int64,
        c_void_p, c_int64,
        c_void_p, c_void_p, c_void_p,
        POINTER(c_int64)
    ]
    lib.scl_mmap_csr_load_indexed.restype = c_int32

    lib.scl_mmap_csr_create_view.argtypes = [
        c_int64, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_int64)
    ]
    lib.scl_mmap_csr_create_view.restype = c_int32

    lib.scl_mmap_view_shape.argtypes = [
        c_int64, POINTER(c_int64), POINTER(c_int64), POINTER(c_int64)
    ]
    lib.scl_mmap_view_shape.restype = c_int32

    lib.scl_mmap_csr_reorder_rows.argtypes = [
        c_int64, c_void_p, c_int64, c_void_p, c_void_p, c_void_p
    ]
    lib.scl_mmap_csr_reorder_rows.restype = c_int32

    lib.scl_mmap_csr_reorder_cols.argtypes = [
        c_int64, c_void_p, c_int64, c_void_p, c_void_p, c_void_p
    ]
    lib.scl_mmap_csr_reorder_cols.restype = c_int32

    lib.scl_mmap_csr_to_csc.argtypes = [c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_to_csc.restype = c_int32

    lib.scl_mmap_csr_to_dense.argtypes = [c_int64, c_void_p]
    lib.scl_mmap_csr_to_dense.restype = c_int32

    lib.scl_mmap_csr_row_sum.argtypes = [c_int64, c_void_p]
    lib.scl_mmap_csr_row_sum.restype = c_int32

    lib.scl_mmap_csr_row_mean.argtypes = [c_int64, c_void_p, c_int32]
    lib.scl_mmap_csr_row_mean.restype = c_int32

    lib.scl_mmap_csr_row_var.argtypes = [c_int64, c_void_p, c_void_p, c_int32]
    lib.scl_mmap_csr_row_var.restype = c_int32

    lib.scl_mmap_csr_col_sum.argtypes = [c_int64, c_void_p]
    lib.scl_mmap_csr_col_sum.restype = c_int32

    lib.scl_mmap_csr_global_sum.argtypes = [c_int64, c_void_p]
    lib.scl_mmap_csr_global_sum.restype = c_int32

    lib.scl_mmap_csr_normalize_l1.argtypes = [c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_normalize_l1.restype = c_int32

    lib.scl_mmap_csr_normalize_l2.argtypes = [c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_normalize_l2.restype = c_int32

    lib.scl_mmap_csr_log1p.argtypes = [c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_log1p.restype = c_int32

    lib.scl_mmap_csr_scale_rows.argtypes = [c_int64, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_scale_rows.restype = c_int32

    lib.scl_mmap_csr_scale_cols.argtypes = [c_int64, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_scale_cols.restype = c_int32

    lib.scl_mmap_csr_spmv.argtypes = [c_int64, c_void_p, c_void_p]
    lib.scl_mmap_csr_spmv.restype = c_int32

    lib.scl_mmap_csr_filter_threshold.argtypes = [
        c_int64, c_double, c_void_p, c_void_p, c_void_p, POINTER(c_int64)
    ]
    lib.scl_mmap_csr_filter_threshold.restype = c_int32

    lib.scl_mmap_csr_top_k.argtypes = [c_int64, c_int64, c_void_p, c_void_p, c_void_p]
    lib.scl_mmap_csr_top_k.restype = c_int32

    lib.scl_mmap_get_config.argtypes = [POINTER(c_int64), POINTER(c_int64)]
    lib.scl_mmap_get_config.restype = c_int32

    lib.scl_mmap_estimate_memory.argtypes = [c_int64, c_int64, POINTER(c_int64)]
    lib.scl_mmap_estimate_memory.restype = c_int32

    lib.scl_mmap_suggest_backend.argtypes = [c_int64, c_int64, POINTER(c_int32)]
    lib.scl_mmap_suggest_backend.restype = c_int32

    return lib


@lru_cache(maxsize=1)
def get_lib_with_signatures():
    """Get library with all function signatures set up."""
    lib = get_lib()
    return _setup_functions(lib)


# =============================================================================
# Configuration Utilities
# =============================================================================

def get_config() -> Tuple[int, int]:
    """Get mmap config: (page_size, default_pool_size)."""
    lib = get_lib_with_signatures()
    page_size = c_int64()
    pool_size = c_int64()
    check_error(lib.scl_mmap_get_config(byref(page_size), byref(pool_size)))
    return page_size.value, pool_size.value


def estimate_memory(rows: int, nnz: int) -> int:
    """Estimate sparse matrix memory requirements (bytes)."""
    lib = get_lib_with_signatures()
    result = c_int64()
    check_error(lib.scl_mmap_estimate_memory(rows, nnz, byref(result)))
    return result.value


class BackendType:
    """Backend type constants."""
    IN_MEMORY = 0
    MAPPED = 1
    STREAMING = 2


def suggest_backend(data_bytes: int, available_mb: int = 4096) -> int:
    """Suggest backend type based on data size."""
    lib = get_lib_with_signatures()
    result = c_int32()
    check_error(lib.scl_mmap_suggest_backend(data_bytes, available_mb, byref(result)))
    return result.value


# =============================================================================
# Workspace Size Utilities
# =============================================================================

def ttest_workspace_size(n_features: int, n_groups: int) -> int:
    """Calculate workspace size for T-test."""
    lib = get_lib_with_signatures()
    return lib.scl_ttest_workspace_size(n_features, n_groups)


def diff_expr_output_size(n_features: int, n_groups: int) -> int:
    """Calculate output size for differential expression."""
    lib = get_lib_with_signatures()
    return lib.scl_diff_expr_output_size(n_features, n_groups)


def group_stats_output_size(n_features: int, n_groups: int) -> int:
    """Calculate output size for group statistics."""
    lib = get_lib_with_signatures()
    return lib.scl_group_stats_output_size(n_features, n_groups)


def gram_output_size(n: int) -> int:
    """Calculate output size for Gram matrix."""
    lib = get_lib_with_signatures()
    return lib.scl_gram_output_size(n)


def correlation_workspace_size(n: int) -> int:
    """Calculate workspace size for correlation."""
    lib = get_lib_with_signatures()
    return lib.scl_correlation_workspace_size(n)


# =============================================================================
# Type Size Utilities
# =============================================================================

def sizeof_real() -> int:
    """Get size of Real type in bytes."""
    lib = get_lib_with_signatures()
    return lib.scl_sizeof_real()


def sizeof_index() -> int:
    """Get size of Index type in bytes."""
    lib = get_lib_with_signatures()
    return lib.scl_sizeof_index()


def alignment() -> int:
    """Get recommended memory alignment in bytes."""
    lib = get_lib_with_signatures()
    return lib.scl_alignment()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Error handling
    "OK",
    "ERR_INVALID_HANDLE",
    "ERR_INVALID_PARAM",
    "ERR_OUT_OF_MEMORY",
    "ERR_IO",
    "ERR_TYPE_MISMATCH",
    "ERR_NOT_IMPLEMENTED",
    "ERR_DIMENSION_MISMATCH",
    "ERR_UNKNOWN",
    "SclError",
    "check_error",
    # Library access
    "get_lib",
    "get_lib_with_signatures",
    # Configuration
    "get_config",
    "estimate_memory",
    "BackendType",
    "suggest_backend",
    # Workspace utilities
    "ttest_workspace_size",
    "diff_expr_output_size",
    "group_stats_output_size",
    "gram_output_size",
    "correlation_workspace_size",
    # Type utilities
    "sizeof_real",
    "sizeof_index",
    "alignment",
]
