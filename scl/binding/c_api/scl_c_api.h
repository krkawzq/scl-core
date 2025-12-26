// =============================================================================
/// @file scl_c_api.h
/// @brief C ABI Interface for SCL Kernels
///
/// Complete C-compatible function exports for Python/Julia/R bindings.
///
/// Design Principles:
///
/// 1. C Linkage: All functions use extern "C"
/// 2. POD Arguments: Only plain C types (pointers, integers, floats)
/// 3. Error Codes: Return 0 on success, error code on failure
/// 4. Thread-Safe: Error messages stored in thread-local storage
/// 5. Zero-Copy: No memory allocations, all arrays pre-allocated by caller
///
/// Error Handling Pattern:
///
/// int status = scl_function(...);
/// if (status != 0) {
///     const char* error = scl_get_last_error();
///     // Handle error
/// }
///
/// Module Organization:
///
/// - Error Handling: scl_get_last_error, scl_clear_error
/// - Version Info: scl_version, scl_precision_type, etc.
/// - Sparse Matrix: primary_sums, primary_means, primary_variances, etc.
/// - QC Metrics: compute_basic_qc
/// - Normalization: scale_primary
/// - Feature Statistics: standard_moments, clipped_moments, detection_rate
/// - Statistical Tests: mwu_test, ttest
/// - Log Transforms: log1p_inplace, log2p1_inplace, expm1_inplace
/// - Gram Matrix: gram
/// - Correlation: pearson
/// - Group Aggregations: group_stats, count_group_sizes
/// - Standardization: standardize
/// - Softmax: softmax_inplace
/// - MMD: mmd_rbf
/// - Spatial Statistics: morans_i
/// - Linear Algebra: spmv, spmv_trans
/// - HVG Selection: hvg_by_dispersion, hvg_by_variance
/// - Reordering: align_secondary
/// - Resampling: downsample_counts
/// - Memory Management: malloc, calloc, free, memcpy, etc.
/// - Helpers: sizeof_real, alignment, workspace calculations
// =============================================================================

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Type Definitions (matching C++ scl:: types)
// =============================================================================

/// @brief Floating-point type (configured at compile time: float or double)
typedef double scl_real_t;

/// @brief Signed 64-bit integer for array indexing
typedef int64_t scl_index_t;

/// @brief Unsigned size type
typedef size_t scl_size_t;

/// @brief Unsigned 8-bit integer
typedef uint8_t scl_byte_t;

// =============================================================================
// Error Query Functions
// =============================================================================

const char* scl_get_last_error(void);
void scl_clear_error(void);

// =============================================================================
// Version and Type Information
// =============================================================================

const char* scl_version(void);
int scl_precision_type(void);
const char* scl_precision_name(void);
int scl_index_type(void);
const char* scl_index_name(void);

// =============================================================================
// Sparse Matrix Statistics (sparse.hpp)
// =============================================================================

int scl_primary_sums_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* output
);

int scl_primary_sums_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* output
);

int scl_primary_means_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* output
);

int scl_primary_means_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* output
);

int scl_primary_variances_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    int ddof,
    scl_real_t* output
);

int scl_primary_variances_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    int ddof,
    scl_real_t* output
);

int scl_primary_nnz_counts_csr(
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t* output
);

int scl_primary_nnz_counts_csc(
    const scl_index_t* indptr,
    scl_index_t cols,
    scl_index_t* output
);

// =============================================================================
// Quality Control Metrics (qc.hpp)
// =============================================================================

int scl_compute_basic_qc_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t* out_n_genes,
    scl_real_t* out_total_counts
);

int scl_compute_basic_qc_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t* out_n_cells,
    scl_real_t* out_total_counts
);

// =============================================================================
// Normalization Operations (normalize.hpp)
// =============================================================================

int scl_scale_primary_csr(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    const scl_real_t* scales
);

int scl_scale_primary_csc(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    const scl_real_t* scales
);

// =============================================================================
// Feature Statistics (feature.hpp)
// =============================================================================

int scl_standard_moments_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof
);

int scl_clipped_moments_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    const scl_real_t* clip_vals,
    scl_real_t* out_means,
    scl_real_t* out_vars
);

int scl_detection_rate_csc(
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* out_rates
);

int scl_dispersion(
    const scl_real_t* means,
    const scl_real_t* vars,
    scl_size_t size,
    scl_real_t* out_dispersion
);

// =============================================================================
// Statistical Tests (mwu.hpp, ttest.hpp)
// =============================================================================

int scl_mwu_test_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    const int32_t* group_ids,
    scl_real_t* out_u_stats,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc
);

int scl_ttest_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    const int32_t* group_ids,
    scl_size_t n_groups,
    scl_real_t* out_t_stats,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc,
    bool use_welch
);

// =============================================================================
// Log Transforms (log1p.hpp)
// =============================================================================

int scl_log1p_inplace_array(scl_real_t* data, scl_size_t size);

int scl_log1p_inplace_csr(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols
);

int scl_log2p1_inplace_array(scl_real_t* data, scl_size_t size);

int scl_expm1_inplace_array(scl_real_t* data, scl_size_t size);

// =============================================================================
// Gram Matrix (gram.hpp)
// =============================================================================

int scl_gram_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* output
);

int scl_gram_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* output
);

// =============================================================================
// Pearson Correlation (correlation.hpp)
// =============================================================================

int scl_pearson_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* output,
    scl_real_t* workspace_means,
    scl_real_t* workspace_inv_stds
);

int scl_pearson_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* output,
    scl_real_t* workspace_means,
    scl_real_t* workspace_inv_stds
);

// =============================================================================
// Group Aggregations (group.hpp)
// =============================================================================

int scl_group_stats_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    const int32_t* group_ids,
    scl_size_t n_groups,
    const scl_size_t* group_sizes,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof,
    bool include_zeros
);

int scl_count_group_sizes(
    const int32_t* group_ids,
    scl_size_t n_elements,
    scl_size_t n_groups,
    scl_size_t* out_sizes
);

// =============================================================================
// Standardization (scale.hpp)
// =============================================================================

int scl_standardize_csc(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    const scl_real_t* means,
    const scl_real_t* stds,
    scl_real_t max_value,
    bool zero_center
);

// =============================================================================
// Softmax (softmax.hpp)
// =============================================================================

int scl_softmax_inplace_csr(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols
);

// =============================================================================
// MMD (mmd.hpp)
// =============================================================================

int scl_mmd_rbf_csc(
    const scl_real_t* data_x,
    const scl_index_t* indices_x,
    const scl_index_t* indptr_x,
    scl_index_t rows_x,
    scl_index_t cols,
    const scl_real_t* data_y,
    const scl_index_t* indices_y,
    const scl_index_t* indptr_y,
    scl_index_t rows_y,
    scl_real_t* output,
    scl_real_t gamma
);

// =============================================================================
// Spatial Statistics (spatial.hpp)
// =============================================================================

int scl_morans_i(
    const scl_real_t* graph_data,
    const scl_index_t* graph_indices,
    const scl_index_t* graph_indptr,
    scl_index_t n_cells,
    const scl_real_t* features_data,
    const scl_index_t* features_indices,
    const scl_index_t* features_indptr,
    scl_index_t n_genes,
    scl_real_t* output
);

// =============================================================================
// Linear Algebra (algebra.hpp)
// =============================================================================

int scl_spmv_csr(
    const scl_real_t* A_data,
    const scl_index_t* A_indices,
    const scl_index_t* A_indptr,
    scl_index_t A_rows,
    scl_index_t A_cols,
    const scl_real_t* x,
    scl_real_t* y,
    scl_real_t alpha,
    scl_real_t beta
);

int scl_spmv_trans_csc(
    const scl_real_t* A_data,
    const scl_index_t* A_indices,
    const scl_index_t* A_indptr,
    scl_index_t A_rows,
    scl_index_t A_cols,
    const scl_real_t* x,
    scl_real_t* y,
    scl_real_t alpha,
    scl_real_t beta
);

// =============================================================================
// HVG Selection (hvg.hpp)
// =============================================================================

int scl_hvg_by_dispersion_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t cols,
    scl_size_t n_top,
    scl_index_t* out_indices,
    scl_byte_t* out_mask,
    scl_real_t* out_dispersions
);

int scl_hvg_by_variance_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_size_t n_top,
    scl_index_t* out_indices,
    scl_byte_t* out_mask
);

// =============================================================================
// Reordering (reorder.hpp)
// =============================================================================

int scl_align_secondary_csc(
    scl_real_t* data,
    scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    const scl_index_t* index_map,
    scl_index_t* out_lengths,
    scl_index_t new_cols
);

// =============================================================================
// Resampling (resample.hpp)
// =============================================================================

int scl_downsample_counts_csc(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t target_sum,
    uint64_t seed
);

// =============================================================================
// Memory Management
// =============================================================================

int scl_malloc(scl_size_t bytes, void** out_ptr);
int scl_calloc(scl_size_t bytes, void** out_ptr);
int scl_malloc_aligned(scl_size_t bytes, scl_size_t alignment, void** out_ptr);
void scl_free(void* ptr);
void scl_free_aligned(void* ptr);
void scl_memzero(void* ptr, scl_size_t bytes);
int scl_memcpy(const void* src, void* dst, scl_size_t bytes);

// =============================================================================
// Helper Functions
// =============================================================================

bool scl_is_valid_value(scl_real_t value);
scl_size_t scl_sizeof_real(void);
scl_size_t scl_sizeof_index(void);
scl_size_t scl_alignment(void);

// =============================================================================
// Workspace Size Calculation Helpers
// =============================================================================

scl_size_t scl_ttest_workspace_size(scl_size_t n_features, scl_size_t n_groups);
scl_size_t scl_diff_expr_output_size(scl_size_t n_features, scl_size_t n_groups);
scl_size_t scl_group_stats_output_size(scl_size_t n_features, scl_size_t n_groups);
scl_size_t scl_gram_output_size(scl_size_t n);
scl_size_t scl_correlation_workspace_size(scl_size_t n);

// =============================================================================
// Memory-Mapped Sparse Matrix (mmap module)
// =============================================================================

/// @brief Handle type for mmap objects
typedef int64_t scl_mmap_handle_t;

// --- Lifecycle ---

int scl_mmap_create_csr_from_ptr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_index_t max_pages,
    scl_mmap_handle_t* out_handle
);

int scl_mmap_open_csr_file(
    const char* filepath,
    scl_index_t max_pages,
    scl_mmap_handle_t* out_handle
);

int scl_mmap_release(scl_mmap_handle_t handle);

const char* scl_mmap_type(scl_mmap_handle_t handle);

// --- Properties ---

int scl_mmap_csr_shape(
    scl_mmap_handle_t handle,
    scl_index_t* out_rows,
    scl_index_t* out_cols,
    scl_index_t* out_nnz
);

// --- Load Operations ---

int scl_mmap_csr_load_full(
    scl_mmap_handle_t handle,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out
);

int scl_mmap_csr_load_masked(
    scl_mmap_handle_t handle,
    const scl_byte_t* row_mask,
    const scl_byte_t* col_mask,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out,
    scl_index_t* out_rows,
    scl_index_t* out_cols,
    scl_index_t* out_nnz
);

int scl_mmap_csr_compute_masked_nnz(
    scl_mmap_handle_t handle,
    const scl_byte_t* row_mask,
    const scl_byte_t* col_mask,
    scl_index_t* out_nnz
);

int scl_mmap_csr_load_indexed(
    scl_mmap_handle_t handle,
    const scl_index_t* row_indices,
    scl_index_t num_rows,
    const scl_index_t* col_indices,
    scl_index_t num_cols,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out,
    scl_index_t* out_nnz
);

// --- View Operations ---

int scl_mmap_csr_create_view(
    scl_mmap_handle_t handle,
    const scl_byte_t* row_mask,
    const scl_byte_t* col_mask,
    scl_mmap_handle_t* out_handle
);

int scl_mmap_view_shape(
    scl_mmap_handle_t handle,
    scl_index_t* out_rows,
    scl_index_t* out_cols,
    scl_index_t* out_nnz
);

// --- Reorder Operations ---

int scl_mmap_csr_reorder_rows(
    scl_mmap_handle_t handle,
    const scl_index_t* order,
    scl_index_t count,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out
);

int scl_mmap_csr_reorder_cols(
    scl_mmap_handle_t handle,
    const scl_index_t* col_order,
    scl_index_t num_cols,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out
);

// --- Format Conversion ---

int scl_mmap_csr_to_csc(
    scl_mmap_handle_t handle,
    scl_real_t* csc_data,
    scl_index_t* csc_indices,
    scl_index_t* csc_indptr
);

int scl_mmap_csr_to_dense(
    scl_mmap_handle_t handle,
    scl_real_t* dense_out
);

// --- Statistics ---

int scl_mmap_csr_row_sum(scl_mmap_handle_t handle, scl_real_t* out);
int scl_mmap_csr_row_mean(scl_mmap_handle_t handle, scl_real_t* out, int count_zeros);
int scl_mmap_csr_row_var(scl_mmap_handle_t handle, scl_real_t* out, const scl_real_t* means, int count_zeros);
int scl_mmap_csr_col_sum(scl_mmap_handle_t handle, scl_real_t* out);
int scl_mmap_csr_global_sum(scl_mmap_handle_t handle, scl_real_t* out);

// --- Normalization ---

int scl_mmap_csr_normalize_l1(
    scl_mmap_handle_t handle,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out
);

int scl_mmap_csr_normalize_l2(
    scl_mmap_handle_t handle,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out
);

// --- Transforms ---

int scl_mmap_csr_log1p(
    scl_mmap_handle_t handle,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out
);

int scl_mmap_csr_scale_rows(
    scl_mmap_handle_t handle,
    const scl_real_t* row_factors,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out
);

int scl_mmap_csr_scale_cols(
    scl_mmap_handle_t handle,
    const scl_real_t* col_factors,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out
);

// --- SpMV ---

int scl_mmap_csr_spmv(
    scl_mmap_handle_t handle,
    const scl_real_t* x,
    scl_real_t* y
);

// --- Filtering ---

int scl_mmap_csr_filter_threshold(
    scl_mmap_handle_t handle,
    scl_real_t threshold,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out,
    scl_index_t* out_nnz
);

int scl_mmap_csr_top_k(
    scl_mmap_handle_t handle,
    scl_index_t k,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out
);

// --- Utility ---

int scl_mmap_get_config(scl_index_t* page_size, scl_index_t* default_pool_size);
int scl_mmap_estimate_memory(scl_index_t rows, scl_index_t nnz, scl_index_t* out_bytes);
int scl_mmap_suggest_backend(scl_index_t data_bytes, scl_index_t available_mb, int* out_backend);

#ifdef __cplusplus
} // extern "C"
#endif
