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
/// - I/O Operations: mmap_array, h5_load, file utilities
/// - Sorting: vqsort, argsort, sort_pairs, topk
/// - Memory Management: malloc, calloc, free, memcpy, etc.
/// - Helpers: sizeof_real, alignment, workspace calculations
// =============================================================================

#pragma once

#ifdef __cplusplus
#include "scl/core/type.hpp"

// Type aliases for C API (directly use C++ types)
using scl_real_t = scl::Real;
using scl_index_t = scl::Index;
using scl_size_t = scl::Size;
using scl_byte_t = scl::Byte;

extern "C" {
#else
// C-only mode: provide fallback type definitions
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef double scl_real_t;
typedef int64_t scl_index_t;
typedef size_t scl_size_t;
typedef uint8_t scl_byte_t;
#endif

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

// --- Mapped Versions ---

int scl_primary_sums_mapped_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* output
);

int scl_primary_sums_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* output
);

int scl_primary_means_mapped_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* output
);

int scl_primary_means_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* output
);

int scl_primary_variances_mapped_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    int ddof,
    scl_real_t* output
);

int scl_primary_variances_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    int ddof,
    scl_real_t* output
);

int scl_primary_nnz_counts_mapped_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_index_t* output
);

int scl_primary_nnz_counts_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
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
// I/O Operations (io.cpp)
// =============================================================================

// --- Memory-Mapped Arrays ---

int scl_mmap_array_open(
    const char* filepath,
    scl_size_t element_size,
    bool writable,
    void** out_ptr,
    scl_size_t* out_size
);

int scl_mmap_array_prefetch(void* ptr, scl_size_t byte_size);
int scl_mmap_array_drop_cache(void* ptr, scl_size_t byte_size);
int scl_mmap_array_advise_sequential(void* ptr, scl_size_t byte_size);
int scl_mmap_array_advise_random(void* ptr, scl_size_t byte_size);

// --- HDF5 Sparse Matrix Loading ---

#ifdef SCL_HAS_HDF5

int scl_h5_load_sparse_csr(
    const char* filepath,
    const char* group_path,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out,
    scl_index_t* out_rows,
    scl_index_t* out_cols,
    scl_index_t* out_nnz
);

int scl_h5_load_sparse_csr_cols(
    const char* filepath,
    const char* group_path,
    const scl_index_t* col_indices,
    scl_index_t num_cols,
    scl_real_t* data_out,
    scl_index_t* indices_out,
    scl_index_t* indptr_out,
    scl_index_t* out_nnz
);

int scl_h5_estimate_masked_nnz(
    const char* filepath,
    const char* group_path,
    const scl_byte_t* row_mask,
    const scl_byte_t* col_mask,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t* out_nnz
);

#endif // SCL_HAS_HDF5

// --- File Utilities ---

int scl_file_exists(const char* filepath, int* out_exists);
int scl_file_size(const char* filepath, scl_size_t* out_size);
int scl_create_directory(const char* dirpath);

int scl_write_binary_array(
    const char* filepath,
    const void* data,
    scl_size_t element_size,
    scl_size_t num_elements
);

int scl_read_binary_array(
    const char* filepath,
    void* data,
    scl_size_t element_size,
    scl_size_t num_elements
);

int scl_get_file_extension(const char* filepath, char* out_ext);
int scl_get_parent_directory(const char* filepath, char* out_dir);
int scl_get_filename_stem(const char* filepath, char* out_name);

// =============================================================================
// Sorting Operations (sorting.cpp)
// =============================================================================

// --- VQSort (SIMD-Optimized Sorting) ---

int scl_vqsort_real_ascending(scl_real_t* data, scl_size_t size);
int scl_vqsort_real_descending(scl_real_t* data, scl_size_t size);
int scl_vqsort_index_ascending(scl_index_t* data, scl_size_t size);
int scl_vqsort_index_descending(scl_index_t* data, scl_size_t size);
int scl_vqsort_int32_ascending(int32_t* data, scl_size_t size);
int scl_vqsort_int32_descending(int32_t* data, scl_size_t size);

// --- Argsort (Indirect Sorting) ---

int scl_argsort_real_ascending(
    const scl_real_t* keys,
    scl_size_t size,
    scl_index_t* indices
);

int scl_argsort_real_descending(
    const scl_real_t* keys,
    scl_size_t size,
    scl_index_t* indices
);

int scl_argsort_real_buffered(
    const scl_real_t* keys,
    scl_size_t size,
    scl_index_t* indices,
    scl_byte_t* buffer
);

int scl_argsort_real_buffered_descending(
    const scl_real_t* keys,
    scl_size_t size,
    scl_index_t* indices,
    scl_byte_t* buffer
);

// --- Pair Sorting ---

int scl_sort_pairs_real_real_ascending(
    scl_real_t* keys,
    scl_real_t* values,
    scl_size_t size
);

int scl_sort_pairs_real_real_descending(
    scl_real_t* keys,
    scl_real_t* values,
    scl_size_t size
);

int scl_sort_pairs_real_index_ascending(
    scl_real_t* keys,
    scl_index_t* values,
    scl_size_t size
);

int scl_sort_pairs_real_index_descending(
    scl_real_t* keys,
    scl_index_t* values,
    scl_size_t size
);

int scl_sort_pairs_index_real_ascending(
    scl_index_t* keys,
    scl_real_t* values,
    scl_size_t size
);

int scl_sort_pairs_index_index_ascending(
    scl_index_t* keys,
    scl_index_t* values,
    scl_size_t size
);

// --- Top-K Selection ---

int scl_topk_real(scl_real_t* data, scl_size_t size, scl_size_t k);

int scl_topk_real_with_indices(
    const scl_real_t* keys,
    scl_size_t size,
    scl_size_t k,
    scl_real_t* out_values,
    scl_index_t* out_indices
);

// --- Utility ---

int scl_is_sorted_real_ascending(
    const scl_real_t* data,
    scl_size_t size,
    int* out_is_sorted
);

int scl_is_sorted_real_descending(
    const scl_real_t* data,
    scl_size_t size,
    int* out_is_sorted
);

scl_size_t scl_argsort_buffer_size(scl_size_t size);

// =============================================================================
// Core Utilities (core.cpp)
// =============================================================================

// --- Memory Operations ---

int scl_fill_real(scl_real_t* data, scl_size_t size, scl_real_t value);
int scl_fill_index(scl_index_t* data, scl_size_t size, scl_index_t value);
int scl_zero_real(scl_real_t* data, scl_size_t size);
int scl_zero_index(scl_index_t* data, scl_size_t size);

int scl_copy_fast_real(const scl_real_t* src, scl_real_t* dst, scl_size_t size);
int scl_copy_safe_real(const scl_real_t* src, scl_real_t* dst, scl_size_t size);
int scl_stream_copy_real(const scl_real_t* src, scl_real_t* dst, scl_size_t size);

// --- Array Utilities ---

int scl_iota_index(scl_index_t* data, scl_size_t size);
int scl_reverse_real(scl_real_t* data, scl_size_t size);
int scl_reverse_index(scl_index_t* data, scl_size_t size);
int scl_unique_real(scl_real_t* data, scl_size_t size, scl_size_t* out_new_size);

// --- Math Reductions ---

int scl_sum_real(const scl_real_t* data, scl_size_t size, scl_real_t* out_sum);
int scl_mean_real(const scl_real_t* data, scl_size_t size, scl_real_t* out_mean);
int scl_variance_real(
    const scl_real_t* data,
    scl_size_t size,
    scl_real_t mean,
    int ddof,
    scl_real_t* out_var
);
int scl_min_real(const scl_real_t* data, scl_size_t size, scl_real_t* out_min);
int scl_max_real(const scl_real_t* data, scl_size_t size, scl_real_t* out_max);
int scl_minmax_real(
    const scl_real_t* data,
    scl_size_t size,
    scl_real_t* out_min,
    scl_real_t* out_max
);
int scl_dot_real(
    const scl_real_t* x,
    const scl_real_t* y,
    scl_size_t size,
    scl_real_t* out_dot
);
int scl_norm_real(const scl_real_t* data, scl_size_t size, scl_real_t* out_norm);

// --- Element-wise Operations ---

int scl_add_scalar_real(
    const scl_real_t* x,
    scl_size_t size,
    scl_real_t scalar,
    scl_real_t* y
);
int scl_mul_scalar_real(
    const scl_real_t* x,
    scl_size_t size,
    scl_real_t scalar,
    scl_real_t* y
);
int scl_add_arrays_real(
    const scl_real_t* x,
    const scl_real_t* y,
    scl_size_t size,
    scl_real_t* z
);
int scl_mul_arrays_real(
    const scl_real_t* x,
    const scl_real_t* y,
    scl_size_t size,
    scl_real_t* z
);
int scl_clip_real(
    scl_real_t* data,
    scl_size_t size,
    scl_real_t min_val,
    scl_real_t max_val
);

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

// =============================================================================
// Log Transforms (Mapped Versions)
// =============================================================================

int scl_log1p_mapped_csr(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz
);

int scl_log1p_mapped_csc(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz
);

int scl_log2p1_mapped_csr(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz
);

int scl_expm1_mapped_csr(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz
);

// =============================================================================
// Standardization and Scaling (Mapped Versions)
// =============================================================================

int scl_scale_rows_mapped_csr(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_real_t* scales
);

int scl_normalize_rows_mapped_csr(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t norm_order
);

int scl_compute_row_sums_mapped_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* output
);

// =============================================================================
// Statistical Tests (Mapped Versions)
// =============================================================================

int scl_mwu_test_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const int32_t* group_ids,
    scl_real_t* out_u_stats,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc
);

int scl_ttest_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const int32_t* group_ids,
    scl_size_t n_groups,
    scl_real_t* out_t_stats,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc,
    int use_welch
);

// =============================================================================
// Feature Statistics (Mapped Versions)
// =============================================================================

int scl_standard_moments_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof
);

int scl_clipped_moments_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_real_t* clip_vals,
    scl_real_t* out_means,
    scl_real_t* out_vars
);

int scl_detection_rate_mapped_csc(
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* out_rates
);

// =============================================================================
// Gram Matrix and Pearson Correlation (Mapped Versions)
// =============================================================================

int scl_gram_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* output
);

int scl_gram_mapped_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* output
);

int scl_pearson_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* output,
    scl_real_t* workspace_means,
    scl_real_t* workspace_inv_stds
);

int scl_pearson_mapped_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t* output,
    scl_real_t* workspace_means,
    scl_real_t* workspace_inv_stds
);

// =============================================================================
// Group Aggregations (Mapped Versions)
// =============================================================================

int scl_group_stats_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const int32_t* group_ids,
    scl_size_t n_groups,
    const scl_size_t* group_sizes,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof,
    int include_zeros
);

int scl_group_stats_mapped_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const int32_t* group_ids,
    scl_size_t n_groups,
    const scl_size_t* group_sizes,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof,
    int include_zeros
);

int scl_standardize_mapped_csc(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_real_t* means,
    const scl_real_t* stds,
    scl_real_t max_value,
    int zero_center
);

// =============================================================================
// Quality Control (Mapped Versions)
// =============================================================================

int scl_compute_basic_qc_mapped_csr(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_index_t* out_n_genes,
    scl_real_t* out_total_counts
);

int scl_compute_basic_qc_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_index_t* out_n_cells,
    scl_real_t* out_total_counts
);

// =============================================================================
// HVG Selection (Mapped Versions)
// =============================================================================

int scl_hvg_by_dispersion_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t cols,
    scl_index_t nnz,
    scl_size_t n_top,
    scl_index_t* out_indices,
    scl_byte_t* out_mask,
    scl_real_t* out_dispersions
);

int scl_hvg_by_variance_mapped_csc(
    const scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_size_t n_top,
    scl_index_t* out_indices,
    scl_byte_t* out_mask
);

// =============================================================================
// Resampling (Mapped Versions)
// =============================================================================

int scl_downsample_counts_mapped_csc(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t target_sum,
    uint64_t seed
);

int scl_binomial_resample_mapped_csc(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_real_t p,
    uint64_t seed
);

// =============================================================================
// Advanced Operations (Mapped Versions)
// =============================================================================

int scl_softmax_mapped_csr(
    scl_real_t* data,
    const scl_index_t* indices,
    const scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz
);

int scl_spmv_mapped_csr(
    const scl_real_t* A_data,
    const scl_index_t* A_indices,
    const scl_index_t* A_indptr,
    scl_index_t A_rows,
    scl_index_t A_cols,
    scl_index_t nnz,
    const scl_real_t* x,
    scl_real_t* y,
    scl_real_t alpha,
    scl_real_t beta
);

int scl_spmv_trans_mapped_csc(
    const scl_real_t* A_data,
    const scl_index_t* A_indices,
    const scl_index_t* A_indptr,
    scl_index_t A_rows,
    scl_index_t A_cols,
    scl_index_t nnz,
    const scl_real_t* x,
    scl_real_t* y,
    scl_real_t alpha,
    scl_real_t beta
);

// =============================================================================
// Callback-Based Sparse Matrices
// =============================================================================

/// @brief Handle type for callback sparse matrices
typedef int64_t scl_callback_handle_t;

/// @brief Callback function types for sparse matrix access
typedef scl_index_t (*scl_get_rows_callback)(void* context);
typedef scl_index_t (*scl_get_cols_callback)(void* context);
typedef scl_index_t (*scl_get_nnz_callback)(void* context);
typedef int (*scl_get_primary_values_callback)(void* context, scl_index_t i, 
                                                scl_real_t** out_data, scl_index_t* out_len);
typedef int (*scl_get_primary_indices_callback)(void* context, scl_index_t i,
                                                 scl_index_t** out_indices, scl_index_t* out_len);
typedef scl_index_t (*scl_get_primary_length_callback)(void* context, scl_index_t i);
typedef int (*scl_prefetch_range_callback)(void* context, scl_index_t start, scl_index_t end);
typedef int (*scl_release_primary_callback)(void* context, scl_index_t i);

/// @brief Virtual function table for callback-based sparse matrices
///
/// Required fields (must be non-null):
/// - get_rows, get_cols, get_nnz
/// - get_primary_values, get_primary_indices
///
/// Optional fields (can be null):
/// - get_primary_length: Fast length query
/// - prefetch_range: Batch prefetch for performance
/// - release_primary: Release resources after use
typedef struct {
    scl_get_rows_callback get_rows;
    scl_get_cols_callback get_cols;
    scl_get_nnz_callback get_nnz;
    scl_get_primary_values_callback get_primary_values;
    scl_get_primary_indices_callback get_primary_indices;
    scl_get_primary_length_callback get_primary_length;     // Optional
    scl_prefetch_range_callback prefetch_range;             // Optional
    scl_release_primary_callback release_primary;           // Optional
} scl_callback_vtable_t;

// --- Lifecycle ---

/// @brief Create a callback-based CSR matrix
/// @param context User context pointer (passed to all callbacks)
/// @param vtable Pointer to callback function table
/// @param out_handle Output: handle for the created matrix
/// @return 0 on success, error code on failure
int scl_create_callback_csr(
    void* context,
    const scl_callback_vtable_t* vtable,
    scl_callback_handle_t* out_handle
);

/// @brief Create a callback-based CSC matrix
int scl_create_callback_csc(
    void* context,
    const scl_callback_vtable_t* vtable,
    scl_callback_handle_t* out_handle
);

/// @brief Destroy a callback CSR matrix
int scl_destroy_callback_csr(scl_callback_handle_t handle);

/// @brief Destroy a callback CSC matrix
int scl_destroy_callback_csc(scl_callback_handle_t handle);

// --- Properties ---

/// @brief Get shape of callback CSR matrix
int scl_callback_csr_shape(
    scl_callback_handle_t handle,
    scl_index_t* out_rows,
    scl_index_t* out_cols,
    scl_index_t* out_nnz
);

/// @brief Get shape of callback CSC matrix
int scl_callback_csc_shape(
    scl_callback_handle_t handle,
    scl_index_t* out_rows,
    scl_index_t* out_cols,
    scl_index_t* out_nnz
);

// --- Statistics (CSR) ---

/// @brief Compute row sums for callback CSR
int scl_callback_csr_row_sums(scl_callback_handle_t handle, scl_real_t* output);

/// @brief Compute row means for callback CSR
int scl_callback_csr_row_means(scl_callback_handle_t handle, scl_real_t* output);

/// @brief Compute row variances for callback CSR
int scl_callback_csr_row_variances(
    scl_callback_handle_t handle,
    scl_real_t* output,
    int ddof
);

/// @brief Compute row nnz counts for callback CSR
int scl_callback_csr_row_nnz(scl_callback_handle_t handle, scl_index_t* output);

// --- Statistics (CSC) ---

/// @brief Compute column sums for callback CSC
int scl_callback_csc_col_sums(scl_callback_handle_t handle, scl_real_t* output);

/// @brief Compute column means for callback CSC
int scl_callback_csc_col_means(scl_callback_handle_t handle, scl_real_t* output);

/// @brief Compute column variances for callback CSC
int scl_callback_csc_col_variances(
    scl_callback_handle_t handle,
    scl_real_t* output,
    int ddof
);

/// @brief Compute column nnz counts for callback CSC
int scl_callback_csc_col_nnz(scl_callback_handle_t handle, scl_index_t* output);

// --- Utility ---

/// @brief Prefetch a range of rows for callback CSR
int scl_callback_csr_prefetch(
    scl_callback_handle_t handle,
    scl_index_t start,
    scl_index_t end
);

/// @brief Prefetch a range of columns for callback CSC
int scl_callback_csc_prefetch(
    scl_callback_handle_t handle,
    scl_index_t start,
    scl_index_t end
);

/// @brief Invalidate cached dimensions for callback CSR
int scl_callback_csr_invalidate_cache(scl_callback_handle_t handle);

/// @brief Invalidate cached dimensions for callback CSC
int scl_callback_csc_invalidate_cache(scl_callback_handle_t handle);

// --- Direct Access (for debugging) ---

/// @brief Get row data from callback CSR
int scl_callback_csr_get_row(
    scl_callback_handle_t handle,
    scl_index_t row_idx,
    scl_real_t** out_values,
    scl_index_t** out_indices,
    scl_index_t* out_len
);

/// @brief Get column data from callback CSC
int scl_callback_csc_get_col(
    scl_callback_handle_t handle,
    scl_index_t col_idx,
    scl_real_t** out_values,
    scl_index_t** out_indices,
    scl_index_t* out_len
);

#ifdef __cplusplus
} // extern "C"
#endif
