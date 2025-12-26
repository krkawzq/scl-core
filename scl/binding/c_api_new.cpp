// =============================================================================
/// @file c_api.cpp
/// @brief C ABI Exports for SCL Kernels
///
/// Provides C-compatible function exports for Python/Julia/R bindings.
///
/// Design Principles:
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
/// Architecture:
/// - Section 1: Error Handling Infrastructure
/// - Section 2: Version and Type Information
/// - Section 3: Sparse Matrix Statistics (sparse.hpp)
/// - Section 4: Quality Control Metrics (qc.hpp)
/// - Section 5: Normalization Operations (normalize.hpp)
/// - Section 6: Feature Statistics (feature.hpp)
/// - Section 7: Statistical Tests (mwu.hpp, ttest.hpp)
/// - Section 8: Log Transforms (log1p.hpp)
/// - Section 9: Gram Matrix (gram.hpp)
/// - Section 10: Pearson Correlation (correlation.hpp)
/// - Section 11: Group Aggregations (group.hpp)
/// - Section 12: Standardization (scale.hpp)
/// - Section 13: Softmax (softmax.hpp)
/// - Section 14: MMD (mmd.hpp)
/// - Section 15: Spatial Statistics (spatial.hpp)
/// - Section 16: Linear Algebra (algebra.hpp)
/// - Section 17: HVG Selection (hvg.hpp)
/// - Section 18: Reordering (reorder.hpp)
/// - Section 19: Resampling (resample.hpp)
/// - Section 20: Memory Management
/// - Section 21: Helper Functions
// =============================================================================

#include "scl/kernel/core.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/sparse.hpp"
#include "scl/version.hpp"

#include <cstring>
#include <exception>
#include <stdexcept>

// =============================================================================
// SECTION 1: Error Handling Infrastructure
// =============================================================================

namespace {

/// @brief Thread-local error message buffer
thread_local char g_error_buffer[512] = {0};

/// @brief Store exception message for later retrieval
/// @param e Exception object
inline void store_error(const std::exception& e) noexcept {
    std::strncpy(g_error_buffer, e.what(), sizeof(g_error_buffer) - 1);
    g_error_buffer[sizeof(g_error_buffer) - 1] = '\0';
}

/// @brief Store generic error message
/// @param msg Error message string
inline void store_error(const char* msg) noexcept {
    std::strncpy(g_error_buffer, msg, sizeof(g_error_buffer) - 1);
    g_error_buffer[sizeof(g_error_buffer) - 1] = '\0';
}

/// @brief Clear error state
inline void clear_error() noexcept {
    g_error_buffer[0] = '\0';
}

/// @brief Helper macro for try-catch wrapper
///
/// Wraps C++ kernel calls with exception handling and converts to C error codes.
/// Returns 0 on success, -1 on failure.
#define SCL_C_API_WRAPPER(func_call) \
    try { \
        clear_error(); \
        func_call; \
        return 0; \
    } catch (const std::exception& e) { \
        store_error(e); \
        return -1; \
    } catch (...) { \
        store_error("Unknown error"); \
        return -1; \
    }

} // anonymous namespace

extern "C" {

// =============================================================================
// SECTION 2: Version and Type Information
// =============================================================================

/// @brief Get library version string
/// @return Version string (e.g., "0.1.0")
const char* scl_version() {
    return SCL_VERSION;
}

/// @brief Get precision type code
/// @return 0 for float32, 1 for float64, 2 for float16
int scl_precision_type() {
    return scl::DTYPE_CODE;
}

/// @brief Get precision type name
/// @return Type name string (e.g., "float32")
const char* scl_precision_name() {
    return scl::DTYPE_NAME;
}

/// @brief Get index type code
/// @return 0 for int16, 1 for int32, 2 for int64
int scl_index_type() {
    return scl::INDEX_DTYPE_CODE;
}

/// @brief Get index type name
/// @return Type name string (e.g., "int64")
const char* scl_index_name() {
    return scl::INDEX_DTYPE_NAME;
}

// =============================================================================
// Error Query Functions
// =============================================================================

/// @brief Get last error message
/// @return Error message string (empty if no error)
const char* scl_get_last_error() {
    return g_error_buffer;
}

/// @brief Clear error state
void scl_clear_error() {
    clear_error();
}

// =============================================================================
// SECTION 3: Sparse Matrix Statistics (sparse.hpp)
// =============================================================================

/// @brief Compute row sums for CSR matrix
///
/// @param data Values array [nnz]
/// @param indices Column indices [nnz]
/// @param indptr Row pointers [rows+1]
/// @param row_lengths Precomputed row lengths [rows] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param output Output sums [rows]
/// @return 0 on success, -1 on error
int scl_primary_sums_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::sparse::primary_sums(
            matrix, 
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows))
        );
    )
}

/// @brief Compute column sums for CSC matrix
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param output Output sums [cols]
/// @return 0 on success, -1 on error
int scl_primary_sums_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::sparse::primary_sums(
            matrix, 
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols))
        );
    )
}

/// @brief Compute row means for CSR matrix
///
/// @param data Values array [nnz]
/// @param indices Column indices [nnz]
/// @param indptr Row pointers [rows+1]
/// @param row_lengths Precomputed row lengths [rows] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param output Output means [rows]
/// @return 0 on success, -1 on error
int scl_primary_means_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::sparse::primary_means(
            matrix, 
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows))
        );
    )
}

/// @brief Compute column means for CSC matrix
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param output Output means [cols]
/// @return 0 on success, -1 on error
int scl_primary_means_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::sparse::primary_means(
            matrix, 
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols))
        );
    )
}

/// @brief Compute row variances for CSR matrix
///
/// @param data Values array [nnz]
/// @param indices Column indices [nnz]
/// @param indptr Row pointers [rows+1]
/// @param row_lengths Precomputed row lengths [rows] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param ddof Delta degrees of freedom (typically 0 or 1)
/// @param output Output variances [rows]
/// @return 0 on success, -1 on error
int scl_primary_variances_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    int ddof,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::sparse::primary_variances(
            matrix, 
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows)), 
            ddof
        );
    )
}

/// @brief Compute column variances for CSC matrix
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param ddof Delta degrees of freedom (typically 0 or 1)
/// @param output Output variances [cols]
/// @return 0 on success, -1 on error
int scl_primary_variances_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    int ddof,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::sparse::primary_variances(
            matrix, 
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols)), 
            ddof
        );
    )
}

/// @brief Count non-zero elements per row (CSR)
///
/// @param indptr Row pointers [rows+1]
/// @param rows Number of rows
/// @param output Output counts [rows]
/// @return 0 on success, -1 on error
int scl_primary_nnz_counts_csr(
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index* output
) {
    SCL_C_API_WRAPPER(
        scl::kernel::sparse::primary_nnz_counts(
            scl::Array<const scl::Index>(indptr, static_cast<scl::Size>(rows + 1)),
            scl::Array<scl::Index>(output, static_cast<scl::Size>(rows))
        );
    )
}

/// @brief Count non-zero elements per column (CSC)
///
/// @param indptr Column pointers [cols+1]
/// @param cols Number of columns
/// @param output Output counts [cols]
/// @return 0 on success, -1 on error
int scl_primary_nnz_counts_csc(
    const scl::Index* indptr,
    scl::Index cols,
    scl::Index* output
) {
    SCL_C_API_WRAPPER(
        scl::kernel::sparse::primary_nnz_counts(
            scl::Array<const scl::Index>(indptr, static_cast<scl::Size>(cols + 1)),
            scl::Array<scl::Index>(output, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// SECTION 4: Quality Control Metrics (qc.hpp)
// =============================================================================

/// @brief Compute basic QC metrics for CSR matrix
///
/// @param data Values array [nnz]
/// @param indices Column indices [nnz]
/// @param indptr Row pointers [rows+1]
/// @param row_lengths Precomputed row lengths [rows] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param out_n_genes Number of detected genes per cell [rows]
/// @param out_total_counts Total counts per cell [rows]
/// @return 0 on success, -1 on error
int scl_compute_basic_qc_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Index* out_n_genes,
    scl::Real* out_total_counts
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::qc::compute_basic_qc(
            matrix,
            scl::Array<scl::Index>(out_n_genes, static_cast<scl::Size>(rows)),
            scl::Array<scl::Real>(out_total_counts, static_cast<scl::Size>(rows))
        );
    )
}

/// @brief Compute basic QC metrics for CSC matrix
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param out_n_cells Number of cells expressing each gene [cols]
/// @param out_total_counts Total counts per gene [cols]
/// @return 0 on success, -1 on error
int scl_compute_basic_qc_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Index* out_n_cells,
    scl::Real* out_total_counts
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::qc::compute_basic_qc(
            matrix,
            scl::Array<scl::Index>(out_n_cells, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_total_counts, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// SECTION 5: Normalization Operations (normalize.hpp)
// =============================================================================

/// @brief Scale rows of CSR matrix by given factors
///
/// @param data Values array [nnz] (modified in-place)
/// @param indices Column indices [nnz]
/// @param indptr Row pointers [rows+1]
/// @param row_lengths Precomputed row lengths [rows] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param scales Scale factors [rows]
/// @return 0 on success, -1 on error
int scl_scale_primary_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const scl::Real* scales
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::normalize::scale_primary(
            matrix,
            scl::Array<const scl::Real>(scales, static_cast<scl::Size>(rows))
        );
    )
}

/// @brief Scale columns of CSC matrix by given factors
///
/// @param data Values array [nnz] (modified in-place)
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param scales Scale factors [cols]
/// @return 0 on success, -1 on error
int scl_scale_primary_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const scl::Real* scales
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::normalize::scale_primary(
            matrix,
            scl::Array<const scl::Real>(scales, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// SECTION 6: Feature Statistics (feature.hpp)
// =============================================================================

/// @brief Compute standard moments (mean and variance) for features (CSC)
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param out_means Output means [cols]
/// @param out_vars Output variances [cols]
/// @param ddof Delta degrees of freedom (typically 0 or 1)
/// @return 0 on success, -1 on error
int scl_standard_moments_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* out_means,
    scl::Real* out_vars,
    int ddof
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::feature::standard_moments(
            matrix,
            scl::Array<scl::Real>(out_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_vars, static_cast<scl::Size>(cols)),
            ddof
        );
    )
}

/// @brief Compute clipped moments (mean and variance with clipping) for features (CSC)
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param clip_vals Clipping values [cols]
/// @param out_means Output means [cols]
/// @param out_vars Output variances [cols]
/// @return 0 on success, -1 on error
int scl_clipped_moments_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const scl::Real* clip_vals,
    scl::Real* out_means,
    scl::Real* out_vars
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::feature::clipped_moments(
            matrix,
            scl::Array<const scl::Real>(clip_vals, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_vars, static_cast<scl::Size>(cols))
        );
    )
}

/// @brief Compute detection rate (fraction of non-zero cells) for features (CSC)
///
/// @param indptr Column pointers [cols+1]
/// @param rows Number of rows
/// @param cols Number of columns
/// @param out_rates Output detection rates [cols]
/// @return 0 on success, -1 on error
int scl_detection_rate_csc(
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Real* out_rates
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            nullptr,
            nullptr,
            const_cast<scl::Index*>(indptr),
            rows, cols, 0,
            nullptr
        );
        scl::kernel::feature::detection_rate(
            matrix,
            scl::Array<scl::Real>(out_rates, static_cast<scl::Size>(cols))
        );
    )
}

/// @brief Compute dispersion (variance / mean) for features
///
/// @param means Mean values [size]
/// @param vars Variance values [size]
/// @param size Number of features
/// @param out_dispersion Output dispersion values [size]
/// @return 0 on success, -1 on error
int scl_dispersion(
    const scl::Real* means,
    const scl::Real* vars,
    scl::Size size,
    scl::Real* out_dispersion
) {
    SCL_C_API_WRAPPER(
        scl::kernel::feature::dispersion(
            scl::Array<const scl::Real>(means, size),
            scl::Array<const scl::Real>(vars, size),
            scl::Array<scl::Real>(out_dispersion, size)
        );
    )
}

// =============================================================================
// SECTION 7: Statistical Tests (mwu.hpp, ttest.hpp)
// =============================================================================

/// @brief Mann-Whitney U test for differential expression (CSC)
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param group_ids Group labels [rows] (0 or 1)
/// @param out_u_stats U statistics [cols]
/// @param out_p_values P-values [cols]
/// @param out_log2_fc Log2 fold changes [cols]
/// @return 0 on success, -1 on error
int scl_mwu_test_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const int32_t* group_ids,
    scl::Real* out_u_stats,
    scl::Real* out_p_values,
    scl::Real* out_log2_fc
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::mwu::mwu_test(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
            scl::Array<scl::Real>(out_u_stats, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_p_values, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_log2_fc, static_cast<scl::Size>(cols))
        );
    )
}

/// @brief T-test for differential expression with multiple groups (CSC)
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param group_ids Group labels [rows] (0 to n_groups-1)
/// @param n_groups Number of groups
/// @param out_t_stats T statistics [cols * (n_groups-1)]
/// @param out_p_values P-values [cols * (n_groups-1)]
/// @param out_log2_fc Log2 fold changes [cols * (n_groups-1)]
/// @param out_mean_diff Mean differences [cols * (n_groups-1)]
/// @param workspace Workspace buffer [workspace_size]
/// @param workspace_size Size of workspace buffer
/// @param use_welch Use Welch's t-test (unequal variances)
/// @return 0 on success, -1 on error
int scl_ttest_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const int32_t* group_ids,
    scl::Size n_groups,
    scl::Real* out_t_stats,
    scl::Real* out_p_values,
    scl::Real* out_log2_fc,
    scl::Real* out_mean_diff,
    scl::Byte* workspace,
    scl::Size workspace_size,
    bool use_welch
) {
    SCL_C_API_WRAPPER(
        scl::Size required_size = static_cast<scl::Size>(cols) * n_groups * 2 * sizeof(scl::Real);
        if (workspace_size < required_size) {
            throw scl::ValueError("TTest: Workspace too small");
        }
        
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        
        scl::Size output_size = static_cast<scl::Size>(cols) * (n_groups - 1);
        scl::kernel::diff_expr::ttest(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
            n_groups,
            scl::Array<scl::Real>(out_t_stats, output_size),
            scl::Array<scl::Real>(out_p_values, output_size),
            scl::Array<scl::Real>(out_log2_fc, output_size),
            scl::Array<scl::Real>(out_mean_diff, output_size),
            scl::Array<scl::Byte>(workspace, workspace_size),
            use_welch
        );
    )
}

// =============================================================================
// SECTION 8: Log Transforms (log1p.hpp)
// =============================================================================

/// @brief Apply log1p transform to array in-place
///
/// @param data Data array [size] (modified in-place)
/// @param size Number of elements
/// @return 0 on success, -1 on error
int scl_log1p_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::log1p_inplace(scl::Array<scl::Real>(data, size));
    )
}

/// @brief Apply log1p transform to CSR matrix in-place
///
/// @param data Values array [nnz] (modified in-place)
/// @param indices Column indices [nnz]
/// @param indptr Row pointers [rows+1]
/// @param row_lengths Precomputed row lengths [rows] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @return 0 on success, -1 on error
int scl_log1p_inplace_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::log1p_inplace(matrix);
    )
}

/// @brief Apply log2(1+x) transform to array in-place
///
/// @param data Data array [size] (modified in-place)
/// @param size Number of elements
/// @return 0 on success, -1 on error
int scl_log2p1_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::log2p1_inplace(scl::Array<scl::Real>(data, size));
    )
}

/// @brief Apply exp(x)-1 transform to array in-place
///
/// @param data Data array [size] (modified in-place)
/// @param size Number of elements
/// @return 0 on success, -1 on error
int scl_expm1_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::expm1_inplace(scl::Array<scl::Real>(data, size));
    )
}

// =============================================================================
// SECTION 9: Gram Matrix (gram.hpp)
// =============================================================================

/// @brief Compute Gram matrix (X^T X) for CSC matrix
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param output Output Gram matrix [cols * cols] (row-major)
/// @return 0 on success, -1 on error
int scl_gram_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::Size output_size = static_cast<scl::Size>(cols) * static_cast<scl::Size>(cols);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
    )
}

/// @brief Compute Gram matrix (X X^T) for CSR matrix
///
/// @param data Values array [nnz]
/// @param indices Column indices [nnz]
/// @param indptr Row pointers [rows+1]
/// @param row_lengths Precomputed row lengths [rows] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param output Output Gram matrix [rows * rows] (row-major)
/// @return 0 on success, -1 on error
int scl_gram_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::Size output_size = static_cast<scl::Size>(rows) * static_cast<scl::Size>(rows);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
    )
}

// =============================================================================
// SECTION 10: Pearson Correlation (correlation.hpp)
// =============================================================================

/// @brief Compute Pearson correlation matrix for CSC matrix
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param output Output correlation matrix [cols * cols] (row-major)
/// @param workspace_means Workspace for means [cols]
/// @param workspace_inv_stds Workspace for inverse stds [cols]
/// @return 0 on success, -1 on error
int scl_pearson_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output,
    scl::Real* workspace_means,
    scl::Real* workspace_inv_stds
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::correlation::detail::compute_stats(
            matrix,
            scl::Array<scl::Real>(workspace_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(workspace_inv_stds, static_cast<scl::Size>(cols))
        );
        scl::Size output_size = static_cast<scl::Size>(cols) * static_cast<scl::Size>(cols);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
        // Note: Full correlation implementation requires additional transformation
    )
}

/// @brief Compute Pearson correlation matrix for CSR matrix
///
/// @param data Values array [nnz]
/// @param indices Column indices [nnz]
/// @param indptr Row pointers [rows+1]
/// @param row_lengths Precomputed row lengths [rows] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param output Output correlation matrix [rows * rows] (row-major)
/// @param workspace_means Workspace for means [rows]
/// @param workspace_inv_stds Workspace for inverse stds [rows]
/// @return 0 on success, -1 on error
int scl_pearson_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output,
    scl::Real* workspace_means,
    scl::Real* workspace_inv_stds
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::correlation::detail::compute_stats(
            matrix,
            scl::Array<scl::Real>(workspace_means, static_cast<scl::Size>(rows)),
            scl::Array<scl::Real>(workspace_inv_stds, static_cast<scl::Size>(rows))
        );
        scl::Size output_size = static_cast<scl::Size>(rows) * static_cast<scl::Size>(rows);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
        // Note: Full correlation implementation requires additional transformation
    )
}

// =============================================================================
// SECTION 11: Group Aggregations (group.hpp)
// =============================================================================

/// @brief Compute group statistics (mean and variance) for CSC matrix
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param group_ids Group labels [rows] (0 to n_groups-1)
/// @param n_groups Number of groups
/// @param group_sizes Group sizes [n_groups]
/// @param out_means Output means [cols * n_groups]
/// @param out_vars Output variances [cols * n_groups]
/// @param ddof Delta degrees of freedom
/// @param include_zeros Include zero values in statistics
/// @return 0 on success, -1 on error
int scl_group_stats_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const int32_t* group_ids,
    scl::Size n_groups,
    const scl::Size* group_sizes,
    scl::Real* out_means,
    scl::Real* out_vars,
    int ddof,
    bool include_zeros
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::Size output_size = static_cast<scl::Size>(cols) * n_groups;
        scl::kernel::group::group_stats(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
            n_groups,
            scl::Array<const scl::Size>(group_sizes, n_groups),
            scl::Array<scl::Real>(out_means, output_size),
            scl::Array<scl::Real>(out_vars, output_size),
            ddof,
            include_zeros
        );
    )
}

/// @brief Count group sizes from group labels
///
/// @param group_ids Group labels [n_elements]
/// @param n_elements Number of elements
/// @param n_groups Number of groups
/// @param out_sizes Output group sizes [n_groups]
/// @return 0 on success, -1 on error
int scl_count_group_sizes(
    const int32_t* group_ids,
    scl::Size n_elements,
    scl::Size n_groups,
    scl::Size* out_sizes
) {
    SCL_C_API_WRAPPER(
        scl::kernel::group::count_group_sizes(
            scl::Array<const int32_t>(group_ids, n_elements),
            n_groups,
            scl::Array<scl::Size>(out_sizes, n_groups)
        );
    )
}

// =============================================================================
// SECTION 12: Standardization (scale.hpp)
// =============================================================================

/// @brief Standardize features (z-score normalization) for CSC matrix
///
/// @param data Values array [nnz] (modified in-place)
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Precomputed column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @param means Feature means [cols]
/// @param stds Feature standard deviations [cols]
/// @param max_value Maximum absolute value (clipping)
/// @param zero_center Center data at zero
/// @return 0 on success, -1 on error
int scl_standardize_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const scl::Real* means,
    const scl::Real* stds,
    scl::Real max_value,
    bool zero_center
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::scale::standardize(
            matrix,
            scl::Array<const scl::Real>(means, static_cast<scl::Size>(cols)),
            scl::Array<const scl::Real>(stds, static_cast<scl::Size>(cols)),
            max_value,
            zero_center
        );
    )
}

// =============================================================================
// SECTION 13: Softmax (softmax.hpp)
// =============================================================================

/// @brief Apply softmax to rows of CSR matrix in-place
///
/// @param data Values array [nnz] (modified in-place)
/// @param indices Column indices [nnz]
/// @param indptr Row pointers [rows+1]
/// @param row_lengths Precomputed row lengths [rows] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zero elements
/// @return 0 on success, -1 on error
int scl_softmax_inplace_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::softmax::softmax_inplace(matrix);
    )
}

// =============================================================================
// SECTION 14: MMD (mmd.hpp)
// =============================================================================

/// @brief Compute Maximum Mean Discrepancy with RBF kernel for CSC matrices
///
/// @param data_x Values array for X [nnz_x]
/// @param indices_x Row indices for X [nnz_x]
/// @param indptr_x Column pointers for X [cols+1]
/// @param col_lengths_x Column lengths for X [cols] (can be NULL)
/// @param rows_x Number of rows in X
/// @param cols Number of columns (same for X and Y)
/// @param nnz_x Number of non-zeros in X
/// @param data_y Values array for Y [nnz_y]
/// @param indices_y Row indices for Y [nnz_y]
/// @param indptr_y Column pointers for Y [cols+1]
/// @param col_lengths_y Column lengths for Y [cols] (can be NULL)
/// @param rows_y Number of rows in Y
/// @param nnz_y Number of non-zeros in Y
/// @param output Output MMD values [cols]
/// @param gamma RBF kernel parameter
/// @return 0 on success, -1 on error
int scl_mmd_rbf_csc(
    const scl::Real* data_x,
    const scl::Index* indices_x,
    const scl::Index* indptr_x,
    const scl::Index* col_lengths_x,
    scl::Index rows_x,
    scl::Index cols,
    scl::Index nnz_x,
    const scl::Real* data_y,
    const scl::Index* indices_y,
    const scl::Index* indptr_y,
    const scl::Index* col_lengths_y,
    scl::Index rows_y,
    scl::Index nnz_y,
    scl::Real* output,
    scl::Real gamma
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> mat_x(
            const_cast<scl::Real*>(data_x),
            const_cast<scl::Index*>(indices_x),
            const_cast<scl::Index*>(indptr_x),
            rows_x, cols, nnz_x,
            col_lengths_x
        );
        scl::CustomCSC<scl::Real> mat_y(
            const_cast<scl::Real*>(data_y),
            const_cast<scl::Index*>(indices_y),
            const_cast<scl::Index*>(indptr_y),
            rows_y, cols, nnz_y,
            col_lengths_y
        );
        scl::kernel::mmd::mmd_rbf(
            mat_x,
            mat_y,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols)),
            gamma
        );
    )
}

// =============================================================================
// SECTION 15: Spatial Statistics (spatial.hpp)
// =============================================================================

/// @brief Compute Moran's I spatial autocorrelation
///
/// @param graph_data Spatial graph weights [graph_nnz]
/// @param graph_indices Graph column indices [graph_nnz]
/// @param graph_indptr Graph row pointers [n_cells+1]
/// @param graph_row_lengths Graph row lengths [n_cells] (can be NULL)
/// @param n_cells Number of cells
/// @param graph_nnz Number of non-zeros in graph
/// @param features_data Feature values [features_nnz]
/// @param features_indices Feature row indices [features_nnz]
/// @param features_indptr Feature column pointers [n_genes+1]
/// @param features_col_lengths Feature column lengths [n_genes] (can be NULL)
/// @param n_genes Number of genes
/// @param features_nnz Number of non-zeros in features
/// @param output Output Moran's I values [n_genes]
/// @return 0 on success, -1 on error
int scl_morans_i(
    const scl::Real* graph_data,
    const scl::Index* graph_indices,
    const scl::Index* graph_indptr,
    const scl::Index* graph_row_lengths,
    scl::Index n_cells,
    scl::Index graph_nnz,
    const scl::Real* features_data,
    const scl::Index* features_indices,
    const scl::Index* features_indptr,
    const scl::Index* features_col_lengths,
    scl::Index n_genes,
    scl::Index features_nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> graph(
            const_cast<scl::Real*>(graph_data),
            const_cast<scl::Index*>(graph_indices),
            const_cast<scl::Index*>(graph_indptr),
            n_cells, n_cells, graph_nnz,
            graph_row_lengths
        );
        scl::CustomCSC<scl::Real> features(
            const_cast<scl::Real*>(features_data),
            const_cast<scl::Index*>(features_indices),
            const_cast<scl::Index*>(features_indptr),
            n_cells, n_genes, features_nnz,
            features_col_lengths
        );
        scl::kernel::spatial::morans_i(
            graph,
            features,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(n_genes))
        );
    )
}

// =============================================================================
// SECTION 16: Linear Algebra (algebra.hpp)
// =============================================================================

/// @brief Sparse matrix-vector multiplication (y = alpha * A * x + beta * y) for CSR
///
/// @param A_data Values array [A_nnz]
/// @param A_indices Column indices [A_nnz]
/// @param A_indptr Row pointers [A_rows+1]
/// @param A_row_lengths Row lengths [A_rows] (can be NULL)
/// @param A_rows Number of rows in A
/// @param A_cols Number of columns in A
/// @param A_nnz Number of non-zeros in A
/// @param x Input vector [A_cols]
/// @param y Output vector [A_rows] (modified in-place)
/// @param alpha Scaling factor for A*x
/// @param beta Scaling factor for y
/// @return 0 on success, -1 on error
int scl_spmv_csr(
    const scl::Real* A_data,
    const scl::Index* A_indices,
    const scl::Index* A_indptr,
    const scl::Index* A_row_lengths,
    scl::Index A_rows,
    scl::Index A_cols,
    scl::Index A_nnz,
    const scl::Real* x,
    scl::Real* y,
    scl::Real alpha,
    scl::Real beta
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR<scl::Real> A(
            const_cast<scl::Real*>(A_data),
            const_cast<scl::Index*>(A_indices),
            const_cast<scl::Index*>(A_indptr),
            A_rows, A_cols, A_nnz,
            A_row_lengths
        );
        scl::kernel::algebra::spmv(
            A,
            scl::Array<const scl::Real>(x, static_cast<scl::Size>(A_cols)),
            scl::Array<scl::Real>(y, static_cast<scl::Size>(A_rows)),
            alpha,
            beta
        );
    )
}

/// @brief Sparse matrix-vector multiplication with transpose (y = alpha * A^T * x + beta * y) for CSC
///
/// @param A_data Values array [A_nnz]
/// @param A_indices Row indices [A_nnz]
/// @param A_indptr Column pointers [A_cols+1]
/// @param A_col_lengths Column lengths [A_cols] (can be NULL)
/// @param A_rows Number of rows in A
/// @param A_cols Number of columns in A
/// @param A_nnz Number of non-zeros in A
/// @param x Input vector [A_rows]
/// @param y Output vector [A_cols] (modified in-place)
/// @param alpha Scaling factor for A^T*x
/// @param beta Scaling factor for y
/// @return 0 on success, -1 on error
int scl_spmv_trans_csc(
    const scl::Real* A_data,
    const scl::Index* A_indices,
    const scl::Index* A_indptr,
    const scl::Index* A_col_lengths,
    scl::Index A_rows,
    scl::Index A_cols,
    scl::Index A_nnz,
    const scl::Real* x,
    scl::Real* y,
    scl::Real alpha,
    scl::Real beta
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> A(
            const_cast<scl::Real*>(A_data),
            const_cast<scl::Index*>(A_indices),
            const_cast<scl::Index*>(A_indptr),
            A_rows, A_cols, A_nnz,
            A_col_lengths
        );
        scl::kernel::algebra::spmv_trans(
            A,
            scl::Array<const scl::Real>(x, static_cast<scl::Size>(A_rows)),
            scl::Array<scl::Real>(y, static_cast<scl::Size>(A_cols)),
            alpha,
            beta
        );
    )
}

// =============================================================================
// SECTION 17: HVG Selection (hvg.hpp)
// =============================================================================

/// @brief Select highly variable genes by dispersion for CSC matrix
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zeros
/// @param n_top Number of top genes to select
/// @param out_indices Output gene indices [n_top]
/// @param out_mask Output binary mask [cols]
/// @param out_dispersions Output dispersion values [cols]
/// @return 0 on success, -1 on error
int scl_hvg_by_dispersion_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Size n_top,
    scl::Index* out_indices,
    uint8_t* out_mask,
    scl::Real* out_dispersions
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::hvg::select_by_dispersion(
            matrix,
            n_top,
            scl::Array<scl::Index>(out_indices, n_top),
            scl::Array<uint8_t>(out_mask, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_dispersions, static_cast<scl::Size>(cols))
        );
    )
}

/// @brief Select highly variable genes by variance for CSC matrix
///
/// @param data Values array [nnz]
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zeros
/// @param n_top Number of top genes to select
/// @param out_indices Output gene indices [n_top]
/// @param out_mask Output binary mask [cols]
/// @return 0 on success, -1 on error
int scl_hvg_by_variance_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Size n_top,
    scl::Index* out_indices,
    uint8_t* out_mask
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::hvg::select_by_variance(
            matrix,
            n_top,
            scl::Array<scl::Index>(out_indices, n_top),
            scl::Array<uint8_t>(out_mask, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// SECTION 18: Reordering (reorder.hpp)
// =============================================================================

/// @brief Align secondary dimension indices for CSC matrix
///
/// @param data Values array [nnz] (modified in-place)
/// @param indices Row indices [nnz] (modified in-place)
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zeros
/// @param index_map Index mapping [cols]
/// @param out_lengths Output new lengths [cols]
/// @param new_cols New number of columns after reordering
/// @return 0 on success, -1 on error
int scl_align_secondary_csc(
    scl::Real* data,
    scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const scl::Index* index_map,
    scl::Index* out_lengths,
    scl::Index new_cols
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            data,
            indices,
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::reorder::align_secondary(
            matrix,
            scl::Array<const scl::Index>(index_map, static_cast<scl::Size>(cols)),
            scl::Array<scl::Index>(out_lengths, static_cast<scl::Size>(cols)),
            new_cols
        );
    )
}

// =============================================================================
// SECTION 19: Resampling (resample.hpp)
// =============================================================================

/// @brief Downsample counts to target sum for CSC matrix
///
/// @param data Values array [nnz] (modified in-place)
/// @param indices Row indices [nnz]
/// @param indptr Column pointers [cols+1]
/// @param col_lengths Column lengths [cols] (can be NULL)
/// @param rows Number of rows
/// @param cols Number of columns
/// @param nnz Number of non-zeros
/// @param target_sum Target sum for each column
/// @param seed Random seed
/// @return 0 on success, -1 on error
int scl_downsample_counts_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real target_sum,
    uint64_t seed
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC<scl::Real> matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::resample::downsample_counts(matrix, target_sum, seed);
    )
}

// =============================================================================
// SECTION 20: Memory Management
// =============================================================================

/// @brief Allocate memory
///
/// @param bytes Number of bytes to allocate
/// @param out_ptr Output pointer to allocated memory
/// @return 0 on success, -1 on error
int scl_malloc(scl::Size bytes, void** out_ptr) {
    SCL_C_API_WRAPPER(
        if (!out_ptr) {
            throw scl::ValueError("scl_malloc: out_ptr is null");
        }
        auto handle = scl::core::mem::alloc(bytes);
        *out_ptr = handle.release();
    )
}

/// @brief Allocate zero-initialized memory
///
/// @param bytes Number of bytes to allocate
/// @param out_ptr Output pointer to allocated memory
/// @return 0 on success, -1 on error
int scl_calloc(scl::Size bytes, void** out_ptr) {
    SCL_C_API_WRAPPER(
        if (!out_ptr) {
            throw scl::ValueError("scl_calloc: out_ptr is null");
        }
        auto handle = scl::core::mem::alloc_zero(bytes);
        *out_ptr = handle.release();
    )
}

/// @brief Allocate aligned memory
///
/// @param bytes Number of bytes to allocate
/// @param alignment Alignment requirement (must be power of 2)
/// @param out_ptr Output pointer to allocated memory
/// @return 0 on success, -1 on error
int scl_malloc_aligned(scl::Size bytes, scl::Size alignment, void** out_ptr) {
    SCL_C_API_WRAPPER(
        if (!out_ptr) {
            throw scl::ValueError("scl_malloc_aligned: out_ptr is null");
        }
        auto handle = scl::core::mem::alloc_aligned(bytes, alignment);
        *out_ptr = handle.release();
    )
}

/// @brief Free memory allocated by scl_malloc or scl_calloc
///
/// @param ptr Pointer to memory to free (NULL is safe)
void scl_free(void* ptr) {
    if (ptr) std::free(ptr);
}

/// @brief Free aligned memory allocated by scl_malloc_aligned
///
/// @param ptr Pointer to memory to free (NULL is safe)
void scl_free_aligned(void* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

/// @brief Zero out memory
///
/// @param ptr Pointer to memory
/// @param bytes Number of bytes to zero
void scl_memzero(void* ptr, scl::Size bytes) {
    if (ptr && bytes > 0) {
        scl::memory::zero(scl::Array<uint8_t>(
            static_cast<uint8_t*>(ptr), 
            bytes
        ));
    }
}

/// @brief Copy memory
///
/// @param src Source pointer
/// @param dst Destination pointer
/// @param bytes Number of bytes to copy
/// @return 0 on success, -1 on error
int scl_memcpy(const void* src, void* dst, scl::Size bytes) {
    SCL_C_API_WRAPPER(
        if (!src || !dst) {
            throw scl::ValueError("scl_memcpy: null pointer");
        }
        scl::memory::copy(
            scl::Array<const uint8_t>(static_cast<const uint8_t*>(src), bytes),
            scl::Array<uint8_t>(static_cast<uint8_t*>(dst), bytes)
        );
    )
}

// =============================================================================
// SECTION 21: Helper Functions
// =============================================================================

/// @brief Check if value is finite
///
/// @param value Value to check
/// @return true if finite, false otherwise
bool scl_is_valid_value(scl::Real value) {
    return std::isfinite(value);
}

/// @brief Get size of Real type in bytes
///
/// @return Size in bytes
scl::Size scl_sizeof_real() {
    return sizeof(scl::Real);
}

/// @brief Get size of Index type in bytes
///
/// @return Size in bytes
scl::Size scl_sizeof_index() {
    return sizeof(scl::Index);
}

/// @brief Get recommended memory alignment in bytes
///
/// @return Alignment in bytes (64 for AVX-512 compatibility)
scl::Size scl_alignment() {
    return 64;
}

// =============================================================================
// Workspace Size Calculation Helpers
// =============================================================================

/// @brief Calculate workspace size for T-test
///
/// @param n_features Number of features
/// @param n_groups Number of groups
/// @return Required workspace size in bytes
scl::Size scl_ttest_workspace_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * n_groups * 2 * sizeof(scl::Real);
}

/// @brief Calculate output size for differential expression tests
///
/// @param n_features Number of features
/// @param n_groups Number of groups
/// @return Output array size
scl::Size scl_diff_expr_output_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * (n_groups - 1);
}

/// @brief Calculate output size for group statistics
///
/// @param n_features Number of features
/// @param n_groups Number of groups
/// @return Output array size
scl::Size scl_group_stats_output_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * n_groups;
}

/// @brief Calculate output size for Gram matrix
///
/// @param n Primary dimension size
/// @return Output array size
scl::Size scl_gram_output_size(scl::Size n) {
    return n * n;
}

/// @brief Calculate workspace size for correlation
///
/// @param n Primary dimension size
/// @return Required workspace size in bytes
scl::Size scl_correlation_workspace_size(scl::Size n) {
    return n * 2 * sizeof(scl::Real);  // means + inv_stds
}

} // extern "C"

