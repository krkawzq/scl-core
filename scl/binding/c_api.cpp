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
#include "scl/core/sparse.hpp"
#include "scl/version.hpp"

#include <cstring>
#include <exception>

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

const char* scl_version() {
    return SCL_VERSION;
}

int scl_precision_type() {
    return scl::DTYPE_CODE;
}

const char* scl_precision_name() {
    return scl::DTYPE_NAME;
}

int scl_index_type() {
    return scl::INDEX_DTYPE_CODE;
}

const char* scl_index_name() {
    return scl::INDEX_DTYPE_NAME;
}

// =============================================================================
// Error Query Functions
// =============================================================================

const char* scl_get_last_error() {
    return g_error_buffer;
}

void scl_clear_error() {
    clear_error();
}

// =============================================================================
// SECTION 3: Sparse Matrix Statistics (sparse.hpp)
// =============================================================================

int scl_primary_sums_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::sparse::primary_sums(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows))
        );
    )
}

int scl_primary_sums_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::sparse::primary_sums(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols))
        );
    )
}

int scl_primary_means_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::sparse::primary_means(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows))
        );
    )
}

int scl_primary_means_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::sparse::primary_means(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols))
        );
    )
}

int scl_primary_variances_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    int ddof,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::sparse::primary_variances(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows)),
            ddof
        );
    )
}

int scl_primary_variances_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    int ddof,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::sparse::primary_variances(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols)),
            ddof
        );
    )
}

int scl_primary_nnz_counts_csr(
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            nullptr,
            nullptr,
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::sparse::primary_nnz_counts(
            matrix,
            scl::Array<scl::Index>(output, static_cast<scl::Size>(rows))
        );
    )
}

int scl_primary_nnz_counts_csc(
    const scl::Index* indptr,
    scl::Index cols,
    scl::Index* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            nullptr,
            nullptr,
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::sparse::primary_nnz_counts(
            matrix,
            scl::Array<scl::Index>(output, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// SECTION 4: Quality Control Metrics (qc.hpp)
// =============================================================================

int scl_compute_basic_qc_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    scl::Index* out_n_genes,
    scl::Real* out_total_counts
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::qc::compute_basic_qc(
            matrix,
            scl::Array<scl::Index>(out_n_genes, static_cast<scl::Size>(rows)),
            scl::Array<scl::Real>(out_total_counts, static_cast<scl::Size>(rows))
        );
    )
}

int scl_compute_basic_qc_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Index* out_n_cells,
    scl::Real* out_total_counts
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
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

int scl_scale_primary_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    const scl::Real* scales
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::normalize::scale_primary(
            matrix,
            scl::Array<const scl::Real>(scales, static_cast<scl::Size>(rows))
        );
    )
}

int scl_scale_primary_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    const scl::Real* scales
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
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

int scl_standard_moments_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Real* out_means,
    scl::Real* out_vars,
    int ddof
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::feature::standard_moments(
            matrix,
            scl::Array<scl::Real>(out_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_vars, static_cast<scl::Size>(cols)),
            ddof
        );
    )
}

int scl_clipped_moments_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    const scl::Real* clip_vals,
    scl::Real* out_means,
    scl::Real* out_vars
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::feature::clipped_moments(
            matrix,
            scl::Array<const scl::Real>(clip_vals, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_vars, static_cast<scl::Size>(cols))
        );
    )
}

int scl_detection_rate_csc(
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Real* out_rates
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            nullptr,
            nullptr,
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::feature::detection_rate(
            matrix,
            scl::Array<scl::Real>(out_rates, static_cast<scl::Size>(cols))
        );
    )
}

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

int scl_mwu_test_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    const int32_t* group_ids,
    scl::Real* out_u_stats,
    scl::Real* out_p_values,
    scl::Real* out_log2_fc
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols
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

int scl_ttest_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    const int32_t* group_ids,
    scl::Size n_groups,
    scl::Real* out_t_stats,
    scl::Real* out_p_values,
    scl::Real* out_log2_fc,
    bool use_welch
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols
        );

        scl::Size output_size = static_cast<scl::Size>(cols) * (n_groups - 1);
        scl::kernel::diff_expr::ttest(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
            n_groups,
            scl::Array<scl::Real>(out_t_stats, output_size),
            scl::Array<scl::Real>(out_p_values, output_size),
            scl::Array<scl::Real>(out_log2_fc, output_size),
            use_welch
        );
    )
}

// =============================================================================
// SECTION 8: Log Transforms (log1p.hpp)
// =============================================================================

int scl_log1p_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::log1p_inplace(scl::Array<scl::Real>(data, size));
    )
}

int scl_log1p_inplace_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::log1p_inplace(matrix);
    )
}

int scl_log2p1_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::log2p1_inplace(scl::Array<scl::Real>(data, size));
    )
}

int scl_expm1_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::expm1_inplace(scl::Array<scl::Real>(data, size));
    )
}

// =============================================================================
// SECTION 9: Gram Matrix (gram.hpp)
// =============================================================================

int scl_gram_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::Size output_size = static_cast<scl::Size>(cols) * static_cast<scl::Size>(cols);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
    )
}

int scl_gram_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::Size output_size = static_cast<scl::Size>(rows) * static_cast<scl::Size>(rows);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
    )
}

// =============================================================================
// SECTION 10: Pearson Correlation (correlation.hpp)
// =============================================================================

int scl_pearson_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Real* output,
    scl::Real* workspace_means,
    scl::Real* workspace_inv_stds
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
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

int scl_pearson_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    scl::Real* output,
    scl::Real* workspace_means,
    scl::Real* workspace_inv_stds
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
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

int scl_group_stats_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    const int32_t* group_ids,
    scl::Size n_groups,
    const scl::Size* group_sizes,
    scl::Real* out_means,
    scl::Real* out_vars,
    int ddof,
    bool include_zeros
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::Size output_size = static_cast<scl::Size>(cols) * n_groups;
        scl::kernel::group::group_stats(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(0)), // rows=0
            n_groups,
            scl::Array<const scl::Size>(group_sizes, n_groups),
            scl::Array<scl::Real>(out_means, output_size),
            scl::Array<scl::Real>(out_vars, output_size),
            ddof,
            include_zeros
        );
    )
}

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

int scl_standardize_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    const scl::Real* means,
    const scl::Real* stds,
    scl::Real max_value,
    bool zero_center
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
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

int scl_softmax_inplace_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::softmax::softmax_inplace(matrix);
    )
}

// =============================================================================
// SECTION 14: MMD (mmd.hpp)
// =============================================================================

int scl_mmd_rbf_csc(
    const scl::Real* data_x,
    const scl::Index* indices_x,
    const scl::Index* indptr_x,
    scl::Index rows_x,
    scl::Index cols,
    const scl::Real* data_y,
    const scl::Index* indices_y,
    const scl::Index* indptr_y,
    scl::Index rows_y,
    scl::Real* output,
    scl::Real gamma
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC mat_x(
            const_cast<scl::Real*>(data_x),
            const_cast<scl::Index*>(indices_x),
            const_cast<scl::Index*>(indptr_x),
            rows_x, cols
        );
        scl::CustomCSC mat_y(
            const_cast<scl::Real*>(data_y),
            const_cast<scl::Index*>(indices_y),
            const_cast<scl::Index*>(indptr_y),
            rows_y, cols
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

int scl_morans_i(
    const scl::Real* graph_data,
    const scl::Index* graph_indices,
    const scl::Index* graph_indptr,
    scl::Index n_cells,
    const scl::Real* features_data,
    const scl::Index* features_indices,
    const scl::Index* features_indptr,
    scl::Index n_genes,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR graph(
            const_cast<scl::Real*>(graph_data),
            const_cast<scl::Index*>(graph_indices),
            const_cast<scl::Index*>(graph_indptr),
            n_cells, n_cells
        );
        scl::CustomCSC features(
            const_cast<scl::Real*>(features_data),
            const_cast<scl::Index*>(features_indices),
            const_cast<scl::Index*>(features_indptr),
            n_cells, n_genes
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

int scl_spmv_csr(
    const scl::Real* A_data,
    const scl::Index* A_indices,
    const scl::Index* A_indptr,
    scl::Index A_rows,
    scl::Index A_cols,
    const scl::Real* x,
    scl::Real* y,
    scl::Real alpha,
    scl::Real beta
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR A(
            const_cast<scl::Real*>(A_data),
            const_cast<scl::Index*>(A_indices),
            const_cast<scl::Index*>(A_indptr),
            A_rows, A_cols
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

int scl_spmv_trans_csc(
    const scl::Real* A_data,
    const scl::Index* A_indices,
    const scl::Index* A_indptr,
    scl::Index A_rows,
    scl::Index A_cols,
    const scl::Real* x,
    scl::Real* y,
    scl::Real alpha,
    scl::Real beta
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC A(
            const_cast<scl::Real*>(A_data),
            const_cast<scl::Index*>(A_indices),
            const_cast<scl::Index*>(A_indptr),
            A_rows, A_cols
        );
        // For CSC matrix, spmv computes y = A^T * x naturally
        // because primary_dim = cols, secondary_dim = rows
        scl::kernel::algebra::spmv(
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

int scl_hvg_by_dispersion_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index cols,
    scl::Size n_top,
    scl::Index* out_indices,
    uint8_t* out_mask,
    scl::Real* out_dispersions
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
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

int scl_hvg_by_variance_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Size n_top,
    scl::Index* out_indices,
    uint8_t* out_mask
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
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

int scl_align_secondary_csc(
    scl::Real* data,
    scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    const scl::Index* index_map,
    scl::Index* out_lengths,
    scl::Index new_cols
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            data,
            indices,
            const_cast<scl::Index*>(indptr),
            0, cols
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

int scl_downsample_counts_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Real target_sum,
    uint64_t seed
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::resample::downsample_counts(matrix, target_sum, seed);
    )
}

// =============================================================================
// SECTION 20: Memory Management
// =============================================================================

int scl_malloc(scl::Size bytes, void** out_ptr) {
    SCL_C_API_WRAPPER(
        if (!out_ptr) {
            throw scl::ValueError("scl_malloc: out_ptr is null");
        }
        auto handle = scl::core::mem::alloc(bytes);
        *out_ptr = handle.release();
    )
}

int scl_calloc(scl::Size bytes, void** out_ptr) {
    SCL_C_API_WRAPPER(
        if (!out_ptr) {
            throw scl::ValueError("scl_calloc: out_ptr is null");
        }
        auto handle = scl::core::mem::alloc_zero(bytes);
        *out_ptr = handle.release();
    )
}

int scl_malloc_aligned(scl::Size bytes, scl::Size alignment, void** out_ptr) {
    SCL_C_API_WRAPPER(
        if (!out_ptr) {
            throw scl::ValueError("scl_malloc_aligned: out_ptr is null");
        }
        auto handle = scl::core::mem::alloc_aligned(bytes, alignment);
        *out_ptr = handle.release();
    )
}

void scl_free(void* ptr) {
    if (ptr) std::free(ptr);
}

void scl_free_aligned(void* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void scl_memzero(void* ptr, scl::Size bytes) {
    if (ptr && bytes > 0) {
        std::memset(ptr, 0, bytes);
    }
}

int scl_memcpy(const void* src, void* dst, scl::Size bytes) {
    SCL_C_API_WRAPPER(
        if (!src || !dst) {
            throw scl::ValueError("scl_memcpy: null pointer");
        }
        std::memcpy(dst, src, bytes);
    )
}

// =============================================================================
// SECTION 21: Helper Functions
// =============================================================================

bool scl_is_valid_value(scl::Real value) {
    return std::isfinite(value);
}

scl::Size scl_sizeof_real() {
    return sizeof(scl::Real);
}

scl::Size scl_sizeof_index() {
    return sizeof(scl::Index);
}

scl::Size scl_alignment() {
    return 64;
}

// =============================================================================
// Workspace Size Calculation Helpers
// =============================================================================

scl::Size scl_ttest_workspace_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * n_groups * 2 * sizeof(scl::Real);
}

scl::Size scl_diff_expr_output_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * (n_groups - 1);
}

scl::Size scl_group_stats_output_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * n_groups;
}

scl::Size scl_gram_output_size(scl::Size n) {
    return n * n;
}

scl::Size scl_correlation_workspace_size(scl::Size n) {
    return n * 2 * sizeof(scl::Real);  // means + inv_stds
}

} // extern "C"

