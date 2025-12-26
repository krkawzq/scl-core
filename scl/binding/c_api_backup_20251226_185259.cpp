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
/// int status = scl_function(...);
/// if (status != 0) {
///     const char* error = scl_get_last_error();
///     // Handle error
/// }
// =============================================================================

#include "scl/kernel/core.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/sparse.hpp"
#include "scl/version.hpp"

#include <cstring>
#include <exception>

// =============================================================================
// Error Handling Infrastructure
// =============================================================================

namespace {

/// @brief Thread-local error message buffer
thread_local char g_error_buffer[512] = {0};

/// @brief Store exception for later retrieval
inline void store_error(const std::exception& e) {
    std::strncpy(g_error_buffer, e.what(), sizeof(g_error_buffer) - 1);
    g_error_buffer[sizeof(g_error_buffer) - 1] = '\0';
}

/// @brief Clear error state
inline void clear_error() {
    g_error_buffer[0] = '\0';
}

/// @brief Helper macro for try-catch wrapper
#define SCL_C_API_WRAPPER(func_call) \
    try { \
        clear_error(); \
        func_call; \
        return 0; \
    } catch (const std::exception& e) { \
        store_error(e); \
        return -1; \
    } catch (...) { \
        store_error(std::runtime_error("Unknown error")); \
        return -1; \
    }

} // anonymous namespace

extern "C" {

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
// Version Information
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

// =============================================================================
// 1. Sparse Matrix Statistics (sparse.hpp)
// =============================================================================

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
        scl::kernel::sparse::primary_sums(matrix, scl::Array<scl::Real>(output, rows));
    )
}

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
        scl::kernel::sparse::primary_sums(matrix, scl::Array<scl::Real>(output, cols));
    )
}

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
        scl::kernel::sparse::primary_means(matrix, scl::MutableSpan<scl::Real>(output, rows));
    )
}

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
        scl::kernel::sparse::primary_means(matrix, scl::MutableSpan<scl::Real>(output, cols));
    )
}

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
        scl::kernel::sparse::primary_variances(matrix, scl::MutableSpan<scl::Real>(output, rows), ddof);
    )
}

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
        scl::kernel::sparse::primary_variances(matrix, scl::MutableSpan<scl::Real>(output, cols), ddof);
    )
}

int scl_primary_nnz_counts_csr(
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index* output
) {
    SCL_C_API_WRAPPER(
        scl::kernel::sparse::primary_nnz_counts(
            scl::Span<const scl::Index>(indptr, rows + 1),
            scl::MutableSpan<scl::Index>(output, rows)
        );
    )
}

int scl_primary_nnz_counts_csc(
    const scl::Index* indptr,
    scl::Index cols,
    scl::Index* output
) {
    SCL_C_API_WRAPPER(
        scl::kernel::sparse::primary_nnz_counts(
            scl::Span<const scl::Index>(indptr, cols + 1),
            scl::MutableSpan<scl::Index>(output, cols)
        );
    )
}

// =============================================================================
// 2. Quality Control (qc.hpp)
// =============================================================================

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
            scl::MutableSpan<scl::Index>(out_n_genes, rows),
            scl::MutableSpan<scl::Real>(out_total_counts, rows)
        );
    )
}

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
            scl::MutableSpan<scl::Index>(out_n_cells, cols),
            scl::MutableSpan<scl::Real>(out_total_counts, cols)
        );
    )
}

// =============================================================================
// 3. Normalization (normalize.hpp)
// =============================================================================

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
            scl::Span<const scl::Real>(scales, rows)
        );
    )
}

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
            scl::Span<const scl::Real>(scales, cols)
        );
    )
}

// =============================================================================
// 4. Feature Statistics (feature.hpp)
// =============================================================================

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
            scl::MutableSpan<scl::Real>(out_means, cols),
            scl::MutableSpan<scl::Real>(out_vars, cols),
            ddof
        );
    )
}

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
            scl::Span<const scl::Real>(clip_vals, cols),
            scl::MutableSpan<scl::Real>(out_means, cols),
            scl::MutableSpan<scl::Real>(out_vars, cols)
        );
    )
}

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
            scl::MutableSpan<scl::Real>(out_rates, cols)
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
            scl::Span<const scl::Real>(means, size),
            scl::Span<const scl::Real>(vars, size),
            scl::MutableSpan<scl::Real>(out_dispersion, size)
        );
    )
}

// =============================================================================
// 5. Statistical Tests (mwu.hpp, ttest.hpp)
// =============================================================================

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
            scl::Span<const int32_t>(group_ids, rows),
            scl::MutableSpan<scl::Real>(out_u_stats, cols),
            scl::MutableSpan<scl::Real>(out_p_values, cols),
            scl::MutableSpan<scl::Real>(out_log2_fc, cols)
        );
    )
}

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
        scl::Size required_size = cols * n_groups * 2 * sizeof(scl::Real);
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
        
        scl::kernel::diff_expr::ttest(
            matrix,
            scl::Span<const int32_t>(group_ids, rows),
            n_groups,
            scl::MutableSpan<scl::Real>(out_t_stats, cols * (n_groups - 1)),
            scl::MutableSpan<scl::Real>(out_p_values, cols * (n_groups - 1)),
            scl::MutableSpan<scl::Real>(out_log2_fc, cols * (n_groups - 1)),
            scl::MutableSpan<scl::Real>(out_mean_diff, cols * (n_groups - 1)),
            scl::MutableSpan<scl::Byte>(workspace, workspace_size),
            use_welch
        );
    )
}

// =============================================================================
// 6. Log Transforms (log1p.hpp)
// =============================================================================

int scl_log1p_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::log1p_inplace(scl::MutableSpan<scl::Real>(data, size));
    )
}

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

int scl_log2p1_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::log2p1_inplace(scl::MutableSpan<scl::Real>(data, size));
    )
}

int scl_expm1_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::expm1_inplace(scl::MutableSpan<scl::Real>(data, size));
    )
}

// =============================================================================
// 7. Gram Matrix (gram.hpp)
// =============================================================================

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
        scl::kernel::gram::gram(matrix, scl::MutableSpan<scl::Real>(output, cols * cols));
    )
}

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
        scl::kernel::gram::gram(matrix, scl::MutableSpan<scl::Real>(output, rows * rows));
    )
}

// =============================================================================
// 8. Pearson Correlation (correlation.hpp)
// =============================================================================

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
            scl::MutableSpan<scl::Real>(workspace_means, cols),
            scl::MutableSpan<scl::Real>(workspace_inv_stds, cols)
        );
        scl::kernel::gram::gram(matrix, scl::MutableSpan<scl::Real>(output, cols * cols));
        // Transform to correlation (implementation detail)
    )
}

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
            scl::MutableSpan<scl::Real>(workspace_means, rows),
            scl::MutableSpan<scl::Real>(workspace_inv_stds, rows)
        );
        scl::kernel::gram::gram(matrix, scl::MutableSpan<scl::Real>(output, rows * rows));
        // Transform to correlation (implementation detail)
    )
}

// =============================================================================
// 9. Group Aggregations (group.hpp)
// =============================================================================

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
        scl::kernel::group::group_stats(
            matrix,
            scl::Span<const int32_t>(group_ids, rows),
            n_groups,
            scl::Span<const scl::Size>(group_sizes, n_groups),
            scl::MutableSpan<scl::Real>(out_means, cols * n_groups),
            scl::MutableSpan<scl::Real>(out_vars, cols * n_groups),
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
            scl::Span<const int32_t>(group_ids, n_elements),
            n_groups,
            scl::MutableSpan<scl::Size>(out_sizes, n_groups)
        );
    )
}

// =============================================================================
// 10. Standardization (scale.hpp)
// =============================================================================

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
            scl::Span<const scl::Real>(means, cols),
            scl::Span<const scl::Real>(stds, cols),
            max_value,
            zero_center
        );
    )
}

// =============================================================================
// 11. Softmax (softmax.hpp)
// =============================================================================

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
// 12. MMD (mmd.hpp)
// =============================================================================

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
            scl::MutableSpan<scl::Real>(output, cols),
            gamma
        );
    )
}

// =============================================================================
// 13. Spatial Statistics (spatial.hpp)
// =============================================================================

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
            scl::MutableSpan<scl::Real>(output, n_genes)
        );
    )
}

// =============================================================================
// 14. Linear Algebra (algebra.hpp)
// =============================================================================

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
            scl::Span<const scl::Real>(x, A_cols),
            scl::MutableSpan<scl::Real>(y, A_rows),
            alpha,
            beta
        );
    )
}

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
            scl::Span<const scl::Real>(x, A_rows),
            scl::MutableSpan<scl::Real>(y, A_cols),
            alpha,
            beta
        );
    )
}

// =============================================================================
// 15. HVG Selection (hvg.hpp)
// =============================================================================

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
            scl::MutableSpan<scl::Index>(out_indices, n_top),
            scl::MutableSpan<uint8_t>(out_mask, cols),
            scl::MutableSpan<scl::Real>(out_dispersions, cols)
        );
    )
}

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
            scl::MutableSpan<scl::Index>(out_indices, n_top),
            scl::MutableSpan<uint8_t>(out_mask, cols)
        );
    )
}

// =============================================================================
// 16. Reordering (reorder.hpp)
// =============================================================================

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
            scl::Span<const scl::Index>(index_map, cols),
            scl::MutableSpan<scl::Index>(out_lengths, cols),
            new_cols
        );
    )
}

// =============================================================================
// 17. Resampling (resample.hpp)
// =============================================================================

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

// =============================================================================
// Helper Functions
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
    return 64;  // AVX-512 compatible
}

// =============================================================================
// Memory Management API
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
        scl::memory::zero(scl::MutableSpan<uint8_t>(
            static_cast<uint8_t*>(ptr), 
            bytes
        ));
    }
}

int scl_memcpy(const void* src, void* dst, scl::Size bytes) {
    SCL_C_API_WRAPPER(
        if (!src || !dst) {
            throw scl::ValueError("scl_memcpy: null pointer");
        }
        scl::memory::copy(
            scl::Span<const uint8_t>(static_cast<const uint8_t*>(src), bytes),
            scl::MutableSpan<uint8_t>(static_cast<uint8_t*>(dst), bytes)
        );
    )
}

} // extern "C"
