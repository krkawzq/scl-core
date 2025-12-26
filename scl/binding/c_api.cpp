// =============================================================================
/// @file c_api.cpp
/// @brief C ABI Exports for SCL Kernels
///
/// Provides C-compatible function exports for Python/Julia/R bindings.
///
/// Design Principles:
/// 1. **C Linkage**: All functions use extern "C"
/// 2. **POD Arguments**: Only plain C types (pointers, integers, floats)
/// 3. **Error Codes**: Return 0 on success, error code on failure
/// 4. **Thread-Safe**: Error messages stored in thread-local storage
/// 5. **Zero-Copy**: No memory allocations, all arrays pre-allocated by caller
///
/// Error Handling Pattern:
/// ```c
/// int status = scl_function(...);
/// if (status != 0) {
///     const char* error = scl_get_last_error();
///     // Handle error
/// }
/// ```
// =============================================================================

#include "scl/kernel/sparse.hpp"
#include "scl/kernel/feature.hpp"
#include "scl/kernel/normalize.hpp"
#include "scl/kernel/qc.hpp"
#include "scl/kernel/correlation.hpp"
#include "scl/kernel/group.hpp"
#include "scl/kernel/algebra.hpp"
#include "scl/kernel/gram.hpp"
#include "scl/kernel/scale.hpp"
#include "scl/kernel/softmax.hpp"
#include "scl/kernel/log1p.hpp"
#include "scl/kernel/mwu.hpp"
#include "scl/kernel/ttest.hpp"
#include "scl/kernel/mmd.hpp"
#include "scl/kernel/hvg.hpp"
#include "scl/kernel/reorder.hpp"
#include "scl/kernel/merge.hpp"
#include "scl/kernel/spatial.hpp"
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

int scl_row_sums_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    try {
        clear_error();
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::sparse::row_sums(matrix, scl::MutableSpan<scl::Real>(output, rows));
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_col_sums_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    try {
        clear_error();
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::sparse::col_sums(matrix, scl::MutableSpan<scl::Real>(output, cols));
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_row_statistics_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* out_means,
    scl::Real* out_vars
) {
    try {
        clear_error();
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::sparse::row_statistics(
            matrix,
            scl::MutableSpan<scl::Real>(out_means, rows),
            scl::MutableSpan<scl::Real>(out_vars, rows)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_col_statistics_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* out_means,
    scl::Real* out_vars
) {
    try {
        clear_error();
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::sparse::col_statistics(
            matrix,
            scl::MutableSpan<scl::Real>(out_means, cols),
            scl::MutableSpan<scl::Real>(out_vars, cols)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
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
    try {
        clear_error();
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
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_compute_basic_gene_qc_csc(
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
    try {
        clear_error();
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::qc::compute_basic_gene_qc(
            matrix,
            scl::MutableSpan<scl::Index>(out_n_cells, cols),
            scl::MutableSpan<scl::Real>(out_total_counts, cols)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 3. Normalization (normalize.hpp)
// =============================================================================

int scl_scale_rows_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const scl::Real* scales
) {
    try {
        clear_error();
        scl::CustomCSR<scl::Real> matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::normalize::scale_rows(
            matrix,
            scl::Span<const scl::Real>(scales, rows)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_scale_cols_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const scl::Real* scales
) {
    try {
        clear_error();
        scl::CustomCSC<scl::Real> matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::normalize::scale_cols(
            matrix,
            scl::Span<const scl::Real>(scales, cols)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 4. Statistical Tests (mwu.hpp, ttest.hpp)
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
    try {
        clear_error();
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
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_mwu_test_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const int32_t* group_ids,
    scl::Real* out_u_stats,
    scl::Real* out_p_values,
    scl::Real* out_log2_fc
) {
    try {
        clear_error();
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::mwu::mwu_test(
            matrix,
            scl::Span<const int32_t>(group_ids, cols),
            scl::MutableSpan<scl::Real>(out_u_stats, rows),
            scl::MutableSpan<scl::Real>(out_p_values, rows),
            scl::MutableSpan<scl::Real>(out_log2_fc, rows)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
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
    try {
        clear_error();
        
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
        
        scl::Size n_targets = n_groups - 1;
        scl::kernel::diff_expr::ttest(
            matrix,
            scl::Span<const int32_t>(group_ids, rows),
            n_groups,
            scl::MutableSpan<scl::Real>(out_t_stats, cols * n_targets),
            scl::MutableSpan<scl::Real>(out_p_values, cols * n_targets),
            scl::MutableSpan<scl::Real>(out_log2_fc, cols * n_targets),
            scl::MutableSpan<scl::Real>(out_mean_diff, cols * n_targets),
            scl::MutableSpan<scl::Byte>(workspace, workspace_size),
            use_welch
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 5. Log Transforms (log1p.hpp)
// =============================================================================

int scl_log1p_inplace(scl::Real* data, scl::Size size) {
    try {
        clear_error();
        scl::kernel::log1p_inplace(scl::MutableSpan<scl::Real>(data, size));
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_log2p1_inplace(scl::Real* data, scl::Size size) {
    try {
        clear_error();
        scl::kernel::log2p1_inplace(scl::MutableSpan<scl::Real>(data, size));
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_expm1_inplace(scl::Real* data, scl::Size size) {
    try {
        clear_error();
        scl::kernel::expm1_inplace(scl::MutableSpan<scl::Real>(data, size));
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 6. Gram Matrix (gram.hpp)
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
    try {
        clear_error();
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::gram::gram(
            matrix,
            scl::MutableSpan<scl::Real>(output, cols * cols)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
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
    try {
        clear_error();
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::gram::gram(
            matrix,
            scl::MutableSpan<scl::Real>(output, rows * rows)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 7. Pearson Correlation (correlation.hpp)
// =============================================================================

int scl_pearson_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    try {
        clear_error();
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::correlation::pearson(
            matrix,
            scl::MutableSpan<scl::Real>(output, cols * cols)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_pearson_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    try {
        clear_error();
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::correlation::pearson(
            matrix,
            scl::MutableSpan<scl::Real>(output, rows * rows)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 8. Group Aggregations (group.hpp)
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
    try {
        clear_error();
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
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 9. HVG Selection (hvg.hpp)
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
    try {
        clear_error();
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
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
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
    try {
        clear_error();
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
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 10. Spatial Statistics (spatial.hpp)
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
    try {
        clear_error();
        scl::CSRMatrix<scl::Real> graph(
            const_cast<scl::Real*>(graph_data),
            const_cast<scl::Index*>(graph_indices),
            const_cast<scl::Index*>(graph_indptr),
            n_cells, n_cells, graph_nnz,
            graph_row_lengths
        );
        scl::CSCMatrix<scl::Real> features(
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
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_gearys_c(
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
    try {
        clear_error();
        scl::CSRMatrix<scl::Real> graph(
            const_cast<scl::Real*>(graph_data),
            const_cast<scl::Index*>(graph_indices),
            const_cast<scl::Index*>(graph_indptr),
            n_cells, n_cells, graph_nnz,
            graph_row_lengths
        );
        scl::CSCMatrix<scl::Real> features(
            const_cast<scl::Real*>(features_data),
            const_cast<scl::Index*>(features_indices),
            const_cast<scl::Index*>(features_indptr),
            n_cells, n_genes, features_nnz,
            features_col_lengths
        );
        scl::kernel::spatial::gearys_c(
            graph,
            features,
            scl::MutableSpan<scl::Real>(output, n_genes)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 11. Linear Algebra (algebra.hpp)
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
    try {
        clear_error();
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
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
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
    try {
        clear_error();
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
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 12. Feature Selection (feature.hpp)
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
    try {
        clear_error();
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
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_dispersion(
    const scl::Real* means,
    const scl::Real* vars,
    scl::Size size,
    scl::Real* out_dispersion
) {
    try {
        clear_error();
        scl::kernel::feature::dispersion(
            scl::Span<const scl::Real>(means, size),
            scl::Span<const scl::Real>(vars, size),
            scl::MutableSpan<scl::Real>(out_dispersion, size)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 13. Standardization (scale.hpp)
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
    try {
        clear_error();
        scl::CustomCSC<scl::Real> matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::scale::standardize_csc(
            matrix,
            scl::Span<const scl::Real>(means, cols),
            scl::Span<const scl::Real>(stds, cols),
            max_value,
            zero_center
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 14. Softmax (softmax.hpp)
// =============================================================================

int scl_softmax_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* row_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    try {
        clear_error();
        scl::CustomCSR<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            row_lengths
        );
        scl::kernel::softmax::softmax(
            matrix,
            scl::MutableSpan<scl::Real>(output, rows * cols)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

int scl_softmax_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    const scl::Index* col_lengths,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    try {
        clear_error();
        scl::CustomCSC<scl::Real> matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols, nnz,
            col_lengths
        );
        scl::kernel::softmax::softmax(
            matrix,
            scl::MutableSpan<scl::Real>(output, rows * cols)
        );
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// 15. MMD (mmd.hpp)
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
    try {
        clear_error();
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
        return 0;
    } catch (const std::exception& e) {
        store_error(e);
        return -1;
    }
}

// =============================================================================
// Workspace Size Calculation Helpers
// =============================================================================

/// @brief Calculate required workspace size for T-test
scl::Size scl_ttest_workspace_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * n_groups * 2 * sizeof(scl::Real);
}

/// @brief Calculate output size for differential expression (n_targets = n_groups - 1)
scl::Size scl_diff_expr_output_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * (n_groups - 1);
}

/// @brief Calculate output size for group statistics
scl::Size scl_group_stats_output_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * n_groups;
}

/// @brief Calculate output size for Gram matrix
scl::Size scl_gram_output_size(scl::Size n) {
    return n * n;
}

/// @brief Calculate output size for softmax (dense output)
scl::Size scl_softmax_output_size(scl::Index rows, scl::Index cols) {
    return static_cast<scl::Size>(rows) * static_cast<scl::Size>(cols);
}

// =============================================================================
// Helper Functions
// =============================================================================

/// @brief Check if a value is finite and valid
bool scl_is_valid_value(scl::Real value) {
    return std::isfinite(value);
}

/// @brief Get size of Real type in bytes
scl::Size scl_sizeof_real() {
    return sizeof(scl::Real);
}

/// @brief Get size of Index type in bytes
scl::Size scl_sizeof_index() {
    return sizeof(scl::Index);
}

/// @brief Get alignment requirement for SIMD operations
scl::Size scl_alignment() {
    return 64;  // AVX-512 compatible
}

} // extern "C"

