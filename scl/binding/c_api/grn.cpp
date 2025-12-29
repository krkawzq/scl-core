// =============================================================================
// FILE: scl/binding/c_api/grn/grn.cpp
// BRIEF: C API implementation for Gene Regulatory Network inference
// =============================================================================

#include "scl/binding/c_api/grn.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/grn.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

[[nodiscard]] constexpr auto convert_grn_method(
    scl_grn_method_t method) noexcept -> scl::kernel::grn::GRNMethod {
    switch (method) {
        case SCL_GRN_METHOD_CORRELATION:
            return scl::kernel::grn::GRNMethod::Correlation;
        case SCL_GRN_METHOD_PARTIAL_CORRELATION:
            return scl::kernel::grn::GRNMethod::PartialCorrelation;
        case SCL_GRN_METHOD_MUTUAL_INFORMATION:
            return scl::kernel::grn::GRNMethod::MutualInformation;
        case SCL_GRN_METHOD_GENIE3:
            return scl::kernel::grn::GRNMethod::GENIE3;
        case SCL_GRN_METHOD_COMBINED:
            return scl::kernel::grn::GRNMethod::Combined;
        default:
            return scl::kernel::grn::GRNMethod::Correlation;
    }
}

} // anonymous namespace

extern "C" {

// =============================================================================
// Correlation Network
// =============================================================================

SCL_EXPORT scl_error_t scl_grn_correlation_network(
    scl_sparse_t expression,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_real_t threshold,
    scl_real_t* correlation_matrix,
    const int use_spearman) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(correlation_matrix, "Output correlation matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& expr) {
            scl::kernel::grn::correlation_network(
                expr, n_cells, n_genes, static_cast<Real>(threshold),
                reinterpret_cast<Real*>(correlation_matrix),
                use_spearman != 0
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Sparse Correlation Network
// =============================================================================

SCL_EXPORT scl_error_t scl_grn_correlation_network_sparse(
    scl_sparse_t expression,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_real_t threshold,
    scl_index_t* edge_row,
    scl_index_t* edge_col,
    scl_real_t* edge_weight,
    const scl_index_t max_edges,
    scl_index_t* out_n_edges,
    const int use_spearman) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(edge_row, "Output edge row array is null");
    SCL_C_API_CHECK_NULL(edge_col, "Output edge col array is null");
    SCL_C_API_CHECK_NULL(edge_weight, "Output edge weight array is null");
    SCL_C_API_CHECK_NULL(out_n_edges, "Output n_edges pointer is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && max_edges > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        Index n_edges = 0;
        
        expression->visit([&](auto& expr) {
            n_edges = scl::kernel::grn::correlation_network_sparse(
                expr, n_cells, n_genes, static_cast<Real>(threshold),
                edge_row, edge_col, reinterpret_cast<Real*>(edge_weight),
                max_edges, use_spearman != 0
            );
        });
        
        *out_n_edges = n_edges;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Partial Correlation Network
// =============================================================================

SCL_EXPORT scl_error_t scl_grn_partial_correlation_network(
    scl_sparse_t expression,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_real_t threshold,
    scl_real_t* partial_corr_matrix) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(partial_corr_matrix, "Output partial correlation matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& expr) {
            scl::kernel::grn::partial_correlation_network(
                expr, n_cells, n_genes, static_cast<Real>(threshold),
                reinterpret_cast<Real*>(partial_corr_matrix)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// TF-Target Score
// =============================================================================

SCL_EXPORT scl_error_t scl_grn_tf_target_score(
    scl_sparse_t expression,
    const scl_index_t* tf_genes,
    const scl_index_t n_tfs,
    const scl_index_t* target_genes,
    const scl_index_t n_targets,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    scl_real_t* scores) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(tf_genes, "TF genes array is null");
    SCL_C_API_CHECK_NULL(target_genes, "Target genes array is null");
    SCL_C_API_CHECK_NULL(scores, "Output scores array is null");
    SCL_C_API_CHECK(n_tfs > 0 && n_targets > 0 && n_cells > 0 && n_genes > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<const Index> tfs(tf_genes, static_cast<Size>(n_tfs));
        Array<const Index> targets(target_genes, static_cast<Size>(n_targets));
        
        expression->visit([&](auto& expr) {
            scl::kernel::grn::tf_target_score(
                expr, tfs, targets, n_cells, n_genes,
                reinterpret_cast<Real*>(scores)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Infer GRN
// =============================================================================

SCL_EXPORT scl_error_t scl_grn_infer(
    scl_sparse_t expression,
    const scl_index_t* tf_genes,
    const scl_index_t n_tfs,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    scl_real_t* grn_matrix,
    const scl_grn_method_t method,
    const scl_real_t threshold) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(tf_genes, "TF genes array is null");
    SCL_C_API_CHECK_NULL(grn_matrix, "Output GRN matrix is null");
    SCL_C_API_CHECK(n_tfs > 0 && n_cells > 0 && n_genes > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<const Index> tfs(tf_genes, static_cast<Size>(n_tfs));
        
        expression->visit([&](auto& expr) {
            scl::kernel::grn::infer_grn(
                expr, tfs, n_cells, n_genes,
                reinterpret_cast<Real*>(grn_matrix),
                convert_grn_method(method),
                static_cast<Real>(threshold)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
