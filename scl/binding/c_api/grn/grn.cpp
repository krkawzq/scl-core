// =============================================================================
// FILE: scl/binding/c_api/grn/grn.cpp
// BRIEF: C API implementation for Gene Regulatory Network inference
// =============================================================================

#include "scl/binding/c_api/grn/grn.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/grn.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::grn;

extern "C" {

// =============================================================================
// Helper: Convert GRN method type
// =============================================================================

static GRNMethod convert_grn_method(scl_grn_method_t method) {
    switch (method) {
        case SCL_GRN_METHOD_CORRELATION:
            return GRNMethod::Correlation;
        case SCL_GRN_METHOD_PARTIAL_CORRELATION:
            return GRNMethod::PartialCorrelation;
        case SCL_GRN_METHOD_MUTUAL_INFORMATION:
            return GRNMethod::MutualInformation;
        case SCL_GRN_METHOD_GENIE3:
            return GRNMethod::GENIE3;
        case SCL_GRN_METHOD_COMBINED:
            return GRNMethod::Combined;
        default:
            return GRNMethod::Correlation;
    }
}

// =============================================================================
// Correlation Network
// =============================================================================

scl_error_t scl_grn_correlation_network(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_real_t* correlation_matrix,
    int use_spearman)
{
    if (!expression || !correlation_matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(expression);
        
        scl_size_t expected_size = static_cast<scl_size_t>(n_genes) * static_cast<scl_size_t>(n_genes);

        wrapper->visit([&](auto& expr) {
            correlation_network(
                expr, n_cells, n_genes, static_cast<Real>(threshold),
                reinterpret_cast<Real*>(correlation_matrix),
                use_spearman != 0
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Sparse Correlation Network
// =============================================================================

scl_error_t scl_grn_correlation_network_sparse(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_index_t* edge_row,
    scl_index_t* edge_col,
    scl_real_t* edge_weight,
    scl_index_t max_edges,
    scl_index_t* out_n_edges,
    int use_spearman)
{
    if (!expression || !edge_row || !edge_col || !edge_weight || !out_n_edges) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(expression);
        
        Index n_edges = 0;
        wrapper->visit([&](auto& expr) {
            n_edges = correlation_network_sparse(
                expr, n_cells, n_genes, static_cast<Real>(threshold),
                reinterpret_cast<Index*>(edge_row),
                reinterpret_cast<Index*>(edge_col),
                reinterpret_cast<Real*>(edge_weight),
                max_edges,
                use_spearman != 0
            );
        });

        *out_n_edges = n_edges;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// TF-Target Score
// =============================================================================

scl_error_t scl_grn_tf_target_score(
    scl_sparse_t expression,
    const scl_index_t* tf_genes,
    scl_index_t n_tfs,
    const scl_index_t* target_genes,
    scl_index_t n_targets,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* scores)
{
    if (!expression || !tf_genes || !target_genes || !scores) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(expression);
        
        Array<const Index> tf_arr(
            reinterpret_cast<const Index*>(tf_genes),
            static_cast<scl_size_t>(n_tfs)
        );
        Array<const Index> target_arr(
            reinterpret_cast<const Index*>(target_genes),
            static_cast<scl_size_t>(n_targets)
        );

        wrapper->visit([&](auto& expr) {
            tf_target_score(
                expr, tf_arr, target_arr, n_cells, n_genes,
                reinterpret_cast<Real*>(scores)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// GRN Inference
// =============================================================================

scl_error_t scl_grn_infer(
    scl_sparse_t expression,
    const scl_index_t* tf_genes,
    scl_index_t n_tfs,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* grn_matrix,
    scl_grn_method_t method,
    scl_real_t threshold)
{
    if (!expression || !tf_genes || !grn_matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(expression);
        
        Array<const Index> tf_arr(
            reinterpret_cast<const Index*>(tf_genes),
            static_cast<scl_size_t>(n_tfs)
        );

        wrapper->visit([&](auto& expr) {
            infer_grn(
                expr, tf_arr, n_cells, n_genes,
                reinterpret_cast<Real*>(grn_matrix),
                convert_grn_method(method),
                static_cast<Real>(threshold)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

