// =============================================================================
// FILE: scl/binding/c_api/grn.cpp
// BRIEF: C API implementation for gene regulatory network inference
// =============================================================================

#include "scl/binding/c_api/grn.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/grn.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

static scl_error_t handle_exception() {
    try {
        throw;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::TypeError&) {
        return SCL_ERROR_TYPE;
    } catch (const scl::IOError&) {
        return SCL_ERROR_IO;
    } catch (const scl::NotImplementedError&) {
        return SCL_ERROR_NOT_IMPLEMENTED;
    } catch (const scl::InternalError&) {
        return SCL_ERROR_INTERNAL;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

static scl::kernel::grn::GRNMethod convert_grn_method(scl_grn_method_t method) {
    switch (method) {
        case SCL_GRN_CORRELATION: return scl::kernel::grn::GRNMethod::Correlation;
        case SCL_GRN_PARTIAL_CORRELATION: return scl::kernel::grn::GRNMethod::PartialCorrelation;
        case SCL_GRN_MUTUAL_INFORMATION: return scl::kernel::grn::GRNMethod::MutualInformation;
        case SCL_GRN_GENIE3: return scl::kernel::grn::GRNMethod::GENIE3;
        case SCL_GRN_COMBINED: return scl::kernel::grn::GRNMethod::Combined;
        default: return scl::kernel::grn::GRNMethod::Correlation;
    }
}

scl_error_t scl_grn_correlation_network(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_real_t* correlation_matrix,
    int use_spearman
) {
    if (!expression || !correlation_matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(expression);
        scl::kernel::grn::correlation_network(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Real>(threshold),
            reinterpret_cast<scl::Real*>(correlation_matrix),
            use_spearman != 0
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_grn_correlation_network_sparse(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_index_t* edge_row,
    scl_index_t* edge_col,
    scl_real_t* edge_weight,
    scl_index_t max_edges,
    scl_index_t* n_edges_out,
    int use_spearman
) {
    if (!expression || !edge_row || !edge_col || !edge_weight || !n_edges_out) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(expression);
        scl::Index n_edges = scl::kernel::grn::correlation_network_sparse(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Real>(threshold),
            reinterpret_cast<scl::Index*>(edge_row),
            reinterpret_cast<scl::Index*>(edge_col),
            reinterpret_cast<scl::Real*>(edge_weight),
            static_cast<scl::Index>(max_edges),
            use_spearman != 0
        );
        *n_edges_out = static_cast<scl_index_t>(n_edges);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_grn_partial_correlation_network(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_real_t* partial_corr_matrix,
    scl_real_t regularization
) {
    if (!expression || !partial_corr_matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(expression);
        scl::kernel::grn::partial_correlation_network(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Real>(threshold),
            reinterpret_cast<scl::Real*>(partial_corr_matrix),
            static_cast<scl::Real>(regularization)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_grn_mutual_information_network(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_real_t* mi_matrix,
    scl_index_t n_bins
) {
    if (!expression || !mi_matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(expression);
        scl::kernel::grn::mutual_information_network(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Real>(threshold),
            reinterpret_cast<scl::Real*>(mi_matrix),
            static_cast<scl::Index>(n_bins)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_grn_genie3_importance(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_tfs,
    const scl_index_t* tf_indices,
    scl_real_t* importance_matrix,
    scl_index_t n_trees,
    scl_index_t subsample
) {
    if (!expression || !tf_indices || !importance_matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(expression);
        scl::kernel::grn::genie3_importance(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Index>(n_tfs),
            reinterpret_cast<const scl::Index*>(tf_indices),
            reinterpret_cast<scl::Real*>(importance_matrix),
            static_cast<scl::Index>(n_trees),
            static_cast<scl::Index>(subsample)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_grn_regulon_activity(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* grn_matrix,
    scl_index_t n_tfs,
    scl_real_t threshold,
    scl_real_t* regulon_activity
) {
    if (!expression || !grn_matrix || !regulon_activity) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(expression);
        scl::kernel::grn::regulon_activity(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<const scl::Real*>(grn_matrix),
            static_cast<scl::Index>(n_tfs),
            static_cast<scl::Real>(threshold),
            reinterpret_cast<scl::Real*>(regulon_activity)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_grn_infer_grn(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_grn_method_t method,
    scl_real_t threshold,
    scl_real_t* grn_matrix
) {
    if (!expression || !grn_matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(expression);
        scl::kernel::grn::infer_grn(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            convert_grn_method(method),
            static_cast<scl::Real>(threshold),
            reinterpret_cast<scl::Real*>(grn_matrix)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_grn_tf_activity_from_regulons(
    scl_sparse_matrix_t expression,
    const scl_index_t* regulon_tf,
    const scl_index_t* regulon_offsets,
    const scl_index_t* regulon_targets,
    const scl_real_t* grn_matrix,
    scl_index_t n_regulons,
    scl_index_t n_tfs,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* tf_activity
) {
    if (!expression || !regulon_tf || !regulon_offsets || !regulon_targets || !grn_matrix || !tf_activity) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(expression);
        scl::kernel::grn::tf_activity_from_regulons(
            *sparse,
            reinterpret_cast<const scl::Index*>(regulon_tf),
            reinterpret_cast<const scl::Index*>(regulon_offsets),
            reinterpret_cast<const scl::Index*>(regulon_targets),
            reinterpret_cast<const scl::Real*>(grn_matrix),
            static_cast<scl::Index>(n_regulons),
            static_cast<scl::Index>(n_tfs),
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            reinterpret_cast<scl::Real*>(tf_activity)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::grn::correlation_network<scl::Real, true>(
    const scl::CSR&, scl::Index, scl::Index, scl::Real, scl::Real*, bool);
template scl::Index scl::kernel::grn::correlation_network_sparse<scl::Real, true>(
    const scl::CSR&, scl::Index, scl::Index, scl::Real, scl::Index*, scl::Index*, scl::Real*, scl::Index, bool);
template void scl::kernel::grn::partial_correlation_network<scl::Real, true>(
    const scl::CSR&, scl::Index, scl::Index, scl::Real, scl::Real*, scl::Real);
template void scl::kernel::grn::mutual_information_network<scl::Real, true>(
    const scl::CSR&, scl::Index, scl::Index, scl::Real, scl::Real*, scl::Index);
template void scl::kernel::grn::genie3_importance<scl::Real, true>(
    const scl::CSR&, scl::Index, scl::Index, scl::Index, const scl::Index*, scl::Real*, scl::Index, scl::Index);
template void scl::kernel::grn::regulon_activity<scl::Real, true>(
    const scl::CSR&, scl::Index, scl::Index, const scl::Real*, scl::Index, scl::Real, scl::Real*);
template void scl::kernel::grn::infer_grn<scl::Real, true>(
    const scl::CSR&, scl::Index, scl::Index, scl::kernel::grn::GRNMethod, scl::Real, scl::Real*);
template void scl::kernel::grn::tf_activity_from_regulons<scl::Real, true>(
    const scl::CSR&, const scl::Index*, const scl::Index*, const scl::Index*, const scl::Real*, scl::Index, scl::Index, scl::Index, scl::Index, scl::Real*);

} // extern "C"
