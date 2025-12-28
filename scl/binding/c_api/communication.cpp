// =============================================================================
// FILE: scl/binding/c_api/communication.cpp
// BRIEF: C API implementation for cell-cell communication analysis
// =============================================================================

#include "scl/binding/c_api/communication.h"
#include "scl/kernel/communication.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <cstring>

extern "C" {

// Internal helper to convert C++ exception to error code
static scl_error_t handle_exception() {
    try {
        throw;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::TypeError&) {
        return SCL_ERROR_TYPE_ERROR;
    } catch (const scl::IOError&) {
        return SCL_ERROR_IO_ERROR;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

// Convert C score method to C++
static scl::kernel::communication::ScoreMethod to_cpp_method(scl_comm_score_method_t method) {
    switch (method) {
        case SCL_COMM_SCORE_MEAN_PRODUCT: return scl::kernel::communication::ScoreMethod::MeanProduct;
        case SCL_COMM_SCORE_GEOMETRIC_MEAN: return scl::kernel::communication::ScoreMethod::GeometricMean;
        case SCL_COMM_SCORE_MIN_MEAN: return scl::kernel::communication::ScoreMethod::MinMean;
        case SCL_COMM_SCORE_PRODUCT: return scl::kernel::communication::ScoreMethod::Product;
        case SCL_COMM_SCORE_NATMI: return scl::kernel::communication::ScoreMethod::Natmi;
        default: return scl::kernel::communication::ScoreMethod::MeanProduct;
    }
}

scl_error_t scl_communication_lr_score_matrix(
    scl_sparse_matrix_t expression,
    const scl_index_t* cell_type_labels,
    scl_index_t ligand_gene,
    scl_index_t receptor_gene,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* score_matrix,
    scl_comm_score_method_t method
) {
    if (!expression || !cell_type_labels || !score_matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(expression);
        scl::Array<const scl::Index> labels_arr(cell_type_labels, static_cast<scl::Size>(n_cells));
        
        scl::kernel::communication::lr_score_matrix(
            *sparse,
            labels_arr,
            ligand_gene,
            receptor_gene,
            n_cells,
            n_types,
            score_matrix,
            to_cpp_method(method)
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_communication_lr_score_batch(
    scl_sparse_matrix_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    const scl_index_t* receptor_genes,
    scl_index_t n_pairs,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores,
    scl_comm_score_method_t method
) {
    if (!expression || !cell_type_labels || !ligand_genes || !receptor_genes || !scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(expression);
        scl::Array<const scl::Index> labels_arr(cell_type_labels, static_cast<scl::Size>(n_cells));
        
        scl::kernel::communication::lr_score_batch(
            *sparse,
            labels_arr,
            ligand_genes,
            receptor_genes,
            n_pairs,
            n_cells,
            n_types,
            scores,
            to_cpp_method(method)
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_communication_probability(
    scl_sparse_matrix_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    const scl_index_t* receptor_genes,
    scl_index_t n_pairs,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* p_values,
    scl_real_t* scores,
    scl_index_t n_permutations,
    scl_comm_score_method_t method,
    uint64_t seed
) {
    if (!expression || !cell_type_labels || !ligand_genes || !receptor_genes || !p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(expression);
        scl::Array<const scl::Index> labels_arr(cell_type_labels, static_cast<scl::Size>(n_cells));
        
        scl::kernel::communication::communication_probability(
            *sparse,
            labels_arr,
            ligand_genes,
            receptor_genes,
            n_pairs,
            n_cells,
            n_types,
            p_values,
            scores,
            n_permutations,
            to_cpp_method(method),
            seed
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_communication_sender_score(
    scl_sparse_matrix_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    scl_index_t n_ligands,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores
) {
    if (!expression || !cell_type_labels || !ligand_genes || !scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(expression);
        scl::Array<const scl::Index> labels_arr(cell_type_labels, static_cast<scl::Size>(n_cells));
        
        scl::kernel::communication::sender_score(
            *sparse,
            labels_arr,
            ligand_genes,
            n_ligands,
            n_cells,
            n_types,
            scores
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_communication_receiver_score(
    scl_sparse_matrix_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* receptor_genes,
    scl_index_t n_receptors,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores
) {
    if (!expression || !cell_type_labels || !receptor_genes || !scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(expression);
        scl::Array<const scl::Index> labels_arr(cell_type_labels, static_cast<scl::Size>(n_cells));
        
        scl::kernel::communication::receiver_score(
            *sparse,
            labels_arr,
            receptor_genes,
            n_receptors,
            n_cells,
            n_types,
            scores
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_index_t scl_communication_filter_significant(
    const scl_real_t* p_values,
    scl_index_t n_pairs,
    scl_index_t n_types,
    scl_real_t p_threshold,
    scl_index_t* pair_indices,
    scl_index_t* sender_types,
    scl_index_t* receiver_types,
    scl_real_t* filtered_pvalues,
    scl_index_t max_results
) {
    if (!p_values || !pair_indices || !sender_types || !receiver_types || !filtered_pvalues) {
        return 0;
    }
    
    try {
        return scl::kernel::communication::filter_significant(
            p_values,
            n_pairs,
            n_types,
            p_threshold,
            pair_indices,
            sender_types,
            receiver_types,
            filtered_pvalues,
            max_results
        );
    } catch (...) {
        return 0;
    }
}

// Explicit instantiation
template void scl::kernel::communication::lr_score_matrix<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Index, scl::Index, scl::Index, scl::Index,
    scl::Real*,
    scl::kernel::communication::ScoreMethod
);

template void scl::kernel::communication::lr_score_batch<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    const scl::Index*,
    const scl::Index*,
    scl::Index, scl::Index, scl::Index,
    scl::Real*,
    scl::kernel::communication::ScoreMethod
);

template void scl::kernel::communication::communication_probability<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    const scl::Index*,
    const scl::Index*,
    scl::Index, scl::Index, scl::Index,
    scl::Real*,
    scl::Real*,
    scl::Index,
    scl::kernel::communication::ScoreMethod,
    uint64_t
);

template void scl::kernel::communication::sender_score<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    const scl::Index*,
    scl::Index, scl::Index, scl::Index,
    scl::Real*
);

template void scl::kernel::communication::receiver_score<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    const scl::Index*,
    scl::Index, scl::Index, scl::Index,
    scl::Real*
);

} // extern "C"
