// =============================================================================
// FILE: scl/binding/c_api/comparison.cpp
// BRIEF: C API implementation for group comparison and differential abundance
// =============================================================================

#include "scl/binding/c_api/comparison.h"
#include "scl/kernel/comparison.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

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
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_comparison_composition_analysis(
    const scl_index_t* cell_types,
    const scl_index_t* conditions,
    scl_size_t n_cells,
    scl_size_t n_types,
    scl_size_t n_conditions,
    scl_real_t* proportions,
    scl_real_t* p_values
) {
    if (!cell_types || !conditions || !proportions || !p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> cell_types_arr(cell_types, n_cells);
        scl::Array<const scl::Index> conditions_arr(conditions, n_cells);
        
        scl::kernel::comparison::composition_analysis(
            cell_types_arr,
            conditions_arr,
            proportions,
            p_values,
            n_types,
            n_conditions
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_comparison_abundance_test(
    const scl_index_t* cluster_labels,
    const scl_index_t* conditions,
    scl_size_t n_cells,
    scl_real_t* fold_changes,
    scl_real_t* p_values,
    scl_size_t max_clusters
) {
    if (!cluster_labels || !conditions || !fold_changes || !p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> labels_arr(cluster_labels, n_cells);
        scl::Array<const scl::Index> cond_arr(conditions, n_cells);
        scl::Array<scl::Real> fc_arr(fold_changes, max_clusters);
        scl::Array<scl::Real> pv_arr(p_values, max_clusters);
        
        scl::kernel::comparison::abundance_test(
            labels_arr,
            cond_arr,
            fc_arr,
            pv_arr
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_comparison_differential_abundance(
    const scl_index_t* cluster_labels,
    const scl_index_t* sample_ids,
    const scl_index_t* conditions,
    scl_size_t n_cells,
    scl_real_t* da_scores,
    scl_real_t* p_values,
    scl_size_t max_clusters
) {
    if (!cluster_labels || !sample_ids || !conditions || !da_scores || !p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> labels_arr(cluster_labels, n_cells);
        scl::Array<const scl::Index> samples_arr(sample_ids, n_cells);
        scl::Array<const scl::Index> cond_arr(conditions, n_cells);
        scl::Array<scl::Real> da_arr(da_scores, max_clusters);
        scl::Array<scl::Real> pv_arr(p_values, max_clusters);
        
        scl::kernel::comparison::differential_abundance(
            labels_arr,
            samples_arr,
            cond_arr,
            da_arr,
            pv_arr
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_comparison_condition_response(
    scl_sparse_matrix_t expression,
    const scl_index_t* conditions,
    scl_size_t n_genes,
    scl_real_t* response_scores,
    scl_real_t* p_values
) {
    if (!expression || !conditions || !response_scores || !p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(expression);
        scl::Array<const scl::Index> cond_arr(conditions, static_cast<scl::Size>(sparse->rows()));
        
        scl::kernel::comparison::condition_response(
            *sparse,
            cond_arr,
            response_scores,
            p_values,
            n_genes
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_real_t scl_comparison_effect_size(
    const scl_real_t* group1,
    scl_size_t n1,
    const scl_real_t* group2,
    scl_size_t n2
) {
    if (!group1 || !group2) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Real> g1_arr(group1, n1);
        scl::Array<const scl::Real> g2_arr(group2, n2);
        
        return scl::kernel::comparison::effect_size(g1_arr, g2_arr);
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_comparison_glass_delta(
    const scl_real_t* control,
    scl_size_t n_control,
    const scl_real_t* treatment,
    scl_size_t n_treatment
) {
    if (!control || !treatment) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Real> ctrl_arr(control, n_control);
        scl::Array<const scl::Real> treat_arr(treatment, n_treatment);
        
        return scl::kernel::comparison::glass_delta(ctrl_arr, treat_arr);
    } catch (...) {
        return 0.0;
    }
}

scl_real_t scl_comparison_hedges_g(
    const scl_real_t* group1,
    scl_size_t n1,
    const scl_real_t* group2,
    scl_size_t n2
) {
    if (!group1 || !group2) {
        return 0.0;
    }
    
    try {
        scl::Array<const scl::Real> g1_arr(group1, n1);
        scl::Array<const scl::Real> g2_arr(group2, n2);
        
        return scl::kernel::comparison::hedges_g(g1_arr, g2_arr);
    } catch (...) {
        return 0.0;
    }
}

// Explicit instantiation
template void scl::kernel::comparison::condition_response<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Real*,
    scl::Real*,
    scl::Size
);

} // extern "C"
