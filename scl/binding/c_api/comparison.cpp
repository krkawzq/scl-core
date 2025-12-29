// =============================================================================
// FILE: scl/binding/c_api/comparison/comparison.cpp
// BRIEF: C API implementation for comparison analysis
// =============================================================================

#include "scl/binding/c_api/comparison.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/comparison.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Composition Analysis
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_composition_analysis(
    const scl_index_t* cell_types,
    const scl_index_t* conditions,
    const scl_size_t n_cells,
    const scl_size_t n_types,
    const scl_size_t n_conditions,
    scl_real_t* proportions,
    scl_real_t* p_values) {
    
    SCL_C_API_CHECK_NULL(cell_types, "Cell types array is null");
    SCL_C_API_CHECK_NULL(conditions, "Conditions array is null");
    SCL_C_API_CHECK_NULL(proportions, "Output proportions array is null");
    SCL_C_API_CHECK_NULL(p_values, "Output p-values array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_types > 0 && n_conditions > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<const Index> types(cell_types, n_cells);
        Array<const Index> conds(conditions, n_cells);
        Real* props_ptr = reinterpret_cast<Real*>(proportions);
        Real* pvals_ptr = reinterpret_cast<Real*>(p_values);
        
        scl::kernel::comparison::composition_analysis(
            types, conds,
            props_ptr, pvals_ptr,
            static_cast<Size>(n_types), static_cast<Size>(n_conditions)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Abundance Test
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_abundance_test(
    const scl_index_t* cluster_labels,
    const scl_index_t* condition,
    const scl_size_t n_cells,
    scl_real_t* fold_changes,
    scl_real_t* p_values) {
    
    SCL_C_API_CHECK_NULL(cluster_labels, "Cluster labels array is null");
    SCL_C_API_CHECK_NULL(condition, "Condition array is null");
    SCL_C_API_CHECK_NULL(fold_changes, "Output fold changes array is null");
    SCL_C_API_CHECK_NULL(p_values, "Output p-values array is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    
    SCL_C_API_TRY
        Array<const Index> labels(cluster_labels, n_cells);
        Array<const Index> cond(condition, n_cells);
        // Size determined internally by kernel function
        Array<Real> fc(reinterpret_cast<Real*>(fold_changes), 0);
        Array<Real> pvals(reinterpret_cast<Real*>(p_values), 0);
        
        scl::kernel::comparison::abundance_test(labels, cond, fc, pvals);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Differential Abundance
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_differential_abundance(
    const scl_index_t* cluster_labels,
    const scl_index_t* sample_ids,
    const scl_index_t* conditions,
    const scl_size_t n_cells,
    scl_real_t* da_scores,
    scl_real_t* p_values) {
    
    SCL_C_API_CHECK_NULL(cluster_labels, "Cluster labels array is null");
    SCL_C_API_CHECK_NULL(sample_ids, "Sample IDs array is null");
    SCL_C_API_CHECK_NULL(conditions, "Conditions array is null");
    SCL_C_API_CHECK_NULL(da_scores, "Output DA scores array is null");
    SCL_C_API_CHECK_NULL(p_values, "Output p-values array is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");
    
    SCL_C_API_TRY
        Array<const Index> labels(cluster_labels, n_cells);
        Array<const Index> samples(sample_ids, n_cells);
        Array<const Index> conds(conditions, n_cells);
        // Size determined internally
        Array<Real> da(reinterpret_cast<Real*>(da_scores), 0);
        Array<Real> pvals(reinterpret_cast<Real*>(p_values), 0);
        
        scl::kernel::comparison::differential_abundance(labels, samples, conds, da, pvals);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Condition Response
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_condition_response(
    scl_sparse_t expression,
    const scl_index_t* conditions,
    const scl_size_t n_genes,
    scl_real_t* response_scores,
    scl_real_t* p_values) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(conditions, "Conditions array is null");
    SCL_C_API_CHECK_NULL(response_scores, "Output response scores array is null");
    SCL_C_API_CHECK_NULL(p_values, "Output p-values array is null");
    SCL_C_API_CHECK(n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of genes must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            Array<const Index> conds(conditions, static_cast<Size>(m.rows()));
            Array<Real> scores(reinterpret_cast<Real*>(response_scores), static_cast<Size>(n_genes));
            Array<Real> pvals(reinterpret_cast<Real*>(p_values), static_cast<Size>(n_genes));
            
            scl::kernel::comparison::condition_response(
                m, conds, scores.ptr, pvals.ptr, static_cast<Size>(n_genes)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Effect Size
// =============================================================================

SCL_EXPORT scl_error_t scl_comp_effect_size(
    const scl_real_t* group1,
    const scl_size_t n1,
    const scl_real_t* group2,
    const scl_size_t n2,
    scl_real_t* effect_size) {
    
    SCL_C_API_CHECK_NULL(group1, "Group 1 array is null");
    SCL_C_API_CHECK_NULL(group2, "Group 2 array is null");
    SCL_C_API_CHECK_NULL(effect_size, "Output effect size pointer is null");
    SCL_C_API_CHECK(n1 > 0 && n2 > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Group sizes must be positive");
    
    SCL_C_API_TRY
        Array<const Real> g1(reinterpret_cast<const Real*>(group1), n1);
        Array<const Real> g2(reinterpret_cast<const Real*>(group2), n2);
        
        Real d = scl::kernel::comparison::effect_size(g1, g2);
        *effect_size = static_cast<scl_real_t>(d);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comp_glass_delta(
    const scl_real_t* control,
    const scl_size_t n_control,
    const scl_real_t* treatment,
    const scl_size_t n_treatment,
    scl_real_t* delta) {
    
    SCL_C_API_CHECK_NULL(control, "Control array is null");
    SCL_C_API_CHECK_NULL(treatment, "Treatment array is null");
    SCL_C_API_CHECK_NULL(delta, "Output delta pointer is null");
    SCL_C_API_CHECK(n_control > 0 && n_treatment > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Group sizes must be positive");
    
    SCL_C_API_TRY
        Array<const Real> ctrl(reinterpret_cast<const Real*>(control), n_control);
        Array<const Real> treat(reinterpret_cast<const Real*>(treatment), n_treatment);
        
        Real d = scl::kernel::comparison::glass_delta(ctrl, treat);
        *delta = static_cast<scl_real_t>(d);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_comp_hedges_g(
    const scl_real_t* group1,
    const scl_size_t n1,
    const scl_real_t* group2,
    const scl_size_t n2,
    scl_real_t* hedges_g) {
    
    SCL_C_API_CHECK_NULL(group1, "Group 1 array is null");
    SCL_C_API_CHECK_NULL(group2, "Group 2 array is null");
    SCL_C_API_CHECK_NULL(hedges_g, "Output Hedges' g pointer is null");
    SCL_C_API_CHECK(n1 > 0 && n2 > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Group sizes must be positive");
    
    SCL_C_API_TRY
        Array<const Real> g1(reinterpret_cast<const Real*>(group1), n1);
        Array<const Real> g2(reinterpret_cast<const Real*>(group2), n2);
        
        Real g = scl::kernel::comparison::hedges_g(g1, g2);
        *hedges_g = static_cast<scl_real_t>(g);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
