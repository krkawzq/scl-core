// =============================================================================
// FILE: scl/binding/c_api/comparison/comparison.cpp
// BRIEF: C API implementation for comparison analysis
// =============================================================================

#include "scl/binding/c_api/comparison.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/comparison.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_comp_composition_analysis(
    const scl_index_t* cell_types,
    const scl_index_t* conditions,
    scl_size_t n_cells,
    scl_size_t n_types,
    scl_size_t n_conditions,
    scl_real_t* proportions,
    scl_real_t* p_values
) {
    if (!cell_types || !conditions || !proportions || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> types(cell_types, n_cells);
        scl::Array<const scl::Index> conds(conditions, n_cells);
        scl::Array<scl::Real> props(reinterpret_cast<scl::Real*>(proportions),
                                    static_cast<scl::Size>(n_types) * n_conditions);
        scl::Array<scl::Real> pvals(reinterpret_cast<scl::Real*>(p_values), static_cast<scl::Size>(n_types));

        scl::kernel::comparison::composition_analysis(
            types, conds, props, pvals, static_cast<scl::Size>(n_types), static_cast<scl::Size>(n_conditions)
        );

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_abundance_test(
    const scl_index_t* cluster_labels,
    const scl_index_t* condition,
    scl_size_t n_cells,
    scl_real_t* fold_changes,
    scl_real_t* p_values
) {
    if (!cluster_labels || !condition || !fold_changes || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> labels(cluster_labels, n_cells);
        scl::Array<const scl::Index> cond(condition, n_cells);
        scl::Array<scl::Real> fc(reinterpret_cast<scl::Real*>(fold_changes), 0);  // Size determined internally
        scl::Array<scl::Real> pvals(reinterpret_cast<scl::Real*>(p_values), 0);

        scl::kernel::comparison::abundance_test(labels, cond, fc, pvals);

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_differential_abundance(
    const scl_index_t* cluster_labels,
    const scl_index_t* sample_ids,
    const scl_index_t* conditions,
    scl_size_t n_cells,
    scl_real_t* da_scores,
    scl_real_t* p_values
) {
    if (!cluster_labels || !sample_ids || !conditions || !da_scores || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> labels(cluster_labels, n_cells);
        scl::Array<const scl::Index> samples(sample_ids, n_cells);
        scl::Array<const scl::Index> conds(conditions, n_cells);
        scl::Array<scl::Real> da(reinterpret_cast<scl::Real*>(da_scores), 0);
        scl::Array<scl::Real> pvals(reinterpret_cast<scl::Real*>(p_values), 0);

        scl::kernel::comparison::differential_abundance(labels, samples, conds, da, pvals);

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_condition_response(
    scl_sparse_t expression,
    const scl_index_t* conditions,
    scl_size_t n_genes,
    scl_real_t* response_scores,
    scl_real_t* p_values
) {
    if (!expression || !conditions || !response_scores || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Array<const scl::Index> conds(conditions, static_cast<scl::Size>(sparse->rows()));
        scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(response_scores), static_cast<scl::Size>(n_genes));
        scl::Array<scl::Real> pvals(reinterpret_cast<scl::Real*>(p_values), static_cast<scl::Size>(n_genes));

        sparse->visit([&](auto& m) {
            scl::kernel::comparison::condition_response(
                m, conds, scores.ptr, pvals.ptr, static_cast<scl::Size>(n_genes)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_effect_size(
    const scl_real_t* group1,
    scl_size_t n1,
    const scl_real_t* group2,
    scl_size_t n2,
    scl_real_t* effect_size
) {
    if (!group1 || !group2 || !effect_size) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> g1(reinterpret_cast<const scl::Real*>(group1), n1);
        scl::Array<const scl::Real> g2(reinterpret_cast<const scl::Real*>(group2), n2);

        scl::Real d = scl::kernel::comparison::effect_size(g1, g2);
        *effect_size = static_cast<scl_real_t>(d);

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_glass_delta(
    const scl_real_t* control,
    scl_size_t n_control,
    const scl_real_t* treatment,
    scl_size_t n_treatment,
    scl_real_t* delta
) {
    if (!control || !treatment || !delta) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> ctrl(reinterpret_cast<const scl::Real*>(control), n_control);
        scl::Array<const scl::Real> treat(reinterpret_cast<const scl::Real*>(treatment), n_treatment);

        scl::Real d = scl::kernel::comparison::glass_delta(ctrl, treat);
        *delta = static_cast<scl_real_t>(d);

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_comp_hedges_g(
    const scl_real_t* group1,
    scl_size_t n1,
    const scl_real_t* group2,
    scl_size_t n2,
    scl_real_t* hedges_g
) {
    if (!group1 || !group2 || !hedges_g) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> g1(reinterpret_cast<const scl::Real*>(group1), n1);
        scl::Array<const scl::Real> g2(reinterpret_cast<const scl::Real*>(group2), n2);

        scl::Real g = scl::kernel::comparison::hedges_g(g1, g2);
        *hedges_g = static_cast<scl_real_t>(g);

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
