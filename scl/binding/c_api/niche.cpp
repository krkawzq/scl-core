// =============================================================================
// FILE: scl/binding/c_api/niche/niche.cpp
// BRIEF: C API implementation for niche analysis
// =============================================================================

#include "scl/binding/c_api/niche.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/niche.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_niche_neighborhood_composition(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* composition_output
) {
    if (!spatial_neighbors || !cell_type_labels || !composition_output) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(spatial_neighbors);
        scl::Index n_cells = sparse->rows();
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> output(reinterpret_cast<scl::Real*>(composition_output),
                                     static_cast<scl::Size>(n_cells) * n_cell_types);

        sparse->visit([&](auto& m) {
            scl::kernel::niche::neighborhood_composition(
                m, labels, n_cell_types, output
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_niche_neighborhood_enrichment(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* enrichment_scores,
    scl_real_t* p_values,
    scl_index_t n_permutations
) {
    if (!spatial_neighbors || !cell_type_labels || !enrichment_scores || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(spatial_neighbors);
        scl::Size n_pairs = static_cast<scl::Size>(n_cell_types) * n_cell_types;
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(sparse->rows()));
        scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(enrichment_scores), n_pairs);
        scl::Array<scl::Real> pvals(reinterpret_cast<scl::Real*>(p_values), n_pairs);

        sparse->visit([&](auto& m) {
            scl::kernel::niche::neighborhood_enrichment(
                m, labels, n_cell_types, scores, pvals, n_permutations
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_niche_cell_cell_contact(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* contact_matrix
) {
    if (!spatial_neighbors || !cell_type_labels || !contact_matrix) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(spatial_neighbors);
        scl::Size n_pairs = static_cast<scl::Size>(n_cell_types) * n_cell_types;
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(sparse->rows()));
        scl::Array<scl::Real> matrix(reinterpret_cast<scl::Real*>(contact_matrix), n_pairs);

        sparse->visit([&](auto& m) {
            scl::kernel::niche::cell_cell_contact(m, labels, n_cell_types, matrix);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_niche_colocalization_score(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_index_t type_a,
    scl_index_t type_b,
    scl_real_t* colocalization,
    scl_real_t* p_value,
    scl_index_t n_permutations
) {
    if (!spatial_neighbors || !cell_type_labels || !colocalization || !p_value) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(spatial_neighbors);
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(sparse->rows()));
        scl::Real coloc, pval;

        sparse->visit([&](auto& m) {
            scl::kernel::niche::colocalization_score(
                m, labels, n_cell_types, type_a, type_b, coloc, pval, n_permutations
            );
        });

        *colocalization = static_cast<scl_real_t>(coloc);
        *p_value = static_cast<scl_real_t>(pval);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_niche_colocalization_matrix(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* coloc_matrix
) {
    if (!spatial_neighbors || !cell_type_labels || !coloc_matrix) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(spatial_neighbors);
        scl::Size n_pairs = static_cast<scl::Size>(n_cell_types) * n_cell_types;
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(sparse->rows()));
        scl::Array<scl::Real> matrix(reinterpret_cast<scl::Real*>(coloc_matrix), n_pairs);

        sparse->visit([&](auto& m) {
            scl::kernel::niche::colocalization_matrix(m, labels, n_cell_types, matrix);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_niche_similarity(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    const scl_index_t* query_cells,
    scl_size_t n_query,
    scl_real_t* similarity_output
) {
    if (!spatial_neighbors || !cell_type_labels || !query_cells || !similarity_output) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(spatial_neighbors);
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(sparse->rows()));
        scl::Array<scl::Real> output(reinterpret_cast<scl::Real*>(similarity_output),
                                    static_cast<scl::Size>(n_query) * n_query);

        sparse->visit([&](auto& m) {
            scl::kernel::niche::niche_similarity(
                m, labels, n_cell_types, query_cells, n_query, output
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_niche_diversity(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* diversity_output
) {
    if (!spatial_neighbors || !cell_type_labels || !diversity_output) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(spatial_neighbors);
        scl::Index n_cells = sparse->rows();
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> output(reinterpret_cast<scl::Real*>(diversity_output),
                                    static_cast<scl::Size>(n_cells));

        sparse->visit([&](auto& m) {
            scl::kernel::niche::niche_diversity(m, labels, n_cell_types, output);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_niche_boundary_score(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* boundary_scores
) {
    if (!spatial_neighbors || !cell_type_labels || !boundary_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(spatial_neighbors);
        scl::Index n_cells = sparse->rows();
        scl::Array<const scl::Index> labels(cell_type_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(boundary_scores),
                                     static_cast<scl::Size>(n_cells));

        sparse->visit([&](auto& m) {
            scl::kernel::niche::niche_boundary_score(m, labels, n_cell_types, scores);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
