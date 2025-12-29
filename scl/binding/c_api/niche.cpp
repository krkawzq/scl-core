// =============================================================================
// FILE: scl/binding/c_api/niche/niche.cpp
// BRIEF: C API implementation for niche analysis
// =============================================================================

#include "scl/binding/c_api/niche.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/niche.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Neighborhood Composition
// =============================================================================

SCL_EXPORT scl_error_t scl_niche_neighborhood_composition(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    const scl_index_t n_cell_types,
    scl_real_t* composition_output) {
    
    SCL_C_API_CHECK_NULL(spatial_neighbors, "Spatial neighbors matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(composition_output, "Composition output array is null");
    SCL_C_API_CHECK(n_cell_types > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cell types must be positive");

    SCL_C_API_TRY
        const Index n_cells = spatial_neighbors->rows();
        Array<const Index> labels(cell_type_labels, static_cast<Size>(n_cells));
        Array<Real> output(
            reinterpret_cast<Real*>(composition_output),
            static_cast<Size>(n_cells) * static_cast<Size>(n_cell_types)
        );

        spatial_neighbors->visit([&](auto& m) {
            scl::kernel::niche::neighborhood_composition(m, labels, n_cell_types, output);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Neighborhood Enrichment
// =============================================================================

SCL_EXPORT scl_error_t scl_niche_neighborhood_enrichment(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    const scl_index_t n_cell_types,
    scl_real_t* enrichment_scores,
    scl_real_t* p_values,
    const scl_index_t n_permutations) {
    
    SCL_C_API_CHECK_NULL(spatial_neighbors, "Spatial neighbors matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(enrichment_scores, "Enrichment scores array is null");
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK(n_cell_types > 0 && n_permutations > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");

    SCL_C_API_TRY
        const Size n_pairs = static_cast<Size>(n_cell_types) * static_cast<Size>(n_cell_types);
        Array<const Index> labels(cell_type_labels, static_cast<Size>(spatial_neighbors->rows()));
        Array<Real> scores(reinterpret_cast<Real*>(enrichment_scores), n_pairs);
        Array<Real> pvals(reinterpret_cast<Real*>(p_values), n_pairs);

        spatial_neighbors->visit([&](auto& m) {
            scl::kernel::niche::neighborhood_enrichment(
                m, labels, n_cell_types, scores, pvals, n_permutations
            );
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Cell-Cell Contact
// =============================================================================

SCL_EXPORT scl_error_t scl_niche_cell_cell_contact(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    const scl_index_t n_cell_types,
    scl_real_t* contact_matrix) {
    
    SCL_C_API_CHECK_NULL(spatial_neighbors, "Spatial neighbors matrix is null");
    SCL_C_API_CHECK_NULL(cell_type_labels, "Cell type labels array is null");
    SCL_C_API_CHECK_NULL(contact_matrix, "Contact matrix array is null");
    SCL_C_API_CHECK(n_cell_types > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cell types must be positive");

    SCL_C_API_TRY
        const Size n_pairs = static_cast<Size>(n_cell_types) * static_cast<Size>(n_cell_types);
        Array<const Index> labels(cell_type_labels, static_cast<Size>(spatial_neighbors->rows()));
        Array<Real> contact(reinterpret_cast<Real*>(contact_matrix), n_pairs);

        spatial_neighbors->visit([&](auto& m) {
            scl::kernel::niche::cell_cell_contact(m, labels, n_cell_types, contact);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
