// =============================================================================
// FILE: scl/binding/c_api/tissue.cpp
// BRIEF: C API implementation for tissue architecture analysis
// =============================================================================

#include "scl/binding/c_api/tissue.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/tissue.hpp"
#include "scl/core/type.hpp"

#include <exception>

extern "C" {

static scl_error_t get_sparse_matrix(
    scl_sparse_t handle,
    scl::binding::SparseWrapper*& wrapper
) {
    if (!handle) {
        return SCL_ERROR_NULL_POINTER;
    }
    wrapper = static_cast<scl::binding::SparseWrapper*>(handle);
    if (!wrapper->valid()) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    return SCL_OK;
}

scl_error_t scl_tissue_architecture(
    const scl_real_t* coordinates,
    scl_size_t n_cells,
    scl_size_t n_dims,
    const scl_index_t* cell_types,
    scl_real_t* density,
    scl_real_t* heterogeneity,
    scl_real_t* clustering_coef,
    scl_size_t n_neighbors
) {
    if (!coordinates || !cell_types || !density || !heterogeneity || !clustering_coef) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::kernel::tissue::tissue_architecture(
            reinterpret_cast<const scl::Real*>(coordinates),
            static_cast<scl::Size>(n_cells),
            static_cast<scl::Size>(n_dims),
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(cell_types),
                static_cast<scl::Size>(n_cells)
            ),
            reinterpret_cast<scl::Real*>(density),
            reinterpret_cast<scl::Real*>(heterogeneity),
            reinterpret_cast<scl::Real*>(clustering_coef),
            static_cast<scl::Size>(n_neighbors)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_tissue_layer_assignment(
    const scl_real_t* coordinates,
    scl_size_t n_cells,
    scl_size_t n_dims,
    scl_index_t n_layers,
    scl_index_t* layer_labels,
    scl_index_t reference_dim
) {
    if (!coordinates || !layer_labels) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::kernel::tissue::layer_assignment(
            reinterpret_cast<const scl::Real*>(coordinates),
            static_cast<scl::Size>(n_cells),
            static_cast<scl::Size>(n_dims),
            static_cast<scl::Index>(n_layers),
            scl::Array<scl::Index>(
                reinterpret_cast<scl::Index*>(layer_labels),
                static_cast<scl::Size>(n_cells)
            ),
            static_cast<scl::Index>(reference_dim)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_tissue_radial_layer_assignment(
    const scl_real_t* coordinates,
    scl_size_t n_cells,
    scl_size_t n_dims,
    const scl_real_t* center,
    scl_index_t n_layers,
    scl_index_t* layer_labels
) {
    if (!coordinates || !center || !layer_labels) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::kernel::tissue::radial_layer_assignment(
            reinterpret_cast<const scl::Real*>(coordinates),
            static_cast<scl::Size>(n_cells),
            static_cast<scl::Size>(n_dims),
            reinterpret_cast<const scl::Real*>(center),
            static_cast<scl::Index>(n_layers),
            scl::Array<scl::Index>(
                reinterpret_cast<scl::Index*>(layer_labels),
                static_cast<scl::Size>(n_cells)
            )
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_tissue_zonation_score(
    scl_sparse_t expression,
    const scl_real_t* coordinates,
    scl_size_t n_dims,
    const scl_real_t* reference_point,
    scl_real_t* zonation_scores
) {
    if (!expression || !coordinates || !reference_point || !zonation_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper{};
        scl_error_t err = get_sparse_matrix(expression, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& expr) {
            scl::kernel::tissue::zonation_score(
                expr,
                reinterpret_cast<const scl::Real*>(coordinates),
                static_cast<scl::Size>(n_dims),
                reinterpret_cast<const scl::Real*>(reference_point),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(zonation_scores),
                    static_cast<scl::Size>(expr.rows())
                )
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_tissue_morphological_features(
    const scl_real_t* coordinates,
    scl_size_t n_cells,
    scl_size_t n_dims,
    const scl_index_t* labels,
    scl_size_t n_groups,
    scl_real_t* area,
    scl_real_t* perimeter,
    scl_real_t* circularity,
    scl_real_t* eccentricity
) {
    if (!coordinates || !labels || !area || !perimeter || !circularity || !eccentricity) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::kernel::tissue::morphological_features(
            reinterpret_cast<const scl::Real*>(coordinates),
            static_cast<scl::Size>(n_cells),
            static_cast<scl::Size>(n_dims),
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(labels),
                static_cast<scl::Size>(n_cells)
            ),
            reinterpret_cast<scl::Real*>(area),
            reinterpret_cast<scl::Real*>(perimeter),
            reinterpret_cast<scl::Real*>(circularity),
            reinterpret_cast<scl::Real*>(eccentricity),
            static_cast<scl::Size>(n_groups)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
