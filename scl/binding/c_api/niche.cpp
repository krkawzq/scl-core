// =============================================================================
// FILE: scl/binding/c_api/niche.cpp
// BRIEF: C API implementation for cellular neighborhood analysis
// =============================================================================

#include "scl/binding/c_api/niche.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/niche.hpp"
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

scl_error_t scl_niche_neighborhood_composition(
    scl_sparse_matrix_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cells,
    scl_index_t n_cell_types,
    scl_real_t* composition_output
) {
    if (!spatial_neighbors || !cell_type_labels || !composition_output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(spatial_neighbors);
        scl::Array<const scl::Index> labels_arr(
            reinterpret_cast<const scl::Index*>(cell_type_labels),
            static_cast<scl::Size>(n_cells)
        );
        scl::Array<scl::Real> comp_arr(
            reinterpret_cast<scl::Real*>(composition_output),
            static_cast<scl::Size>(n_cells) * static_cast<scl::Size>(n_cell_types)
        );
        scl::kernel::niche::neighborhood_composition(
            *sparse,
            labels_arr,
            static_cast<scl::Index>(n_cell_types),
            comp_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_niche_colocalization_score(
    scl_sparse_matrix_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cells,
    scl_index_t n_cell_types,
    scl_real_t* colocalization_matrix
) {
    if (!spatial_neighbors || !cell_type_labels || !colocalization_matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(spatial_neighbors);
        scl::Array<const scl::Index> labels_arr(
            reinterpret_cast<const scl::Index*>(cell_type_labels),
            static_cast<scl::Size>(n_cells)
        );
        scl::Array<scl::Real> coloc_arr(
            reinterpret_cast<scl::Real*>(colocalization_matrix),
            static_cast<scl::Size>(n_cell_types) * static_cast<scl::Size>(n_cell_types)
        );
        scl::kernel::niche::colocalization_score(
            *sparse,
            labels_arr,
            static_cast<scl::Index>(n_cell_types),
            coloc_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::niche::neighborhood_composition<scl::Real, true>(
    const scl::CSR&, scl::Array<const scl::Index>, scl::Index, scl::Array<scl::Real>);
template void scl::kernel::niche::colocalization_score<scl::Real, true>(
    const scl::CSR&, scl::Array<const scl::Index>, scl::Index, scl::Array<scl::Real>);

} // extern "C"
