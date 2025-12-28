// =============================================================================
// FILE: scl/binding/c_api/sampling.cpp
// BRIEF: C API implementation for advanced sampling strategies
// =============================================================================

#include "scl/binding/c_api/sampling.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/sampling.hpp"
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

scl_error_t scl_sampling_kmeans_pp_init(
    scl_sparse_matrix_t data,
    scl_index_t* centers,
    scl_size_t k,
    uint64_t seed
) {
    if (!data || !centers) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(data);
        scl::Array<scl::Index> centers_arr(
            reinterpret_cast<scl::Index*>(centers),
            k
        );
        scl::kernel::sampling::kmeans_pp_init(*sparse, k, centers_arr, seed);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_geometric_sketching(
    scl_sparse_matrix_t data,
    scl_index_t* selected,
    scl_size_t n_selected,
    scl_size_t n_bins,
    uint64_t seed
) {
    if (!data || !selected) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(data);
        scl::Array<scl::Index> selected_arr(
            reinterpret_cast<scl::Index*>(selected),
            n_selected
        );
        scl::kernel::sampling::geometric_sketching(*sparse, selected_arr, n_bins, seed);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_density_preserving(
    scl_sparse_matrix_t data,
    scl_index_t* selected,
    scl_size_t n_selected,
    uint64_t seed
) {
    if (!data || !selected) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(data);
        scl::Array<scl::Index> selected_arr(
            reinterpret_cast<scl::Index*>(selected),
            n_selected
        );
        scl::kernel::sampling::density_preserving(*sparse, selected_arr, seed);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_landmark_selection(
    scl_sparse_matrix_t data,
    scl_index_t* landmarks,
    scl_size_t n_landmarks,
    uint64_t seed
) {
    if (!data || !landmarks) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(data);
        scl::Array<scl::Index> landmarks_arr(
            reinterpret_cast<scl::Index*>(landmarks),
            n_landmarks
        );
        scl::kernel::sampling::landmark_selection(*sparse, landmarks_arr, seed);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_representative_cells(
    scl_sparse_matrix_t data,
    const scl_index_t* cluster_labels,
    scl_index_t* representatives,
    scl_size_t n_clusters,
    scl_size_t n_cells
) {
    if (!data || !cluster_labels || !representatives) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(data);
        scl::Array<const scl::Index> labels_arr(
            reinterpret_cast<const scl::Index*>(cluster_labels),
            n_cells
        );
        scl::Array<scl::Index> reps_arr(
            reinterpret_cast<scl::Index*>(representatives),
            n_clusters
        );
        scl::kernel::sampling::representative_cells(*sparse, labels_arr, reps_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::sampling::kmeans_pp_init<scl::Real, true>(
    const scl::CSR&,
    scl::Size,
    scl::Array<scl::Index>,
    uint64_t
);

template void scl::kernel::sampling::geometric_sketching<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Index>,
    scl::Size,
    uint64_t
);

template void scl::kernel::sampling::density_preserving<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Index>,
    uint64_t
);

template void scl::kernel::sampling::landmark_selection<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Index>,
    uint64_t
);

template void scl::kernel::sampling::representative_cells<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Array<scl::Index>
);

} // extern "C"

