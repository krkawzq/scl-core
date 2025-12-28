// =============================================================================
// FILE: scl/binding/c_api/neighbors.cpp
// BRIEF: C API implementation for K-Nearest Neighbors
// =============================================================================

#include "scl/binding/c_api/neighbors.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/neighbors.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

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

scl_error_t scl_compute_norms(
    scl_sparse_matrix_t matrix,
    scl_real_t* norms_sq
) {
    if (!matrix || !norms_sq) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* sparse = reinterpret_cast<scl::CSR*>(matrix);

        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        const scl::Index primary_dim = sparse->primary_dim();
        scl::Array<scl::Real> norms_arr(norms_sq, static_cast<scl::Size>(primary_dim));

        scl::kernel::neighbors::compute_norms(*sparse, norms_arr);

        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_knn(
    scl_sparse_matrix_t matrix,
    const scl_real_t* norms_sq,
    scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances
) {
    if (!matrix || !norms_sq || !out_indices || !out_distances) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* sparse = reinterpret_cast<scl::CSR*>(matrix);

        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        const scl::Index primary_dim = sparse->primary_dim();
        const scl::Size n = static_cast<scl::Size>(primary_dim);

        scl::Array<const scl::Real> norms_arr(norms_sq, n);
        scl::Array<scl::Index> indices_arr(out_indices, n * k);
        scl::Array<scl::Real> distances_arr(out_distances, n * k);

        scl::kernel::neighbors::knn(*sparse, norms_arr, k, indices_arr, distances_arr);

        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

// Explicit instantiation
template void scl::kernel::neighbors::compute_norms<scl::Real, true>(
    const scl::Sparse<scl::Real, true>&,
    scl::Array<scl::Real>
);

template void scl::kernel::neighbors::knn<scl::Real, true>(
    const scl::Sparse<scl::Real, true>&,
    scl::Array<const scl::Real>,
    scl::Size,
    scl::Array<scl::Index>,
    scl::Array<scl::Real>
);

