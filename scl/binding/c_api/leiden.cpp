// =============================================================================
// FILE: scl/binding/c_api/leiden.cpp
// BRIEF: C API implementation for Leiden clustering
// =============================================================================

#include "scl/binding/c_api/leiden.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/leiden.hpp"
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

scl_error_t scl_leiden_cluster(
    const scl_sparse_matrix_t* adjacency,
    scl_index_t* labels,
    scl_real_t resolution,
    scl_index_t max_iter,
    uint64_t seed
) {
    if (!adjacency || !labels) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(adjacency);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<scl::Index> labels_arr(reinterpret_cast<scl::Index*>(labels), n);
        scl::kernel::leiden::cluster(
            *sparse, labels_arr,
            static_cast<scl::Real>(resolution),
            static_cast<scl::Index>(max_iter),
            seed
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_leiden_cluster_multilevel(
    const scl_sparse_matrix_t* adjacency,
    scl_index_t* labels,
    scl_real_t resolution,
    scl_index_t max_levels,
    uint64_t seed
) {
    if (!adjacency || !labels) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(adjacency);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<scl::Index> labels_arr(reinterpret_cast<scl::Index*>(labels), n);
        scl::kernel::leiden::cluster_multilevel(
            *sparse, labels_arr,
            static_cast<scl::Real>(resolution),
            static_cast<scl::Index>(max_levels),
            seed
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_leiden_compute_modularity(
    const scl_sparse_matrix_t* adjacency,
    const scl_index_t* labels,
    scl_real_t* modularity,
    scl_real_t resolution
) {
    if (!adjacency || !labels || !modularity) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = reinterpret_cast<const scl::CSR*>(adjacency);
        scl::Size n = static_cast<scl::Size>(sparse->rows());
        scl::Array<const scl::Index> labels_arr(reinterpret_cast<const scl::Index*>(labels), n);
        scl::Real q = scl::kernel::leiden::compute_modularity(
            *sparse, labels_arr,
            static_cast<scl::Real>(resolution)
        );
        *modularity = static_cast<scl_real_t>(q);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_leiden_community_sizes(
    const scl_index_t* labels,
    scl_index_t* sizes,
    scl_size_t n_nodes,
    scl_index_t* n_communities
) {
    if (!labels || !sizes || !n_communities) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> labels_arr(reinterpret_cast<const scl::Index*>(labels), n_nodes);
        scl::Index max_label = 0;
        for (scl::Size i = 0; i < n_nodes; ++i) {
            if (labels[i] > max_label) max_label = labels[i];
        }
        scl::Index n_comm = max_label + 1;
        
        scl::Array<scl::Index> sizes_arr(reinterpret_cast<scl::Index*>(sizes), static_cast<scl::Size>(n_comm));
        scl::kernel::leiden::community_sizes(labels_arr, sizes_arr, n_comm);
        *n_communities = n_comm;
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_leiden_sort_communities_by_size(
    scl_index_t* labels,
    scl_size_t n_nodes,
    scl_index_t n_communities
) {
    if (!labels) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<scl::Index> labels_arr(reinterpret_cast<scl::Index*>(labels), n_nodes);
        scl::kernel::leiden::sort_communities_by_size(labels_arr, n_communities);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::leiden::cluster(scl::CSR const&, scl::Array<scl::Index>, scl::Real, scl::Index, uint64_t);
template void scl::kernel::leiden::cluster_multilevel(scl::CSR const&, scl::Array<scl::Index>, scl::Real, scl::Index, uint64_t);
template scl::Real scl::kernel::leiden::compute_modularity(scl::CSR const&, scl::Array<const scl::Index>, scl::Real);

} // extern "C"

