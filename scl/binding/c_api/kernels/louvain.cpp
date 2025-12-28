// =============================================================================
// FILE: scl/binding/c_api/kernels/louvain.cpp
// BRIEF: C API implementation for Louvain clustering
// =============================================================================

#include "louvain.h"
#include "scl/kernel/louvain.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/type.hpp"

#include <cstring>

namespace {

// Convert C error code to scl::ErrorCode
inline scl::ErrorCode to_error_code(scl_error_t code) {
    return static_cast<scl::ErrorCode>(code);
}

// Convert scl::ErrorCode to C error code
inline scl_error_t from_error_code(scl::ErrorCode code) {
    return static_cast<scl_error_t>(code);
}

// Convert C sparse matrix handle to C++ Sparse
inline scl::CSR* to_sparse(scl_sparse_matrix_t handle) {
    return static_cast<scl::CSR*>(handle);
}

} // anonymous namespace

extern "C" {

scl_error_t scl_louvain_cluster(
    scl_sparse_matrix_t adjacency,
    scl_index_t* labels,
    scl_index_t n_nodes,
    scl_real_t resolution,
    scl_index_t max_iter
) {
    try {
        scl::CSR* adj = to_sparse(adjacency);
        if (!adj || !labels) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<scl::Index> labels_array(labels, static_cast<scl::Size>(n_nodes));
        scl::kernel::louvain::cluster(*adj, labels_array, resolution, max_iter);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_louvain_compute_modularity(
    scl_sparse_matrix_t adjacency,
    const scl_index_t* labels,
    scl_index_t n_nodes,
    scl_real_t resolution,
    scl_real_t* modularity
) {
    try {
        scl::CSR* adj = to_sparse(adjacency);
        if (!adj || !labels || !modularity) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(labels, static_cast<scl::Size>(n_nodes));
        *modularity = scl::kernel::louvain::compute_modularity(*adj, labels_array, resolution);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_louvain_community_sizes(
    const scl_index_t* labels,
    scl_index_t n_nodes,
    scl_index_t* sizes,
    scl_index_t* n_communities
) {
    try {
        if (!labels || !sizes || !n_communities) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(labels, static_cast<scl::Size>(n_nodes));
        scl::Index max_label = 0;
        for (scl::Size i = 0; i < labels_array.len; ++i) {
            if (labels_array[i] > max_label) {
                max_label = labels_array[i];
            }
        }
        scl::Index n_comm = max_label + 1;

        scl::Array<scl::Index> sizes_array(sizes, static_cast<scl::Size>(n_comm));
        scl::kernel::louvain::community_sizes(labels_array, sizes_array, n_comm);
        *n_communities = n_comm;
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_louvain_get_community_members(
    const scl_index_t* labels,
    scl_index_t n_nodes,
    scl_index_t community,
    scl_index_t* members,
    scl_index_t* n_members
) {
    try {
        if (!labels || !members || !n_members) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(labels, static_cast<scl::Size>(n_nodes));
        
        // First pass: count members
        scl::Index count = 0;
        for (scl::Size i = 0; i < labels_array.len; ++i) {
            if (labels_array[i] == community) {
                ++count;
            }
        }
        
        // Second pass: fill members array
        scl::Index idx = 0;
        for (scl::Size i = 0; i < labels_array.len && idx < count; ++i) {
            if (labels_array[i] == community) {
                members[idx++] = static_cast<scl_index_t>(i);
            }
        }
        
        *n_members = count;
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

} // extern "C"

