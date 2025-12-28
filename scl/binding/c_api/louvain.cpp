// =============================================================================
// FILE: scl/binding/c_api/louvain.cpp
// BRIEF: C API implementation for Louvain clustering
// =============================================================================

#include "scl/binding/c_api/louvain.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/louvain.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

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

scl_error_t scl_louvain_clustering(
    scl_sparse_t adjacency,
    scl_index_t* labels,
    scl_size_t n_nodes,
    scl_real_t resolution,
    scl_index_t max_iter
) {
    if (!adjacency || !labels) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(adjacency, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& adj) {
            scl::kernel::louvain::compute_modularity(
                adj,
                scl::Array<scl::Index>(
                    reinterpret_cast<scl::Index*>(labels),
                    static_cast<scl::Size>(n_nodes)
                ),
                static_cast<scl::Real>(resolution),
                static_cast<scl::Index>(max_iter)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_louvain_compute_modularity(
    scl_sparse_t adjacency,
    const scl_index_t* labels,
    scl_size_t n_nodes,
    scl_real_t resolution,
    scl_real_t* modularity
) {
    if (!adjacency || !labels || !modularity) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(adjacency, wrapper);
        if (err != SCL_OK) return err;

        scl::Real mod = scl::Real(0);
        wrapper->visit([&](auto& adj) {
            mod = scl::kernel::louvain::compute_modularity(
                adj,
                scl::Array<const scl::Index>(
                    reinterpret_cast<const scl::Index*>(labels),
                    static_cast<scl::Size>(n_nodes)
                ),
                static_cast<scl::Real>(resolution)
            );
        });
        *modularity = static_cast<scl_real_t>(mod);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_louvain_community_sizes(
    const scl_index_t* labels,
    scl_size_t n_nodes,
    scl_index_t* sizes,
    scl_size_t sizes_size,
    scl_index_t* n_communities
) {
    if (!labels || !sizes || !n_communities) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Index n_comm = 0;
        scl::kernel::louvain::community_sizes(
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(labels),
                static_cast<scl::Size>(n_nodes)
            ),
            scl::Array<scl::Index>(
                reinterpret_cast<scl::Index*>(sizes),
                static_cast<scl::Size>(sizes_size)
            ),
            n_comm
        );
        *n_communities = static_cast<scl_index_t>(n_comm);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_louvain_get_community_members(
    const scl_index_t* labels,
    scl_size_t n_nodes,
    scl_index_t community,
    scl_index_t* members,
    scl_size_t members_size,
    scl_index_t* n_members
) {
    if (!labels || !members || !n_members) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Index n_mem = 0;
        scl::kernel::louvain::get_community_members(
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(labels),
                static_cast<scl::Size>(n_nodes)
            ),
            static_cast<scl::Index>(community),
            scl::Array<scl::Index>(
                reinterpret_cast<scl::Index*>(members),
                static_cast<scl::Size>(members_size)
            ),
            n_mem
        );
        *n_members = static_cast<scl_index_t>(n_mem);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
