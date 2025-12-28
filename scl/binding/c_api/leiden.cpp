// =============================================================================
// FILE: scl/binding/c_api/leiden/leiden.cpp
// BRIEF: C API implementation for Leiden clustering
// =============================================================================

#include "scl/binding/c_api/leiden.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/leiden.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::leiden;

extern "C" {

scl_error_t scl_leiden_cluster(
    scl_sparse_t adjacency,
    scl_index_t* partition,
    scl_size_t n_nodes,
    scl_real_t resolution,
    scl_index_t max_iter,
    scl_index_t* out_n_communities)
{
    if (!adjacency || !partition || !out_n_communities) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(adjacency);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_nodes) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Node count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Index> labels_arr(
            reinterpret_cast<Index*>(partition),
            n_nodes
        );

        wrapper->visit([&](auto& adj) {
            cluster(
                adj, labels_arr,
                static_cast<Real>(resolution),
                max_iter,
                42  // seed
            );
        });

        // Count unique communities
        Index max_comm = -1;
        for (scl_size_t i = 0; i < n_nodes; ++i) {
            if (partition[i] > max_comm) {
                max_comm = partition[i];
            }
        }
        *out_n_communities = max_comm + 1;

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_leiden_cluster_multilevel(
    scl_sparse_t adjacency,
    scl_index_t* partition,
    scl_size_t n_nodes,
    scl_real_t resolution,
    scl_index_t max_iter,
    scl_index_t* out_n_communities)
{
    if (!adjacency || !partition || !out_n_communities) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(adjacency);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_nodes) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Node count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Index> labels_arr(
            reinterpret_cast<Index*>(partition),
            n_nodes
        );

        wrapper->visit([&](auto& adj) {
            cluster_multilevel(
                adj, labels_arr,
                static_cast<Real>(resolution),
                max_iter,
                42  // seed
            );
        });

        // Count unique communities
        Index max_comm = -1;
        for (scl_size_t i = 0; i < n_nodes; ++i) {
            if (partition[i] > max_comm) {
                max_comm = partition[i];
            }
        }
        *out_n_communities = max_comm + 1;

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_leiden_compute_modularity(
    scl_sparse_t adjacency,
    const scl_index_t* partition,
    scl_size_t n_nodes,
    scl_real_t resolution,
    scl_real_t* out_modularity)
{
    if (!adjacency || !partition || !out_modularity) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(adjacency);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_nodes) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Node count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Real modularity = Real(0);
        wrapper->visit([&](const auto& adj) {
            modularity = compute_modularity(
                adj,
                reinterpret_cast<const Index*>(partition),
                static_cast<Real>(resolution)
            );
        });

        *out_modularity = modularity;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

