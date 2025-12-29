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

extern "C" {

// =============================================================================
// Leiden Clustering
// =============================================================================

SCL_EXPORT scl_error_t scl_leiden_cluster(
    scl_sparse_t adjacency,
    scl_index_t* partition,
    const scl_size_t n_nodes,
    const scl_real_t resolution,
    const scl_index_t max_iter,
    scl_index_t* out_n_communities) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(partition, "Partition array is null");
    SCL_C_API_CHECK_NULL(out_n_communities, "Output n_communities pointer is null");
    SCL_C_API_CHECK(n_nodes > 0 && max_iter > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");

    SCL_C_API_TRY
        const Index n = adjacency->rows();
        SCL_C_API_CHECK(static_cast<scl_size_t>(n) == n_nodes,
                       SCL_ERROR_DIMENSION_MISMATCH, "Node count mismatch");

        Array<Index> labels_arr(
            reinterpret_cast<Index*>(partition),
            n_nodes
        );

        adjacency->visit([&](auto& adj) {
            scl::kernel::leiden::cluster(
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

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Leiden Multilevel
// =============================================================================

SCL_EXPORT scl_error_t scl_leiden_cluster_multilevel(
    scl_sparse_t adjacency,
    scl_index_t* partition,
    const scl_size_t n_nodes,
    const scl_real_t resolution,
    const scl_index_t max_iter,
    scl_index_t* out_n_communities) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(partition, "Partition array is null");
    SCL_C_API_CHECK_NULL(out_n_communities, "Output n_communities pointer is null");
    SCL_C_API_CHECK(n_nodes > 0 && max_iter > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");

    SCL_C_API_TRY
        const Index n = adjacency->rows();
        SCL_C_API_CHECK(static_cast<scl_size_t>(n) == n_nodes,
                       SCL_ERROR_DIMENSION_MISMATCH, "Node count mismatch");

        Array<Index> labels_arr(
            reinterpret_cast<Index*>(partition),
            n_nodes
        );

        adjacency->visit([&](auto& adj) {
            scl::kernel::leiden::cluster_multilevel(
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

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
