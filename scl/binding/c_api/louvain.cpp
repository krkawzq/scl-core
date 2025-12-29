// =============================================================================
// FILE: scl/binding/c_api/louvain.cpp
// BRIEF: C API implementation for Louvain clustering
// =============================================================================

#include "scl/binding/c_api/louvain.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/louvain.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Louvain Clustering
// =============================================================================

SCL_EXPORT scl_error_t scl_louvain_clustering(
    scl_sparse_t adjacency,
    scl_index_t* labels,
    const scl_size_t n_nodes,
    const scl_real_t resolution,
    const scl_index_t max_iter) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(labels, "Labels array is null");
    SCL_C_API_CHECK(n_nodes > 0 && max_iter > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");

    SCL_C_API_TRY
        adjacency->visit([&](auto& adj) {
            scl::kernel::louvain::cluster(
                adj,
                Array<Index>(
                    reinterpret_cast<Index*>(labels),
                    static_cast<Size>(n_nodes)
                ),
                static_cast<Real>(resolution),
                static_cast<Index>(max_iter)
            );
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Compute Modularity (Utility Functions)
// =============================================================================

SCL_EXPORT scl_error_t scl_louvain_compute_modularity(
    scl_sparse_t adjacency,
    const scl_index_t* labels,
    const scl_size_t n_nodes,
    const scl_real_t resolution,
    scl_real_t* modularity) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(labels, "Labels array is null");
    SCL_C_API_CHECK_NULL(modularity, "Output modularity pointer is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");

    SCL_C_API_TRY
        Real mod = Real(0);

        adjacency->visit([&](auto& adj) {
            mod = scl::kernel::louvain::compute_modularity(
                adj,
                Array<const Index>(
                    reinterpret_cast<const Index*>(labels),
                    static_cast<Size>(n_nodes)
                ),
                static_cast<Real>(resolution)
            );
        });

        *modularity = static_cast<scl_real_t>(mod);

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Community Utility Functions
// =============================================================================

SCL_EXPORT scl_error_t scl_louvain_community_sizes(
    const scl_index_t* labels,
    const scl_size_t n_nodes,
    scl_index_t* sizes,
    const scl_size_t sizes_size,
    scl_index_t* n_communities) {
    
    SCL_C_API_CHECK_NULL(labels, "Labels array is null");
    SCL_C_API_CHECK_NULL(sizes, "Sizes array is null");
    SCL_C_API_CHECK_NULL(n_communities, "Output n_communities pointer is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");

    SCL_C_API_TRY
        // Find max community ID
        Index max_comm = -1;
        for (scl_size_t i = 0; i < n_nodes; ++i) {
            if (labels[i] > max_comm) {
                max_comm = labels[i];
            }
        }
        
        const Index n_comm = max_comm + 1;
        SCL_C_API_CHECK(static_cast<scl_size_t>(n_comm) <= sizes_size,
                       SCL_ERROR_INVALID_ARGUMENT, "Sizes buffer too small");
        
        // Initialize sizes
        for (Index i = 0; i < n_comm; ++i) {
            sizes[i] = 0;
        }
        
        // Count members
        for (scl_size_t i = 0; i < n_nodes; ++i) {
            if (labels[i] >= 0 && labels[i] < n_comm) {
                sizes[labels[i]]++;
            }
        }
        
        *n_communities = n_comm;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_louvain_get_community_members(
    const scl_index_t* labels,
    const scl_size_t n_nodes,
    const scl_index_t community,
    scl_index_t* members,
    const scl_size_t members_size,
    scl_index_t* n_members) {
    
    SCL_C_API_CHECK_NULL(labels, "Labels array is null");
    SCL_C_API_CHECK_NULL(members, "Members array is null");
    SCL_C_API_CHECK_NULL(n_members, "Output n_members pointer is null");
    SCL_C_API_CHECK(n_nodes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of nodes must be positive");

    SCL_C_API_TRY
        Index count = 0;
        
        for (scl_size_t i = 0; i < n_nodes; ++i) {
            if (labels[i] == community) {
                SCL_C_API_CHECK(static_cast<scl_size_t>(count) < members_size,
                               SCL_ERROR_INVALID_ARGUMENT, "Members buffer too small");
                members[count++] = static_cast<scl_index_t>(i);
            }
        }
        
        *n_members = count;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
