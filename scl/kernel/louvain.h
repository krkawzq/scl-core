// =============================================================================
// FILE: scl/kernel/louvain.h
// BRIEF: API reference for Louvain clustering algorithm
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::louvain {

// =============================================================================
// Configuration Constants
// =============================================================================

namespace config {
    constexpr Real DEFAULT_RESOLUTION = Real(1.0);
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real MODULARITY_EPSILON = Real(1e-8);
    constexpr Size PARALLEL_THRESHOLD = 1000;
    constexpr Index MAX_LEVELS = 100;
}

/* -----------------------------------------------------------------------------
 * FUNCTION: cluster
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Perform Louvain community detection on a sparse graph.
 *
 * PARAMETERS:
 *     adjacency   [in]  Sparse adjacency matrix (symmetric, weighted or unweighted)
 *     labels      [out] Pre-allocated buffer for community labels, size = n_nodes
 *     resolution  [in]  Resolution parameter (default: 1.0). Higher values yield
 *                       more communities
 *     max_iter    [in]  Maximum number of multi-level iterations (default: 100)
 *
 * PRECONDITIONS:
 *     - adjacency must be a valid sparse matrix (CSR or CSC format)
 *     - adjacency should represent an undirected graph (symmetric matrix)
 *     - labels.len >= adjacency.primary_dim()
 *     - resolution > 0
 *     - max_iter > 0
 *
 * POSTCONDITIONS:
 *     - labels[i] contains the community ID for node i
 *     - Community IDs are contiguous: 0, 1, 2, ..., n_communities-1
 *     - adjacency is unchanged
 *
 * ALGORITHM:
 *     Multi-level Louvain algorithm:
 *     1. Initialize each node as its own community
 *     2. Local moving phase:
 *        a. For each node, compute modularity gain of moving to neighbor communities
 *        b. Move node to community with maximum positive gain
 *        c. Repeat until no improvement
 *     3. Aggregation phase:
 *        a. Build coarsened graph where nodes are communities
 *        b. Edge weights are sum of inter-community edges
 *     4. Repeat from step 2 on coarsened graph until convergence
 *
 * COMPLEXITY:
 *     Time:  O(n * log(n) * avg_degree) expected for sparse graphs
 *     Space: O(n + nnz) for working memory
 *
 * THREAD SAFETY:
 *     Safe - uses parallel processing with thread-local workspaces
 *
 * THROWS:
 *     DimensionError - if labels.len < adjacency.primary_dim()
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void cluster(
    const Sparse<T, IsCSR>& adjacency,  // Sparse adjacency matrix (weighted edges)
    Array<Index> labels,                 // Output: community labels [n_nodes]
    Real resolution = config::DEFAULT_RESOLUTION,  // Resolution parameter
    Index max_iter = config::DEFAULT_MAX_ITER      // Max multi-level iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_modularity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the modularity score for a given clustering.
 *
 * PARAMETERS:
 *     adjacency   [in] Sparse adjacency matrix
 *     labels      [in] Community labels for each node
 *     resolution  [in] Resolution parameter (default: 1.0)
 *
 * PRECONDITIONS:
 *     - adjacency must be a valid sparse matrix
 *     - labels.len >= adjacency.primary_dim()
 *     - labels[i] >= 0 for all i
 *
 * POSTCONDITIONS:
 *     - Returns modularity Q in range [-0.5, 1.0]
 *     - Q > 0 indicates community structure stronger than random
 *     - adjacency and labels are unchanged
 *
 * ALGORITHM:
 *     Q = (1/2m) * sum_ij[ A_ij - resolution * k_i * k_j / (2m) ] * delta(c_i, c_j)
 *
 *     Where:
 *     - A_ij = edge weight between i and j
 *     - m = total edge weight / 2
 *     - k_i = weighted degree of node i
 *     - c_i = community of node i
 *     - delta(x,y) = 1 if x==y, 0 otherwise
 *
 * COMPLEXITY:
 *     Time:  O(n + nnz)
 *     Space: O(n) for degree and sigma_tot arrays
 *
 * THREAD SAFETY:
 *     Safe - uses parallel processing for degree computation
 *
 * RETURNS:
 *     Modularity score Q
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Real compute_modularity(
    const Sparse<T, IsCSR>& adjacency,  // Sparse adjacency matrix
    Array<const Index> labels,           // Community labels [n_nodes]
    Real resolution = config::DEFAULT_RESOLUTION  // Resolution parameter
);

/* -----------------------------------------------------------------------------
 * FUNCTION: community_sizes
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the number of nodes in each community.
 *
 * PARAMETERS:
 *     labels        [in]  Community labels for each node
 *     sizes         [out] Pre-allocated buffer for community sizes
 *     n_communities [out] Number of communities found
 *
 * PRECONDITIONS:
 *     - labels[i] >= 0 for all i
 *     - sizes.len >= max(labels) + 1
 *
 * POSTCONDITIONS:
 *     - sizes[c] = number of nodes with labels[i] == c
 *     - n_communities = max(labels) + 1
 *     - labels is unchanged
 *
 * COMPLEXITY:
 *     Time:  O(n) where n = labels.len
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - single-threaded, no shared state
 *
 * THROWS:
 *     DimensionError - if sizes.len < n_communities
 * -------------------------------------------------------------------------- */
inline void community_sizes(
    Array<const Index> labels,  // Community labels [n_nodes]
    Array<Index> sizes,          // Output: size of each community [n_communities]
    Index& n_communities         // Output: number of communities
);

/* -----------------------------------------------------------------------------
 * FUNCTION: get_community_members
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Get the indices of all nodes belonging to a specific community.
 *
 * PARAMETERS:
 *     labels      [in]  Community labels for each node
 *     community   [in]  Target community ID
 *     members     [out] Pre-allocated buffer for member indices
 *     n_members   [out] Number of members found
 *
 * PRECONDITIONS:
 *     - community >= 0
 *     - members.len should be large enough to hold all members
 *
 * POSTCONDITIONS:
 *     - members[0..n_members-1] contain indices of nodes in the community
 *     - n_members = total count of nodes with labels[i] == community
 *     - If n_members > members.len, only first members.len indices are stored
 *     - labels is unchanged
 *
 * COMPLEXITY:
 *     Time:  O(n) where n = labels.len
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - single-threaded, no shared state
 * -------------------------------------------------------------------------- */
inline void get_community_members(
    Array<const Index> labels,  // Community labels [n_nodes]
    Index community,             // Target community ID
    Array<Index> members,        // Output: member indices [max_members]
    Index& n_members             // Output: actual number of members
);

} // namespace scl::kernel::louvain
