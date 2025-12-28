// =============================================================================
// FILE: scl/kernel/leiden.h
// BRIEF: API reference for high-performance Leiden clustering for community detection
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::leiden {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_RESOLUTION = Real(1.0);
    constexpr Index DEFAULT_MAX_ITER = 10;
    constexpr Index DEFAULT_MAX_MOVES = 100;
    constexpr Real MODULARITY_EPSILON = Real(1e-10);
    constexpr Real THETA = Real(0.05);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size HASH_LOAD_FACTOR_INV = 2;
    constexpr Size PREFETCH_DISTANCE = 4;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr Index MIN_COMMUNITY_SIZE = 1;
    constexpr Real AGGREGATION_THRESHOLD = 0.8;
}

// =============================================================================
// Clustering Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: cluster
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Perform Leiden clustering on adjacency graph.
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     labels     [out] Community labels [n_nodes]
 *     resolution [in]  Resolution parameter (default 1.0)
 *     max_iter   [in]  Maximum iterations
 *     seed        [in]  Random seed
 *
 * PRECONDITIONS:
 *     - labels.len >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - labels[i] contains community ID for node i
 *     - Communities are well-connected
 *
 * ALGORITHM:
 *     Multi-level optimization: local moving, refinement, aggregation
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * nnz * log(n_nodes))
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with atomic updates
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void cluster(
    const Sparse<T, IsCSR>& adjacency,       // Adjacency matrix
    Array<Index> labels,                     // Output community labels [n_nodes]
    Real resolution = config::DEFAULT_RESOLUTION, // Resolution parameter
    Index max_iter = config::DEFAULT_MAX_ITER,    // Max iterations
    uint64_t seed = 42                       // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: modularity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute modularity of a partition.
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     labels     [in]  Community labels [n_nodes]
 *     resolution [in]  Resolution parameter
 *
 * PRECONDITIONS:
 *     - labels.len >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - Returns modularity value Q
 *     - Higher values indicate better community structure
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Real modularity(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<const Index> labels,               // Community labels [n_nodes]
    Real resolution = config::DEFAULT_RESOLUTION // Resolution parameter
);

} // namespace scl::kernel::leiden

