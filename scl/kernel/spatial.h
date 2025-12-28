// =============================================================================
// FILE: scl/kernel/spatial.h
// BRIEF: API reference for spatial statistics kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::spatial {

/* -----------------------------------------------------------------------------
 * CONFIGURATION CONSTANTS
 * -----------------------------------------------------------------------------
 * PREFETCH_DISTANCE          - Number of elements to prefetch ahead (8)
 * SIMD_GATHER_THRESHOLD      - Min length for 8-way unroll path (16)
 * PARALLEL_CELL_THRESHOLD    - Min cells for nested parallelism (1024)
 * CELL_BLOCK_SIZE            - Block size for parallel reduction (256)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: weight_sum
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute sum of all edge weights in a sparse graph.
 *
 * PARAMETERS:
 *     graph    [in]  Sparse graph matrix (CSR or CSC format)
 *     out_sum  [out] Output scalar receiving total weight sum
 *
 * PRECONDITIONS:
 *     - graph must be valid sparse matrix
 *
 * POSTCONDITIONS:
 *     - out_sum contains sum of all non-zero values in graph
 *     - If graph has no edges, out_sum = 0
 *
 * ALGORITHM:
 *     1. Partition rows across threads
 *     2. Each thread computes partial sum using SIMD vectorize::sum
 *     3. Reduce partial sums to final result
 *
 * COMPLEXITY:
 *     Time:  O(nnz / n_threads)
 *     Space: O(n_threads) for partial sums
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for thread-local storage
 * -------------------------------------------------------------------------- */
template <typename T, bool GraphCSR>
void weight_sum(
    const Sparse<T, GraphCSR>& graph,   // Sparse graph matrix
    T& out_sum                           // Output: total weight sum
);

/* -----------------------------------------------------------------------------
 * FUNCTION: morans_i
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Moran's I spatial autocorrelation statistic for each feature.
 *
 * PARAMETERS:
 *     graph    [in]  Spatial weights matrix, shape (n_cells, n_cells)
 *     features [in]  Feature matrix, shape (n_features, n_cells)
 *     output   [out] Output array, size = n_features
 *
 * PRECONDITIONS:
 *     - graph must be square: graph.rows() == graph.cols()
 *     - features.secondary_dim() == graph.primary_dim()
 *     - output.len == features.primary_dim()
 *     - graph weights should be non-negative (typically row-normalized)
 *
 * POSTCONDITIONS:
 *     - output[f] contains Moran's I for feature f
 *     - Values typically in range [-1, 1]
 *     - Positive values indicate spatial clustering
 *     - Negative values indicate spatial dispersion
 *     - Zero indicates random spatial pattern
 *
 * ALGORITHM:
 *     For each feature f:
 *         1. Compute mean of feature values
 *         2. Compute z = x - mean (centered values)
 *         3. Compute numerator: sum_i(z_i * sum_j(w_ij * z_j))
 *         4. Compute denominator: sum_i(z_i^2)
 *         5. Moran's I = (N / W) * (numerator / denominator)
 *
 *     Optimization strategies:
 *         - 8-way unrolled weighted neighbor sum with prefetch
 *         - Multi-accumulator pattern for latency hiding
 *         - Nested parallelism for single-feature large-cell case
 *         - Block-based parallel reduction
 *
 * COMPLEXITY:
 *     Time:  O(n_features * nnz_graph)
 *     Space: O(n_cells * n_threads) for z buffers
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for thread-local z arrays
 *
 * NUMERICAL NOTES:
 *     - Returns 0 for features with zero variance
 *     - Returns 0 if total graph weight is zero
 * -------------------------------------------------------------------------- */
template <typename T, bool GraphCSR, bool FeatCSR>
void morans_i(
    const Sparse<T, GraphCSR>& graph,    // Spatial weights matrix [n_cells x n_cells]
    const Sparse<T, FeatCSR>& features,  // Feature matrix [n_features x n_cells]
    Array<Real> output                    // Output Moran's I values [n_features]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: gearys_c
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Geary's C spatial autocorrelation statistic for each feature.
 *
 * PARAMETERS:
 *     graph    [in]  Spatial weights matrix, shape (n_cells, n_cells)
 *     features [in]  Feature matrix, shape (n_features, n_cells)
 *     output   [out] Output array, size = n_features
 *
 * PRECONDITIONS:
 *     - graph must be square: graph.rows() == graph.cols()
 *     - features.secondary_dim() == graph.primary_dim()
 *     - output.len == features.primary_dim()
 *     - graph weights should be non-negative
 *
 * POSTCONDITIONS:
 *     - output[f] contains Geary's C for feature f
 *     - Values typically in range [0, 2]
 *     - C < 1 indicates positive spatial autocorrelation (clustering)
 *     - C = 1 indicates no spatial autocorrelation (random)
 *     - C > 1 indicates negative spatial autocorrelation (dispersion)
 *
 * ALGORITHM:
 *     For each feature f:
 *         1. Compute mean of feature values
 *         2. Compute z = x - mean (centered values)
 *         3. Compute numerator: sum_ij(w_ij * (z_i - z_j)^2)
 *         4. Compute denominator: 2 * W * sum_i(z_i^2)
 *         5. Geary's C = (N-1) * numerator / denominator
 *
 *     Optimization strategies:
 *         - 8-way unrolled difference squared with multi-accumulator
 *         - Aggressive prefetching for indirect z access
 *         - 4-way cleanup loop for remainder
 *         - Nested parallelism for single-feature large-cell case
 *
 * COMPLEXITY:
 *     Time:  O(n_features * nnz_graph)
 *     Space: O(n_cells * n_threads) for z buffers
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for thread-local z arrays
 *
 * NUMERICAL NOTES:
 *     - Returns 0 for features with zero variance
 *     - Returns 0 if total graph weight is zero
 *     - Geary's C is inversely related to Moran's I
 * -------------------------------------------------------------------------- */
template <typename T, bool GraphCSR, bool FeatCSR>
void gearys_c(
    const Sparse<T, GraphCSR>& graph,    // Spatial weights matrix [n_cells x n_cells]
    const Sparse<T, FeatCSR>& features,  // Feature matrix [n_features x n_cells]
    Array<Real> output                    // Output Geary's C values [n_features]
);

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::compute_weighted_neighbor_sum
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute weighted sum of neighbor values with adaptive optimization.
 *
 * PARAMETERS:
 *     weights  [in] Edge weights array
 *     indices  [in] Neighbor index array
 *     len      [in] Number of neighbors
 *     z        [in] Full z values array (indexed indirectly)
 *
 * ALGORITHM:
 *     Adaptive dispatch based on array length:
 *         - len >= 16: 8-way unrolled with aggressive prefetch
 *         - len < 16:  4-way scalar with prefetch
 *
 *     8-way path uses multi-accumulator pattern:
 *         sum0..sum7 accumulated independently to hide latency,
 *         then reduced at the end.
 *
 * COMPLEXITY:
 *     Time:  O(len)
 *     Space: O(1)
 *
 * NOTE:
 *     Indirect z[indices[k]] access prevents true SIMD gather.
 *     Optimization relies on prefetching and instruction-level parallelism.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::compute_moran_numer_block
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Moran's I numerator for a block of cells.
 *
 * PARAMETERS:
 *     graph      [in] Spatial weights matrix
 *     z          [in] Centered feature values
 *     start_cell [in] First cell index (inclusive)
 *     end_cell   [in] Last cell index (exclusive)
 *
 * ALGORITHM:
 *     For each cell i in [start_cell, end_cell):
 *         1. Prefetch next cell's graph data
 *         2. Compute weighted neighbor sum using 8-way helper
 *         3. Accumulate z[i] * neighbor_sum
 *
 * COMPLEXITY:
 *     Time:  O(sum of neighbor counts in block)
 *     Space: O(1)
 *
 * NOTE:
 *     Used for block-parallel reduction in nested parallelism path.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::compute_geary_numer_block
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Geary's C numerator for a block of cells.
 *
 * PARAMETERS:
 *     graph      [in] Spatial weights matrix
 *     z          [in] Centered feature values
 *     start_cell [in] First cell index (inclusive)
 *     end_cell   [in] Last cell index (exclusive)
 *
 * ALGORITHM:
 *     For each cell i in [start_cell, end_cell):
 *         1. Prefetch next cell's graph data
 *         2. 8-way unrolled loop with multi-accumulator:
 *            - Prefetch z values ahead
 *            - Compute diff = z[i] - z[neighbor]
 *            - Accumulate w * diff^2 into 8 independent accumulators
 *         3. 4-way cleanup for remainder
 *         4. Scalar cleanup for final elements
 *
 * COMPLEXITY:
 *     Time:  O(sum of neighbor counts in block)
 *     Space: O(1)
 *
 * NOTE:
 *     Uses 8 independent accumulators (acc0..acc7) to hide FP latency.
 * -------------------------------------------------------------------------- */

} // namespace scl::kernel::spatial
