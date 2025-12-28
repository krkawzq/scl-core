// =============================================================================
// FILE: scl/kernel/pseudotime.h
// BRIEF: API reference for pseudotime inference kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::pseudotime {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_DCS = 10;
    constexpr Index DEFAULT_N_ITERATIONS = 100;
    constexpr Real DEFAULT_THRESHOLD = Real(0.1);
    constexpr Real DEFAULT_DAMPING = Real(0.85);
    constexpr Real CONVERGENCE_TOL = Real(1e-6);
    constexpr Real INF_DISTANCE = Real(1e30);
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr Index HEAP_ARITY = 4;
}

// =============================================================================
// Pseudotime Methods
// =============================================================================

enum class PseudotimeMethod {
    DiffusionPseudotime,
    ShortestPath,
    GraphDistance,
    WatershedDescent
};

/* -----------------------------------------------------------------------------
 * FUNCTION: dijkstra_shortest_path
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute shortest path distances from a single source node using
 *     Dijkstra's algorithm with a 4-ary heap.
 *
 * PARAMETERS:
 *     adjacency [in]  Graph adjacency matrix (edge weights as distances)
 *     source    [in]  Source node index
 *     distances [out] Shortest distances from source to all nodes
 *
 * PRECONDITIONS:
 *     - distances.len >= adjacency.primary_dim()
 *     - source in [0, adjacency.primary_dim())
 *     - Edge weights should be positive (negative treated as 1)
 *
 * POSTCONDITIONS:
 *     - distances[i] = shortest path distance from source to i
 *     - distances[source] = 0
 *     - Unreachable nodes have distance = INF_DISTANCE
 *
 * ALGORITHM:
 *     4-ary heap Dijkstra:
 *         1. Initialize all distances to INF, source to 0
 *         2. Pop minimum from heap, relax neighbors
 *         3. Continue until heap empty
 *
 * COMPLEXITY:
 *     Time:  O((V + E) * log_4(V))
 *     Space: O(V) for heap and distance arrays
 *
 * THREAD SAFETY:
 *     Unsafe - single-threaded algorithm
 *
 * NUMERICAL NOTES:
 *     4-ary heap typically faster than binary for Dijkstra due to cache
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void dijkstra_shortest_path(
    const Sparse<T, IsCSR>& adjacency,   // Graph adjacency matrix
    Index source,                         // Source node index
    Array<Real> distances                 // Output distances [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: dijkstra_multi_source
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute shortest path distances from multiple source nodes in parallel.
 *
 * PARAMETERS:
 *     adjacency [in]  Graph adjacency matrix
 *     sources   [in]  Array of source node indices
 *     distances [out] Distance matrix [n_sources * n], row-major
 *
 * PRECONDITIONS:
 *     - distances size >= sources.len * adjacency.primary_dim()
 *     - All source indices in valid range
 *
 * POSTCONDITIONS:
 *     - distances[s * n + i] = shortest distance from sources[s] to node i
 *     - Each source computed independently in parallel
 *
 * COMPLEXITY:
 *     Time:  O(n_sources * (V + E) * log_4(V) / n_threads)
 *     Space: O(V * n_threads) for per-thread heaps
 *
 * THREAD SAFETY:
 *     Safe - parallelized over sources with per-thread heaps
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void dijkstra_multi_source(
    const Sparse<T, IsCSR>& adjacency,   // Graph adjacency matrix
    Array<const Index> sources,           // Source node indices
    Real* distances                       // Output [n_sources * n], row-major
);

/* -----------------------------------------------------------------------------
 * FUNCTION: graph_pseudotime
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute pseudotime as normalized shortest path distance from root cell.
 *
 * PARAMETERS:
 *     adjacency  [in]  Cell neighborhood graph
 *     root_cell  [in]  Starting cell for trajectory
 *     pseudotime [out] Normalized pseudotime values [0, 1]
 *
 * PRECONDITIONS:
 *     - pseudotime.len >= adjacency.primary_dim()
 *     - root_cell in valid range
 *     - Graph should be connected from root
 *
 * POSTCONDITIONS:
 *     - pseudotime[root_cell] = 0
 *     - pseudotime[i] in [0, 1] for all reachable cells
 *     - Unreachable cells have pseudotime = 1
 *
 * ALGORITHM:
 *     1. Run Dijkstra from root_cell
 *     2. Normalize distances to [0, 1] by dividing by max
 *     3. Set unreachable cells to 1
 *
 * COMPLEXITY:
 *     Time:  O((V + E) * log_4(V))
 *     Space: O(V) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - uses single-threaded Dijkstra
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void graph_pseudotime(
    const Sparse<T, IsCSR>& adjacency,   // Cell neighborhood graph
    Index root_cell,                      // Root cell index
    Array<Real> pseudotime                // Output pseudotime [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: diffusion_pseudotime
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute diffusion pseudotime (DPT) using diffusion map distance
 *     from root cell in the diffusion component space.
 *
 * PARAMETERS:
 *     transition_matrix [in]  Markov transition matrix
 *     root_cell         [in]  Starting cell for trajectory
 *     pseudotime        [out] DPT values [0, 1]
 *     n_dcs             [in]  Number of diffusion components
 *     n_iterations      [in]  Power iteration iterations
 *
 * PRECONDITIONS:
 *     - pseudotime.len >= transition_matrix.primary_dim()
 *     - root_cell in valid range
 *     - transition_matrix should be row-stochastic (rows sum to 1)
 *
 * POSTCONDITIONS:
 *     - pseudotime based on diffusion distance from root
 *     - Values normalized to [0, 1]
 *     - Captures connectivity structure beyond graph distance
 *
 * ALGORITHM:
 *     1. Initialize random diffusion components [n x n_dcs]
 *     2. Power iteration: DC = T * DC (apply transition)
 *     3. Orthonormalize with modified Gram-Schmidt
 *     4. Compute Euclidean distance from root in DC space
 *     5. Normalize to [0, 1]
 *
 * COMPLEXITY:
 *     Time:  O(n_iterations * nnz * n_dcs)
 *     Space: O(n * n_dcs) for diffusion components
 *
 * THREAD SAFETY:
 *     Safe - uses parallel SpMM and parallel distance computation
 *
 * NUMERICAL NOTES:
 *     More robust to noise than shortest path pseudotime
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void diffusion_pseudotime(
    const Sparse<T, IsCSR>& transition_matrix,  // Markov transition matrix
    Index root_cell,                             // Root cell index
    Array<Real> pseudotime,                      // Output DPT [n]
    Index n_dcs = config::DEFAULT_N_DCS,         // Number of diffusion components
    Index n_iterations = config::DEFAULT_N_ITERATIONS  // Power iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: select_root_cell
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select root cell as the one with minimum marker gene expression.
 *
 * PARAMETERS:
 *     adjacency         [in] Cell neighborhood graph
 *     marker_expression [in] Expression of stem/early marker gene per cell
 *
 * RETURNS:
 *     Index of cell with minimum marker expression
 *
 * PRECONDITIONS:
 *     - marker_expression.len >= adjacency.primary_dim()
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - read-only operation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index select_root_cell(
    const Sparse<T, IsCSR>& adjacency,        // Cell graph (for dimension)
    Array<const Real> marker_expression        // Marker gene expression [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: select_root_peripheral
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select root cell as the most peripheral node (highest average
 *     edge weight to neighbors, indicating isolation).
 *
 * PARAMETERS:
 *     adjacency [in] Cell neighborhood graph
 *
 * RETURNS:
 *     Index of most peripheral cell
 *
 * ALGORITHM:
 *     1. For each cell: compute average edge weight to neighbors
 *     2. Return cell with maximum average (most isolated)
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized average computation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index select_root_peripheral(
    const Sparse<T, IsCSR>& adjacency         // Cell neighborhood graph
);

/* -----------------------------------------------------------------------------
 * FUNCTION: detect_branch_points
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify branch points in trajectory based on pseudotime topology.
 *
 * PARAMETERS:
 *     adjacency     [in]  Cell neighborhood graph
 *     pseudotime    [in]  Pre-computed pseudotime values
 *     branch_points [out] Indices of detected branch point cells
 *     threshold     [in]  Pseudotime difference threshold for neighbor classification
 *
 * RETURNS:
 *     Number of branch points detected
 *
 * PRECONDITIONS:
 *     - branch_points.len >= adjacency.primary_dim()
 *     - pseudotime computed and valid
 *
 * POSTCONDITIONS:
 *     - branch_points[0..return_value) contains branch cell indices
 *     - Branch defined as: (>=1 earlier and >=2 later) or (>=2 earlier and >=1 later)
 *
 * ALGORITHM:
 *     For each cell in parallel:
 *         1. Count neighbors with pseudotime < (pt_i - threshold) (earlier)
 *         2. Count neighbors with pseudotime > (pt_i + threshold) (later)
 *         3. Mark as branch if asymmetric neighbor distribution
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(n * n_threads) for thread-local buffers
 *
 * THREAD SAFETY:
 *     Safe - parallelized with thread-local branch collection
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index detect_branch_points(
    const Sparse<T, IsCSR>& adjacency,        // Cell neighborhood graph
    Array<const Real> pseudotime,              // Pseudotime values [n]
    Array<Index> branch_points,                // Output branch indices [n]
    Real threshold = config::DEFAULT_THRESHOLD // Neighbor classification threshold
);

/* -----------------------------------------------------------------------------
 * FUNCTION: segment_trajectory
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Assign cells to trajectory segments based on branch points.
 *
 * PARAMETERS:
 *     adjacency       [in]  Cell neighborhood graph
 *     pseudotime      [in]  Pseudotime values
 *     branch_points   [in]  Detected branch point indices
 *     n_branch_points [in]  Number of branch points
 *     segment_labels  [out] Segment assignment for each cell
 *
 * PRECONDITIONS:
 *     - segment_labels.len >= adjacency.primary_dim()
 *     - branch_points contains valid indices
 *
 * POSTCONDITIONS:
 *     - segment_labels[i] in [0, n_branch_points]
 *     - Cells before first branch: segment 0
 *     - Cells between branch k and k+1: segment k+1
 *
 * ALGORITHM:
 *     1. Sort branch points by pseudotime
 *     2. For each cell: find segment by comparing to branch pseudotimes
 *
 * COMPLEXITY:
 *     Time:  O(n * n_branch_points)
 *     Space: O(n_branch_points) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized segment assignment
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void segment_trajectory(
    const Sparse<T, IsCSR>& adjacency,        // Cell neighborhood graph
    Array<const Real> pseudotime,              // Pseudotime values [n]
    Array<const Index> branch_points,          // Branch point indices
    Index n_branch_points,                     // Number of branches
    Array<Index> segment_labels                // Output segments [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: smooth_pseudotime
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Smooth pseudotime values using neighborhood averaging.
 *
 * PARAMETERS:
 *     adjacency    [in]     Cell neighborhood graph
 *     pseudotime   [in,out] Pseudotime values to smooth
 *     n_iterations [in]     Number of smoothing iterations
 *     alpha        [in]     Smoothing strength [0, 1]
 *
 * PRECONDITIONS:
 *     - pseudotime.len >= adjacency.primary_dim()
 *     - 0 <= alpha <= 1
 *
 * POSTCONDITIONS:
 *     - Pseudotime smoothed: pt = (1-alpha)*pt + alpha*avg(neighbors)
 *     - Repeated n_iterations times
 *
 * MUTABILITY:
 *     INPLACE - modifies pseudotime array
 *
 * COMPLEXITY:
 *     Time:  O(n_iterations * nnz)
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized averaging
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void smooth_pseudotime(
    const Sparse<T, IsCSR>& adjacency,        // Cell neighborhood graph
    Array<Real> pseudotime,                    // Pseudotime to smooth [n]
    Index n_iterations = 10,                   // Smoothing iterations
    Real alpha = Real(0.5)                     // Smoothing strength
);

/* -----------------------------------------------------------------------------
 * FUNCTION: pseudotime_correlation
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Pearson correlation between pseudotime and each gene.
 *
 * PARAMETERS:
 *     X            [in]  Gene expression matrix (cells x genes, CSR)
 *     pseudotime   [in]  Pseudotime values
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *     correlations [out] Per-gene correlations with pseudotime
 *
 * PRECONDITIONS:
 *     - X.rows() == n_cells
 *     - pseudotime.len >= n_cells
 *     - correlations.len >= n_genes
 *
 * POSTCONDITIONS:
 *     - correlations[g] = Pearson(pseudotime, gene_g_expression)
 *     - Accounts for sparse zeros in variance computation
 *
 * ALGORITHM:
 *     Two-pass algorithm:
 *         1. First pass: compute gene sums and covariances with pseudotime
 *         2. Second pass: compute gene variances
 *         3. Parallel correlation computation
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_genes)
 *     Space: O(n_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized final correlation computation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void pseudotime_correlation(
    const Sparse<T, IsCSR>& X,                // Expression matrix [n_cells x n_genes]
    Array<const Real> pseudotime,              // Pseudotime values [n_cells]
    Index n_cells,                             // Number of cells
    Index n_genes,                             // Number of genes
    Array<Real> correlations                   // Output correlations [n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: velocity_weighted_pseudotime
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Refine pseudotime using RNA velocity direction information.
 *
 * PARAMETERS:
 *     adjacency          [in]     Cell neighborhood graph
 *     initial_pseudotime [in]     Initial pseudotime estimate
 *     velocity_field     [in]     Per-cell velocity magnitude/direction
 *     refined_pseudotime [out]    Velocity-refined pseudotime
 *     n_iterations       [in]     Refinement iterations
 *
 * PRECONDITIONS:
 *     - All arrays have length >= adjacency.primary_dim()
 *     - velocity_field indicates direction (positive = forward in time)
 *
 * POSTCONDITIONS:
 *     - Refined pseudotime incorporates velocity information
 *     - Normalized to [0, 1]
 *
 * ALGORITHM:
 *     For each iteration:
 *         1. For each cell: weighted average with velocity-adjusted weights
 *         2. Neighbors earlier in pseudotime: weight by 1/(1+velocity)
 *         3. Neighbors later: weight by (1+velocity)
 *         4. Renormalize to [0, 1]
 *
 * COMPLEXITY:
 *     Time:  O(n_iterations * nnz)
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized refinement
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void velocity_weighted_pseudotime(
    const Sparse<T, IsCSR>& adjacency,        // Cell neighborhood graph
    Array<const Real> initial_pseudotime,      // Initial pseudotime [n]
    Array<const Real> velocity_field,          // Velocity per cell [n]
    Array<Real> refined_pseudotime,            // Output refined pseudotime [n]
    Index n_iterations = 20                    // Refinement iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: find_terminal_states
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify terminal (end) states as cells above pseudotime percentile.
 *
 * PARAMETERS:
 *     adjacency      [in]  Cell neighborhood graph
 *     pseudotime     [in]  Pseudotime values
 *     terminal_cells [out] Indices of terminal state cells
 *     percentile     [in]  Pseudotime threshold percentile (e.g., 0.95)
 *
 * RETURNS:
 *     Number of terminal cells identified
 *
 * PRECONDITIONS:
 *     - terminal_cells.len >= adjacency.primary_dim()
 *     - 0 < percentile <= 1
 *
 * POSTCONDITIONS:
 *     - terminal_cells[0..return_value) contains cells with pt >= threshold
 *     - threshold = percentile of pseudotime distribution
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for percentile, O(n) for selection
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential sorting for percentile
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index find_terminal_states(
    const Sparse<T, IsCSR>& adjacency,        // Cell neighborhood graph
    Array<const Real> pseudotime,              // Pseudotime values [n]
    Array<Index> terminal_cells,               // Output terminal indices [n]
    Real percentile = Real(0.95)               // Percentile threshold
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_backbone
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select representative backbone cells evenly spaced along pseudotime.
 *
 * PARAMETERS:
 *     adjacency        [in]  Cell neighborhood graph
 *     pseudotime       [in]  Pseudotime values
 *     n_backbone_cells [in]  Number of backbone cells to select
 *     backbone_indices [out] Indices of selected backbone cells
 *
 * RETURNS:
 *     Actual number of backbone cells selected (may be < n_backbone_cells)
 *
 * PRECONDITIONS:
 *     - backbone_indices.len >= n_backbone_cells
 *     - n_backbone_cells > 0
 *
 * POSTCONDITIONS:
 *     - backbone_indices contains cells uniformly sampled in pseudotime
 *     - Covers full range from earliest to latest pseudotime
 *
 * ALGORITHM:
 *     1. Sort cells by pseudotime
 *     2. Select cells at uniform pseudotime intervals
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for sorting
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential sorting
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index compute_backbone(
    const Sparse<T, IsCSR>& adjacency,        // Cell neighborhood graph
    Array<const Real> pseudotime,              // Pseudotime values [n]
    Index n_backbone_cells,                    // Desired backbone size
    Array<Index> backbone_indices              // Output backbone indices
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_pseudotime
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Generic pseudotime computation with method selection.
 *
 * PARAMETERS:
 *     adjacency  [in]  Cell neighborhood graph or transition matrix
 *     root_cell  [in]  Root cell index
 *     pseudotime [out] Computed pseudotime values
 *     method     [in]  Algorithm to use
 *     n_dcs      [in]  Diffusion components (for DPT method)
 *
 * PRECONDITIONS:
 *     - pseudotime.len >= adjacency.primary_dim()
 *     - root_cell in valid range
 *
 * POSTCONDITIONS:
 *     - Pseudotime computed using specified method
 *     - Values normalized to [0, 1]
 *
 * COMPLEXITY:
 *     Depends on method - see individual functions
 *
 * THREAD SAFETY:
 *     Depends on method
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_pseudotime(
    const Sparse<T, IsCSR>& adjacency,        // Graph or transition matrix
    Index root_cell,                           // Root cell index
    Array<Real> pseudotime,                    // Output pseudotime [n]
    PseudotimeMethod method = PseudotimeMethod::DiffusionPseudotime,
    Index n_dcs = config::DEFAULT_N_DCS        // Diffusion components (for DPT)
);

} // namespace scl::kernel::pseudotime
