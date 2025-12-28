// =============================================================================
// FILE: scl/kernel/velocity.h
// BRIEF: API reference for RNA velocity analysis
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::velocity {

/* -----------------------------------------------------------------------------
 * ENUM: VelocityModel
 * -----------------------------------------------------------------------------
 * VALUES:
 *     SteadyState - Simple gamma * s = u model
 *     Dynamical   - Time-dependent model (not yet implemented)
 *     Stochastic  - Stochastic differential equations (not yet implemented)
 * -------------------------------------------------------------------------- */
enum class VelocityModel {
    SteadyState,
    Dynamical,
    Stochastic
};

/* -----------------------------------------------------------------------------
 * FUNCTION: fit_gene_kinetics
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Fit kinetic parameters (gamma) for each gene using spliced/unspliced data.
 *
 * PARAMETERS:
 *     spliced   [in]  Spliced expression matrix (cells x genes or genes x cells)
 *     unspliced [in]  Unspliced expression matrix
 *     n_cells   [in]  Number of cells
 *     n_genes   [in]  Number of genes
 *     gamma     [out] Degradation rate for each gene [n_genes]
 *     r2        [out] R-squared fit quality for each gene [n_genes]
 *     model     [in]  Velocity model to use (default: SteadyState)
 *
 * PRECONDITIONS:
 *     - gamma.len >= n_genes
 *     - r2.len >= n_genes
 *     - spliced and unspliced have same dimensions
 *
 * POSTCONDITIONS:
 *     - gamma[g] = estimated degradation rate for gene g
 *     - r2[g] = fit quality (0 to 1)
 *
 * ALGORITHM:
 *     SteadyState model: Linear regression u = gamma * s
 *     - SIMD-optimized linear regression with 6 accumulators
 *     - CSR: Binary search for gene values across cells
 *     - CSC: Direct column access
 *
 * COMPLEXITY:
 *     Time:  O(n_genes * n_cells)
 *     Space: O(n_cells) per thread for buffers
 *
 * THREAD SAFETY:
 *     Safe - uses DualWorkspacePool for thread-local buffers
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void fit_gene_kinetics(
    const Sparse<T, IsCSR>& spliced,   // Spliced expression
    const Sparse<T, IsCSR>& unspliced, // Unspliced expression
    Index n_cells,                     // Number of cells
    Index n_genes,                     // Number of genes
    Array<Real> gamma,                 // Output: degradation rates [n_genes]
    Array<Real> r2,                    // Output: R-squared values [n_genes]
    VelocityModel model = VelocityModel::SteadyState  // Velocity model
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_velocity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute RNA velocity for each cell-gene pair.
 *
 * PARAMETERS:
 *     spliced      [in]  Spliced expression matrix
 *     unspliced    [in]  Unspliced expression matrix
 *     gamma        [in]  Pre-computed degradation rates [n_genes]
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *     velocity_out [out] Velocity matrix [n_cells x n_genes], row-major
 *
 * POSTCONDITIONS:
 *     - velocity_out[c,g] = unspliced[c,g] - gamma[g] * spliced[c,g]
 *     - Positive velocity = gene being upregulated
 *     - Negative velocity = gene being downregulated
 *
 * ALGORITHM:
 *     CSR: Parallel over cells, process each row
 *     CSC: Parallel over genes, update all cells per gene
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(n_cells * n_genes) for output
 *
 * THREAD SAFETY:
 *     Safe - no race conditions for CSR (row-parallel) or CSC (gene-parallel)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_velocity(
    const Sparse<T, IsCSR>& spliced,   // Spliced expression
    const Sparse<T, IsCSR>& unspliced, // Unspliced expression
    Array<const Real> gamma,           // Degradation rates [n_genes]
    Index n_cells,                     // Number of cells
    Index n_genes,                     // Number of genes
    Real* SCL_RESTRICT velocity_out    // Output velocity [n_cells x n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: splice_ratio
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute unspliced/spliced ratio for each cell-gene pair.
 *
 * PARAMETERS:
 *     spliced   [in]  Spliced expression matrix
 *     unspliced [in]  Unspliced expression matrix
 *     n_cells   [in]  Number of cells
 *     n_genes   [in]  Number of genes
 *     ratio_out [out] Ratio matrix [n_cells x n_genes]
 *
 * POSTCONDITIONS:
 *     - ratio_out[c,g] = unspliced[c,g] / (spliced[c,g] + epsilon)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void splice_ratio(
    const Sparse<T, IsCSR>& spliced,   // Spliced expression
    const Sparse<T, IsCSR>& unspliced, // Unspliced expression
    Index n_cells,                     // Number of cells
    Index n_genes,                     // Number of genes
    Real* SCL_RESTRICT ratio_out       // Output ratio [n_cells x n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: velocity_graph
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Build velocity-based transition graph from kNN.
 *
 * PARAMETERS:
 *     velocity         [in]  Velocity vectors [n_cells x n_genes]
 *     expression       [in]  Expression vectors [n_cells x n_genes]
 *     knn              [in]  KNN graph (sparse)
 *     n_cells          [in]  Number of cells
 *     n_genes          [in]  Number of genes
 *     transition_probs [out] Transition probabilities [n_cells x k_neighbors]
 *     k_neighbors      [in]  Number of neighbors per cell
 *
 * ALGORITHM:
 *     For each cell i (parallel):
 *     1. For each neighbor j: delta = expression[j] - expression[i]
 *     2. Compute cosine similarity between velocity[i] and delta
 *     3. Apply softmax to get transition probabilities
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * k * n_genes)
 *     Space: O(n_genes) per thread
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for delta buffers
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void velocity_graph(
    const Real* SCL_RESTRICT velocity,     // Velocity [n_cells x n_genes]
    const Real* SCL_RESTRICT expression,   // Expression [n_cells x n_genes]
    const Sparse<T, IsCSR>& knn,           // KNN graph
    Index n_cells,                         // Number of cells
    Index n_genes,                         // Number of genes
    Real* SCL_RESTRICT transition_probs,   // Output [n_cells x k_neighbors]
    Index k_neighbors                      // Neighbors per cell
);

/* -----------------------------------------------------------------------------
 * FUNCTION: velocity_graph_cosine
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Build velocity graph using cosine similarity between velocity vectors.
 *
 * PARAMETERS:
 *     velocity         [in]  Velocity vectors [n_cells x n_genes]
 *     knn_indices      [in]  KNN indices [n_cells x k_neighbors]
 *     n_cells          [in]  Number of cells
 *     n_genes          [in]  Number of genes
 *     k_neighbors      [in]  Neighbors per cell
 *     transition_probs [out] Transition probabilities [n_cells x k_neighbors]
 *
 * ALGORITHM:
 *     For each cell pair (i, j):
 *     prob[i,j] = (cos_similarity(vel_i, vel_j) + 1) / 2
 *     Then normalize per row
 * -------------------------------------------------------------------------- */
void velocity_graph_cosine(
    const Real* SCL_RESTRICT velocity,     // Velocity [n_cells x n_genes]
    const Index* SCL_RESTRICT knn_indices, // KNN indices [n_cells x k]
    Index n_cells,                         // Number of cells
    Index n_genes,                         // Number of genes
    Index k_neighbors,                     // Neighbors per cell
    Real* SCL_RESTRICT transition_probs    // Output [n_cells x k]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: velocity_embedding
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Project velocity to low-dimensional embedding space.
 *
 * PARAMETERS:
 *     velocity          [in]  Velocity [n_cells x n_genes]
 *     embedding         [in]  Low-dim embedding [n_cells x n_dims]
 *     knn_indices       [in]  KNN indices [n_cells x k]
 *     n_cells           [in]  Number of cells
 *     n_genes           [in]  Number of genes
 *     n_dims            [in]  Embedding dimensions
 *     k_neighbors       [in]  Neighbors per cell
 *     velocity_embedded [out] Projected velocity [n_cells x n_dims]
 *
 * ALGORITHM:
 *     For each cell: weighted average of neighbor direction vectors
 *     in embedding space, weighted by velocity magnitude.
 * -------------------------------------------------------------------------- */
void velocity_embedding(
    const Real* SCL_RESTRICT velocity,         // Velocity [n_cells x n_genes]
    const Real* SCL_RESTRICT embedding,        // Embedding [n_cells x n_dims]
    const Index* SCL_RESTRICT knn_indices,     // KNN indices [n_cells x k]
    Index n_cells,                             // Number of cells
    Index n_genes,                             // Number of genes
    Index n_dims,                              // Embedding dimensions
    Index k_neighbors,                         // Neighbors per cell
    Real* SCL_RESTRICT velocity_embedded       // Output [n_cells x n_dims]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: velocity_grid
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute velocity on a regular grid for visualization.
 *
 * PARAMETERS:
 *     embedding         [in]  2D embedding [n_cells x 2]
 *     velocity_embedded [in]  2D velocity [n_cells x 2]
 *     n_cells           [in]  Number of cells
 *     n_dims            [in]  Must be 2
 *     grid_size         [in]  Grid resolution per axis
 *     grid_coords       [out] Grid coordinates [grid_size^2 x 2]
 *     grid_velocity     [out] Grid velocities [grid_size^2 x 2]
 *
 * PRECONDITIONS:
 *     - n_dims == 2
 *
 * ALGORITHM:
 *     1. Create regular grid over embedding bounds
 *     2. Assign cells to grid bins
 *     3. Average velocities per bin
 * -------------------------------------------------------------------------- */
void velocity_grid(
    const Real* SCL_RESTRICT embedding,        // 2D embedding [n_cells x 2]
    const Real* SCL_RESTRICT velocity_embedded,// 2D velocity [n_cells x 2]
    Index n_cells,                             // Number of cells
    Index n_dims,                              // Must be 2
    Index grid_size,                           // Grid resolution
    Real* SCL_RESTRICT grid_coords,            // Output grid coords [grid^2 x 2]
    Real* SCL_RESTRICT grid_velocity           // Output grid velocity [grid^2 x 2]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: velocity_confidence
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute velocity confidence as consistency with neighbors.
 *
 * PARAMETERS:
 *     velocity    [in]  Velocity [n_cells x n_genes]
 *     knn_indices [in]  KNN indices [n_cells x k]
 *     n_cells     [in]  Number of cells
 *     n_genes     [in]  Number of genes
 *     k_neighbors [in]  Neighbors per cell
 *     confidence  [out] Confidence scores [n_cells]
 *
 * POSTCONDITIONS:
 *     - confidence[i] = average cosine similarity with neighbors
 *     - Range: [-1, 1], higher = more consistent
 * -------------------------------------------------------------------------- */
void velocity_confidence(
    const Real* SCL_RESTRICT velocity,     // Velocity [n_cells x n_genes]
    const Index* SCL_RESTRICT knn_indices, // KNN indices [n_cells x k]
    Index n_cells,                         // Number of cells
    Index n_genes,                         // Number of genes
    Index k_neighbors,                     // Neighbors per cell
    Real* SCL_RESTRICT confidence          // Output confidence [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: latent_time
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Infer latent time from transition probabilities using shortest paths.
 *
 * PARAMETERS:
 *     transition_probs [in]  Transition probabilities [n_cells x k]
 *     knn_indices      [in]  KNN indices [n_cells x k]
 *     n_cells          [in]  Number of cells
 *     k_neighbors      [in]  Neighbors per cell
 *     root_cell        [in]  Root cell index
 *     latent_time_out  [out] Latent time [n_cells]
 *
 * POSTCONDITIONS:
 *     - latent_time[root_cell] = 0
 *     - latent_time normalized to [0, 1]
 *
 * ALGORITHM:
 *     Bellman-Ford shortest path with -log(prob) as edge weights.
 * -------------------------------------------------------------------------- */
void latent_time(
    const Real* SCL_RESTRICT transition_probs, // Transition probs [n_cells x k]
    const Index* SCL_RESTRICT knn_indices,     // KNN indices [n_cells x k]
    Index n_cells,                             // Number of cells
    Index k_neighbors,                         // Neighbors per cell
    Index root_cell,                           // Root cell index
    Real* SCL_RESTRICT latent_time_out         // Output latent time [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: cell_fate_probability
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute probability of reaching each terminal state.
 *
 * PARAMETERS:
 *     transition_probs [in]  Transition probabilities [n_cells x k]
 *     knn_indices      [in]  KNN indices [n_cells x k]
 *     n_cells          [in]  Number of cells
 *     k_neighbors      [in]  Neighbors per cell
 *     terminal_cells   [in]  Terminal cell indices
 *     fate_probs       [out] Fate probabilities [n_cells x n_terminal]
 *
 * POSTCONDITIONS:
 *     - fate_probs[c, t] = probability that cell c reaches terminal t
 *     - Sum over t equals 1 for each cell
 * -------------------------------------------------------------------------- */
void cell_fate_probability(
    const Real* SCL_RESTRICT transition_probs, // Transition probs [n_cells x k]
    const Index* SCL_RESTRICT knn_indices,     // KNN indices [n_cells x k]
    Index n_cells,                             // Number of cells
    Index k_neighbors,                         // Neighbors per cell
    Array<const Index> terminal_cells,         // Terminal cell indices
    Real* SCL_RESTRICT fate_probs              // Output [n_cells x n_terminal]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: select_velocity_genes
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select genes with reliable velocity estimates.
 *
 * PARAMETERS:
 *     velocity         [in]  Velocity [n_cells x n_genes]
 *     gamma            [in]  Degradation rates [n_genes]
 *     r2               [in]  Fit quality [n_genes]
 *     n_cells          [in]  Number of cells
 *     n_genes          [in]  Number of genes
 *     min_r2           [in]  Minimum R-squared threshold
 *     min_velocity_var [in]  Minimum velocity variance
 *     selected_genes   [out] Selected gene indices
 *     n_selected       [out] Number of selected genes
 * -------------------------------------------------------------------------- */
void select_velocity_genes(
    const Real* SCL_RESTRICT velocity,     // Velocity [n_cells x n_genes]
    const Real* SCL_RESTRICT gamma,        // Degradation rates [n_genes]
    const Real* SCL_RESTRICT r2,           // Fit quality [n_genes]
    Index n_cells,                         // Number of cells
    Index n_genes,                         // Number of genes
    Real min_r2,                           // Min R-squared
    Real min_velocity_var,                 // Min velocity variance
    Array<Index> selected_genes,           // Output selected genes
    Index& n_selected                      // Output count
);

/* -----------------------------------------------------------------------------
 * FUNCTION: velocity_pseudotime
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute velocity-informed pseudotime.
 *
 * PARAMETERS:
 *     transition_probs   [in]  Transition probabilities
 *     knn_indices        [in]  KNN indices
 *     velocity_magnitude [in]  Velocity magnitudes [n_cells]
 *     n_cells            [in]  Number of cells
 *     k_neighbors        [in]  Neighbors per cell
 *     root_cell          [in]  Root cell index
 *     pseudotime         [out] Pseudotime [n_cells]
 *
 * ALGORITHM:
 *     Refines latent time using velocity magnitude weighting.
 * -------------------------------------------------------------------------- */
void velocity_pseudotime(
    const Real* SCL_RESTRICT transition_probs,   // Transition probs
    const Index* SCL_RESTRICT knn_indices,       // KNN indices
    const Real* SCL_RESTRICT velocity_magnitude, // Velocity magnitude [n_cells]
    Index n_cells,                               // Number of cells
    Index k_neighbors,                           // Neighbors per cell
    Index root_cell,                             // Root cell
    Real* SCL_RESTRICT pseudotime                // Output pseudotime [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: velocity_divergence
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute velocity divergence in embedding space.
 *
 * PARAMETERS:
 *     velocity_embedded [in]  Embedded velocity [n_cells x n_dims]
 *     embedding         [in]  Embedding [n_cells x n_dims]
 *     knn_indices       [in]  KNN indices [n_cells x k]
 *     n_cells           [in]  Number of cells
 *     n_dims            [in]  Embedding dimensions
 *     k_neighbors       [in]  Neighbors per cell
 *     divergence        [out] Divergence [n_cells]
 *
 * POSTCONDITIONS:
 *     - Positive divergence = source-like (expanding)
 *     - Negative divergence = sink-like (contracting)
 * -------------------------------------------------------------------------- */
void velocity_divergence(
    const Real* SCL_RESTRICT velocity_embedded,// Embedded velocity [n_cells x n_dims]
    const Real* SCL_RESTRICT embedding,        // Embedding [n_cells x n_dims]
    const Index* SCL_RESTRICT knn_indices,     // KNN indices [n_cells x k]
    Index n_cells,                             // Number of cells
    Index n_dims,                              // Embedding dimensions
    Index k_neighbors,                         // Neighbors per cell
    Real* SCL_RESTRICT divergence              // Output divergence [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: select_root_by_velocity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select root cell as the one with minimum incoming flow.
 *
 * PARAMETERS:
 *     transition_probs [in]  Transition probabilities
 *     knn_indices      [in]  KNN indices
 *     n_cells          [in]  Number of cells
 *     k_neighbors      [in]  Neighbors per cell
 *
 * RETURNS:
 *     Index of selected root cell
 * -------------------------------------------------------------------------- */
Index select_root_by_velocity(
    const Real* SCL_RESTRICT transition_probs, // Transition probs
    const Index* SCL_RESTRICT knn_indices,     // KNN indices
    Index n_cells,                             // Number of cells
    Index k_neighbors                          // Neighbors per cell
);

/* -----------------------------------------------------------------------------
 * FUNCTION: detect_terminal_states
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Detect terminal states based on velocity and outflow.
 *
 * PARAMETERS:
 *     transition_probs   [in]  Transition probabilities
 *     knn_indices        [in]  KNN indices
 *     velocity_magnitude [in]  Velocity magnitudes [n_cells]
 *     n_cells            [in]  Number of cells
 *     k_neighbors        [in]  Neighbors per cell
 *     magnitude_threshold[in]  Low velocity threshold
 *     terminal_cells     [out] Detected terminal cells
 *
 * RETURNS:
 *     Number of detected terminal cells
 * -------------------------------------------------------------------------- */
Index detect_terminal_states(
    const Real* SCL_RESTRICT transition_probs,   // Transition probs
    const Index* SCL_RESTRICT knn_indices,       // KNN indices
    const Real* SCL_RESTRICT velocity_magnitude, // Velocity magnitude
    Index n_cells,                               // Number of cells
    Index k_neighbors,                           // Neighbors per cell
    Real magnitude_threshold,                    // Threshold
    Array<Index> terminal_cells                  // Output terminals
);

} // namespace scl::kernel::velocity
