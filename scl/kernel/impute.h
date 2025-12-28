// =============================================================================
// FILE: scl/kernel/impute.h
// BRIEF: API reference for high-performance imputation kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::impute {

// =============================================================================
// Configuration Constants
// =============================================================================

namespace config {
    constexpr Real DISTANCE_EPSILON = Real(1e-10);
    constexpr Real DEFAULT_ALPHA = Real(1.0);
    constexpr Index DEFAULT_K_NEIGHBORS = 15;
    constexpr Index DEFAULT_N_STEPS = 3;
    constexpr Index DEFAULT_N_COMPONENTS = 50;
    constexpr Size PARALLEL_THRESHOLD = 32;
    constexpr Size GENE_BLOCK_SIZE = 64;
    constexpr Size CELL_BLOCK_SIZE = 32;
}

// =============================================================================
// Enumerations
// =============================================================================

/* -----------------------------------------------------------------------------
 * ENUM: ImputeMethod
 * -----------------------------------------------------------------------------
 * VALUES:
 *     KNN        - K-nearest neighbors imputation
 *     Diffusion  - Diffusion-based imputation (MAGIC-style)
 *     ALRA       - Adaptively-thresholded Low-Rank Approximation
 *     Weighted   - Distance-weighted KNN imputation
 * -------------------------------------------------------------------------- */
enum class ImputeMethod {
    KNN,
    Diffusion,
    ALRA,
    Weighted
};

// =============================================================================
// Core Imputation Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: knn_impute_dense
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Impute missing values using K-nearest neighbor averaging on dense output.
 *
 * PARAMETERS:
 *     X_sparse     [in]  Input sparse expression matrix (n_cells x n_genes)
 *     affinity     [in]  Cell-cell affinity matrix (n_cells x n_cells)
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *     X_imputed    [out] Dense imputed matrix (n_cells x n_genes, row-major)
 *
 * PRECONDITIONS:
 *     - X_sparse must be CSR format (cells x genes)
 *     - affinity must be row-normalized (rows sum to 1)
 *     - X_imputed must be pre-allocated with n_cells * n_genes elements
 *
 * POSTCONDITIONS:
 *     - X_imputed[i, j] = weighted average of gene j across neighbors of cell i
 *     - Weights from affinity matrix define neighbor contributions
 *     - Dense output suitable for downstream dense operations
 *
 * ALGORITHM:
 *     For each cell i in parallel:
 *         1. Compute row sum from affinity matrix
 *         2. If sum < epsilon: copy original row
 *         3. Otherwise: X_out[i] = sum_k(affinity[i,k] * X[k]) / row_sum
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * avg_neighbors * n_genes)
 *     Space: O(n_cells * n_genes) for output
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells, each writes to independent memory
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void knn_impute_dense(
    const Sparse<T, IsCSR>& X_sparse,   // Sparse input (n_cells x n_genes)
    const Sparse<Real, true>& affinity,  // Affinity matrix (n_cells x n_cells)
    Index n_cells,                        // Number of cells
    Index n_genes,                        // Number of genes
    Real* X_imputed                       // Output: Dense matrix [n_cells * n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: knn_impute_weighted_dense
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Impute with distance-weighted KNN contributions.
 *
 * PARAMETERS:
 *     X_sparse     [in]  Input sparse expression matrix
 *     knn_indices  [in]  K-nearest neighbor indices [n_cells x k]
 *     knn_distances [in] K-nearest neighbor distances [n_cells x k]
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *     k            [in]  Number of neighbors
 *     X_imputed    [out] Dense imputed matrix
 *
 * PRECONDITIONS:
 *     - knn_indices and knn_distances pre-computed for all cells
 *     - k <= n_cells - 1
 *     - X_imputed must be pre-allocated
 *
 * POSTCONDITIONS:
 *     - X_imputed[i,j] = sum_k(weight[k] * X[neighbor_k, j]) / sum(weights)
 *     - Weights inversely proportional to distance
 *
 * ALGORITHM:
 *     For each cell i in parallel:
 *         1. Compute weights: w[k] = 1 / (dist[k] + epsilon)
 *         2. Normalize weights
 *         3. Weighted average across k neighbors
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * k * n_genes)
 *     Space: O(n_cells * n_genes)
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void knn_impute_weighted_dense(
    const Sparse<T, IsCSR>& X_sparse,   // Sparse input
    const Index* knn_indices,            // KNN indices [n_cells * k]
    const Real* knn_distances,           // KNN distances [n_cells * k]
    Index n_cells,                        // Number of cells
    Index n_genes,                        // Number of genes
    Index k,                              // Number of neighbors
    Real* X_imputed                       // Output: Dense matrix [n_cells * n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: diffusion_impute_sparse_transition
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Diffusion-based imputation using sparse transition matrix.
 *
 * PARAMETERS:
 *     X_sparse          [in]  Input sparse expression matrix
 *     transition_matrix [in]  Row-stochastic transition matrix
 *     n_cells           [in]  Number of cells
 *     n_genes           [in]  Number of genes
 *     n_steps           [in]  Number of diffusion steps
 *     X_imputed         [out] Dense imputed matrix
 *
 * PRECONDITIONS:
 *     - transition_matrix must be row-stochastic (rows sum to 1)
 *     - n_steps >= 1
 *     - X_imputed must be pre-allocated
 *
 * POSTCONDITIONS:
 *     - X_imputed = T^n_steps * X where T is transition matrix
 *     - Higher n_steps = more smoothing/imputation
 *
 * ALGORITHM:
 *     1. Convert sparse X to dense buffer
 *     2. For t = 1 to n_steps:
 *        - SpMM: buffer_out = T * buffer_in (parallel)
 *        - Swap buffers (double buffering)
 *     3. Copy final result to X_imputed
 *
 * COMPLEXITY:
 *     Time:  O(n_steps * n_cells * avg_nnz_per_row * n_genes)
 *     Space: O(2 * n_cells * n_genes) for double buffering
 *
 * THREAD SAFETY:
 *     Safe - uses double buffering with parallel SpMM
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR, typename TT, bool IsCSR2>
void diffusion_impute_sparse_transition(
    const Sparse<T, IsCSR>& X_sparse,         // Sparse input
    const Sparse<TT, IsCSR2>& transition_matrix, // Transition matrix
    Index n_cells,                              // Number of cells
    Index n_genes,                              // Number of genes
    Index n_steps,                              // Diffusion steps
    Real* X_imputed                             // Output: Dense [n_cells * n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: diffusion_impute_dense
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Diffusion imputation with pre-densified data for efficiency.
 *
 * PARAMETERS:
 *     X_dense           [in]  Dense expression matrix [n_cells x n_genes]
 *     transition_matrix [in]  Sparse transition matrix
 *     n_cells           [in]  Number of cells
 *     n_genes           [in]  Number of genes
 *     n_steps           [in]  Number of diffusion steps
 *     X_imputed         [out] Dense imputed matrix
 *
 * PRECONDITIONS:
 *     - X_dense already in row-major dense format
 *     - transition_matrix row-stochastic
 *
 * POSTCONDITIONS:
 *     - Same as diffusion_impute_sparse_transition
 *     - Faster when input is already dense
 *
 * COMPLEXITY:
 *     Time:  O(n_steps * n_cells * avg_nnz_per_row * n_genes)
 *     Space: O(n_cells * n_genes) additional for buffer
 *
 * THREAD SAFETY:
 *     Safe - parallel SpMM with double buffering
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void diffusion_impute_dense(
    const Real* X_dense,                      // Dense input [n_cells * n_genes]
    const Sparse<T, IsCSR>& transition_matrix, // Transition matrix
    Index n_cells,                             // Number of cells
    Index n_genes,                             // Number of genes
    Index n_steps,                             // Diffusion steps
    Real* X_imputed                            // Output: Dense [n_cells * n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: magic_impute
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     MAGIC (Markov Affinity-based Graph Imputation of Cells) algorithm.
 *
 * PARAMETERS:
 *     X_sparse          [in]  Input sparse expression matrix
 *     transition_matrix [in]  Diffusion operator (from MAGIC)
 *     n_cells           [in]  Number of cells
 *     n_genes           [in]  Number of genes
 *     t                 [in]  Diffusion time parameter
 *     X_imputed         [out] Dense imputed matrix
 *
 * PRECONDITIONS:
 *     - transition_matrix from MAGIC preprocessing (symmetric normalized)
 *     - t >= 1, typically t in [1, 5]
 *     - X_imputed must be pre-allocated
 *
 * POSTCONDITIONS:
 *     - X_imputed = (T^t) * X
 *     - Denoised and imputed expression values
 *     - Preserves overall expression structure
 *
 * ALGORITHM:
 *     1. Initialize dense buffer from sparse X
 *     2. Apply t steps of diffusion
 *     3. Each step: X_new = T * X_old (parallel SpMM)
 *
 * COMPLEXITY:
 *     Time:  O(t * n_cells * avg_nnz * n_genes)
 *     Space: O(2 * n_cells * n_genes)
 *
 * THREAD SAFETY:
 *     Safe - all operations parallelized
 *
 * REFERENCE:
 *     van Dijk et al., MAGIC, Cell 2018
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR, typename TT, bool IsCSR2>
void magic_impute(
    const Sparse<T, IsCSR>& X_sparse,         // Sparse input
    const Sparse<TT, IsCSR2>& transition_matrix, // MAGIC diffusion operator
    Index n_cells,                              // Number of cells
    Index n_genes,                              // Number of genes
    Index t,                                    // Diffusion time
    Real* X_imputed                             // Output: Dense [n_cells * n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: alra_impute
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     ALRA (Adaptively-thresholded Low-Rank Approximation) imputation.
 *
 * PARAMETERS:
 *     X_dense      [in]  Dense normalized expression [n_cells x n_genes]
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *     n_components [in]  Number of SVD components (rank)
 *     X_imputed    [out] Dense imputed matrix
 *     n_iter       [in]  Number of power iterations for SVD
 *     seed         [in]  Random seed
 *
 * PRECONDITIONS:
 *     - X_dense already log-normalized
 *     - n_components <= min(n_cells, n_genes)
 *     - X_imputed must be pre-allocated
 *
 * POSTCONDITIONS:
 *     - X_imputed = U * S * V^T (rank-k approximation)
 *     - Negative values set to zero (biological constraint)
 *     - Original non-zero values preserved where imputed < original
 *
 * ALGORITHM:
 *     1. Randomized SVD via power iteration:
 *        - Random Gaussian projection
 *        - Power iteration for numerical stability
 *        - QR orthogonalization
 *     2. Threshold negative values to zero
 *     3. Preserve original non-zero values where appropriate
 *
 * COMPLEXITY:
 *     Time:  O(n_iter * n_cells * n_genes * n_components)
 *     Space: O(n_cells * n_components + n_genes * n_components)
 *
 * THREAD SAFETY:
 *     Safe - parallel matrix operations
 *
 * REFERENCE:
 *     Linderman et al., ALRA, bioRxiv 2018
 * -------------------------------------------------------------------------- */
void alra_impute(
    const Real* X_dense,               // Dense normalized input
    Index n_cells,                      // Number of cells
    Index n_genes,                      // Number of genes
    Index n_components,                 // SVD rank
    Real* X_imputed,                    // Output: Dense [n_cells * n_genes]
    Index n_iter = 5,                   // Power iterations
    uint64_t seed = 42                  // Random seed
);

// =============================================================================
// Auxiliary Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: impute_selected_genes
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Impute only a subset of genes for efficiency.
 *
 * PARAMETERS:
 *     X_sparse       [in]  Input sparse expression matrix
 *     affinity       [in]  Cell-cell affinity matrix
 *     gene_indices   [in]  Indices of genes to impute
 *     n_selected     [in]  Number of genes to impute
 *     n_cells        [in]  Number of cells
 *     X_imputed      [out] Imputed values for selected genes [n_cells x n_selected]
 *
 * PRECONDITIONS:
 *     - All gene indices in [0, n_genes)
 *     - X_imputed pre-allocated with n_cells * n_selected elements
 *
 * POSTCONDITIONS:
 *     - X_imputed[i, j] contains imputed value for cell i, selected gene j
 *     - Only computes imputation for specified genes (memory efficient)
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * avg_neighbors * n_selected)
 *     Space: O(n_cells * n_selected)
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void impute_selected_genes(
    const Sparse<T, IsCSR>& X_sparse,   // Sparse input
    const Sparse<Real, true>& affinity,  // Affinity matrix
    const Index* gene_indices,           // Genes to impute [n_selected]
    Index n_selected,                     // Number of selected genes
    Index n_cells,                        // Number of cells
    Real* X_imputed                       // Output: [n_cells * n_selected]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: detect_dropouts
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Detect likely dropout events (technical zeros vs biological zeros).
 *
 * PARAMETERS:
 *     X_sparse     [in]  Input sparse expression matrix
 *     gene_means   [in]  Pre-computed gene means [n_genes]
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *     n_dropouts   [out] Count of detected dropouts [n_genes]
 *     threshold    [in]  Detection threshold
 *
 * PRECONDITIONS:
 *     - gene_means pre-computed from normalized data
 *     - n_dropouts must be pre-allocated with n_genes elements
 *
 * POSTCONDITIONS:
 *     - n_dropouts[g] = number of cells where gene g is likely dropout
 *     - Uses gene mean and detection rate to infer dropouts
 *
 * ALGORITHM:
 *     For each gene g in parallel:
 *         1. Count zeros in gene column
 *         2. Estimate expected zeros from Poisson model
 *         3. Excess zeros = dropouts
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_genes)
 *     Space: O(n_genes) for output
 *
 * THREAD SAFETY:
 *     Safe - uses atomic accumulation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void detect_dropouts(
    const Sparse<T, IsCSR>& X_sparse,  // Sparse input
    const Real* gene_means,             // Gene means [n_genes]
    Index n_cells,                       // Number of cells
    Index n_genes,                       // Number of genes
    Index* n_dropouts,                   // Output: Dropout counts [n_genes]
    Real threshold = Real(0.5)           // Detection threshold
);

/* -----------------------------------------------------------------------------
 * FUNCTION: imputation_quality
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute imputation quality metrics (correlation with held-out data).
 *
 * PARAMETERS:
 *     X_original  [in]  Original sparse expression matrix
 *     X_imputed   [in]  Imputed dense matrix
 *     n_cells     [in]  Number of cells
 *     n_genes     [in]  Number of genes
 *
 * RETURNS:
 *     Average Pearson correlation between original and imputed values
 *
 * PRECONDITIONS:
 *     - X_original and X_imputed have same dimensions
 *     - X_imputed in row-major dense format
 *
 * POSTCONDITIONS:
 *     - Returns correlation in [-1, 1]
 *     - Higher values indicate better imputation quality
 *
 * ALGORITHM:
 *     1. For each gene, compute correlation between original and imputed
 *     2. Return mean correlation across genes
 *     Uses Welford's online algorithm for numerical stability
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_genes)
 *     Space: O(n_threads) for thread-local accumulators
 *
 * THREAD SAFETY:
 *     Safe - parallel reduction with thread-local stats
 * -------------------------------------------------------------------------- */
Real imputation_quality(
    const Real* X_original,             // Original dense values
    const Real* X_imputed,              // Imputed dense values
    Index n_cells,                       // Number of cells
    Index n_genes                        // Number of genes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: smooth_expression
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Smooth expression profiles using local averaging.
 *
 * PARAMETERS:
 *     X_sparse  [in]  Input sparse expression matrix
 *     affinity  [in]  Cell-cell affinity matrix
 *     n_cells   [in]  Number of cells
 *     n_genes   [in]  Number of genes
 *     alpha     [in]  Smoothing factor (0 = original, 1 = full neighbor average)
 *     X_smooth  [out] Smoothed dense matrix
 *
 * PRECONDITIONS:
 *     - alpha in [0, 1]
 *     - affinity row-normalized
 *     - X_smooth must be pre-allocated
 *
 * POSTCONDITIONS:
 *     - X_smooth[i] = (1 - alpha) * X[i] + alpha * neighbor_average[i]
 *     - Interpolates between original and fully smoothed
 *
 * ALGORITHM:
 *     For each cell i in parallel:
 *         1. Compute neighbor average from affinity-weighted sum
 *         2. Interpolate: result = (1-alpha)*original + alpha*average
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * avg_neighbors * n_genes)
 *     Space: O(n_cells * n_genes)
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void smooth_expression(
    const Sparse<T, IsCSR>& X_sparse,   // Sparse input
    const Sparse<Real, true>& affinity,  // Affinity matrix
    Index n_cells,                        // Number of cells
    Index n_genes,                        // Number of genes
    Real alpha,                           // Smoothing factor [0, 1]
    Real* X_smooth                        // Output: Dense [n_cells * n_genes]
);

} // namespace scl::kernel::impute
