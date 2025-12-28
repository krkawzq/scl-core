// =============================================================================
// FILE: scl/kernel/transition.h
// BRIEF: API reference for cell state transition analysis
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::transition {

/* -----------------------------------------------------------------------------
 * ENUM: TransitionType
 * -----------------------------------------------------------------------------
 * VALUES:
 *     Forward   - Forward transition (velocity direction)
 *     Backward  - Backward transition (reverse velocity)
 *     Symmetric - Symmetrized transition matrix
 * -------------------------------------------------------------------------- */
enum class TransitionType {
    Forward,
    Backward,
    Symmetric
};

/* -----------------------------------------------------------------------------
 * FUNCTION: sparse_matvec
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sparse matrix-vector product: y = A * x
 *
 * PARAMETERS:
 *     mat [in]  Sparse matrix (n x n)
 *     x   [in]  Input vector [n]
 *     y   [out] Output vector [n]
 *     n   [in]  Dimension
 *
 * ALGORITHM:
 *     Parallel SpMV with 4-way unrolling and prefetch.
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void sparse_matvec(
    const Sparse<T, IsCSR>& mat,   // Sparse transition matrix
    const Real* x,                 // Input vector [n]
    Real* y,                       // Output vector [n]
    Index n                        // Dimension
);

/* -----------------------------------------------------------------------------
 * FUNCTION: sparse_matvec_transpose
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sparse matrix-vector product with transpose: y = A^T * x
 *
 * PARAMETERS:
 *     mat [in]  Sparse matrix (n x n)
 *     x   [in]  Input vector [n]
 *     y   [out] Output vector [n]
 *     n   [in]  Dimension
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void sparse_matvec_transpose(
    const Sparse<T, IsCSR>& mat,   // Sparse transition matrix
    const Real* x,                 // Input vector [n]
    Real* y,                       // Output vector [n]
    Index n                        // Dimension
);

/* -----------------------------------------------------------------------------
 * FUNCTION: is_stochastic
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Check if matrix is row-stochastic (rows sum to 1).
 *
 * PARAMETERS:
 *     mat [in]  Sparse matrix
 *     n   [in]  Dimension
 *     tol [in]  Tolerance for row sum check (default 1e-6)
 *
 * POSTCONDITIONS:
 *     Returns true if all row sums are within tol of 1.0
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
bool is_stochastic(
    const Sparse<T, IsCSR>& mat,   // Sparse matrix to check
    Index n,                       // Dimension
    Real tol = Real(1e-6)          // Tolerance
);

/* -----------------------------------------------------------------------------
 * FUNCTION: transition_matrix_from_velocity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Build transition matrix from velocity vectors and kNN graph.
 *
 * PARAMETERS:
 *     velocity       [in]     Velocity vectors (n_cells x n_genes)
 *     knn            [in]     KNN graph (sparse, n_cells x n_cells)
 *     transition_out [in,out] Output transition matrix
 *     n_cells        [in]     Number of cells
 *     n_genes        [in]     Number of genes
 *     trans_type     [in]     Type of transition (Forward/Backward/Symmetric)
 *
 * ALGORITHM:
 *     For each cell i (parallel):
 *     1. For each neighbor j: compute cosine similarity between
 *        velocity[i] and (expression[j] - expression[i])
 *     2. Apply softmax to get transition probabilities
 *     3. Store in transition matrix
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * k * n_genes)
 *     Space: O(n_genes) per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void transition_matrix_from_velocity(
    const Real* velocity,              // Velocity vectors [n_cells x n_genes]
    const Sparse<T, IsCSR>& knn,       // KNN graph
    Sparse<T, IsCSR>& transition_out,  // Output transition matrix
    Index n_cells,                     // Number of cells
    Index n_genes,                     // Number of genes
    TransitionType trans_type = TransitionType::Forward  // Transition type
);

/* -----------------------------------------------------------------------------
 * FUNCTION: row_normalize_to_stochastic
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Normalize matrix rows to sum to 1 (make row-stochastic).
 *
 * PARAMETERS:
 *     matrix [in,out] Sparse matrix to normalize
 *     n      [in]     Dimension
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix values
 *
 * POSTCONDITIONS:
 *     - Each row sums to 1.0 (or 0.0 if row was all zeros)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void row_normalize_to_stochastic(
    Sparse<T, IsCSR>& matrix,      // Matrix to normalize (modified in-place)
    Index n                        // Dimension
);

/* -----------------------------------------------------------------------------
 * FUNCTION: symmetrize_transition
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Symmetrize transition matrix: T_sym = 0.5 * (T + T^T)
 *
 * PARAMETERS:
 *     matrix [in]     Input transition matrix
 *     output [in,out] Output symmetrized matrix
 *     n      [in]     Dimension
 *
 * POSTCONDITIONS:
 *     - output is symmetric
 *     - output = (matrix + matrix^T) / 2
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void symmetrize_transition(
    const Sparse<T, IsCSR>& matrix,    // Input matrix
    Sparse<T, IsCSR>& output,          // Output symmetrized matrix
    Index n                            // Dimension
);

/* -----------------------------------------------------------------------------
 * FUNCTION: stationary_distribution
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute stationary distribution using power iteration.
 *
 * PARAMETERS:
 *     matrix   [in]  Row-stochastic transition matrix
 *     pi       [out] Output stationary distribution [n]
 *     n        [in]  Dimension
 *     tol      [in]  Convergence tolerance (default 1e-6)
 *     max_iter [in]  Maximum iterations (default 1000)
 *
 * PRECONDITIONS:
 *     - matrix is row-stochastic
 *     - Matrix is irreducible (single ergodic class)
 *
 * POSTCONDITIONS:
 *     - pi * T = pi (stationary condition)
 *     - sum(pi) = 1
 *
 * ALGORITHM:
 *     Power iteration with Aitken delta-squared acceleration:
 *     1. Initialize pi uniformly
 *     2. Iterate: pi_new = pi * T (SpMV transpose)
 *     3. Apply Aitken acceleration every 10 iterations
 *     4. Normalize and check convergence
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * nnz)
 *     Space: O(n)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void stationary_distribution(
    const Sparse<T, IsCSR>& matrix,    // Row-stochastic matrix
    Real* pi,                          // Output distribution [n]
    Index n,                           // Dimension
    Real tol = Real(1e-6),             // Convergence tolerance
    Index max_iter = 1000              // Max iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: identify_terminal_states
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify terminal (absorbing) states in the Markov chain.
 *
 * PARAMETERS:
 *     matrix         [in]  Transition matrix
 *     terminal_mask  [out] Output mask (1 = terminal, 0 = transient)
 *     n              [in]  Dimension
 *     self_loop_thresh [in] Self-loop threshold for terminal detection
 *
 * POSTCONDITIONS:
 *     - terminal_mask[i] = 1 if state i is terminal (high self-loop)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void identify_terminal_states(
    const Sparse<T, IsCSR>& matrix,    // Transition matrix
    uint8_t* terminal_mask,            // Output terminal mask [n]
    Index n,                           // Dimension
    Real self_loop_thresh = Real(0.9)  // Self-loop threshold
);

/* -----------------------------------------------------------------------------
 * FUNCTION: absorption_probability
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute absorption probabilities to terminal states.
 *
 * PARAMETERS:
 *     matrix        [in]  Transition matrix
 *     terminal_mask [in]  Terminal state mask
 *     absorb_probs  [out] Absorption probabilities [n x n_terminal]
 *     n             [in]  Total states
 *     n_terminal    [in]  Number of terminal states
 *     tol           [in]  Convergence tolerance
 *     max_iter      [in]  Maximum iterations
 *
 * ALGORITHM:
 *     Solve (I - Q) * B = R using SOR iteration:
 *     - Q = submatrix of transient-to-transient transitions
 *     - R = submatrix of transient-to-terminal transitions
 *     - omega = 1.5 for over-relaxation (faster convergence)
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * nnz * n_terminal)
 *     Space: O(n * n_terminal)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void absorption_probability(
    const Sparse<T, IsCSR>& matrix,    // Transition matrix
    const uint8_t* terminal_mask,      // Terminal state mask [n]
    Real* absorb_probs,                // Output probabilities [n x n_terminal]
    Index n,                           // Total states
    Index n_terminal,                  // Number of terminal states
    Real tol = Real(1e-6),             // Convergence tolerance
    Index max_iter = 1000              // Max iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: hitting_time
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute expected hitting time to a target state.
 *
 * PARAMETERS:
 *     matrix       [in]  Transition matrix
 *     target_state [in]  Target state index
 *     hitting_time [out] Expected hitting times [n]
 *     n            [in]  Dimension
 *     tol          [in]  Convergence tolerance
 *     max_iter     [in]  Maximum iterations
 *
 * POSTCONDITIONS:
 *     - hitting_time[target_state] = 0
 *     - hitting_time[i] = expected steps to reach target from i
 *     - Unreachable states have hitting_time = INF
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void hitting_time(
    const Sparse<T, IsCSR>& matrix,    // Transition matrix
    Index target_state,                // Target state
    Real* hitting_time,                // Output hitting times [n]
    Index n,                           // Dimension
    Real tol = Real(1e-6),             // Convergence tolerance
    Index max_iter = 1000              // Max iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: time_to_absorption
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute expected time to absorption for transient states.
 *
 * PARAMETERS:
 *     matrix        [in]  Transition matrix
 *     terminal_mask [in]  Terminal state mask
 *     absorb_time   [out] Expected absorption times [n]
 *     n             [in]  Dimension
 *     tol           [in]  Convergence tolerance
 *     max_iter      [in]  Maximum iterations
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void time_to_absorption(
    const Sparse<T, IsCSR>& matrix,    // Transition matrix
    const uint8_t* terminal_mask,      // Terminal mask [n]
    Real* absorb_time,                 // Output absorption times [n]
    Index n,                           // Dimension
    Real tol = Real(1e-6),             // Convergence tolerance
    Index max_iter = 1000              // Max iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_top_eigenvectors
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute top k eigenvectors using power iteration with deflation.
 *
 * PARAMETERS:
 *     matrix       [in]  Transition matrix
 *     eigenvectors [out] Output eigenvectors [k x n], row-major
 *     eigenvalues  [out] Output eigenvalues [k]
 *     n            [in]  Dimension
 *     k            [in]  Number of eigenvectors
 *     tol          [in]  Convergence tolerance
 *     max_iter     [in]  Max iterations per eigenvector
 *
 * ALGORITHM:
 *     For each eigenvector:
 *     1. Power iteration with Aitken acceleration
 *     2. Gram-Schmidt orthogonalization against previous
 *     3. Extract eigenvalue and store
 *
 * COMPLEXITY:
 *     Time:  O(k * max_iter * nnz)
 *     Space: O(k * n)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_top_eigenvectors(
    const Sparse<T, IsCSR>& matrix,    // Transition matrix
    Real* eigenvectors,                // Output eigenvectors [k x n]
    Real* eigenvalues,                 // Output eigenvalues [k]
    Index n,                           // Dimension
    Index k,                           // Number of eigenvectors
    Real tol = Real(1e-6),             // Convergence tolerance
    Index max_iter = 1000              // Max iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: metastable_states
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify metastable states using spectral clustering (PCCA+).
 *
 * PARAMETERS:
 *     eigenvectors [in]  Top k eigenvectors [k x n]
 *     n            [in]  Number of states
 *     k            [in]  Number of eigenvectors / metastable states
 *     assignments  [out] State-to-metastable assignments [n]
 *     membership   [out] Soft membership probabilities [n x k]
 *     seed         [in]  Random seed for k-means++
 *
 * ALGORITHM:
 *     1. K-means++ initialization on eigenvector space
 *     2. Parallel k-means assignment and update
 *     3. Compute soft membership via distance-based weights
 *
 * COMPLEXITY:
 *     Time:  O(n * k^2 * n_iter)
 *     Space: O(k * k) for centroids
 * -------------------------------------------------------------------------- */
void metastable_states(
    const Real* eigenvectors,          // Eigenvectors [k x n]
    Index n,                           // Number of states
    Index k,                           // Number of metastable states
    Index* assignments,                // Hard assignments [n]
    Real* membership,                  // Soft membership [n x k]
    uint64_t seed = 42                 // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: coarse_grain_transition
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute coarse-grained transition matrix between metastable states.
 *
 * PARAMETERS:
 *     fine_matrix   [in]  Fine transition matrix [n x n]
 *     membership    [in]  Soft membership [n x k]
 *     coarse_matrix [out] Coarse transition matrix [k x k]
 *     n             [in]  Fine states
 *     k             [in]  Coarse states
 *
 * ALGORITHM:
 *     T_coarse[i,j] = sum_m sum_l (chi_i[m] * T[m,l] * chi_j[l]) / sum_m chi_i[m]
 *
 * COMPLEXITY:
 *     Time:  O(nnz * k^2)
 *     Space: O(k^2) for accumulation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void coarse_grain_transition(
    const Sparse<T, IsCSR>& fine_matrix,   // Fine transition matrix
    const Real* membership,                // Soft membership [n x k]
    Real* coarse_matrix,                   // Output coarse matrix [k x k]
    Index n,                               // Fine states
    Index k                                // Coarse states
);

/* -----------------------------------------------------------------------------
 * FUNCTION: lineage_drivers
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify genes driving lineage transitions.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (n_cells x n_genes)
 *     absorb_probs  [in]  Absorption probabilities [n_cells x n_terminal]
 *     driver_scores [out] Driver scores [n_genes x n_terminal]
 *     n_cells       [in]  Number of cells
 *     n_genes       [in]  Number of genes
 *     n_terminal    [in]  Number of terminal states
 *
 * ALGORITHM:
 *     For each gene and terminal state:
 *     Compute correlation between gene expression and absorption probability
 *     Using Pearson correlation with SIMD dot products
 *
 * COMPLEXITY:
 *     Time:  O(n_genes * n_terminal * n_cells)
 *     Space: O(n_cells) per thread
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void lineage_drivers(
    const Sparse<T, IsCSR>& expression,    // Expression matrix
    const Real* absorb_probs,              // Absorption probs [n_cells x n_terminal]
    Real* driver_scores,                   // Output scores [n_genes x n_terminal]
    Index n_cells,                         // Number of cells
    Index n_genes,                         // Number of genes
    Index n_terminal                       // Number of terminal states
);

/* -----------------------------------------------------------------------------
 * FUNCTION: forward_committor
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute forward committor probabilities between source and target.
 *
 * PARAMETERS:
 *     matrix       [in]  Transition matrix
 *     source_mask  [in]  Source states mask [n]
 *     target_mask  [in]  Target states mask [n]
 *     committor    [out] Committor probabilities [n]
 *     n            [in]  Dimension
 *     tol          [in]  Convergence tolerance
 *     max_iter     [in]  Maximum iterations
 *
 * POSTCONDITIONS:
 *     - committor[i] = probability of hitting target before source from i
 *     - committor = 0 for source states, 1 for target states
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void forward_committor(
    const Sparse<T, IsCSR>& matrix,    // Transition matrix
    const uint8_t* source_mask,        // Source states [n]
    const uint8_t* target_mask,        // Target states [n]
    Real* committor,                   // Output committor [n]
    Index n,                           // Dimension
    Real tol = Real(1e-6),             // Convergence tolerance
    Index max_iter = 1000              // Max iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: directional_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute directional bias score for each cell.
 *
 * PARAMETERS:
 *     matrix     [in]  Transition matrix
 *     pseudotime [in]  Pseudotime values [n]
 *     scores     [out] Directional scores [n]
 *     n          [in]  Dimension
 *
 * POSTCONDITIONS:
 *     - scores[i] in [-1, 1]
 *     - Positive = forward bias, Negative = backward bias
 *
 * ALGORITHM:
 *     For each cell: weighted sum of forward vs backward transitions
 *     based on pseudotime differences to neighbors.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void directional_score(
    const Sparse<T, IsCSR>& matrix,    // Transition matrix
    const Real* pseudotime,            // Pseudotime values [n]
    Real* scores,                      // Output scores [n]
    Index n                            // Dimension
);

} // namespace scl::kernel::transition
