// =============================================================================
// FILE: scl/kernel/diffusion.h
// BRIEF: API reference for high-performance diffusion processes on sparse graphs
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::diffusion {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_STEPS = 3;
    constexpr Real DEFAULT_ALPHA = Real(0.85);
    constexpr Real CONVERGENCE_TOL = Real(1e-6);
    constexpr Index MAX_ITER = 100;
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Real MIN_PROB = Real(1e-15);
    constexpr Size SPMM_BLOCK_SIZE = 64;
    constexpr Size VECTOR_BLOCK_SIZE = 256;
    constexpr Size PREFETCH_DISTANCE = 4;
    constexpr Real REORTH_TOL = Real(0.7);
    constexpr Index MAX_REORTH = 2;
}

// =============================================================================
// Diffusion Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: diffuse_vector
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply diffusion operator to a dense vector for multiple steps.
 *
 * PARAMETERS:
 *     transition [in]  Transition probability matrix (CSR)
 *     x          [in,out] Input/output vector [n_nodes], modified in-place
 *     n_steps    [in]  Number of diffusion steps
 *
 * PRECONDITIONS:
 *     - x.len >= transition.primary_dim()
 *     - Transition matrix is row-stochastic
 *
 * POSTCONDITIONS:
 *     - x contains diffused vector after n_steps
 *
 * MUTABILITY:
 *     INPLACE - modifies x
 *
 * ALGORITHM:
 *     For each step: x = transition * x
 *
 * COMPLEXITY:
 *     Time:  O(n_steps * nnz)
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized SpMV
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void diffuse_vector(
    const Sparse<T, IsCSR>& transition,     // Transition matrix [n_nodes x n_nodes]
    Array<Real> x,                            // Input/output vector [n_nodes]
    Index n_steps = config::DEFAULT_N_STEPS   // Number of steps
);

/* -----------------------------------------------------------------------------
 * FUNCTION: diffuse_matrix
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply diffusion operator to a dense matrix (multiple features).
 *
 * PARAMETERS:
 *     transition  [in]  Transition probability matrix (CSR)
 *     X           [in,out] Input/output matrix [n_nodes * n_features], modified in-place
 *     n_nodes     [in]  Number of nodes
 *     n_features  [in]  Number of features
 *     n_steps     [in]  Number of diffusion steps
 *
 * PRECONDITIONS:
 *     - X.len >= n_nodes * n_features
 *
 * POSTCONDITIONS:
 *     - X contains diffused feature matrix
 *
 * MUTABILITY:
 *     INPLACE - modifies X
 *
 * COMPLEXITY:
 *     Time:  O(n_steps * nnz * n_features)
 *     Space: O(n_nodes * n_features) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized SpMM
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void diffuse_matrix(
    const Sparse<T, IsCSR>& transition,     // Transition matrix [n_nodes x n_nodes]
    Array<Real> X,                            // Input/output matrix [n_nodes * n_features]
    Index n_nodes,                            // Number of nodes
    Index n_features,                         // Number of features
    Index n_steps = config::DEFAULT_N_STEPS   // Number of steps
);

/* -----------------------------------------------------------------------------
 * FUNCTION: diffusion_distance
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute diffusion distance matrix between all pairs of nodes.
 *
 * PARAMETERS:
 *     transition  [in]  Transition probability matrix (CSR)
 *     distances   [out] Distance matrix [n_nodes * n_nodes]
 *     n_steps     [in]  Number of diffusion steps
 *
 * PRECONDITIONS:
 *     - distances has capacity >= n_nodes * n_nodes
 *
 * POSTCONDITIONS:
 *     - distances[i * n_nodes + j] contains diffusion distance
 *
 * COMPLEXITY:
 *     Time:  O(n_nodes^2 * n_steps * nnz)
 *     Space: O(n_nodes^2) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void diffusion_distance(
    const Sparse<T, IsCSR>& transition,     // Transition matrix [n_nodes x n_nodes]
    Array<Real> distances,                    // Output distances [n_nodes^2]
    Index n_steps = config::DEFAULT_N_STEPS   // Number of steps
);

/* -----------------------------------------------------------------------------
 * FUNCTION: diffusion_pseudotime
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute diffusion pseudotime from root cells.
 *
 * PARAMETERS:
 *     transition  [in]  Transition probability matrix (CSR)
 *     root_cells  [in]  Root cell indices [n_roots]
 *     n_roots     [in]  Number of root cells
 *     pseudotime  [out] Pseudotime values [n_nodes]
 *     max_iter    [in]  Maximum iterations for convergence
 *
 * PRECONDITIONS:
 *     - pseudotime.len >= transition.primary_dim()
 *
 * POSTCONDITIONS:
 *     - pseudotime[i] contains pseudotime from nearest root
 *
 * COMPLEXITY:
 *     Time:  O(n_roots * max_iter * nnz)
 *     Space: O(n_nodes * n_roots) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over roots
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void diffusion_pseudotime(
    const Sparse<T, IsCSR>& transition,     // Transition matrix [n_nodes x n_nodes]
    Array<const Index> root_cells,          // Root cell indices [n_roots]
    Index n_roots,                           // Number of root cells
    Array<Real> pseudotime,                   // Output pseudotime [n_nodes]
    Index max_iter = config::MAX_ITER        // Max iterations
);

/* -----------------------------------------------------------------------------
 * FUNCTION: random_walk_with_restart
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute random walk with restart (RWR) scores.
 *
 * PARAMETERS:
 *     transition  [in]  Transition probability matrix (CSR)
 *     seed_nodes  [in]  Seed node indices [n_seeds]
 *     scores      [out] RWR scores [n_nodes]
 *     alpha       [in]  Restart probability (default 0.85)
 *     max_iter    [in]  Maximum iterations
 *     tol         [in]  Convergence tolerance
 *
 * PRECONDITIONS:
 *     - scores.len >= transition.primary_dim()
 *
 * POSTCONDITIONS:
 *     - scores[i] contains RWR score for node i
 *     - Scores sum to 1.0
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * nnz)
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized SpMV
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void random_walk_with_restart(
    const Sparse<T, IsCSR>& transition,     // Transition matrix [n_nodes x n_nodes]
    Array<const Index> seed_nodes,           // Seed node indices [n_seeds]
    Array<Real> scores,                      // Output RWR scores [n_nodes]
    Real alpha = config::DEFAULT_ALPHA,      // Restart probability
    Index max_iter = config::MAX_ITER,       // Max iterations
    Real tol = config::CONVERGENCE_TOL        // Convergence tolerance
);

/* -----------------------------------------------------------------------------
 * FUNCTION: diffusion_map
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute diffusion map embedding using eigendecomposition.
 *
 * PARAMETERS:
 *     transition  [in]  Transition probability matrix (CSR)
 *     embedding   [out] Diffusion map embedding [n_nodes * n_components]
 *     n_nodes     [in]  Number of nodes
 *     n_components [in]  Number of components to compute
 *     max_iter    [in]  Maximum iterations for power method
 *
 * PRECONDITIONS:
 *     - embedding has capacity >= n_nodes * n_components
 *
 * POSTCONDITIONS:
 *     - embedding contains diffusion map coordinates
 *
 * COMPLEXITY:
 *     Time:  O(n_components * max_iter * nnz)
 *     Space: O(n_nodes * n_components) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void diffusion_map(
    const Sparse<T, IsCSR>& transition,     // Transition matrix [n_nodes x n_nodes]
    Array<Real> embedding,                   // Output embedding [n_nodes * n_components]
    Index n_nodes,                           // Number of nodes
    Index n_components,                      // Number of components
    Index max_iter = config::MAX_ITER        // Max iterations
);

} // namespace scl::kernel::diffusion

