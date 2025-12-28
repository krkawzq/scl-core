// =============================================================================
// FILE: scl/kernel/gnn.h
// BRIEF: API reference for graph neural network operations
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::gnn {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_ALPHA = Real(0.5);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size SIMD_THRESHOLD = 16;
}

// =============================================================================
// GNN Operations
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: graph_convolution
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply graph convolution layer: H' = (D^-1 A) H W
 *
 * PARAMETERS:
 *     adjacency    [in]  Adjacency matrix (CSR)
 *     node_features [in]  Node features [n_nodes * n_features]
 *     n_nodes      [in]  Number of nodes
 *     n_features   [in]  Number of input features
 *     weights      [in]  Weight matrix [n_features * n_output_features]
 *     n_output_features [in]  Number of output features
 *     output       [out] Output features [n_nodes * n_output_features]
 *
 * PRECONDITIONS:
 *     - output has capacity >= n_nodes * n_output_features
 *
 * POSTCONDITIONS:
 *     - output contains convolved features
 *
 * COMPLEXITY:
 *     Time:  O(nnz * n_features + n_nodes * n_features * n_output_features)
 *     Space: O(n_nodes * n_output_features) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void graph_convolution(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix [n_nodes x n_nodes]
    const Real* node_features,                // Node features [n_nodes * n_features]
    Index n_nodes,                            // Number of nodes
    Index n_features,                         // Number of input features
    const Real* weights,                      // Weight matrix [n_features * n_output_features]
    Index n_output_features,                 // Number of output features
    Real* output                               // Output features [n_nodes * n_output_features]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: graph_attention
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply graph attention layer (GAT-style).
 *
 * PARAMETERS:
 *     adjacency    [in]  Adjacency matrix (CSR)
 *     node_features [in]  Node features [n_nodes * n_features]
 *     n_nodes      [in]  Number of nodes
 *     n_features   [in]  Number of features
 *     attention_weights [in]  Attention weight matrix [n_features * n_features]
 *     output       [out] Attention output [n_nodes * n_features]
 *     alpha        [in]  Attention coefficient (LeakyReLU slope)
 *
 * PRECONDITIONS:
 *     - output has capacity >= n_nodes * n_features
 *
 * POSTCONDITIONS:
 *     - output contains attended features
 *
 * COMPLEXITY:
 *     Time:  O(nnz * n_features + n_nodes * n_features^2)
 *     Space: O(n_nodes * n_features) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void graph_attention(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix [n_nodes x n_nodes]
    const Real* node_features,                // Node features [n_nodes * n_features]
    Index n_nodes,                            // Number of nodes
    Index n_features,                         // Number of features
    const Real* attention_weights,            // Attention weights [n_features^2]
    Real* output,                              // Output features [n_nodes * n_features]
    Real alpha = config::DEFAULT_ALPHA         // Attention coefficient
);

} // namespace scl::kernel::gnn

