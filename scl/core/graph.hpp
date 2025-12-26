#pragma once

#include "scl/core/matrix.hpp"
#include "scl/core/macros.hpp"

// =============================================================================
/// @file graph.hpp
/// @brief SCL Graph Views (Adjacency Matrix Wrapper)
///
/// Provides semantic wrappers around Sparse Matrices to represent Graphs.
/// In SCL, a Graph is technically a CSR Adjacency Matrix, but these views
/// provide graph-specific terminology (nodes, neighbors, edges) to make
/// algorithm implementation (like Leiden, Pagerank, Diffusion) more readable.
///
/// @section Storage
/// - CSRGraph: Directed/Undirected graph stored as Compressed Sparse Row.
///   - Row `i` corresponds to Node `i`.
///   - Column indices in Row `i` are the neighbors of Node `i`.
///   - Values in Row `i` are the edge weights.
///
// =============================================================================

namespace scl {

// =============================================================================
// SECTION 1: Graph View
// =============================================================================

/// @brief Lightweight view of a graph backed by a CSR Matrix.
///
/// This struct adds zero overhead over `CSRMatrix`. It purely provides
/// semantic methods (e.g., `neighbors(u)` instead of `row_indices(i)`).
///
/// @tparam T Edge weight type (usually Real)
template <typename T>
struct CSRGraph {
    /// @brief The underlying adjacency matrix.
    CSRMatrix<T> adj;

    // -------------------------------------------------------------------------
    // Constructors (Implicit conversion from CSRMatrix)
    // -------------------------------------------------------------------------
    
    constexpr CSRGraph() noexcept = default;
    
    /// @brief Construct from an existing CSR matrix.
    /// @param matrix The adjacency matrix view.
    constexpr CSRGraph(CSRMatrix<T> matrix) noexcept : adj(matrix) {}

    // -------------------------------------------------------------------------
    // Graph Properties
    // -------------------------------------------------------------------------

    /// @brief Get the number of nodes (vertices).
    SCL_FORCE_INLINE Index num_nodes() const { return adj.rows; }

    /// @brief Get the number of edges.
    SCL_FORCE_INLINE Index num_edges() const { return adj.nnz; }

    // -------------------------------------------------------------------------
    // Traversal API
    // -------------------------------------------------------------------------

    /// @brief Get the degree (out-degree) of a node.
    /// @param u Node index.
    SCL_FORCE_INLINE Index degree(Index u) const {
#if !defined(NDEBUG)
        SCL_ASSERT(u >= 0 && u < adj.rows, "Graph: Node index out of bounds");
#endif
        // Use unified interface to get row length
        return adj.row_length(u);
    }

    /// @brief Get the neighbors of a node.
    /// @param u Node index.
    /// @return Span of indices representing target nodes (v).
    SCL_FORCE_INLINE Span<Index> neighbors(Index u) const {
        return adj.row_indices(u);
    }

    /// @brief Get the weights of outgoing edges for a node.
    /// @param u Node index.
    /// @return Span of weights corresponding to neighbors.
    SCL_FORCE_INLINE Span<T> weights(Index u) const {
        return adj.row_values(u);
    }
    
    // -------------------------------------------------------------------------
    // Utils
    // -------------------------------------------------------------------------
    
    /// @brief Check if the graph has no nodes.
    constexpr bool empty() const noexcept { return adj.empty(); }
};

// =============================================================================
// SECTION 2: Common Aliases
// =============================================================================

/// @brief Standard Weighted Graph (Real weights, e.g., kNN distances/affinities).
/// Directly maps to `adata.obsp['distances']` (float32).
using WeightedGraph = CSRGraph<Real>;

/// @brief Compatibility Unweighted Graph (Real weights fixed to 1.0).
///
/// Why Real?
/// Although logically boolean, most standard libraries (Scanpy, Seurat, Scipy)
/// store connectivity matrices as float32/64 to support direct matrix multiplication.
/// To maintain Zero-Copy layout compatibility with adata.obsp['connectivities'],
/// we must use the same type as the storage (Real).
using UnweightedGraph = CSRGraph<Real>;

/// @brief True Boolean Graph (Masks).
///
/// Use this ONLY if the upstream Python matrix is explicitly `dtype=bool` or `uint8`.
/// Maps to `scipy.sparse.csr_matrix` with `dtype=bool`.
using BooleanGraph = CSRGraph<bool>;

} // namespace scl
