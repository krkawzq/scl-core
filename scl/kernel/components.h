// =============================================================================
// FILE: scl/kernel/components.h
// BRIEF: API reference for high-performance connected components and graph connectivity analysis
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::components {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index INVALID_COMPONENT = -1;
    constexpr Index UNVISITED = -1;
    constexpr Size PARALLEL_NODES_THRESHOLD = 1000;
    constexpr Size PARALLEL_EDGES_THRESHOLD = 10000;
    constexpr Size DENSE_DEGREE_THRESHOLD = 64;
    constexpr Size GALLOP_RATIO_THRESHOLD = 32;
    constexpr Size LINEAR_INTERSECT_THRESHOLD = 16;
    constexpr Size PREFETCH_DISTANCE = 8;
    constexpr Size CACHE_LINE_INDICES = 16;
    constexpr Size BITVEC_WORD_BITS = 64;
    constexpr Size QUEUE_BLOCK_SIZE = 4096;
}

// =============================================================================
// Component Detection
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: connected_components
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Find connected components in undirected graph.
 *
 * PARAMETERS:
 *     adjacency        [in]  Adjacency matrix (CSR or CSC)
 *     component_labels [out] Component labels [n_nodes]
 *     n_components     [out] Number of components found
 *
 * PRECONDITIONS:
 *     - component_labels.len >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - component_labels[i] contains component ID for node i
 *     - n_components contains number of distinct components
 *
 * COMPLEXITY:
 *     Time:  O(nnz) for union-find
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with lock-free union-find
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void connected_components(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Index> component_labels,           // Output component labels [n_nodes]
    Index& n_components                      // Output number of components
);

/* -----------------------------------------------------------------------------
 * FUNCTION: largest_component
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Extract nodes in the largest connected component.
 *
 * PARAMETERS:
 *     adjacency    [in]  Adjacency matrix (CSR or CSC)
 *     node_mask    [out] Binary mask for largest component [n_nodes]
 *     component_size [out] Size of largest component
 *
 * PRECONDITIONS:
 *     - node_mask.len >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - node_mask[i] == 1 if node i is in largest component
 *     - component_size contains size of largest component
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses connected_components
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void largest_component(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Byte> node_mask,                   // Output mask [n_nodes]
    Index& component_size                     // Output component size
);

/* -----------------------------------------------------------------------------
 * FUNCTION: bfs
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Perform breadth-first search from source node.
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     source     [in]  Source node index
 *     distances  [out] Distance from source [n_nodes]
 *     visited    [out] Optional visited mask [n_nodes]
 *
 * PRECONDITIONS:
 *     - distances.len >= adjacency.primary_dim()
 *     - source is valid node index
 *
 * POSTCONDITIONS:
 *     - distances[i] contains shortest path length from source to i
 *     - distances[i] == -1 if node i is unreachable
 *
 * COMPLEXITY:
 *     Time:  O(nnz) for connected component
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential BFS
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void bfs(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Index source,                           // Source node
    Array<Index> distances,                  // Output distances [n_nodes]
    Array<Index> visited                     // Optional visited mask [n_nodes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: parallel_bfs
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Perform parallel BFS using direction-optimizing algorithm.
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     source     [in]  Source node index
 *     distances  [out] Distance from source [n_nodes]
 *
 * PRECONDITIONS:
 *     - distances.len >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - distances[i] contains shortest path length from source to i
 *
 * COMPLEXITY:
 *     Time:  O(nnz) for connected component
 *     Space: O(n_nodes) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized with bit-vector frontiers
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void parallel_bfs(
    const Sparse<T, IsCSR>& adjacency,     // Adjacency matrix
    Index source,                           // Source node
    Array<Index> distances                   // Output distances [n_nodes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: is_connected
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Check if graph is connected.
 *
 * PARAMETERS:
 *     adjacency [in]  Adjacency matrix (CSR or CSC)
 *
 * PRECONDITIONS:
 *     - Graph has at least one node
 *
 * POSTCONDITIONS:
 *     - Returns true if graph has single connected component
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses connected_components
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
bool is_connected(
    const Sparse<T, IsCSR>& adjacency      // Adjacency matrix
);

/* -----------------------------------------------------------------------------
 * FUNCTION: graph_diameter
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute graph diameter (longest shortest path).
 *
 * PARAMETERS:
 *     adjacency [in]  Adjacency matrix (CSR or CSC)
 *
 * PRECONDITIONS:
 *     - Graph is connected
 *
 * POSTCONDITIONS:
 *     - Returns diameter value
 *
 * COMPLEXITY:
 *     Time:  O(n_nodes * nnz)
 *     Space: O(n_nodes) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over source nodes
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index graph_diameter(
    const Sparse<T, IsCSR>& adjacency      // Adjacency matrix
);

/* -----------------------------------------------------------------------------
 * FUNCTION: triangle_count
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count triangles in graph.
 *
 * PARAMETERS:
 *     adjacency [in]  Adjacency matrix (CSR or CSC)
 *
 * PRECONDITIONS:
 *     - Graph is undirected
 *
 * POSTCONDITIONS:
 *     - Returns total number of triangles
 *
 * COMPLEXITY:
 *     Time:  O(nnz^1.5) for sparse graphs
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with atomic accumulation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Size triangle_count(
    const Sparse<T, IsCSR>& adjacency      // Adjacency matrix
);

} // namespace scl::kernel::components
