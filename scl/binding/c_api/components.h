#pragma once

// =============================================================================
// FILE: scl/binding/c_api/components/components.h
// BRIEF: C API for connected components and graph connectivity analysis
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Connected Components
// =============================================================================

scl_error_t scl_comp_connected_components(
    scl_sparse_t adjacency,               // Graph adjacency matrix
    scl_index_t* component_labels,        // Output [n_nodes]
    scl_index_t* n_components             // Output: number of components
);

// =============================================================================
// Graph Connectivity
// =============================================================================

scl_error_t scl_comp_is_connected(
    scl_sparse_t adjacency,
    int* is_connected                     // Output: 1 if connected, 0 otherwise
);

scl_error_t scl_comp_largest_component(
    scl_sparse_t adjacency,
    scl_index_t* node_mask,               // Output [n_nodes]: 1 if in largest component
    scl_index_t* component_size            // Output: size of largest component
);

scl_error_t scl_comp_component_sizes(
    scl_sparse_t adjacency,
    scl_index_t* sizes,                   // Output [n_components]
    scl_index_t* n_components
);

// =============================================================================
// Breadth-First Search
// =============================================================================

scl_error_t scl_comp_bfs(
    scl_sparse_t adjacency,
    scl_index_t source,
    scl_index_t* distances,               // Output [n_nodes]
    scl_index_t* predecessors             // Optional output [n_nodes], can be NULL
);

scl_error_t scl_comp_multi_source_bfs(
    scl_sparse_t adjacency,
    const scl_index_t* sources,           // [n_sources]
    scl_size_t n_sources,
    scl_index_t* distances                 // Output [n_nodes]
);

scl_error_t scl_comp_parallel_bfs(
    scl_sparse_t adjacency,
    scl_index_t source,
    scl_index_t* distances                // Output [n_nodes]
);

// =============================================================================
// Depth-First Search
// =============================================================================

scl_error_t scl_comp_dfs(
    scl_sparse_t adjacency,
    scl_index_t source,
    scl_index_t* discovery_time,          // Output [n_nodes]
    scl_index_t* finish_time              // Output [n_nodes]
);

// =============================================================================
// Topological Sort
// =============================================================================

scl_error_t scl_comp_topological_sort(
    scl_sparse_t adjacency,
    scl_index_t* order,                    // Output [n_nodes]
    int* is_valid                         // Output: 1 if valid DAG, 0 otherwise
);

// =============================================================================
// Graph Metrics
// =============================================================================

scl_error_t scl_comp_graph_diameter(
    scl_sparse_t adjacency,
    scl_index_t* diameter                 // Output: graph diameter
);

scl_error_t scl_comp_average_path_length(
    scl_sparse_t adjacency,
    scl_size_t max_samples,                // 0 = all pairs
    scl_real_t* avg_length                // Output: average path length
);

// =============================================================================
// Clustering Coefficient
// =============================================================================

scl_error_t scl_comp_clustering_coefficient(
    scl_sparse_t adjacency,
    scl_real_t* coefficients              // Output [n_nodes]
);

scl_error_t scl_comp_global_clustering_coefficient(
    scl_sparse_t adjacency,
    scl_real_t* coefficient               // Output: global clustering coefficient
);

// =============================================================================
// Triangle Counting
// =============================================================================

scl_error_t scl_comp_count_triangles(
    scl_sparse_t adjacency,
    scl_size_t* n_triangles               // Output: number of triangles
);

// =============================================================================
// Degree Statistics
// =============================================================================

scl_error_t scl_comp_degree_sequence(
    scl_sparse_t adjacency,
    scl_index_t* degrees                  // Output [n_nodes]
);

scl_error_t scl_comp_degree_statistics(
    scl_sparse_t adjacency,
    scl_real_t* mean_degree,
    scl_real_t* max_degree,
    scl_real_t* min_degree,
    scl_real_t* std_degree
);

scl_error_t scl_comp_degree_distribution(
    scl_sparse_t adjacency,
    scl_size_t* histogram,                // Output [max_degree + 1]
    scl_index_t max_degree
);

// =============================================================================
// Graph Properties
// =============================================================================

scl_error_t scl_comp_graph_density(
    scl_sparse_t adjacency,
    scl_real_t* density                   // Output: graph density
);

// =============================================================================
// K-Core Decomposition
// =============================================================================

scl_error_t scl_comp_kcore_decomposition(
    scl_sparse_t adjacency,
    scl_index_t* core_numbers             // Output [n_nodes]
);

#ifdef __cplusplus
}
#endif
