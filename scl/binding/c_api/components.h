#pragma once

// =============================================================================
// FILE: scl/binding/c_api/components.h
// BRIEF: C API for connected components and graph connectivity analysis
// =============================================================================

#include "scl/binding/c_api/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Connected Components
// =============================================================================

scl_error_t scl_components_connected_components(
    scl_sparse_matrix_t adjacency,
    scl_index_t* component_labels,
    scl_index_t* n_components
);

// =============================================================================
// Graph Connectivity
// =============================================================================

int scl_components_is_connected(scl_sparse_matrix_t adjacency);

// =============================================================================
// Largest Connected Component
// =============================================================================

scl_error_t scl_components_largest_component(
    scl_sparse_matrix_t adjacency,
    scl_index_t* node_mask,
    scl_index_t* component_size
);

// =============================================================================
// Component Sizes
// =============================================================================

scl_error_t scl_components_component_sizes(
    scl_sparse_matrix_t adjacency,
    scl_index_t* sizes,
    scl_index_t* n_components,
    scl_index_t max_components
);

// =============================================================================
// Breadth-First Search
// =============================================================================

scl_error_t scl_components_bfs(
    scl_sparse_matrix_t adjacency,
    scl_index_t source,
    scl_index_t* distances,
    scl_index_t* predecessors
);

// =============================================================================
// Multi-Source BFS
// =============================================================================

scl_error_t scl_components_multi_source_bfs(
    scl_sparse_matrix_t adjacency,
    const scl_index_t* sources,
    scl_size_t n_sources,
    scl_index_t* distances
);

// =============================================================================
// Parallel BFS
// =============================================================================

scl_error_t scl_components_parallel_bfs(
    scl_sparse_matrix_t adjacency,
    scl_index_t source,
    scl_index_t* distances
);

// =============================================================================
// Graph Statistics
// =============================================================================

scl_index_t scl_components_graph_diameter(scl_sparse_matrix_t adjacency);

scl_real_t scl_components_average_path_length(
    scl_sparse_matrix_t adjacency,
    scl_size_t max_samples
);

// =============================================================================
// Clustering Coefficient
// =============================================================================

scl_error_t scl_components_clustering_coefficient(
    scl_sparse_matrix_t adjacency,
    scl_real_t* coefficients
);

scl_real_t scl_components_global_clustering_coefficient(scl_sparse_matrix_t adjacency);

// =============================================================================
// Triangle Counting
// =============================================================================

scl_size_t scl_components_count_triangles(scl_sparse_matrix_t adjacency);

// =============================================================================
// Degree Statistics
// =============================================================================

scl_error_t scl_components_degree_sequence(
    scl_sparse_matrix_t adjacency,
    scl_index_t* degrees
);

scl_error_t scl_components_degree_statistics(
    scl_sparse_matrix_t adjacency,
    scl_real_t* mean_degree,
    scl_real_t* max_degree,
    scl_real_t* min_degree,
    scl_real_t* std_degree
);

#ifdef __cplusplus
}
#endif
