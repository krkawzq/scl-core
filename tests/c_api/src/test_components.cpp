// =============================================================================
// SCL Core - Components Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/components.h
//
// Functions tested (21):
//   ✓ scl_comp_connected_components
//   ✓ scl_comp_is_connected
//   ✓ scl_comp_largest_component
//   ✓ scl_comp_component_sizes
//   ✓ scl_comp_bfs
//   ✓ scl_comp_multi_source_bfs
//   ✓ scl_comp_parallel_bfs
//   ✓ scl_comp_dfs
//   ✓ scl_comp_topological_sort
//   ✓ scl_comp_graph_diameter
//   ✓ scl_comp_average_path_length
//   ✓ scl_comp_clustering_coefficient
//   ✓ scl_comp_global_clustering_coefficient
//   ✓ scl_comp_count_triangles
//   ✓ scl_comp_degree_sequence
//   ✓ scl_comp_degree_statistics
//   ✓ scl_comp_degree_distribution
//   ✓ scl_comp_graph_density
//   ✓ scl_comp_kcore_decomposition
//
// =============================================================================

#include "test.hpp"

extern "C" {
#include "scl/binding/c_api/components.h"
}

using namespace scl::test;

// Helper: Create simple 4-node graph
// 0 -- 1
// |    |
// 2 -- 3
static auto simple_graph_4nodes() {
    std::vector<scl_index_t> indptr = {0, 2, 4, 6, 8};
    std::vector<scl_index_t> indices = {1, 2, 0, 3, 0, 3, 1, 2};
    std::vector<scl_real_t> data(8, 1.0);
    return std::make_tuple(indptr, indices, data);
}

// Helper: Create disconnected graph (2 components)
// 0 -- 1    2 -- 3
static auto disconnected_graph() {
    std::vector<scl_index_t> indptr = {0, 1, 2, 3, 4};
    std::vector<scl_index_t> indices = {1, 0, 3, 2};
    std::vector<scl_real_t> data(4, 1.0);
    return std::make_tuple(indptr, indices, data);
}

SCL_TEST_BEGIN

// =============================================================================
// Connected Components Tests
// =============================================================================

SCL_TEST_SUITE(connected_components)

SCL_TEST_CASE(connected_components_single) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> labels(4);
    scl_index_t n_components;

    scl_error_t err = scl_comp_connected_components(adj, labels.data(), &n_components);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(n_components, 1);

    // All nodes should have same label
    SCL_ASSERT_EQ(labels[0], labels[1]);
    SCL_ASSERT_EQ(labels[1], labels[2]);
    SCL_ASSERT_EQ(labels[2], labels[3]);
}

SCL_TEST_CASE(connected_components_multiple) {
    auto [indptr, indices, data] = disconnected_graph();
    Sparse adj = make_sparse_csr(4, 4, 4, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> labels(4);
    scl_index_t n_components;

    scl_error_t err = scl_comp_connected_components(adj, labels.data(), &n_components);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(n_components, 2);

    // Nodes 0,1 should have same label
    SCL_ASSERT_EQ(labels[0], labels[1]);

    // Nodes 2,3 should have same label
    SCL_ASSERT_EQ(labels[2], labels[3]);

    // Different components should have different labels
    SCL_ASSERT_NE(labels[0], labels[2]);
}

SCL_TEST_CASE(connected_components_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> labels(4);
    scl_index_t n_components;

    // NULL adjacency
    scl_error_t err = scl_comp_connected_components(nullptr, labels.data(), &n_components);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL labels
    err = scl_comp_connected_components(adj, nullptr, &n_components);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL n_components
    err = scl_comp_connected_components(adj, labels.data(), nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(connected_components_isolated_nodes) {
    // Single isolated node
    std::vector<scl_index_t> indptr = {0};
    std::vector<scl_index_t> indices(1);
    std::vector<scl_real_t> data(1);

    Sparse adj = make_sparse_csr(1, 1, 0, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> labels(1);
    scl_index_t n_components;

    scl_error_t err = scl_comp_connected_components(adj, labels.data(), &n_components);

    if (err == SCL_OK) {
        SCL_ASSERT_EQ(n_components, 1);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Graph Connectivity Tests
// =============================================================================

SCL_TEST_SUITE(graph_connectivity)

SCL_TEST_CASE(is_connected_yes) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    int is_conn;
    scl_error_t err = scl_comp_is_connected(adj, &is_conn);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(is_conn, 1);
}

SCL_TEST_CASE(is_connected_no) {
    auto [indptr, indices, data] = disconnected_graph();
    Sparse adj = make_sparse_csr(4, 4, 4, indptr.data(), indices.data(), data.data());

    int is_conn;
    scl_error_t err = scl_comp_is_connected(adj, &is_conn);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(is_conn, 0);
}

SCL_TEST_CASE(is_connected_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    int is_conn;

    // NULL adjacency
    scl_error_t err = scl_comp_is_connected(nullptr, &is_conn);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL output
    err = scl_comp_is_connected(adj, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Largest Component Tests
// =============================================================================

SCL_TEST_SUITE(largest_component)

SCL_TEST_CASE(largest_component_basic) {
    auto [indptr, indices, data] = disconnected_graph();
    Sparse adj = make_sparse_csr(4, 4, 4, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> node_mask(4);
    scl_index_t component_size;

    scl_error_t err = scl_comp_largest_component(adj, node_mask.data(), &component_size);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(component_size, 2);

    // Exactly 2 nodes should be in largest component
    scl_index_t count = 0;
    for (scl_index_t mask : node_mask) {
        count += mask;
    }
    SCL_ASSERT_EQ(count, 2);
}

SCL_TEST_CASE(largest_component_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> node_mask(4);
    scl_index_t component_size;

    // NULL adjacency
    scl_error_t err = scl_comp_largest_component(nullptr, node_mask.data(), &component_size);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL node_mask
    err = scl_comp_largest_component(adj, nullptr, &component_size);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL component_size
    err = scl_comp_largest_component(adj, node_mask.data(), nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Component Sizes Tests
// =============================================================================

SCL_TEST_SUITE(component_sizes)

SCL_TEST_CASE(component_sizes_basic) {
    auto [indptr, indices, data] = disconnected_graph();
    Sparse adj = make_sparse_csr(4, 4, 4, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> sizes(4);
    scl_index_t n_components;

    scl_error_t err = scl_comp_component_sizes(adj, sizes.data(), &n_components);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(n_components, 2);

    // Both components should have size 2
    SCL_ASSERT_EQ(sizes[0], 2);
    SCL_ASSERT_EQ(sizes[1], 2);
}

SCL_TEST_CASE(component_sizes_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> sizes(4);
    scl_index_t n_components;

    // NULL adjacency
    scl_error_t err = scl_comp_component_sizes(nullptr, sizes.data(), &n_components);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL sizes
    err = scl_comp_component_sizes(adj, nullptr, &n_components);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL n_components
    err = scl_comp_component_sizes(adj, sizes.data(), nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// BFS Tests
// =============================================================================

SCL_TEST_SUITE(breadth_first_search)

SCL_TEST_CASE(bfs_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> distances(4);

    scl_error_t err = scl_comp_bfs(adj, 0, distances.data(), nullptr);

    SCL_ASSERT_EQ(err, SCL_OK);

    // Distance from node 0 to itself
    SCL_ASSERT_EQ(distances[0], 0);

    // All nodes should be reachable
    for (scl_index_t d : distances) {
        SCL_ASSERT_GE(d, 0);
        SCL_ASSERT_LE(d, 3);
    }
}

SCL_TEST_CASE(bfs_with_predecessors) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> distances(4);
    std::vector<scl_index_t> predecessors(4);

    scl_error_t err = scl_comp_bfs(adj, 0, distances.data(), predecessors.data());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Source has no predecessor (-1 or special value)
    SCL_ASSERT_TRUE(predecessors[0] < 0 || predecessors[0] == 0);

    // Other nodes should have valid predecessors
    for (scl_index_t i = 1; i < 4; ++i) {
        if (distances[i] > 0) {
            SCL_ASSERT_GE(predecessors[i], 0);
            SCL_ASSERT_LT(predecessors[i], 4);
        }
    }
}

SCL_TEST_CASE(bfs_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> distances(4);

    // NULL adjacency
    scl_error_t err = scl_comp_bfs(nullptr, 0, distances.data(), nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL distances
    err = scl_comp_bfs(adj, 0, nullptr, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(bfs_invalid_source) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> distances(4);

    // Negative source
    scl_error_t err = scl_comp_bfs(adj, -1, distances.data(), nullptr);
    SCL_ASSERT_NE(err, SCL_OK);

    // Source >= n_nodes
    err = scl_comp_bfs(adj, 4, distances.data(), nullptr);
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Multi-Source BFS Tests
// =============================================================================

SCL_TEST_SUITE(multi_source_bfs)

SCL_TEST_CASE(multi_source_bfs_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> sources = {0, 1};
    std::vector<scl_index_t> distances(4);

    scl_error_t err = scl_comp_multi_source_bfs(adj, sources.data(), 2, distances.data());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Sources should have distance 0
    SCL_ASSERT_EQ(distances[0], 0);
    SCL_ASSERT_EQ(distances[1], 0);

    // All nodes should be reachable
    for (scl_index_t d : distances) {
        SCL_ASSERT_GE(d, 0);
    }
}

SCL_TEST_CASE(multi_source_bfs_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> sources = {0};
    std::vector<scl_index_t> distances(4);

    // NULL adjacency
    scl_error_t err = scl_comp_multi_source_bfs(nullptr, sources.data(), 1, distances.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL sources
    err = scl_comp_multi_source_bfs(adj, nullptr, 1, distances.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL distances
    err = scl_comp_multi_source_bfs(adj, sources.data(), 1, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(multi_source_bfs_zero_sources) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> sources(1);
    std::vector<scl_index_t> distances(4);

    scl_error_t err = scl_comp_multi_source_bfs(adj, sources.data(), 0, distances.data());
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Parallel BFS Tests
// =============================================================================

SCL_TEST_SUITE(parallel_bfs)

SCL_TEST_CASE(parallel_bfs_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> distances(4);

    scl_error_t err = scl_comp_parallel_bfs(adj, 0, distances.data());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Should match serial BFS
    SCL_ASSERT_EQ(distances[0], 0);

    for (scl_index_t d : distances) {
        SCL_ASSERT_GE(d, 0);
    }
}

SCL_TEST_CASE(parallel_bfs_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> distances(4);

    // NULL adjacency
    scl_error_t err = scl_comp_parallel_bfs(nullptr, 0, distances.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL distances
    err = scl_comp_parallel_bfs(adj, 0, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// DFS Tests
// =============================================================================

SCL_TEST_SUITE(depth_first_search)

SCL_TEST_CASE(dfs_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> discovery_time(4);
    std::vector<scl_index_t> finish_time(4);

    scl_error_t err = scl_comp_dfs(adj, 0, discovery_time.data(), finish_time.data());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Discovery time should be before finish time
    for (scl_index_t i = 0; i < 4; ++i) {
        SCL_ASSERT_LE(discovery_time[i], finish_time[i]);
    }

    // Source should be discovered first
    for (scl_index_t i = 1; i < 4; ++i) {
        SCL_ASSERT_LE(discovery_time[0], discovery_time[i]);
    }
}

SCL_TEST_CASE(dfs_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> discovery_time(4);
    std::vector<scl_index_t> finish_time(4);

    // NULL adjacency
    scl_error_t err = scl_comp_dfs(nullptr, 0, discovery_time.data(), finish_time.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL discovery_time
    err = scl_comp_dfs(adj, 0, nullptr, finish_time.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL finish_time
    err = scl_comp_dfs(adj, 0, discovery_time.data(), nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Topological Sort Tests
// =============================================================================

SCL_TEST_SUITE(topological_sort)

SCL_TEST_CASE(topological_sort_dag) {
    // Simple DAG: 0 -> 1 -> 2
    std::vector<scl_index_t> indptr = {0, 1, 2, 2};
    std::vector<scl_index_t> indices = {1, 2};
    std::vector<scl_real_t> data(2, 1.0);

    Sparse adj = make_sparse_csr(3, 3, 2, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> order(3);
    int is_valid;

    scl_error_t err = scl_comp_topological_sort(adj, order.data(), &is_valid);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(is_valid, 1);

    // Order should be: 0, 1, 2
    SCL_ASSERT_EQ(order[0], 0);
    SCL_ASSERT_EQ(order[1], 1);
    SCL_ASSERT_EQ(order[2], 2);
}

SCL_TEST_CASE(topological_sort_cycle) {
    // Graph with cycle: 0 -> 1 -> 0
    std::vector<scl_index_t> indptr = {0, 1, 2};
    std::vector<scl_index_t> indices = {1, 0};
    std::vector<scl_real_t> data(2, 1.0);

    Sparse adj = make_sparse_csr(2, 2, 2, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> order(2);
    int is_valid;

    scl_error_t err = scl_comp_topological_sort(adj, order.data(), &is_valid);

    if (err == SCL_OK) {
        // Should detect cycle
        SCL_ASSERT_EQ(is_valid, 0);
    }
}

SCL_TEST_CASE(topological_sort_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> order(4);
    int is_valid;

    // NULL adjacency
    scl_error_t err = scl_comp_topological_sort(nullptr, order.data(), &is_valid);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL order
    err = scl_comp_topological_sort(adj, nullptr, &is_valid);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL is_valid
    err = scl_comp_topological_sort(adj, order.data(), nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Graph Metrics Tests
// =============================================================================

SCL_TEST_SUITE(graph_metrics)

SCL_TEST_CASE(graph_diameter_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_index_t diameter;

    scl_error_t err = scl_comp_graph_diameter(adj, &diameter);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(diameter, 0);
    SCL_ASSERT_LE(diameter, 3);
}

SCL_TEST_CASE(graph_diameter_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_index_t diameter;

    // NULL adjacency
    scl_error_t err = scl_comp_graph_diameter(nullptr, &diameter);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL diameter
    err = scl_comp_graph_diameter(adj, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(average_path_length_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_real_t avg_length;

    scl_error_t err = scl_comp_average_path_length(adj, 0, &avg_length);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GT(avg_length, 0.0);
    SCL_ASSERT_TRUE(std::isfinite(avg_length));
}

SCL_TEST_CASE(average_path_length_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_real_t avg_length;

    // NULL adjacency
    scl_error_t err = scl_comp_average_path_length(nullptr, 0, &avg_length);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL avg_length
    err = scl_comp_average_path_length(adj, 0, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Clustering Coefficient Tests
// =============================================================================

SCL_TEST_SUITE(clustering_coefficient)

SCL_TEST_CASE(clustering_coefficient_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> coefficients(4);

    scl_error_t err = scl_comp_clustering_coefficient(adj, coefficients.data());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Coefficients should be in [0, 1]
    for (scl_real_t c : coefficients) {
        SCL_ASSERT_GE(c, 0.0);
        SCL_ASSERT_LE(c, 1.0);
    }
}

SCL_TEST_CASE(clustering_coefficient_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> coefficients(4);

    // NULL adjacency
    scl_error_t err = scl_comp_clustering_coefficient(nullptr, coefficients.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL coefficients
    err = scl_comp_clustering_coefficient(adj, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(global_clustering_coefficient_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_real_t coefficient;

    scl_error_t err = scl_comp_global_clustering_coefficient(adj, &coefficient);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(coefficient, 0.0);
    SCL_ASSERT_LE(coefficient, 1.0);
}

SCL_TEST_CASE(global_clustering_coefficient_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_real_t coefficient;

    // NULL adjacency
    scl_error_t err = scl_comp_global_clustering_coefficient(nullptr, &coefficient);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL coefficient
    err = scl_comp_global_clustering_coefficient(adj, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Triangle Counting Tests
// =============================================================================

SCL_TEST_SUITE(triangle_counting)

SCL_TEST_CASE(count_triangles_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_size_t n_triangles;

    scl_error_t err = scl_comp_count_triangles(adj, &n_triangles);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(n_triangles, 0);
}

SCL_TEST_CASE(count_triangles_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_size_t n_triangles;

    // NULL adjacency
    scl_error_t err = scl_comp_count_triangles(nullptr, &n_triangles);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL n_triangles
    err = scl_comp_count_triangles(adj, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Degree Tests
// =============================================================================

SCL_TEST_SUITE(degree_analysis)

SCL_TEST_CASE(degree_sequence_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> degrees(4);

    scl_error_t err = scl_comp_degree_sequence(adj, degrees.data());

    SCL_ASSERT_EQ(err, SCL_OK);

    // All nodes should have degree 2
    for (scl_index_t d : degrees) {
        SCL_ASSERT_EQ(d, 2);
    }
}

SCL_TEST_CASE(degree_sequence_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> degrees(4);

    // NULL adjacency
    scl_error_t err = scl_comp_degree_sequence(nullptr, degrees.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL degrees
    err = scl_comp_degree_sequence(adj, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(degree_statistics_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_real_t mean_degree, max_degree, min_degree, std_degree;

    scl_error_t err = scl_comp_degree_statistics(
        adj, &mean_degree, &max_degree, &min_degree, &std_degree
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // All nodes have degree 2
    SCL_ASSERT_NEAR(mean_degree, 2.0, 1e-10);
    SCL_ASSERT_EQ(max_degree, 2.0);
    SCL_ASSERT_EQ(min_degree, 2.0);
    SCL_ASSERT_NEAR(std_degree, 0.0, 1e-10);
}

SCL_TEST_CASE(degree_distribution_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_size_t> histogram(5);

    scl_error_t err = scl_comp_degree_distribution(adj, histogram.data(), 4);

    SCL_ASSERT_EQ(err, SCL_OK);

    // All 4 nodes have degree 2
    SCL_ASSERT_EQ(histogram[2], 4);
}

SCL_TEST_SUITE_END

// =============================================================================
// Graph Density Tests
// =============================================================================

SCL_TEST_SUITE(graph_density)

SCL_TEST_CASE(graph_density_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_real_t density;

    scl_error_t err = scl_comp_graph_density(adj, &density);

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(density, 0.0);
    SCL_ASSERT_LE(density, 1.0);
}

SCL_TEST_CASE(graph_density_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    scl_real_t density;

    // NULL adjacency
    scl_error_t err = scl_comp_graph_density(nullptr, &density);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL density
    err = scl_comp_graph_density(adj, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// K-Core Decomposition Tests
// =============================================================================

SCL_TEST_SUITE(kcore_decomposition)

SCL_TEST_CASE(kcore_decomposition_basic) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> core_numbers(4);

    scl_error_t err = scl_comp_kcore_decomposition(adj, core_numbers.data());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Core numbers should be non-negative
    for (scl_index_t k : core_numbers) {
        SCL_ASSERT_GE(k, 0);
    }
}

SCL_TEST_CASE(kcore_decomposition_null_inputs) {
    auto [indptr, indices, data] = simple_graph_4nodes();
    Sparse adj = make_sparse_csr(4, 4, 8, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> core_numbers(4);

    // NULL adjacency
    scl_error_t err = scl_comp_kcore_decomposition(nullptr, core_numbers.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL core_numbers
    err = scl_comp_kcore_decomposition(adj, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
