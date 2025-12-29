// =============================================================================
// SCL Core - Leiden Clustering Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/leiden.h
//
// Functions tested:
//   ✓ scl_leiden_cluster - Basic Leiden clustering
//   ✓ scl_leiden_cluster_multilevel - Multilevel Leiden clustering
//   ✓ scl_leiden_compute_modularity - Modularity computation
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

using namespace scl::test;
using precision::Tolerance;

SCL_TEST_BEGIN

// =============================================================================
// Helper: Create adjacency matrix (sparse, symmetric)
// =============================================================================

static Sparse create_adjacency_matrix(scl_size_t n_nodes, 
                                     const std::vector<std::pair<scl_index_t, scl_index_t>>& edges) {
    // Create symmetric adjacency matrix
    std::vector<scl_index_t> indptr(n_nodes + 1);
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    
    // Build adjacency list
    std::vector<std::vector<scl_index_t>> adj_list(n_nodes);
    for (const auto& edge : edges) {
        adj_list[edge.first].push_back(edge.second);
        if (edge.first != edge.second) {
            adj_list[edge.second].push_back(edge.first);
        }
    }
    
    // Convert to CSR
    scl_index_t nnz = 0;
    for (scl_index_t i = 0; i < static_cast<scl_index_t>(n_nodes); ++i) {
        indptr[i] = nnz;
        std::sort(adj_list[i].begin(), adj_list[i].end());
        for (scl_index_t j : adj_list[i]) {
            indices.push_back(j);
            data.push_back(1.0);  // Unweighted
            ++nnz;
        }
    }
    indptr[n_nodes] = nnz;
    
    return make_sparse_csr(n_nodes, n_nodes, nnz, indptr.data(), indices.data(), data.data());
}

// =============================================================================
// Helper: Create simple graph (path graph)
// =============================================================================

static Sparse create_path_graph(scl_size_t n_nodes) {
    std::vector<std::pair<scl_index_t, scl_index_t>> edges;
    for (scl_size_t i = 0; i < n_nodes - 1; ++i) {
        edges.push_back({static_cast<scl_index_t>(i), static_cast<scl_index_t>(i + 1)});
    }
    return create_adjacency_matrix(n_nodes, edges);
}

// =============================================================================
// Helper: Create complete graph
// =============================================================================

static Sparse create_complete_graph(scl_size_t n_nodes) {
    std::vector<std::pair<scl_index_t, scl_index_t>> edges;
    for (scl_size_t i = 0; i < n_nodes; ++i) {
        for (scl_size_t j = i + 1; j < n_nodes; ++j) {
            edges.push_back({static_cast<scl_index_t>(i), static_cast<scl_index_t>(j)});
        }
    }
    return create_adjacency_matrix(n_nodes, edges);
}

// =============================================================================
// Helper: Create two disconnected components
// =============================================================================

static Sparse create_two_components(scl_size_t n1, scl_size_t n2) {
    std::vector<std::pair<scl_index_t, scl_index_t>> edges;
    
    // First component (complete)
    for (scl_size_t i = 0; i < n1; ++i) {
        for (scl_size_t j = i + 1; j < n1; ++j) {
            edges.push_back({static_cast<scl_index_t>(i), static_cast<scl_index_t>(j)});
        }
    }
    
    // Second component (complete, offset by n1)
    for (scl_size_t i = 0; i < n2; ++i) {
        for (scl_size_t j = i + 1; j < n2; ++j) {
            edges.push_back({static_cast<scl_index_t>(n1 + i), static_cast<scl_index_t>(n1 + j)});
        }
    }
    
    return create_adjacency_matrix(n1 + n2, edges);
}

// =============================================================================
// Leiden Clustering Tests
// =============================================================================

SCL_TEST_SUITE(leiden_cluster)

SCL_TEST_CASE(leiden_basic) {
    scl_size_t n_nodes = 10;
    Sparse adjacency = create_path_graph(n_nodes);
    
    std::vector<scl_index_t> partition(n_nodes);
    scl_real_t resolution = 1.0;
    scl_index_t max_iter = 10;
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_leiden_cluster(
        adjacency, partition.data(), n_nodes, resolution, max_iter, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GT(n_communities, 0);
    SCL_ASSERT_LE(n_communities, static_cast<scl_index_t>(n_nodes));
    
    // All nodes should be assigned to a community
    for (size_t i = 0; i < n_nodes; ++i) {
        SCL_ASSERT_GE(partition[i], 0);
        SCL_ASSERT_LT(partition[i], n_communities);
    }
}

SCL_TEST_CASE(leiden_complete_graph) {
    scl_size_t n_nodes = 5;
    Sparse adjacency = create_complete_graph(n_nodes);
    
    std::vector<scl_index_t> partition(n_nodes);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_leiden_cluster(
        adjacency, partition.data(), n_nodes, 1.0, 10, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Complete graph should form one community at resolution=1.0
    // (or multiple depending on resolution)
    SCL_ASSERT_GT(n_communities, 0);
}

SCL_TEST_CASE(leiden_two_components) {
    scl_size_t n1 = 5, n2 = 5;
    Sparse adjacency = create_two_components(n1, n2);
    
    std::vector<scl_index_t> partition(n1 + n2);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_leiden_cluster(
        adjacency, partition.data(), n1 + n2, 1.0, 10, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should detect at least 2 communities
    SCL_ASSERT_GE(n_communities, 2);
}

SCL_TEST_CASE(leiden_different_resolutions) {
    scl_size_t n_nodes = 10;
    Sparse adjacency = create_path_graph(n_nodes);
    
    std::vector<scl_real_t> resolutions = {0.1, 0.5, 1.0, 2.0, 5.0};
    
    for (scl_real_t resolution : resolutions) {
        std::vector<scl_index_t> partition(n_nodes);
        scl_index_t n_communities = 0;
        
        scl_error_t err = scl_leiden_cluster(
            adjacency, partition.data(), n_nodes, resolution, 10, &n_communities
        );
        
        SCL_ASSERT_EQ(err, SCL_OK);
        SCL_ASSERT_GT(n_communities, 0);
    }
}

SCL_TEST_CASE(leiden_single_node) {
    scl_size_t n_nodes = 1;
    // Single node with self-loop
    std::vector<std::pair<scl_index_t, scl_index_t>> edges = {{0, 0}};
    Sparse adjacency = create_adjacency_matrix(n_nodes, edges);
    
    std::vector<scl_index_t> partition(n_nodes);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_leiden_cluster(
        adjacency, partition.data(), n_nodes, 1.0, 10, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(n_communities, 1);
    SCL_ASSERT_EQ(partition[0], 0);
}

SCL_TEST_CASE(leiden_null_handle) {
    std::vector<scl_index_t> partition(10);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_leiden_cluster(
        nullptr, partition.data(), 10, 1.0, 10, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(leiden_null_partition) {
    scl_size_t n_nodes = 10;
    Sparse adjacency = create_path_graph(n_nodes);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_leiden_cluster(
        adjacency, nullptr, n_nodes, 1.0, 10, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(leiden_zero_nodes) {
    std::vector<scl_index_t> partition(1);
    scl_index_t n_communities = 0;
    
    // Empty adjacency matrix
    std::vector<scl_index_t> indptr = {0};
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    Sparse adjacency = make_sparse_csr(0, 0, 0, indptr.data(), indices.data(), data.data());
    
    scl_error_t err = scl_leiden_cluster(
        adjacency, partition.data(), 0, 1.0, 10, &n_communities
    );
    
    // Should handle gracefully
    if (err == SCL_OK) {
        SCL_ASSERT_EQ(n_communities, 0);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Multilevel Leiden Clustering Tests
// =============================================================================

SCL_TEST_SUITE(leiden_cluster_multilevel)

SCL_TEST_CASE(multilevel_basic) {
    scl_size_t n_nodes = 10;
    Sparse adjacency = create_path_graph(n_nodes);
    
    std::vector<scl_index_t> partition(n_nodes);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_leiden_cluster_multilevel(
        adjacency, partition.data(), n_nodes, 1.0, 10, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GT(n_communities, 0);
}

SCL_TEST_CASE(multilevel_two_components) {
    scl_size_t n1 = 5, n2 = 5;
    Sparse adjacency = create_two_components(n1, n2);
    
    std::vector<scl_index_t> partition(n1 + n2);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_leiden_cluster_multilevel(
        adjacency, partition.data(), n1 + n2, 1.0, 10, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(n_communities, 2);
}

SCL_TEST_CASE(multilevel_null_handle) {
    std::vector<scl_index_t> partition(10);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_leiden_cluster_multilevel(
        nullptr, partition.data(), 10, 1.0, 10, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(multilevel_null_partition) {
    scl_size_t n_nodes = 10;
    Sparse adjacency = create_path_graph(n_nodes);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_leiden_cluster_multilevel(
        adjacency, nullptr, n_nodes, 1.0, 10, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Modularity Computation Tests
// =============================================================================

SCL_TEST_SUITE(compute_modularity)

SCL_TEST_CASE(modularity_basic) {
    scl_size_t n_nodes = 10;
    Sparse adjacency = create_path_graph(n_nodes);
    
    // Create a simple partition (all nodes in one community)
    std::vector<scl_index_t> partition(n_nodes, 0);
    scl_real_t modularity = 0.0;
    
    scl_error_t err = scl_leiden_compute_modularity(
        adjacency, partition.data(), n_nodes, 1.0, &modularity
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Modularity should be in [-1, 1]
    SCL_ASSERT_GE(modularity, -1.0);
    SCL_ASSERT_LE(modularity, 1.0);
}

SCL_TEST_CASE(modularity_two_communities) {
    scl_size_t n1 = 5, n2 = 5;
    Sparse adjacency = create_two_components(n1, n2);
    
    // Partition matching the two components
    std::vector<scl_index_t> partition(n1 + n2);
    for (size_t i = 0; i < n1; ++i) partition[i] = 0;
    for (size_t i = 0; i < n2; ++i) partition[n1 + i] = 1;
    
    scl_real_t modularity = 0.0;
    scl_error_t err = scl_leiden_compute_modularity(
        adjacency, partition.data(), n1 + n2, 1.0, &modularity
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Perfect partition should have high modularity
    SCL_ASSERT_GT(modularity, 0.0);
}

SCL_TEST_CASE(modularity_single_community) {
    scl_size_t n_nodes = 5;
    Sparse adjacency = create_complete_graph(n_nodes);
    
    // All nodes in one community
    std::vector<scl_index_t> partition(n_nodes, 0);
    scl_real_t modularity = 0.0;
    
    scl_error_t err = scl_leiden_compute_modularity(
        adjacency, partition.data(), n_nodes, 1.0, &modularity
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Single community should have modularity around 0
    SCL_ASSERT_GE(modularity, -0.1);
    SCL_ASSERT_LE(modularity, 0.1);
}

SCL_TEST_CASE(modularity_different_resolutions) {
    scl_size_t n_nodes = 10;
    Sparse adjacency = create_path_graph(n_nodes);
    std::vector<scl_index_t> partition(n_nodes, 0);
    
    std::vector<scl_real_t> resolutions = {0.1, 0.5, 1.0, 2.0};
    
    for (scl_real_t resolution : resolutions) {
        scl_real_t modularity = 0.0;
        scl_error_t err = scl_leiden_compute_modularity(
            adjacency, partition.data(), n_nodes, resolution, &modularity
        );
        
        SCL_ASSERT_EQ(err, SCL_OK);
        SCL_ASSERT_GE(modularity, -1.0);
        SCL_ASSERT_LE(modularity, 1.0);
    }
}

SCL_TEST_CASE(modularity_null_handle) {
    std::vector<scl_index_t> partition(10, 0);
    scl_real_t modularity = 0.0;
    
    scl_error_t err = scl_leiden_compute_modularity(
        nullptr, partition.data(), 10, 1.0, &modularity
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(modularity_null_partition) {
    scl_size_t n_nodes = 10;
    Sparse adjacency = create_path_graph(n_nodes);
    scl_real_t modularity = 0.0;
    
    scl_error_t err = scl_leiden_compute_modularity(
        adjacency, nullptr, n_nodes, 1.0, &modularity
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(modularity_null_output) {
    scl_size_t n_nodes = 10;
    Sparse adjacency = create_path_graph(n_nodes);
    std::vector<scl_index_t> partition(n_nodes, 0);
    
    scl_error_t err = scl_leiden_compute_modularity(
        adjacency, partition.data(), n_nodes, 1.0, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

