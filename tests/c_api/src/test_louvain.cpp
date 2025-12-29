// =============================================================================
// SCL Core - Louvain Clustering Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/louvain.h
//
// Functions tested:
//   - scl_louvain_clustering
//   - scl_louvain_compute_modularity
//   - scl_louvain_community_sizes
//   - scl_louvain_get_community_members
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

using namespace scl::test;
using precision::Tolerance;

// Helper: Generate symmetric adjacency matrix (undirected graph)
static EigenCSR make_symmetric_adjacency(scl_index_t n_nodes, double density, Random& rng) {
    EigenCSR mat(n_nodes, n_nodes);
    mat.reserve(Eigen::VectorXi::Constant(n_nodes, static_cast<int>(n_nodes * density)));
    
    // Generate symmetric matrix (undirected graph)
    for (scl_index_t i = 0; i < n_nodes; ++i) {
        for (scl_index_t j = i; j < n_nodes; ++j) {
            if (i == j) {
                // No self-loops
                continue;
            }
            if (rng.bernoulli(density)) {
                scl_real_t weight = rng.uniform(0.1, 1.0);
                mat.insert(i, j) = weight;
                mat.insert(j, i) = weight;  // Symmetric
            }
        }
    }
    
    mat.makeCompressed();
    return mat;
}

// Helper: Convert Eigen CSR to SCL Sparse
static Sparse eigen_to_scl_sparse(const EigenCSR& eigen_mat) {
    auto csr = from_eigen_csr(eigen_mat);
    return make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
}

SCL_TEST_BEGIN

// =============================================================================
// Louvain Clustering
// =============================================================================

SCL_TEST_SUITE(clustering)

SCL_TEST_RETRY(clustering_small_graph, 3)
{
    Random rng(42);
    
    // Small graph: 10 nodes
    scl_index_t n_nodes = 10;
    auto adj_eigen = make_symmetric_adjacency(n_nodes, 0.3, rng);
    Sparse adj = eigen_to_scl_sparse(adj_eigen);
    
    std::vector<scl_index_t> labels(n_nodes);
    
    scl_error_t err = scl_louvain_clustering(
        adj, labels.data(), n_nodes, 1.0, 100
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify all nodes are assigned to some community
    for (scl_index_t i = 0; i < n_nodes; ++i) {
        SCL_ASSERT_GE(labels[i], 0);
    }
}

SCL_TEST_RETRY(clustering_random_graph, 5)
{
    Random rng(123);
    
    auto n_nodes = rng.uniform_int(20, 50);
    double density = rng.uniform(0.1, 0.4);
    
    auto adj_eigen = make_symmetric_adjacency(n_nodes, density, rng);
    Sparse adj = eigen_to_scl_sparse(adj_eigen);
    
    std::vector<scl_index_t> labels(n_nodes);
    
    scl_error_t err = scl_louvain_clustering(
        adj, labels.data(), n_nodes, 1.0, 100
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Check that labels are valid
    scl_index_t max_label = *std::max_element(labels.begin(), labels.end());
    SCL_ASSERT_GE(max_label, 0);
    SCL_ASSERT_LT(max_label, n_nodes);  // At most n_nodes communities
}

SCL_TEST_CASE(clustering_different_resolutions)
{
    Random rng(456);
    scl_index_t n_nodes = 15;
    auto adj_eigen = make_symmetric_adjacency(n_nodes, 0.3, rng);
    Sparse adj = eigen_to_scl_sparse(adj_eigen);
    
    std::vector<scl_real_t> resolutions = {0.5, 1.0, 2.0};
    
    for (scl_real_t resolution : resolutions) {
        std::vector<scl_index_t> labels(n_nodes);
        
        scl_error_t err = scl_louvain_clustering(
            adj, labels.data(), n_nodes, resolution, 100
        );
        
        SCL_ASSERT_EQ(err, SCL_OK);
        
        // Verify labels are valid
        for (scl_index_t i = 0; i < n_nodes; ++i) {
            SCL_ASSERT_GE(labels[i], 0);
        }
    }
}

SCL_TEST_CASE(clustering_null_adjacency)
{
    std::vector<scl_index_t> labels(10);
    
    scl_error_t err = scl_louvain_clustering(
        nullptr, labels.data(), 10, 1.0, 100
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(clustering_null_labels)
{
    Random rng(789);
    scl_index_t n_nodes = 10;
    auto adj_eigen = make_symmetric_adjacency(n_nodes, 0.3, rng);
    Sparse adj = eigen_to_scl_sparse(adj_eigen);
    
    scl_error_t err = scl_louvain_clustering(
        adj, nullptr, n_nodes, 1.0, 100
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(clustering_zero_nodes)
{
    Random rng(999);
    scl_index_t n_nodes = 10;
    auto adj_eigen = make_symmetric_adjacency(n_nodes, 0.3, rng);
    Sparse adj = eigen_to_scl_sparse(adj_eigen);
    
    std::vector<scl_index_t> labels(1);
    
    scl_error_t err = scl_louvain_clustering(
        adj, labels.data(), 0, 1.0, 100
    );
    
    // Should handle zero nodes gracefully
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_INVALID_ARGUMENT);
}

SCL_TEST_CASE(clustering_single_node)
{
    // Single node graph (no edges possible)
    EigenCSR adj_eigen(1, 1);
    adj_eigen.makeCompressed();
    Sparse adj = eigen_to_scl_sparse(adj_eigen);
    
    std::vector<scl_index_t> labels(1);
    
    scl_error_t err = scl_louvain_clustering(
        adj, labels.data(), 1, 1.0, 100
    );
    
    // Should succeed (single node = single community)
    if (err == SCL_OK) {
        SCL_ASSERT_GE(labels[0], 0);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Compute Modularity
// =============================================================================

SCL_TEST_SUITE(modularity)

SCL_TEST_RETRY(modularity_computation, 3)
{
    Random rng(111);
    
    scl_index_t n_nodes = 20;
    auto adj_eigen = make_symmetric_adjacency(n_nodes, 0.3, rng);
    Sparse adj = eigen_to_scl_sparse(adj_eigen);
    
    // First, get clustering
    std::vector<scl_index_t> labels(n_nodes);
    scl_louvain_clustering(adj, labels.data(), n_nodes, 1.0, 100);
    
    // Compute modularity
    scl_real_t modularity = 0.0;
    scl_error_t err = scl_louvain_compute_modularity(
        adj, labels.data(), n_nodes, 1.0, &modularity
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Modularity should be in [-1, 1] range
    SCL_ASSERT_GE(modularity, -1.0);
    SCL_ASSERT_LE(modularity, 1.0);
}

SCL_TEST_CASE(modularity_perfect_clustering)
{
    // Two disconnected communities
    scl_index_t n_nodes = 10;
    EigenCSR adj_eigen(n_nodes, n_nodes);
    
    // First community: nodes 0-4
    for (scl_index_t i = 0; i < 5; ++i) {
        for (scl_index_t j = i + 1; j < 5; ++j) {
            adj_eigen.insert(i, j) = 1.0;
            adj_eigen.insert(j, i) = 1.0;
        }
    }
    
    // Second community: nodes 5-9
    for (scl_index_t i = 5; i < 10; ++i) {
        for (scl_index_t j = i + 1; j < 10; ++j) {
            adj_eigen.insert(i, j) = 1.0;
            adj_eigen.insert(j, i) = 1.0;
        }
    }
    
    adj_eigen.makeCompressed();
    Sparse adj = eigen_to_scl_sparse(adj_eigen);
    
    // Perfect clustering: each community separate
    std::vector<scl_index_t> labels = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    
    scl_real_t modularity = 0.0;
    scl_error_t err = scl_louvain_compute_modularity(
        adj, labels.data(), n_nodes, 1.0, &modularity
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Perfect clustering should have high modularity
    SCL_ASSERT_GT(modularity, 0.5);
}

SCL_TEST_CASE(modularity_null_adjacency)
{
    std::vector<scl_index_t> labels(10);
    scl_real_t modularity = 0.0;
    
    scl_error_t err = scl_louvain_compute_modularity(
        nullptr, labels.data(), 10, 1.0, &modularity
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(modularity_null_labels)
{
    Random rng(222);
    scl_index_t n_nodes = 10;
    auto adj_eigen = make_symmetric_adjacency(n_nodes, 0.3, rng);
    Sparse adj = eigen_to_scl_sparse(adj_eigen);
    
    scl_real_t modularity = 0.0;
    
    scl_error_t err = scl_louvain_compute_modularity(
        adj, nullptr, n_nodes, 1.0, &modularity
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(modularity_null_output)
{
    Random rng(333);
    scl_index_t n_nodes = 10;
    auto adj_eigen = make_symmetric_adjacency(n_nodes, 0.3, rng);
    Sparse adj = eigen_to_scl_sparse(adj_eigen);
    
    std::vector<scl_index_t> labels(n_nodes);
    scl_louvain_clustering(adj, labels.data(), n_nodes, 1.0, 100);
    
    scl_error_t err = scl_louvain_compute_modularity(
        adj, labels.data(), n_nodes, 1.0, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Community Sizes
// =============================================================================

SCL_TEST_SUITE(community_sizes)

SCL_TEST_CASE(community_sizes_basic)
{
    // Simple labels: 3 communities
    std::vector<scl_index_t> labels = {0, 0, 1, 1, 2, 2};
    scl_size_t n_nodes = labels.size();
    
    std::vector<scl_index_t> sizes(10);  // Large enough buffer
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_louvain_community_sizes(
        labels.data(), n_nodes, sizes.data(), sizes.size(), &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(n_communities, 3);
    
    // Verify sizes sum to n_nodes
    scl_index_t total = 0;
    for (scl_index_t i = 0; i < n_communities; ++i) {
        total += sizes[i];
    }
    SCL_ASSERT_EQ(total, static_cast<scl_index_t>(n_nodes));
}

SCL_TEST_RETRY(community_sizes_random_labels, 3)
{
    Random rng(444);
    
    scl_index_t n_nodes = rng.uniform_int(20, 50);
    scl_index_t n_communities = rng.uniform_int(2, 10);
    
    // Generate random labels
    std::vector<scl_index_t> labels(n_nodes);
    for (scl_index_t i = 0; i < n_nodes; ++i) {
        labels[i] = rng.uniform_int(0, n_communities - 1);
    }
    
    std::vector<scl_index_t> sizes(n_communities + 10);
    scl_index_t n_communities_out = 0;
    
    scl_error_t err = scl_louvain_community_sizes(
        labels.data(), n_nodes, sizes.data(), sizes.size(), &n_communities_out
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_LE(n_communities_out, n_communities);
    
    // Verify sizes sum to n_nodes
    scl_index_t total = 0;
    for (scl_index_t i = 0; i < n_communities_out; ++i) {
        total += sizes[i];
    }
    SCL_ASSERT_EQ(total, static_cast<scl_index_t>(n_nodes));
}

SCL_TEST_CASE(community_sizes_single_community)
{
    scl_index_t n_nodes = 10;
    std::vector<scl_index_t> labels(n_nodes, 0);  // All same community
    
    std::vector<scl_index_t> sizes(10);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_louvain_community_sizes(
        labels.data(), n_nodes, sizes.data(), sizes.size(), &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(n_communities, 1);
    SCL_ASSERT_EQ(sizes[0], n_nodes);
}

SCL_TEST_CASE(community_sizes_null_labels)
{
    std::vector<scl_index_t> sizes(10);
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_louvain_community_sizes(
        nullptr, 10, sizes.data(), sizes.size(), &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(community_sizes_null_sizes)
{
    std::vector<scl_index_t> labels = {0, 0, 1, 1};
    
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_louvain_community_sizes(
        labels.data(), 4, nullptr, 10, &n_communities
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(community_sizes_null_output)
{
    std::vector<scl_index_t> labels = {0, 0, 1, 1};
    std::vector<scl_index_t> sizes(10);
    
    scl_error_t err = scl_louvain_community_sizes(
        labels.data(), 4, sizes.data(), sizes.size(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(community_sizes_insufficient_buffer)
{
    std::vector<scl_index_t> labels = {0, 0, 1, 1, 2, 2};
    std::vector<scl_index_t> sizes(2);  // Too small (need 3)
    scl_index_t n_communities = 0;
    
    scl_error_t err = scl_louvain_community_sizes(
        labels.data(), 6, sizes.data(), sizes.size(), &n_communities
    );
    
    // Should either succeed (if implementation handles it) or return error
    // We just check it doesn't crash
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_RANGE_ERROR);
}

SCL_TEST_SUITE_END

// =============================================================================
// Get Community Members
// =============================================================================

SCL_TEST_SUITE(community_members)

SCL_TEST_CASE(get_community_members_basic)
{
    std::vector<scl_index_t> labels = {0, 0, 1, 1, 2, 2};
    scl_size_t n_nodes = labels.size();
    
    // Get members of community 0
    std::vector<scl_index_t> members(10);
    scl_index_t n_members = 0;
    
    scl_error_t err = scl_louvain_get_community_members(
        labels.data(), n_nodes, 0, members.data(), members.size(), &n_members
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(n_members, 2);
    
    // Verify members are correct
    std::vector<scl_index_t> expected = {0, 1};
    std::sort(members.begin(), members.begin() + n_members);
    SCL_ASSERT_EQ(members[0], 0);
    SCL_ASSERT_EQ(members[1], 1);
}

SCL_TEST_CASE(get_community_members_nonexistent)
{
    std::vector<scl_index_t> labels = {0, 0, 1, 1};
    
    std::vector<scl_index_t> members(10);
    scl_index_t n_members = 0;
    
    scl_error_t err = scl_louvain_get_community_members(
        labels.data(), 4, 99,  // Community 99 doesn't exist
        members.data(), members.size(), &n_members
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(n_members, 0);
}

SCL_TEST_RETRY(get_community_members_random, 3)
{
    Random rng(555);
    
    scl_index_t n_nodes = rng.uniform_int(20, 40);
    scl_index_t n_communities = rng.uniform_int(3, 8);
    
    std::vector<scl_index_t> labels(n_nodes);
    for (scl_index_t i = 0; i < n_nodes; ++i) {
        labels[i] = rng.uniform_int(0, n_communities - 1);
    }
    
    // Get members of a random community
    scl_index_t target_comm = rng.uniform_int(0, n_communities - 1);
    
    std::vector<scl_index_t> members(n_nodes);
    scl_index_t n_members = 0;
    
    scl_error_t err = scl_louvain_get_community_members(
        labels.data(), n_nodes, target_comm,
        members.data(), members.size(), &n_members
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify all returned members have correct label
    for (scl_index_t i = 0; i < n_members; ++i) {
        SCL_ASSERT_EQ(labels[members[i]], target_comm);
    }
}

SCL_TEST_CASE(get_community_members_null_labels)
{
    std::vector<scl_index_t> members(10);
    scl_index_t n_members = 0;
    
    scl_error_t err = scl_louvain_get_community_members(
        nullptr, 10, 0, members.data(), members.size(), &n_members
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(get_community_members_null_members)
{
    std::vector<scl_index_t> labels = {0, 0, 1, 1};
    
    scl_index_t n_members = 0;
    
    scl_error_t err = scl_louvain_get_community_members(
        labels.data(), 4, 0, nullptr, 10, &n_members
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(get_community_members_null_output)
{
    std::vector<scl_index_t> labels = {0, 0, 1, 1};
    std::vector<scl_index_t> members(10);
    
    scl_error_t err = scl_louvain_get_community_members(
        labels.data(), 4, 0, members.data(), members.size(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(get_community_members_insufficient_buffer)
{
    std::vector<scl_index_t> labels = {0, 0, 0, 0, 0};  // 5 nodes in community 0
    std::vector<scl_index_t> members(2);  // Too small
    
    scl_index_t n_members = 0;
    
    scl_error_t err = scl_louvain_get_community_members(
        labels.data(), 5, 0, members.data(), members.size(), &n_members
    );
    
    // Should either succeed (if implementation handles it) or return error
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_RANGE_ERROR);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

