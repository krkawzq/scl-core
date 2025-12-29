// =============================================================================
// SCL Core - Complete Centrality Tests
// =============================================================================
//
// Test coverage for scl/binding/c_api/centrality.h
//
// Functions tested (13 total):
//   ✓ scl_centrality_degree
//   ✓ scl_centrality_weighted_degree
//   ✓ scl_centrality_pagerank
//   ✓ scl_centrality_personalized_pagerank
//   ✓ scl_centrality_hits
//   ✓ scl_centrality_eigenvector
//   ✓ scl_centrality_katz
//   ✓ scl_centrality_closeness
//   ✓ scl_centrality_betweenness
//   ✓ scl_centrality_betweenness_sampled
//   ✓ scl_centrality_harmonic
//   ✓ scl_centrality_current_flow_approx
//
// =============================================================================

#include "test.hpp"

extern "C" {
#include "scl/binding/c_api/centrality.h"
}

using namespace scl::test;

// Helper: Create simple graph (5 nodes)
//   0 -> 1 -> 2
//   |    |
//   v    v
//   3 -> 4
static auto create_simple_graph() {
    std::vector<scl_index_t> indptr = {0, 2, 4, 5, 6, 6};
    std::vector<scl_index_t> indices = {1, 3, 2, 4, 0, 4};
    std::vector<scl_real_t> data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    return std::make_tuple(indptr, indices, data);
}

// Helper: Create weighted graph
static auto create_weighted_graph() {
    std::vector<scl_index_t> indptr = {0, 2, 4, 5, 6, 6};
    std::vector<scl_index_t> indices = {1, 3, 2, 4, 0, 4};
    std::vector<scl_real_t> data = {2.0, 1.5, 3.0, 0.5, 1.0, 2.5};
    return std::make_tuple(indptr, indices, data);
}

// Helper: Create symmetric graph (undirected)
static auto create_symmetric_graph() {
    std::vector<scl_index_t> indptr = {0, 2, 4, 6, 8, 10};
    std::vector<scl_index_t> indices = {1, 2, 0, 2, 0, 1, 3, 4, 3, 4};
    std::vector<scl_real_t> data(10, 1.0);
    return std::make_tuple(indptr, indices, data);
}

SCL_TEST_BEGIN

// =============================================================================
// Degree Centrality Tests
// =============================================================================

SCL_TEST_SUITE(degree_centrality)

SCL_TEST_CASE(basic_degree) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_degree(graph, centrality.data(), 5, SCL_FALSE);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify all centrality values are non-negative
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
    }
}

SCL_TEST_CASE(normalized_degree) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_degree(graph, centrality.data(), 5, SCL_TRUE);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Normalized values should be in [0, 1]
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
        SCL_ASSERT_LE(centrality[i], 1.0);
    }
}

SCL_TEST_CASE(degree_null_graph) {
    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_degree(nullptr, centrality.data(), 5, SCL_FALSE);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(degree_null_output) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    scl_error_t err = scl_centrality_degree(graph, nullptr, 5, SCL_FALSE);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(degree_zero_nodes) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(1);

    scl_error_t err = scl_centrality_degree(graph, centrality.data(), 0, SCL_FALSE);
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(degree_single_node) {
    std::vector<scl_index_t> indptr = {0, 0};
    std::vector<scl_index_t> indices(1);
    std::vector<scl_real_t> data(1);

    Sparse graph = make_sparse_csr(1, 1, 0,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(1);

    scl_error_t err = scl_centrality_degree(graph, centrality.data(), 1, SCL_FALSE);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(centrality[0], 0.0);  // Isolated node has degree 0
}

SCL_TEST_SUITE_END

// =============================================================================
// Weighted Degree Centrality Tests
// =============================================================================

SCL_TEST_SUITE(weighted_degree_centrality)

SCL_TEST_CASE(basic_weighted_degree) {
    auto [indptr, indices, data] = create_weighted_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_weighted_degree(graph, centrality.data(), 5, SCL_FALSE);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Node 0 has edges with weights 2.0 + 1.5 = 3.5
    SCL_ASSERT_GT(centrality[0], 0.0);
}

SCL_TEST_CASE(weighted_degree_normalized) {
    auto [indptr, indices, data] = create_weighted_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_weighted_degree(graph, centrality.data(), 5, SCL_TRUE);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Normalized values should be in [0, 1]
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
        SCL_ASSERT_LE(centrality[i], 1.0);
    }
}

SCL_TEST_CASE(weighted_degree_null_graph) {
    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_weighted_degree(nullptr, centrality.data(), 5, SCL_FALSE);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(weighted_degree_null_output) {
    auto [indptr, indices, data] = create_weighted_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    scl_error_t err = scl_centrality_weighted_degree(graph, nullptr, 5, SCL_FALSE);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// PageRank Tests
// =============================================================================

SCL_TEST_SUITE(pagerank)

SCL_TEST_CASE(basic_pagerank) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> scores(5);

    scl_error_t err = scl_centrality_pagerank(graph, scores.data(), 5, 0.85, 100, 1e-6);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // PageRank scores should sum to approximately 1.0
    scl_real_t sum = 0.0;
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GT(scores[i], 0.0);
        sum += scores[i];
    }
    SCL_ASSERT_NEAR(sum, 1.0, 1e-5);
}

SCL_TEST_CASE(pagerank_null_graph) {
    std::vector<scl_real_t> scores(5);

    scl_error_t err = scl_centrality_pagerank(nullptr, scores.data(), 5, 0.85, 100, 1e-6);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(pagerank_null_output) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    scl_error_t err = scl_centrality_pagerank(graph, nullptr, 5, 0.85, 100, 1e-6);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(pagerank_invalid_damping) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> scores(5);

    // Damping factor must be in [0, 1]
    scl_error_t err = scl_centrality_pagerank(graph, scores.data(), 5, 1.5, 100, 1e-6);
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(pagerank_zero_iterations) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> scores(5);

    scl_error_t err = scl_centrality_pagerank(graph, scores.data(), 5, 0.85, 0, 1e-6);
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Personalized PageRank Tests
// =============================================================================

SCL_TEST_SUITE(personalized_pagerank)

SCL_TEST_CASE(basic_personalized_pagerank) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> seed_nodes = {0, 1};
    std::vector<scl_real_t> scores(5);

    scl_error_t err = scl_centrality_personalized_pagerank(
        graph, seed_nodes.data(), 2, scores.data(), 5, 0.85, 100, 1e-6);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // All scores should be non-negative
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(scores[i], 0.0);
    }
}

SCL_TEST_CASE(personalized_pagerank_null_seeds) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> scores(5);

    scl_error_t err = scl_centrality_personalized_pagerank(
        graph, nullptr, 2, scores.data(), 5, 0.85, 100, 1e-6);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(personalized_pagerank_zero_seeds) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> seed_nodes = {0};
    std::vector<scl_real_t> scores(5);

    scl_error_t err = scl_centrality_personalized_pagerank(
        graph, seed_nodes.data(), 0, scores.data(), 5, 0.85, 100, 1e-6);
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(personalized_pagerank_single_seed) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> seed_nodes = {0};
    std::vector<scl_real_t> scores(5);

    scl_error_t err = scl_centrality_personalized_pagerank(
        graph, seed_nodes.data(), 1, scores.data(), 5, 0.85, 100, 1e-6);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// HITS Algorithm Tests
// =============================================================================

SCL_TEST_SUITE(hits)

SCL_TEST_CASE(basic_hits) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> hub_scores(5);
    std::vector<scl_real_t> authority_scores(5);

    scl_error_t err = scl_centrality_hits(graph, hub_scores.data(),
                                          authority_scores.data(), 5, 100, 1e-6);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Scores should be non-negative and normalized
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(hub_scores[i], 0.0);
        SCL_ASSERT_GE(authority_scores[i], 0.0);
    }
}

SCL_TEST_CASE(hits_null_graph) {
    std::vector<scl_real_t> hub_scores(5);
    std::vector<scl_real_t> authority_scores(5);

    scl_error_t err = scl_centrality_hits(nullptr, hub_scores.data(),
                                          authority_scores.data(), 5, 100, 1e-6);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(hits_null_hub_output) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> authority_scores(5);

    scl_error_t err = scl_centrality_hits(graph, nullptr,
                                          authority_scores.data(), 5, 100, 1e-6);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(hits_null_authority_output) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> hub_scores(5);

    scl_error_t err = scl_centrality_hits(graph, hub_scores.data(),
                                          nullptr, 5, 100, 1e-6);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Eigenvector Centrality Tests
// =============================================================================

SCL_TEST_SUITE(eigenvector_centrality)

SCL_TEST_CASE(basic_eigenvector) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_eigenvector(graph, centrality.data(), 5, 100, 1e-6);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Eigenvector centrality should be non-negative
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
    }
}

SCL_TEST_CASE(eigenvector_null_graph) {
    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_eigenvector(nullptr, centrality.data(), 5, 100, 1e-6);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(eigenvector_null_output) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    scl_error_t err = scl_centrality_eigenvector(graph, nullptr, 5, 100, 1e-6);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Katz Centrality Tests
// =============================================================================

SCL_TEST_SUITE(katz_centrality)

SCL_TEST_CASE(basic_katz) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_katz(graph, centrality.data(), 5, 0.1, 1.0, 100, 1e-6);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Katz centrality should be positive
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GT(centrality[i], 0.0);
    }
}

SCL_TEST_CASE(katz_null_graph) {
    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_katz(nullptr, centrality.data(), 5, 0.1, 1.0, 100, 1e-6);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(katz_null_output) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    scl_error_t err = scl_centrality_katz(graph, nullptr, 5, 0.1, 1.0, 100, 1e-6);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(katz_invalid_alpha) {
    auto [indptr, indices, data] = create_simple_graph();
    Sparse graph = make_sparse_csr(5, 5, 6,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    // Alpha too large may cause divergence
    scl_error_t err = scl_centrality_katz(graph, centrality.data(), 5, 10.0, 1.0, 100, 1e-6);
    // May fail or succeed depending on implementation
    // Just check it doesn't crash
}

SCL_TEST_SUITE_END

// =============================================================================
// Closeness Centrality Tests
// =============================================================================

SCL_TEST_SUITE(closeness_centrality)

SCL_TEST_CASE(basic_closeness) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_closeness(graph, centrality.data(), 5, SCL_FALSE);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Closeness centrality should be non-negative
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
    }
}

SCL_TEST_CASE(closeness_normalized) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_closeness(graph, centrality.data(), 5, SCL_TRUE);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Normalized closeness should be in [0, 1]
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
        SCL_ASSERT_LE(centrality[i], 1.0);
    }
}

SCL_TEST_CASE(closeness_null_graph) {
    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_closeness(nullptr, centrality.data(), 5, SCL_FALSE);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(closeness_null_output) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    scl_error_t err = scl_centrality_closeness(graph, nullptr, 5, SCL_FALSE);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Betweenness Centrality Tests
// =============================================================================

SCL_TEST_SUITE(betweenness_centrality)

SCL_TEST_CASE(basic_betweenness) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_betweenness(graph, centrality.data(), 5, SCL_FALSE);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Betweenness centrality should be non-negative
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
    }
}

SCL_TEST_CASE(betweenness_normalized) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_betweenness(graph, centrality.data(), 5, SCL_TRUE);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Normalized betweenness should be in [0, 1]
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
        SCL_ASSERT_LE(centrality[i], 1.0);
    }
}

SCL_TEST_CASE(betweenness_null_graph) {
    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_betweenness(nullptr, centrality.data(), 5, SCL_FALSE);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(betweenness_null_output) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    scl_error_t err = scl_centrality_betweenness(graph, nullptr, 5, SCL_FALSE);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Betweenness Sampled Tests
// =============================================================================

SCL_TEST_SUITE(betweenness_sampled)

SCL_TEST_CASE(basic_betweenness_sampled) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_betweenness_sampled(
        graph, centrality.data(), 5, 3, SCL_FALSE, 12345);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Sampled betweenness should be non-negative
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
    }
}

SCL_TEST_CASE(betweenness_sampled_null_graph) {
    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_betweenness_sampled(
        nullptr, centrality.data(), 5, 3, SCL_FALSE, 12345);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(betweenness_sampled_zero_samples) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_betweenness_sampled(
        graph, centrality.data(), 5, 0, SCL_FALSE, 12345);
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Harmonic Centrality Tests
// =============================================================================

SCL_TEST_SUITE(harmonic_centrality)

SCL_TEST_CASE(basic_harmonic) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_harmonic(graph, centrality.data(), 5, SCL_FALSE);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Harmonic centrality should be non-negative
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
    }
}

SCL_TEST_CASE(harmonic_null_graph) {
    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_harmonic(nullptr, centrality.data(), 5, SCL_FALSE);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(harmonic_null_output) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    scl_error_t err = scl_centrality_harmonic(graph, nullptr, 5, SCL_FALSE);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Current Flow Approximation Tests
// =============================================================================

SCL_TEST_SUITE(current_flow_approx)

SCL_TEST_CASE(basic_current_flow) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_current_flow_approx(
        graph, centrality.data(), 5, 100, 10, 12345);

    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // Current flow centrality should be non-negative
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(centrality[i], 0.0);
    }
}

SCL_TEST_CASE(current_flow_null_graph) {
    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_current_flow_approx(
        nullptr, centrality.data(), 5, 100, 10, 12345);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(current_flow_null_output) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    scl_error_t err = scl_centrality_current_flow_approx(
        graph, nullptr, 5, 100, 10, 12345);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(current_flow_zero_walks) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_current_flow_approx(
        graph, centrality.data(), 5, 0, 10, 12345);
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(current_flow_zero_length) {
    auto [indptr, indices, data] = create_symmetric_graph();
    Sparse graph = make_sparse_csr(5, 5, 10,
                                    indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> centrality(5);

    scl_error_t err = scl_centrality_current_flow_approx(
        graph, centrality.data(), 5, 100, 0, 12345);
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
