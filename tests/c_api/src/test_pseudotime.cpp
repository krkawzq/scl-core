// =============================================================================
// SCL Core - Pseudotime Module Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/pseudotime.h
//
// Functions tested:
//   ✓ scl_pseudotime_compute
//   ✓ scl_pseudotime_diffusion
//   ✓ scl_pseudotime_graph
//   ✓ scl_pseudotime_multi_source
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/pseudotime.h"

using namespace scl::test;
using precision::Tolerance;

// Helper: Create a simple path graph (linear chain)
static EigenCSR create_path_graph(scl_index_t n) {
    EigenCSR adj(n, n);
    adj.reserve(Eigen::VectorXi::Constant(n, 2));
    
    for (scl_index_t i = 0; i < n - 1; ++i) {
        adj.insert(i, i + 1) = 1.0;
        adj.insert(i + 1, i) = 1.0;
    }
    
    adj.makeCompressed();
    return adj;
}

// Helper: Create a ring graph
static EigenCSR create_ring_graph(scl_index_t n) {
    EigenCSR adj(n, n);
    adj.reserve(Eigen::VectorXi::Constant(n, 2));
    
    for (scl_index_t i = 0; i < n; ++i) {
        scl_index_t next = (i + 1) % n;
        adj.insert(i, next) = 1.0;
        adj.insert(next, i) = 1.0;
    }
    
    adj.makeCompressed();
    return adj;
}

SCL_TEST_BEGIN

// =============================================================================
// Graph Distance Pseudotime
// =============================================================================

SCL_TEST_SUITE(graph_distance)

SCL_TEST_CASE(graph_distance_path_graph) {
    // Path graph: 0-1-2-3-4
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    // Root at node 0
    scl_error_t err = scl_pseudotime_graph(adj, 0, pseudotime.data(), 5);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Pseudotime should increase with distance from root
    SCL_ASSERT_NEAR(pseudotime[0], 0.0, 1e-10);  // Root has time 0
    for (scl_index_t i = 1; i < 5; ++i) {
        SCL_ASSERT_GT(pseudotime[i], pseudotime[i - 1]);
    }
}

SCL_TEST_CASE(graph_distance_ring_graph) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    scl_error_t err = scl_pseudotime_graph(adj, 0, pseudotime.data(), 5);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Root should have time 0
    SCL_ASSERT_NEAR(pseudotime[0], 0.0, 1e-10);
    
    // All other nodes should have positive time
    for (scl_index_t i = 1; i < 5; ++i) {
        SCL_ASSERT_GT(pseudotime[i], 0.0);
    }
}

SCL_TEST_CASE(graph_distance_different_roots) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime1(5);
    std::vector<scl_real_t> pseudotime2(5);
    
    // Root at 0
    scl_pseudotime_graph(adj, 0, pseudotime1.data(), 5);
    
    // Root at 4
    scl_pseudotime_graph(adj, 4, pseudotime2.data(), 5);
    
    // Results should be different
    bool different = false;
    for (scl_index_t i = 0; i < 5; ++i) {
        if (std::abs(pseudotime1[i] - pseudotime2[i]) > 1e-10) {
            different = true;
            break;
        }
    }
    SCL_ASSERT_TRUE(different);
}

SCL_TEST_RETRY(graph_distance_random, 3)
{
    Random rng(60);
    
    scl_index_t n = rng.uniform_int(10, 30);
    auto adj_eigen = random_sparse_csr(n, n, 0.1, rng);
    // Make symmetric
    EigenCSR adj_trans = adj_eigen.transpose();
    adj_eigen = adj_eigen + adj_trans;
    adj_eigen.prune(0.0, 0.0);
    
    auto csr = from_eigen_csr(adj_eigen);
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(n);
    scl_index_t root = rng.uniform_int(0, n - 1);
    
    scl_error_t err = scl_pseudotime_graph(adj, root, pseudotime.data(), n);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Root should have minimum time
    SCL_ASSERT_NEAR(pseudotime[root], 0.0, 1e-10);
}

SCL_TEST_CASE(graph_distance_null_handle) {
    std::vector<scl_real_t> pseudotime(10);
    
    scl_error_t err = scl_pseudotime_graph(nullptr, 0, pseudotime.data(), 10);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(graph_distance_null_output) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_pseudotime_graph(adj, 0, nullptr, 5);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(graph_distance_invalid_root) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    // Root out of bounds
    scl_error_t err = scl_pseudotime_graph(adj, 10, pseudotime.data(), 5);
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Diffusion Pseudotime
// =============================================================================

SCL_TEST_SUITE(diffusion_pseudotime)

SCL_TEST_CASE(diffusion_pseudotime_path_graph) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    scl_error_t err = scl_pseudotime_diffusion(
        adj, 0, pseudotime.data(), 5, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Root should have minimum time
    SCL_ASSERT_GE(pseudotime[0], 0.0);
}

SCL_TEST_CASE(diffusion_pseudotime_ring_graph) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    scl_error_t err = scl_pseudotime_diffusion(
        adj, 0, pseudotime.data(), 5, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_RETRY(diffusion_pseudotime_random, 3)
{
    Random rng(61);
    
    scl_index_t n = rng.uniform_int(10, 30);
    auto adj_eigen = random_sparse_csr(n, n, 0.1, rng);
    EigenCSR adj_trans = adj_eigen.transpose();
    adj_eigen = adj_eigen + adj_trans;
    adj_eigen.prune(0.0, 0.0);
    
    auto csr = from_eigen_csr(adj_eigen);
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(n);
    scl_index_t root = rng.uniform_int(0, n - 1);
    scl_index_t n_dcs = rng.uniform_int(5, 15);
    
    scl_error_t err = scl_pseudotime_diffusion(
        adj, root, pseudotime.data(), n, n_dcs
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(diffusion_pseudotime_null_handle) {
    std::vector<scl_real_t> pseudotime(10);
    
    scl_error_t err = scl_pseudotime_diffusion(
        nullptr, 0, pseudotime.data(), 10, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(diffusion_pseudotime_null_output) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_pseudotime_diffusion(
        adj, 0, nullptr, 5, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(diffusion_pseudotime_invalid_root) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    scl_error_t err = scl_pseudotime_diffusion(
        adj, 10, pseudotime.data(), 5, 10
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Generic Compute Function
// =============================================================================

SCL_TEST_SUITE(compute_pseudotime)

SCL_TEST_CASE(compute_diffusion_method) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    scl_error_t err = scl_pseudotime_compute(
        adj, 0, pseudotime.data(), 5,
        SCL_PSEUDOTIME_DIFFUSION, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(compute_shortest_path_method) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    scl_error_t err = scl_pseudotime_compute(
        adj, 0, pseudotime.data(), 5,
        SCL_PSEUDOTIME_SHORTEST_PATH, 0
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(compute_graph_distance_method) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    scl_error_t err = scl_pseudotime_compute(
        adj, 0, pseudotime.data(), 5,
        SCL_PSEUDOTIME_GRAPH_DISTANCE, 0
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(compute_watershed_method) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    scl_error_t err = scl_pseudotime_compute(
        adj, 0, pseudotime.data(), 5,
        SCL_PSEUDOTIME_WATERSHED, 0
    );
    
    // Watershed may or may not be implemented
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_NOT_IMPLEMENTED);
}

SCL_TEST_CASE(compute_invalid_method) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pseudotime(5);
    
    scl_pseudotime_method_t invalid_method = static_cast<scl_pseudotime_method_t>(99);
    scl_error_t err = scl_pseudotime_compute(
        adj, 0, pseudotime.data(), 5, invalid_method, 0
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(compute_null_handle) {
    std::vector<scl_real_t> pseudotime(10);
    
    scl_error_t err = scl_pseudotime_compute(
        nullptr, 0, pseudotime.data(), 10,
        SCL_PSEUDOTIME_DIFFUSION, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(compute_null_output) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_pseudotime_compute(
        adj, 0, nullptr, 5, SCL_PSEUDOTIME_DIFFUSION, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Multi-Source Shortest Paths
// =============================================================================

SCL_TEST_SUITE(multi_source)

SCL_TEST_CASE(multi_source_simple) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> sources = {0, 4};
    std::vector<scl_real_t> distances(5);
    
    scl_error_t err = scl_pseudotime_multi_source(
        adj, sources.data(), 2, distances.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Sources should have distance 0
    SCL_ASSERT_NEAR(distances[0], 0.0, 1e-10);
    SCL_ASSERT_NEAR(distances[4], 0.0, 1e-10);
}

SCL_TEST_CASE(multi_source_single_source) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> sources = {2};
    std::vector<scl_real_t> distances(5);
    
    scl_error_t err = scl_pseudotime_multi_source(
        adj, sources.data(), 1, distances.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Source should have distance 0
    SCL_ASSERT_NEAR(distances[2], 0.0, 1e-10);
}

SCL_TEST_RETRY(multi_source_random, 3)
{
    Random rng(62);
    
    scl_index_t n = rng.uniform_int(10, 30);
    auto adj_eigen = random_sparse_csr(n, n, 0.1, rng);
    EigenCSR adj_trans = adj_eigen.transpose();
    adj_eigen = adj_eigen + adj_trans;
    adj_eigen.prune(0.0, 0.0);
    
    auto csr = from_eigen_csr(adj_eigen);
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_index_t n_sources = rng.uniform_int(1, std::min(5, n));
    std::vector<scl_index_t> sources(n_sources);
    for (scl_index_t i = 0; i < n_sources; ++i) {
        sources[i] = rng.uniform_int(0, n - 1);
    }
    
    std::vector<scl_real_t> distances(n);
    
    scl_error_t err = scl_pseudotime_multi_source(
        adj, sources.data(), n_sources, distances.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All sources should have distance 0
    for (scl_index_t i = 0; i < n_sources; ++i) {
        SCL_ASSERT_NEAR(distances[sources[i]], 0.0, 1e-10);
    }
}

SCL_TEST_CASE(multi_source_null_handle) {
    std::vector<scl_index_t> sources = {0};
    std::vector<scl_real_t> distances(10);
    
    scl_error_t err = scl_pseudotime_multi_source(
        nullptr, sources.data(), 1, distances.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(multi_source_null_sources) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> distances(5);
    
    scl_error_t err = scl_pseudotime_multi_source(
        adj, nullptr, 1, distances.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(multi_source_null_distances) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> sources = {0};
    
    scl_error_t err = scl_pseudotime_multi_source(
        adj, sources.data(), 1, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(multi_source_zero_sources) {
    auto adj_eigen = create_path_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> sources;
    std::vector<scl_real_t> distances(5);
    
    scl_error_t err = scl_pseudotime_multi_source(
        adj, sources.data(), 0, distances.data()
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

