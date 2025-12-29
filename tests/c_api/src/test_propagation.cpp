// =============================================================================
// SCL Core - Propagation Module Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/propagation.h
//
// Functions tested:
//   ✓ scl_propagation_label_propagation
//   ✓ scl_propagation_label_spreading
//   ✓ scl_propagation_harmonic_function
//   ✓ scl_propagation_confidence
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/propagation.h"

using namespace scl::test;
using precision::Tolerance;

// Helper: Create a simple adjacency matrix (ring graph)
static EigenCSR create_ring_graph(scl_index_t n) {
    EigenCSR adj(n, n);
    adj.reserve(Eigen::VectorXi::Constant(n, 2));
    
    for (scl_index_t i = 0; i < n; ++i) {
        scl_index_t next = (i + 1) % n;
        scl_index_t prev = (i - 1 + n) % n;
        adj.insert(i, next) = 1.0;
        adj.insert(i, prev) = 1.0;
    }
    
    adj.makeCompressed();
    return adj;
}

// Helper: Create a star graph
static EigenCSR create_star_graph(scl_index_t n) {
    EigenCSR adj(n, n);
    adj.reserve(Eigen::VectorXi::Constant(n, n));
    
    // Node 0 is center, connected to all others
    for (scl_index_t i = 1; i < n; ++i) {
        adj.insert(0, i) = 1.0;
        adj.insert(i, 0) = 1.0;
    }
    
    adj.makeCompressed();
    return adj;
}

SCL_TEST_BEGIN

// =============================================================================
// Label Propagation
// =============================================================================

SCL_TEST_SUITE(label_propagation)

SCL_TEST_CASE(label_propagation_simple) {
    // Create a simple 5-node ring graph
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Label nodes 0 and 2, leave others unlabeled (-1)
    std::vector<scl_index_t> labels = {0, -1, 1, -1, -1};
    
    scl_error_t err = scl_propagation_label_propagation(
        adj, labels.data(), 5, 10, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // After propagation, all nodes should have labels (0 or 1)
    for (scl_index_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(labels[i], 0);
        SCL_ASSERT_LE(labels[i], 1);
    }
}

SCL_TEST_CASE(label_propagation_all_labeled) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // All nodes already labeled
    std::vector<scl_index_t> labels = {0, 1, 0, 1, 0};
    std::vector<scl_index_t> labels_orig = labels;
    
    scl_error_t err = scl_propagation_label_propagation(
        adj, labels.data(), 5, 10, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Labels should remain the same (all already labeled)
    for (scl_index_t i = 0; i < 5; ++i) {
        SCL_ASSERT_EQ(labels[i], labels_orig[i]);
    }
}

SCL_TEST_CASE(label_propagation_none_labeled) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // No nodes labeled
    std::vector<scl_index_t> labels = {-1, -1, -1, -1, -1};
    
    scl_error_t err = scl_propagation_label_propagation(
        adj, labels.data(), 5, 10, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Without any seed labels, propagation may not work
    // Behavior depends on implementation
}

SCL_TEST_RETRY(label_propagation_random, 3)
{
    Random rng(50);
    
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
    
    // Randomly label some nodes
    std::vector<scl_index_t> labels(n, -1);
    scl_index_t n_labeled = rng.uniform_int(1, n / 2);
    for (scl_index_t i = 0; i < n_labeled; ++i) {
        scl_index_t idx = rng.uniform_int(0, n - 1);
        labels[idx] = rng.uniform_int(0, 2);
    }
    
    scl_error_t err = scl_propagation_label_propagation(
        adj, labels.data(), n, 20, rng.uniform_int(0, 10000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(label_propagation_null_handle) {
    std::vector<scl_index_t> labels(10);
    
    scl_error_t err = scl_propagation_label_propagation(
        nullptr, labels.data(), 10, 10, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(label_propagation_null_labels) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_propagation_label_propagation(
        adj, nullptr, 5, 10, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(label_propagation_zero_iter) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> labels = {0, -1, 1, -1, -1};
    
    scl_error_t err = scl_propagation_label_propagation(
        adj, labels.data(), 5, 0, 42
    );
    
    // Should either succeed (no-op) or return error
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_INVALID_ARGUMENT);
}

SCL_TEST_SUITE_END

// =============================================================================
// Label Spreading
// =============================================================================

SCL_TEST_SUITE(label_spreading)

SCL_TEST_CASE(label_spreading_simple) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_index_t n_classes = 2;
    std::vector<scl_real_t> label_probs(5 * n_classes, 0.0);
    std::vector<unsigned char> is_labeled(5, 0);
    
    // Label nodes 0 and 2
    is_labeled[0] = 1;
    label_probs[0 * n_classes + 0] = 1.0;  // Node 0: class 0
    is_labeled[2] = 1;
    label_probs[2 * n_classes + 1] = 1.0;  // Node 2: class 1
    
    scl_error_t err = scl_propagation_label_spreading(
        adj, label_probs.data(), is_labeled.data(), 5, n_classes,
        0.2, 10, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(label_spreading_all_labeled) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_index_t n_classes = 2;
    std::vector<scl_real_t> label_probs(5 * n_classes, 0.0);
    std::vector<unsigned char> is_labeled(5, 1);
    
    // All nodes labeled
    for (scl_index_t i = 0; i < 5; ++i) {
        label_probs[i * n_classes + (i % 2)] = 1.0;
    }
    
    scl_error_t err = scl_propagation_label_spreading(
        adj, label_probs.data(), is_labeled.data(), 5, n_classes,
        0.2, 10, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_RETRY(label_spreading_random, 3)
{
    Random rng(51);
    
    scl_index_t n = rng.uniform_int(10, 20);
    scl_index_t n_classes = rng.uniform_int(2, 5);
    
    auto adj_eigen = random_sparse_csr(n, n, 0.15, rng);
    EigenCSR adj_trans = adj_eigen.transpose();
    adj_eigen = adj_eigen + adj_trans;
    adj_eigen.prune(0.0, 0.0);
    
    auto csr = from_eigen_csr(adj_eigen);
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> label_probs(n * n_classes, 0.0);
    std::vector<unsigned char> is_labeled(n, 0);
    
    // Randomly label some nodes
    scl_index_t n_labeled = rng.uniform_int(1, n / 2);
    for (scl_index_t i = 0; i < n_labeled; ++i) {
        scl_index_t idx = rng.uniform_int(0, n - 1);
        is_labeled[idx] = 1;
        scl_index_t cls = rng.uniform_int(0, n_classes - 1);
        label_probs[idx * n_classes + cls] = 1.0;
    }
    
    scl_error_t err = scl_propagation_label_spreading(
        adj, label_probs.data(), is_labeled.data(), n, n_classes,
        0.2, 20, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(label_spreading_null_handle) {
    std::vector<scl_real_t> label_probs(20);
    std::vector<unsigned char> is_labeled(10);
    
    scl_error_t err = scl_propagation_label_spreading(
        nullptr, label_probs.data(), is_labeled.data(), 10, 2,
        0.2, 10, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(label_spreading_null_probs) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<unsigned char> is_labeled(5);
    
    scl_error_t err = scl_propagation_label_spreading(
        adj, nullptr, is_labeled.data(), 5, 2, 0.2, 10, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(label_spreading_null_mask) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> label_probs(10);
    
    scl_error_t err = scl_propagation_label_spreading(
        adj, label_probs.data(), nullptr, 5, 2, 0.2, 10, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(label_spreading_invalid_alpha) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> label_probs(10);
    std::vector<unsigned char> is_labeled(5);
    
    // Alpha should be in [0, 1]
    scl_error_t err1 = scl_propagation_label_spreading(
        adj, label_probs.data(), is_labeled.data(), 5, 2, -0.1, 10, 1e-6
    );
    SCL_ASSERT_NE(err1, SCL_OK);
    
    scl_error_t err2 = scl_propagation_label_spreading(
        adj, label_probs.data(), is_labeled.data(), 5, 2, 1.5, 10, 1e-6
    );
    SCL_ASSERT_NE(err2, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Harmonic Function
// =============================================================================

SCL_TEST_SUITE(harmonic_function)

SCL_TEST_CASE(harmonic_function_simple) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> values = {1.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<unsigned char> is_known = {1, 0, 0, 0, 0};
    
    scl_error_t err = scl_propagation_harmonic_function(
        adj, values.data(), is_known.data(), 5, 20, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(harmonic_function_all_known) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<unsigned char> is_known(5, 1);
    
    std::vector<scl_real_t> values_orig = values;
    
    scl_error_t err = scl_propagation_harmonic_function(
        adj, values.data(), is_known.data(), 5, 20, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Known values should remain unchanged
    for (scl_index_t i = 0; i < 5; ++i) {
        SCL_ASSERT_NEAR(values[i], values_orig[i], 1e-10);
    }
}

SCL_TEST_RETRY(harmonic_function_random, 3)
{
    Random rng(52);
    
    scl_index_t n = rng.uniform_int(10, 20);
    auto adj_eigen = random_sparse_csr(n, n, 0.15, rng);
    EigenCSR adj_trans = adj_eigen.transpose();
    adj_eigen = adj_eigen + adj_trans;
    adj_eigen.prune(0.0, 0.0);
    
    auto csr = from_eigen_csr(adj_eigen);
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> values(n);
    std::vector<unsigned char> is_known(n, 0);
    
    // Randomly set some known values
    scl_index_t n_known = rng.uniform_int(1, n / 2);
    for (scl_index_t i = 0; i < n_known; ++i) {
        scl_index_t idx = rng.uniform_int(0, n - 1);
        is_known[idx] = 1;
        values[idx] = rng.uniform(-5.0, 5.0);
    }
    
    scl_error_t err = scl_propagation_harmonic_function(
        adj, values.data(), is_known.data(), n, 30, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(harmonic_function_null_handle) {
    std::vector<scl_real_t> values(10);
    std::vector<unsigned char> is_known(10);
    
    scl_error_t err = scl_propagation_harmonic_function(
        nullptr, values.data(), is_known.data(), 10, 20, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(harmonic_function_null_values) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<unsigned char> is_known(5);
    
    scl_error_t err = scl_propagation_harmonic_function(
        adj, nullptr, is_known.data(), 5, 20, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(harmonic_function_null_mask) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> values(5);
    
    scl_error_t err = scl_propagation_harmonic_function(
        adj, values.data(), nullptr, 5, 20, 1e-6
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Confidence Propagation
// =============================================================================

SCL_TEST_SUITE(confidence_propagation)

SCL_TEST_CASE(confidence_propagation_simple) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> labels = {0, 1, 0, 1, 0};
    std::vector<scl_real_t> confidences(5);
    
    scl_error_t err = scl_propagation_confidence(
        adj, labels.data(), confidences.data(), 5, 2, 0.2, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Confidences should be in [0, 1]
    for (auto conf : confidences) {
        SCL_ASSERT_GE(conf, 0.0);
        SCL_ASSERT_LE(conf, 1.0);
    }
}

SCL_TEST_RETRY(confidence_propagation_random, 3)
{
    Random rng(53);
    
    scl_index_t n = rng.uniform_int(10, 20);
    scl_index_t n_classes = rng.uniform_int(2, 5);
    
    auto adj_eigen = random_sparse_csr(n, n, 0.15, rng);
    EigenCSR adj_trans = adj_eigen.transpose();
    adj_eigen = adj_eigen + adj_trans;
    adj_eigen.prune(0.0, 0.0);
    
    auto csr = from_eigen_csr(adj_eigen);
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> labels(n);
    std::vector<scl_real_t> confidences(n);
    
    // Random initial labels
    for (scl_index_t i = 0; i < n; ++i) {
        labels[i] = rng.uniform_int(0, n_classes - 1);
    }
    
    scl_error_t err = scl_propagation_confidence(
        adj, labels.data(), confidences.data(), n, n_classes, 0.2, 20
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Check confidences are valid
    for (auto conf : confidences) {
        SCL_ASSERT_GE(conf, 0.0);
        SCL_ASSERT_LE(conf, 1.0);
    }
}

SCL_TEST_CASE(confidence_propagation_null_handle) {
    std::vector<scl_index_t> labels(10);
    std::vector<scl_real_t> confidences(10);
    
    scl_error_t err = scl_propagation_confidence(
        nullptr, labels.data(), confidences.data(), 10, 2, 0.2, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(confidence_propagation_null_labels) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> confidences(5);
    
    scl_error_t err = scl_propagation_confidence(
        adj, nullptr, confidences.data(), 5, 2, 0.2, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(confidence_propagation_null_confidences) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> labels = {0, 1, 0, 1, 0};
    
    scl_error_t err = scl_propagation_confidence(
        adj, labels.data(), nullptr, 5, 2, 0.2, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(confidence_propagation_invalid_alpha) {
    auto adj_eigen = create_ring_graph(5);
    auto csr = from_eigen_csr(adj_eigen);
    
    Sparse adj = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> labels = {0, 1, 0, 1, 0};
    std::vector<scl_real_t> confidences(5);
    
    scl_error_t err = scl_propagation_confidence(
        adj, labels.data(), confidences.data(), 5, 2, -0.1, 10
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

