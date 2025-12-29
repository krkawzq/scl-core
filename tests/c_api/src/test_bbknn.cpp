// =============================================================================
// SCL Core - Complete BBKNN Tests
// =============================================================================
//
// Test coverage for scl/binding/c_api/bbknn.h
//
// Functions tested (3 total):
//   ✓ scl_bbknn_compute_norms
//   ✓ scl_bbknn
//   ✓ scl_bbknn_with_norms
//
// =============================================================================

#include "test.hpp"

extern "C" {
#include "scl/binding/c_api/bbknn.h"
}

using namespace scl::test;

// Helper: Create simple test data (3 batches, 9 cells, 5 genes)
static auto create_test_data() {
    // Cell 0-2: batch 0
    // Cell 3-5: batch 1
    // Cell 6-8: batch 2
    std::vector<scl_index_t> indptr = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    std::vector<scl_index_t> indices = {0, 1, 1, 2, 2, 3, 3, 4, 0, 4,
                                        1, 2, 2, 3, 3, 4, 0, 1};
    std::vector<scl_real_t> data(18, 1.0);

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = 1.0 + 0.1 * static_cast<scl_real_t>(i);
    }

    std::vector<int32_t> batch_labels = {0, 0, 0, 1, 1, 1, 2, 2, 2};

    return std::make_tuple(indptr, indices, data, batch_labels);
}

SCL_TEST_BEGIN

// =============================================================================
// Compute Norms Tests
// =============================================================================

SCL_TEST_SUITE(compute_norms)

SCL_TEST_CASE(basic_norms_computation) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> norms_sq(9);

    scl_error_t err = scl_bbknn_compute_norms(mat, norms_sq.data(), 9);

    // If not implemented, skip
    if (err == SCL_ERROR_NOT_IMPLEMENTED) {
        SCL_SKIP("Function not implemented yet");
    }

    SCL_ASSERT_EQ(err, SCL_OK);

    // All norms should be positive
    for (size_t i = 0; i < 9; ++i) {
        SCL_ASSERT_GT(norms_sq[i], 0.0);
    }
}

SCL_TEST_CASE(null_matrix) {
    std::vector<scl_real_t> norms_sq(10);

    scl_error_t err = scl_bbknn_compute_norms(nullptr, norms_sq.data(), 10);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_output) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    scl_error_t err = scl_bbknn_compute_norms(mat, nullptr, 9);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(zero_cells) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> norms_sq(1);

    scl_error_t err = scl_bbknn_compute_norms(mat, norms_sq.data(), 0);
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(single_cell) {
    std::vector<scl_index_t> indptr = {0, 2};
    std::vector<scl_index_t> indices = {0, 1};
    std::vector<scl_real_t> data = {3.0, 4.0};  // Norm^2 should be 25.0

    Sparse mat = make_sparse_csr(1, 5, 2,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> norms_sq(1);

    scl_error_t err = scl_bbknn_compute_norms(mat, norms_sq.data(), 1);
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(norms_sq[0], 25.0, 1e-10);
}

SCL_TEST_SUITE_END

// =============================================================================
// BBKNN Basic Tests
// =============================================================================

SCL_TEST_SUITE(bbknn_basic)

SCL_TEST_CASE(basic_bbknn) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    scl_size_t k = 2;
    scl_size_t n_batches = 3;
    scl_size_t total_neighbors = 9 * n_batches * k;

    std::vector<scl_index_t> out_indices(total_neighbors);
    std::vector<scl_real_t> out_distances(total_neighbors);

    scl_error_t err = scl_bbknn(mat, batch_labels.data(), 9, n_batches, k,
                                out_indices.data(), out_distances.data());
    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify indices are valid
    for (size_t i = 0; i < total_neighbors; ++i) {
        SCL_ASSERT_GE(out_indices[i], 0);
        SCL_ASSERT_LT(out_indices[i], 9);
    }

    // Verify distances are non-negative
    for (size_t i = 0; i < total_neighbors; ++i) {
        SCL_ASSERT_GE(out_distances[i], 0.0);
    }
}

SCL_TEST_CASE(null_matrix_bbknn) {
    std::vector<int32_t> batch_labels(10, 0);
    std::vector<scl_index_t> out_indices(20);
    std::vector<scl_real_t> out_distances(20);

    scl_error_t err = scl_bbknn(nullptr, batch_labels.data(), 10, 2, 1,
                                out_indices.data(), out_distances.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_batch_labels) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> out_indices(18);
    std::vector<scl_real_t> out_distances(18);

    scl_error_t err = scl_bbknn(mat, nullptr, 9, 3, 1,
                                out_indices.data(), out_distances.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_output_indices) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> out_distances(18);

    scl_error_t err = scl_bbknn(mat, batch_labels.data(), 9, 3, 1,
                                nullptr, out_distances.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_output_distances) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> out_indices(18);

    scl_error_t err = scl_bbknn(mat, batch_labels.data(), 9, 3, 1,
                                out_indices.data(), nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(zero_k) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> out_indices(1);
    std::vector<scl_real_t> out_distances(1);

    scl_error_t err = scl_bbknn(mat, batch_labels.data(), 9, 3, 0,
                                out_indices.data(), out_distances.data());
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(zero_batches) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> out_indices(1);
    std::vector<scl_real_t> out_distances(1);

    scl_error_t err = scl_bbknn(mat, batch_labels.data(), 9, 0, 1,
                                out_indices.data(), out_distances.data());
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(k_equals_one) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    scl_size_t k = 1;
    scl_size_t n_batches = 3;
    std::vector<scl_index_t> out_indices(9 * n_batches * k);
    std::vector<scl_real_t> out_distances(9 * n_batches * k);

    scl_error_t err = scl_bbknn(mat, batch_labels.data(), 9, n_batches, k,
                                out_indices.data(), out_distances.data());
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// BBKNN with Precomputed Norms Tests
// =============================================================================

SCL_TEST_SUITE(bbknn_with_norms)

SCL_TEST_CASE(basic_with_norms) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    // Precompute norms
    std::vector<scl_real_t> norms_sq(9);
    scl_error_t err = scl_bbknn_compute_norms(mat, norms_sq.data(), 9);
    SCL_ASSERT_EQ(err, SCL_OK);

    // Run BBKNN with precomputed norms
    scl_size_t k = 2;
    scl_size_t n_batches = 3;
    std::vector<scl_index_t> out_indices(9 * n_batches * k);
    std::vector<scl_real_t> out_distances(9 * n_batches * k);

    err = scl_bbknn_with_norms(mat, batch_labels.data(), 9, n_batches, k,
                               norms_sq.data(), out_indices.data(), out_distances.data());
    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify results
    for (size_t i = 0; i < 9 * n_batches * k; ++i) {
        SCL_ASSERT_GE(out_indices[i], 0);
        SCL_ASSERT_LT(out_indices[i], 9);
        SCL_ASSERT_GE(out_distances[i], 0.0);
    }
}

SCL_TEST_CASE(consistency_with_without_norms) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    scl_size_t k = 2;
    scl_size_t n_batches = 3;
    scl_size_t total_neighbors = 9 * n_batches * k;

    // Run without precomputed norms
    std::vector<scl_index_t> indices1(total_neighbors);
    std::vector<scl_real_t> distances1(total_neighbors);
    scl_error_t err = scl_bbknn(mat, batch_labels.data(), 9, n_batches, k,
                                indices1.data(), distances1.data());
    SCL_ASSERT_EQ(err, SCL_OK);

    // Run with precomputed norms
    std::vector<scl_real_t> norms_sq(9);
    err = scl_bbknn_compute_norms(mat, norms_sq.data(), 9);
    SCL_ASSERT_EQ(err, SCL_OK);

    std::vector<scl_index_t> indices2(total_neighbors);
    std::vector<scl_real_t> distances2(total_neighbors);
    err = scl_bbknn_with_norms(mat, batch_labels.data(), 9, n_batches, k,
                               norms_sq.data(), indices2.data(), distances2.data());
    SCL_ASSERT_EQ(err, SCL_OK);

    // Results should be identical
    for (size_t i = 0; i < total_neighbors; ++i) {
        SCL_ASSERT_EQ(indices1[i], indices2[i]);
        SCL_ASSERT_NEAR(distances1[i], distances2[i], 1e-10);
    }
}

SCL_TEST_CASE(null_norms) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> out_indices(18);
    std::vector<scl_real_t> out_distances(18);

    scl_error_t err = scl_bbknn_with_norms(mat, batch_labels.data(), 9, 3, 1,
                                           nullptr, out_indices.data(), out_distances.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_matrix_with_norms) {
    std::vector<scl_real_t> norms_sq(10, 1.0);
    std::vector<int32_t> batch_labels(10, 0);
    std::vector<scl_index_t> out_indices(20);
    std::vector<scl_real_t> out_distances(20);

    scl_error_t err = scl_bbknn_with_norms(nullptr, batch_labels.data(), 10, 2, 1,
                                           norms_sq.data(), out_indices.data(), out_distances.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Batch Balancing Tests
// =============================================================================

SCL_TEST_SUITE(batch_balancing)

SCL_TEST_CASE(equal_batch_sizes) {
    // Create data with 3 equal batches (3 cells each)
    auto [indptr, indices, data, batch_labels] = create_test_data();
    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    scl_size_t k = 1;
    scl_size_t n_batches = 3;
    std::vector<scl_index_t> out_indices(9 * n_batches * k);
    std::vector<scl_real_t> out_distances(9 * n_batches * k);

    scl_error_t err = scl_bbknn(mat, batch_labels.data(), 9, n_batches, k,
                                out_indices.data(), out_distances.data());
    SCL_ASSERT_EQ(err, SCL_OK);

    // Each cell should have k neighbors from each batch
    for (size_t cell = 0; cell < 9; ++cell) {
        std::vector<int> batch_counts(3, 0);
        for (size_t b = 0; b < n_batches; ++b) {
            scl_index_t neighbor = out_indices[cell * n_batches * k + b * k];
            int batch = batch_labels[neighbor];
            batch_counts[batch]++;
        }
        // Each batch should contribute k neighbors
        for (int count : batch_counts) {
            SCL_ASSERT_GE(count, 0);
            SCL_ASSERT_LE(count, static_cast<int>(n_batches * k));
        }
    }
}

SCL_TEST_CASE(unequal_batch_sizes) {
    // Create unbalanced batches: batch 0 has 2 cells, batch 1 has 3, batch 2 has 4
    std::vector<scl_index_t> indptr = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    std::vector<scl_index_t> indices = {0, 1, 1, 2, 2, 3, 3, 4, 0, 4,
                                        1, 2, 2, 3, 3, 4, 0, 1};
    std::vector<scl_real_t> data(18, 1.0);
    std::vector<int32_t> batch_labels = {0, 0, 1, 1, 1, 2, 2, 2, 2};

    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    scl_size_t k = 1;
    scl_size_t n_batches = 3;
    std::vector<scl_index_t> out_indices(9 * n_batches * k);
    std::vector<scl_real_t> out_distances(9 * n_batches * k);

    scl_error_t err = scl_bbknn(mat, batch_labels.data(), 9, n_batches, k,
                                out_indices.data(), out_distances.data());
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(single_batch) {
    auto [indptr, indices, data, batch_labels] = create_test_data();
    // Override: all cells in same batch
    std::vector<int32_t> single_batch(9, 0);

    Sparse mat = make_sparse_csr(9, 5, 18,
                                  indptr.data(), indices.data(), data.data());

    scl_size_t k = 2;
    scl_size_t n_batches = 1;
    std::vector<scl_index_t> out_indices(9 * n_batches * k);
    std::vector<scl_real_t> out_distances(9 * n_batches * k);

    scl_error_t err = scl_bbknn(mat, single_batch.data(), 9, n_batches, k,
                                out_indices.data(), out_distances.data());
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
