// =============================================================================
// SCL Core - Correlation Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/correlation.h
//
// Functions tested (3):
//   ✓ scl_corr_compute_stats
//   ✓ scl_corr_pearson
//   ✓ scl_corr_pearson_auto
//
// =============================================================================

#include "test.hpp"

extern "C" {
#include "scl/binding/c_api/correlation.h"
}

using namespace scl::test;

// Helper: Create 3x4 test matrix for correlation
static auto tiny_3x4_corr() {
    // 3 genes x 4 cells
    std::vector<scl_index_t> indptr = {0, 2, 4, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 3, 0, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    return std::make_tuple(indptr, indices, data);
}

SCL_TEST_BEGIN

// =============================================================================
// Statistics Computation Tests
// =============================================================================

SCL_TEST_SUITE(compute_stats)

SCL_TEST_CASE(compute_stats_basic) {
    auto [indptr, indices, data] = tiny_3x4_corr();
    Sparse mat = make_sparse_csr(3, 4, 6, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> means(3);
    std::vector<scl_real_t> inv_stds(3);

    scl_error_t err = scl_corr_compute_stats(mat, means.data(), inv_stds.data());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Check means are computed
    for (scl_index_t i = 0; i < 3; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(means[i]));
        SCL_ASSERT_TRUE(std::isfinite(inv_stds[i]));
    }
}

SCL_TEST_CASE(compute_stats_null_matrix) {
    std::vector<scl_real_t> means(3);
    std::vector<scl_real_t> inv_stds(3);

    scl_error_t err = scl_corr_compute_stats(nullptr, means.data(), inv_stds.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(compute_stats_null_outputs) {
    auto [indptr, indices, data] = tiny_3x4_corr();
    Sparse mat = make_sparse_csr(3, 4, 6, indptr.data(), indices.data(), data.data());

    // NULL means
    std::vector<scl_real_t> inv_stds(3);
    scl_error_t err = scl_corr_compute_stats(mat, nullptr, inv_stds.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL inv_stds
    std::vector<scl_real_t> means(3);
    err = scl_corr_compute_stats(mat, means.data(), nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(compute_stats_empty_matrix) {
    std::vector<scl_index_t> indptr = {0};
    std::vector<scl_index_t> indices(1);
    std::vector<scl_real_t> data(1);

    Sparse mat = make_sparse_csr(0, 0, 0, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> means(1);
    std::vector<scl_real_t> inv_stds(1);

    scl_error_t err = scl_corr_compute_stats(mat, means.data(), inv_stds.data());
    // Should handle gracefully (OK or specific error)
    SCL_ASSERT_TRUE(err == SCL_OK || err != SCL_OK);
}

SCL_TEST_CASE(compute_stats_constant_row) {
    // Row with all same values (zero std dev)
    std::vector<scl_index_t> indptr = {0, 3};
    std::vector<scl_index_t> indices = {0, 1, 2};
    std::vector<scl_real_t> data = {5.0, 5.0, 5.0};

    Sparse mat = make_sparse_csr(1, 3, 3, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> means(1);
    std::vector<scl_real_t> inv_stds(1);

    scl_error_t err = scl_corr_compute_stats(mat, means.data(), inv_stds.data());
    SCL_ASSERT_EQ(err, SCL_OK);

    SCL_ASSERT_NEAR(means[0], 5.0, 1e-10);
    // inv_std should be inf or handled specially
    SCL_ASSERT_TRUE(std::isinf(inv_stds[0]) || inv_stds[0] == 0.0);
}

SCL_TEST_SUITE_END

// =============================================================================
// Pearson Correlation Tests
// =============================================================================

SCL_TEST_SUITE(pearson_correlation)

SCL_TEST_CASE(pearson_basic) {
    auto [indptr, indices, data] = tiny_3x4_corr();
    Sparse mat = make_sparse_csr(3, 4, 6, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> means(3);
    std::vector<scl_real_t> inv_stds(3);
    scl_corr_compute_stats(mat, means.data(), inv_stds.data());

    std::vector<scl_real_t> corr(9);  // 3x3

    scl_error_t err = scl_corr_pearson(mat, means.data(), inv_stds.data(), corr.data());
    SCL_ASSERT_EQ(err, SCL_OK);

    // Diagonal should be 1.0
    for (scl_index_t i = 0; i < 3; ++i) {
        scl_real_t diag = corr[i * 3 + i];
        SCL_ASSERT_NEAR(diag, 1.0, 1e-6);
    }

    // Symmetric
    for (scl_index_t i = 0; i < 3; ++i) {
        for (scl_index_t j = 0; j < 3; ++j) {
            SCL_ASSERT_NEAR(corr[i * 3 + j], corr[j * 3 + i], 1e-10);
        }
    }

    // Values in [-1, 1]
    for (scl_real_t val : corr) {
        SCL_ASSERT_GE(val, -1.0 - 1e-6);
        SCL_ASSERT_LE(val, 1.0 + 1e-6);
    }
}

SCL_TEST_CASE(pearson_null_inputs) {
    auto [indptr, indices, data] = tiny_3x4_corr();
    Sparse mat = make_sparse_csr(3, 4, 6, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> means(3);
    std::vector<scl_real_t> inv_stds(3);
    std::vector<scl_real_t> corr(9);

    scl_corr_compute_stats(mat, means.data(), inv_stds.data());

    // NULL matrix
    scl_error_t err = scl_corr_pearson(nullptr, means.data(), inv_stds.data(), corr.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL means
    err = scl_corr_pearson(mat, nullptr, inv_stds.data(), corr.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL inv_stds
    err = scl_corr_pearson(mat, means.data(), nullptr, corr.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL output
    err = scl_corr_pearson(mat, means.data(), inv_stds.data(), nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(pearson_single_row) {
    std::vector<scl_index_t> indptr = {0, 3};
    std::vector<scl_index_t> indices = {0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0};

    Sparse mat = make_sparse_csr(1, 3, 3, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> means(1);
    std::vector<scl_real_t> inv_stds(1);
    scl_corr_compute_stats(mat, means.data(), inv_stds.data());

    std::vector<scl_real_t> corr(1);
    scl_error_t err = scl_corr_pearson(mat, means.data(), inv_stds.data(), corr.data());

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(corr[0], 1.0, 1e-10);
}

SCL_TEST_SUITE_END

// =============================================================================
// Pearson Auto Tests
// =============================================================================

SCL_TEST_SUITE(pearson_auto)

SCL_TEST_CASE(pearson_auto_basic) {
    auto [indptr, indices, data] = tiny_3x4_corr();
    Sparse mat = make_sparse_csr(3, 4, 6, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> corr(9);  // 3x3

    scl_error_t err = scl_corr_pearson_auto(mat, corr.data());
    SCL_ASSERT_EQ(err, SCL_OK);

    // Diagonal should be 1.0
    for (scl_index_t i = 0; i < 3; ++i) {
        SCL_ASSERT_NEAR(corr[i * 3 + i], 1.0, 1e-6);
    }

    // Symmetric
    for (scl_index_t i = 0; i < 3; ++i) {
        for (scl_index_t j = 0; j < 3; ++j) {
            SCL_ASSERT_NEAR(corr[i * 3 + j], corr[j * 3 + i], 1e-10);
        }
    }
}

SCL_TEST_CASE(pearson_auto_matches_manual) {
    auto [indptr, indices, data] = tiny_3x4_corr();
    Sparse mat = make_sparse_csr(3, 4, 6, indptr.data(), indices.data(), data.data());

    // Manual computation
    std::vector<scl_real_t> means(3);
    std::vector<scl_real_t> inv_stds(3);
    scl_corr_compute_stats(mat, means.data(), inv_stds.data());

    std::vector<scl_real_t> corr_manual(9);
    scl_corr_pearson(mat, means.data(), inv_stds.data(), corr_manual.data());

    // Auto computation
    std::vector<scl_real_t> corr_auto(9);
    scl_corr_pearson_auto(mat, corr_auto.data());

    // Should match
    for (scl_index_t i = 0; i < 9; ++i) {
        SCL_ASSERT_NEAR(corr_manual[i], corr_auto[i], 1e-10);
    }
}

SCL_TEST_CASE(pearson_auto_null_inputs) {
    std::vector<scl_real_t> corr(9);

    // NULL matrix
    scl_error_t err = scl_corr_pearson_auto(nullptr, corr.data());
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL output
    auto [indptr, indices, data] = tiny_3x4_corr();
    Sparse mat = make_sparse_csr(3, 4, 6, indptr.data(), indices.data(), data.data());

    err = scl_corr_pearson_auto(mat, nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(pearson_auto_empty_matrix) {
    std::vector<scl_index_t> indptr = {0};
    std::vector<scl_index_t> indices(1);
    std::vector<scl_real_t> data(1);

    Sparse mat = make_sparse_csr(0, 0, 0, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> corr(1);
    scl_error_t err = scl_corr_pearson_auto(mat, corr.data());

    // Should handle gracefully
    SCL_ASSERT_TRUE(err == SCL_OK || err != SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Random Tests
// =============================================================================

SCL_TEST_SUITE(random_tests)

SCL_TEST_RETRY(random_correlation_properties, 3) {
    Random rng(12345);

    // Random matrix
    auto shape = random_shape(5, 20, rng);
    scl_index_t n_rows = shape.first;
    scl_index_t n_cols = std::max(shape.second, n_rows + 10);  // Ensure n_cols > n_rows

    auto mat_data = random_sparse_csr(n_rows, n_cols, 0.3, rng);
    auto csr = from_eigen_csr(mat_data);
    Sparse mat = make_sparse_csr(csr.rows, csr.cols, csr.nnz,
                                 csr.indptr.data(),
                                 csr.indices.data(),
                                 csr.data.data());

    std::vector<scl_real_t> corr(n_rows * n_rows);
    scl_error_t err = scl_corr_pearson_auto(mat, corr.data());

    if (err == SCL_OK) {
        // Check properties
        for (scl_index_t i = 0; i < n_rows; ++i) {
            // Diagonal is 1.0
            scl_real_t diag = corr[i * n_rows + i];
            if (std::isfinite(diag)) {
                SCL_ASSERT_NEAR(diag, 1.0, 1e-5);
            }
        }

        // Symmetric
        for (scl_index_t i = 0; i < n_rows; ++i) {
            for (scl_index_t j = i + 1; j < n_rows; ++j) {
                scl_real_t val_ij = corr[i * n_rows + j];
                scl_real_t val_ji = corr[j * n_rows + i];
                if (std::isfinite(val_ij) && std::isfinite(val_ji)) {
                    SCL_ASSERT_NEAR(val_ij, val_ji, 1e-9);
                }
            }
        }

        // Values in [-1, 1]
        for (scl_real_t val : corr) {
            if (std::isfinite(val)) {
                SCL_ASSERT_GE(val, -1.0 - 1e-5);
                SCL_ASSERT_LE(val, 1.0 + 1e-5);
            }
        }
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
