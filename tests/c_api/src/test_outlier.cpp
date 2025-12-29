// =============================================================================
// SCL Core - Outlier Detection Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/outlier.h
//
// Functions tested:
//   ✓ scl_outlier_isolation_score
//   ✓ scl_outlier_ambient_detection
//   ✓ scl_outlier_empty_drops
//   ✓ scl_outlier_outlier_genes
//   ✓ scl_outlier_doublet_score
//   ✓ scl_outlier_mitochondrial_outliers
//   ✓ scl_outlier_qc_filter
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/outlier.h"

using namespace scl::test;
using precision::Tolerance;

SCL_TEST_BEGIN

// =============================================================================
// Isolation Score Tests
// =============================================================================

SCL_TEST_SUITE(isolation_score)

SCL_TEST_CASE(isolation_score_basic) {
    // Create a simple expression matrix
    auto mat_eigen = random_sparse_csr(20, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> scores(20, 0.0);
    
    scl_error_t err = scl_outlier_isolation_score(mat, scores.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Scores should be non-negative
    for (scl_index_t i = 0; i < 20; ++i) {
        SCL_ASSERT_GE(scores[i], 0.0);
    }
}

SCL_TEST_CASE(isolation_score_null_matrix) {
    std::vector<scl_real_t> scores(10);
    
    scl_error_t err = scl_outlier_isolation_score(nullptr, scores.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(isolation_score_null_output) {
    auto mat_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_outlier_isolation_score(mat, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(isolation_score_empty_matrix) {
    // Zero matrix
    auto Z = zero_csr(10, 20);
    auto csr = from_eigen_csr(Z);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> scores(10, 0.0);
    
    scl_error_t err = scl_outlier_isolation_score(mat, scores.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All scores should be similar (or zero)
    // (Implementation dependent)
}

SCL_TEST_RETRY(isolation_score_random, 3)
{
    Random rng(42);
    auto [rows, cols] = random_shape(30, 100, rng);
    double density = random_density(0.05, 0.15, rng);
    
    auto mat_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> scores(rows, 0.0);
    
    scl_error_t err = scl_outlier_isolation_score(mat, scores.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify scores are reasonable
    for (scl_index_t i = 0; i < rows; ++i) {
        SCL_ASSERT_GE(scores[i], 0.0);
        SCL_ASSERT_TRUE(std::isfinite(scores[i]));
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Ambient RNA Detection Tests
// =============================================================================

SCL_TEST_SUITE(ambient_detection)

SCL_TEST_CASE(ambient_detection_basic) {
    auto mat_eigen = random_sparse_csr(20, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> scores(20, 0.0);
    
    scl_error_t err = scl_outlier_ambient_detection(mat, scores.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Scores should be non-negative
    for (scl_index_t i = 0; i < 20; ++i) {
        SCL_ASSERT_GE(scores[i], 0.0);
    }
}

SCL_TEST_CASE(ambient_detection_null_matrix) {
    std::vector<scl_real_t> scores(10);
    
    scl_error_t err = scl_outlier_ambient_detection(nullptr, scores.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(ambient_detection_null_output) {
    auto mat_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_outlier_ambient_detection(mat, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Empty Droplet Detection Tests
// =============================================================================

SCL_TEST_SUITE(empty_drops)

SCL_TEST_CASE(empty_drops_basic) {
    // Create matrix with some empty-like cells (very low counts)
    auto mat_eigen = random_sparse_csr(20, 50, 0.05);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<unsigned char> is_empty(20, 0);
    
    scl_error_t err = scl_outlier_empty_drops(mat, is_empty.data(), 0.01);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Some cells might be marked as empty (implementation dependent)
    // Just verify no crash
}

SCL_TEST_CASE(empty_drops_null_matrix) {
    std::vector<unsigned char> is_empty(10, 0);
    
    scl_error_t err = scl_outlier_empty_drops(nullptr, is_empty.data(), 0.01);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(empty_drops_null_output) {
    auto mat_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_outlier_empty_drops(mat, nullptr, 0.01);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(empty_drops_high_threshold) {
    auto mat_eigen = random_sparse_csr(20, 50, 0.2);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<unsigned char> is_empty(20, 0);
    
    // Very high threshold should mark more cells as empty
    scl_error_t err = scl_outlier_empty_drops(mat, is_empty.data(), 0.5);
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(empty_drops_zero_threshold) {
    auto mat_eigen = random_sparse_csr(20, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<unsigned char> is_empty(20, 0);
    
    // Zero threshold should mark few cells as empty
    scl_error_t err = scl_outlier_empty_drops(mat, is_empty.data(), 0.0);
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Outlier Genes Tests
// =============================================================================

SCL_TEST_SUITE(outlier_genes)

SCL_TEST_CASE(outlier_genes_basic) {
    auto mat_eigen = random_sparse_csr(100, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indices(10, 0);
    scl_size_t n_outliers = 0;
    
    scl_error_t err = scl_outlier_outlier_genes(
        mat, indices.data(), 10, &n_outliers, 2.0
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // n_outliers should be <= max_outliers
    SCL_ASSERT_LE(n_outliers, 10);
    
    // Indices should be valid
    for (scl_size_t i = 0; i < n_outliers; ++i) {
        SCL_ASSERT_GE(indices[i], static_cast<scl_index_t>(0));
        SCL_ASSERT_LT(indices[i], static_cast<scl_index_t>(50));
    }
}

SCL_TEST_CASE(outlier_genes_null_matrix) {
    std::vector<scl_index_t> indices(10, 0);
    scl_size_t n_outliers = 0;
    
    scl_error_t err = scl_outlier_outlier_genes(
        nullptr, indices.data(), 10, &n_outliers, 2.0
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(outlier_genes_null_indices) {
    auto mat_eigen = random_sparse_csr(100, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t n_outliers = 0;
    
    scl_error_t err = scl_outlier_outlier_genes(
        mat, nullptr, 10, &n_outliers, 2.0
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(outlier_genes_null_count) {
    auto mat_eigen = random_sparse_csr(100, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indices(10, 0);
    
    scl_error_t err = scl_outlier_outlier_genes(
        mat, indices.data(), 10, nullptr, 2.0
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(outlier_genes_zero_max) {
    auto mat_eigen = random_sparse_csr(100, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indices(1, 0);
    scl_size_t n_outliers = 0;
    
    scl_error_t err = scl_outlier_outlier_genes(
        mat, indices.data(), 0, &n_outliers, 2.0
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(n_outliers, 0);
}

SCL_TEST_CASE(outlier_genes_high_threshold) {
    auto mat_eigen = random_sparse_csr(100, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indices(10, 0);
    scl_size_t n_outliers = 0;
    
    // Very high threshold should find fewer outliers
    scl_error_t err = scl_outlier_outlier_genes(
        mat, indices.data(), 10, &n_outliers, 10.0
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_LE(n_outliers, 10);
}

SCL_TEST_SUITE_END

// =============================================================================
// Doublet Score Tests
// =============================================================================

SCL_TEST_SUITE(doublet_score)

SCL_TEST_CASE(doublet_score_basic) {
    auto expr_eigen = random_sparse_csr(20, 50, 0.1);
    auto expr_csr = from_eigen_csr(expr_eigen);
    Sparse expr = make_sparse_csr(
        expr_csr.rows, expr_csr.cols, expr_csr.nnz,
        expr_csr.indptr.data(), expr_csr.indices.data(), expr_csr.data.data()
    );
    
    // Create neighbor graph
    auto neighbors_eigen = random_sparse_csr(20, 20, 0.2);
    auto neighbors_csr = from_eigen_csr(neighbors_eigen);
    Sparse neighbors = make_sparse_csr(
        neighbors_csr.rows, neighbors_csr.cols, neighbors_csr.nnz,
        neighbors_csr.indptr.data(), neighbors_csr.indices.data(),
        neighbors_csr.data.data()
    );
    
    std::vector<scl_real_t> scores(20, 0.0);
    
    scl_error_t err = scl_outlier_doublet_score(expr, neighbors, scores.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Scores should be non-negative
    for (scl_index_t i = 0; i < 20; ++i) {
        SCL_ASSERT_GE(scores[i], 0.0);
    }
}

SCL_TEST_CASE(doublet_score_null_expression) {
    auto neighbors_eigen = random_sparse_csr(10, 10, 0.2);
    auto neighbors_csr = from_eigen_csr(neighbors_eigen);
    Sparse neighbors = make_sparse_csr(
        neighbors_csr.rows, neighbors_csr.cols, neighbors_csr.nnz,
        neighbors_csr.indptr.data(), neighbors_csr.indices.data(),
        neighbors_csr.data.data()
    );
    
    std::vector<scl_real_t> scores(10);
    
    scl_error_t err = scl_outlier_doublet_score(nullptr, neighbors, scores.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(doublet_score_null_neighbors) {
    auto expr_eigen = random_sparse_csr(10, 20, 0.1);
    auto expr_csr = from_eigen_csr(expr_eigen);
    Sparse expr = make_sparse_csr(
        expr_csr.rows, expr_csr.cols, expr_csr.nnz,
        expr_csr.indptr.data(), expr_csr.indices.data(), expr_csr.data.data()
    );
    
    std::vector<scl_real_t> scores(10);
    
    scl_error_t err = scl_outlier_doublet_score(expr, nullptr, scores.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(doublet_score_null_output) {
    auto expr_eigen = random_sparse_csr(10, 20, 0.1);
    auto expr_csr = from_eigen_csr(expr_eigen);
    Sparse expr = make_sparse_csr(
        expr_csr.rows, expr_csr.cols, expr_csr.nnz,
        expr_csr.indptr.data(), expr_csr.indices.data(), expr_csr.data.data()
    );
    
    auto neighbors_eigen = random_sparse_csr(10, 10, 0.2);
    auto neighbors_csr = from_eigen_csr(neighbors_eigen);
    Sparse neighbors = make_sparse_csr(
        neighbors_csr.rows, neighbors_csr.cols, neighbors_csr.nnz,
        neighbors_csr.indptr.data(), neighbors_csr.indices.data(),
        neighbors_csr.data.data()
    );
    
    scl_error_t err = scl_outlier_doublet_score(expr, neighbors, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Mitochondrial Outliers Tests
// =============================================================================

SCL_TEST_SUITE(mitochondrial_outliers)

SCL_TEST_CASE(mitochondrial_outliers_basic) {
    auto mat_eigen = random_sparse_csr(20, 100, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Mark genes 0-4 as mitochondrial
    std::vector<scl_index_t> mito_genes = {0, 1, 2, 3, 4};
    
    std::vector<scl_real_t> mito_fraction(20, 0.0);
    std::vector<unsigned char> is_outlier(20, 0);
    
    scl_error_t err = scl_outlier_mitochondrial_outliers(
        mat, mito_genes.data(), mito_genes.size(),
        mito_fraction.data(), is_outlier.data(), 0.2
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Fractions should be in [0, 1]
    for (scl_index_t i = 0; i < 20; ++i) {
        SCL_ASSERT_GE(mito_fraction[i], 0.0);
        SCL_ASSERT_LE(mito_fraction[i], 1.0);
    }
}

SCL_TEST_CASE(mitochondrial_outliers_null_matrix) {
    std::vector<scl_index_t> mito_genes = {0, 1, 2};
    std::vector<scl_real_t> mito_fraction(10);
    std::vector<unsigned char> is_outlier(10);
    
    scl_error_t err = scl_outlier_mitochondrial_outliers(
        nullptr, mito_genes.data(), mito_genes.size(),
        mito_fraction.data(), is_outlier.data(), 0.2
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(mitochondrial_outliers_null_genes) {
    auto mat_eigen = random_sparse_csr(10, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> mito_fraction(10);
    std::vector<unsigned char> is_outlier(10);
    
    scl_error_t err = scl_outlier_mitochondrial_outliers(
        mat, nullptr, 5,
        mito_fraction.data(), is_outlier.data(), 0.2
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(mitochondrial_outliers_empty_mito) {
    auto mat_eigen = random_sparse_csr(10, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> mito_genes;  // Empty
    std::vector<scl_real_t> mito_fraction(10);
    std::vector<unsigned char> is_outlier(10);
    
    scl_error_t err = scl_outlier_mitochondrial_outliers(
        mat, mito_genes.data(), 0,
        mito_fraction.data(), is_outlier.data(), 0.2
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All fractions should be zero
    for (scl_index_t i = 0; i < 10; ++i) {
        SCL_ASSERT_NEAR(mito_fraction[i], 0.0, 1e-10);
    }
}

SCL_TEST_CASE(mitochondrial_outliers_high_threshold) {
    auto mat_eigen = random_sparse_csr(20, 100, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> mito_genes = {0, 1, 2, 3, 4};
    std::vector<scl_real_t> mito_fraction(20);
    std::vector<unsigned char> is_outlier(20, 0);
    
    // Very high threshold should mark fewer cells
    scl_error_t err = scl_outlier_mitochondrial_outliers(
        mat, mito_genes.data(), mito_genes.size(),
        mito_fraction.data(), is_outlier.data(), 0.9
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// QC Filter Tests
// =============================================================================

SCL_TEST_SUITE(qc_filter)

SCL_TEST_CASE(qc_filter_basic) {
    auto mat_eigen = random_sparse_csr(20, 100, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> mito_genes = {0, 1, 2, 3, 4};
    std::vector<unsigned char> pass_qc(20, 0);
    
    scl_error_t err = scl_outlier_qc_filter(
        mat,
        10.0,    // min_genes
        5000.0,  // max_genes
        100.0,   // min_counts
        100000.0, // max_counts
        0.2,     // max_mito_fraction
        mito_genes.data(), mito_genes.size(),
        pass_qc.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Some cells should pass QC (implementation dependent)
    // Just verify no crash
}

SCL_TEST_CASE(qc_filter_null_matrix) {
    std::vector<scl_index_t> mito_genes = {0, 1, 2};
    std::vector<unsigned char> pass_qc(10);
    
    scl_error_t err = scl_outlier_qc_filter(
        nullptr,
        10.0, 5000.0, 100.0, 100000.0, 0.2,
        mito_genes.data(), mito_genes.size(),
        pass_qc.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(qc_filter_null_output) {
    auto mat_eigen = random_sparse_csr(10, 50, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> mito_genes = {0, 1, 2};
    
    scl_error_t err = scl_outlier_qc_filter(
        mat,
        10.0, 5000.0, 100.0, 100000.0, 0.2,
        mito_genes.data(), mito_genes.size(),
        nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(qc_filter_strict) {
    auto mat_eigen = random_sparse_csr(20, 100, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> mito_genes = {0, 1, 2};
    std::vector<unsigned char> pass_qc(20, 0);
    
    // Very strict criteria
    scl_error_t err = scl_outlier_qc_filter(
        mat,
        1000.0,   // min_genes (very high)
        2000.0,   // max_genes (very low)
        10000.0,  // min_counts (very high)
        20000.0,  // max_counts (very low)
        0.01,     // max_mito_fraction (very low)
        mito_genes.data(), mito_genes.size(),
        pass_qc.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Fewer cells should pass with strict criteria
}

SCL_TEST_CASE(qc_filter_lenient) {
    auto mat_eigen = random_sparse_csr(20, 100, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> mito_genes = {0, 1, 2};
    std::vector<unsigned char> pass_qc(20, 0);
    
    // Very lenient criteria
    scl_error_t err = scl_outlier_qc_filter(
        mat,
        0.0,      // min_genes (no minimum)
        100000.0, // max_genes (very high)
        0.0,      // min_counts (no minimum)
        1000000.0, // max_counts (very high)
        1.0,      // max_mito_fraction (very high)
        mito_genes.data(), mito_genes.size(),
        pass_qc.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // More cells should pass with lenient criteria
}

SCL_TEST_CASE(qc_filter_no_mito) {
    auto mat_eigen = random_sparse_csr(20, 100, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> mito_genes;  // Empty
    std::vector<unsigned char> pass_qc(20, 0);
    
    scl_error_t err = scl_outlier_qc_filter(
        mat,
        10.0, 5000.0, 100.0, 100000.0, 0.2,
        mito_genes.data(), 0,
        pass_qc.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

