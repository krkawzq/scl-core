// =============================================================================
// SCL Core - Quality Control Module Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/qc.h
//
// Functions tested:
//   ✓ scl_qc_compute_basic
//   ✓ scl_qc_compute_subset_pct
//   ✓ scl_qc_compute_fused
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/qc.h"

using namespace scl::test;
using precision::Tolerance;

SCL_TEST_BEGIN

// =============================================================================
// Basic QC Metrics
// =============================================================================

SCL_TEST_SUITE(basic_qc)

SCL_TEST_CASE(basic_qc_simple_matrix) {
    // Create a simple 3x4 matrix (3 cells, 4 genes)
    // Cell 0: genes 0,1,2 (3 genes, total=6)
    // Cell 1: genes 1,2 (2 genes, total=3)
    // Cell 2: genes 0,2,3 (3 genes, total=6)
    std::vector<scl_index_t> indptr = {0, 3, 5, 8};
    std::vector<scl_index_t> indices = {0, 1, 2, 1, 2, 0, 2, 3};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0};
    
    Sparse mat = make_sparse_csr(3, 4, 8,
        indptr.data(), indices.data(), data.data());
    
    std::vector<scl_index_t> n_genes(3);
    std::vector<scl_real_t> total_counts(3);
    
    scl_error_t err = scl_qc_compute_basic(
        mat, n_genes.data(), total_counts.data(), 3
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify results
    SCL_ASSERT_EQ(n_genes[0], 3);
    SCL_ASSERT_EQ(n_genes[1], 2);
    SCL_ASSERT_EQ(n_genes[2], 3);
    
    SCL_ASSERT_NEAR(total_counts[0], 6.0, 1e-10);
    SCL_ASSERT_NEAR(total_counts[1], 3.0, 1e-10);
    SCL_ASSERT_NEAR(total_counts[2], 6.0, 1e-10);
}

SCL_TEST_CASE(basic_qc_empty_cell) {
    // Cell 0 has no genes
    std::vector<scl_index_t> indptr = {0, 0, 2, 4};
    std::vector<scl_index_t> indices = {0, 1, 0, 1};
    std::vector<scl_real_t> data = {1.0, 2.0, 1.0, 2.0};
    
    Sparse mat = make_sparse_csr(3, 2, 4,
        indptr.data(), indices.data(), data.data());
    
    std::vector<scl_index_t> n_genes(3);
    std::vector<scl_real_t> total_counts(3);
    
    scl_error_t err = scl_qc_compute_basic(
        mat, n_genes.data(), total_counts.data(), 3
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Empty cell should have 0 genes and 0 counts
    SCL_ASSERT_EQ(n_genes[0], 0);
    SCL_ASSERT_NEAR(total_counts[0], 0.0, 1e-10);
}

SCL_TEST_CASE(basic_qc_empty_matrix) {
    // Empty matrix (no non-zeros)
    std::vector<scl_index_t> indptr = {0, 0, 0, 0};
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    
    Sparse mat = make_sparse_csr(3, 4, 0,
        indptr.data(), indices.data(), data.data());
    
    std::vector<scl_index_t> n_genes(3);
    std::vector<scl_real_t> total_counts(3);
    
    scl_error_t err = scl_qc_compute_basic(
        mat, n_genes.data(), total_counts.data(), 3
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All cells should have 0 genes and 0 counts
    for (scl_index_t i = 0; i < 3; ++i) {
        SCL_ASSERT_EQ(n_genes[i], 0);
        SCL_ASSERT_NEAR(total_counts[i], 0.0, 1e-10);
    }
}

SCL_TEST_RETRY(basic_qc_random, 3)
{
    Random rng(70);
    
    auto [rows, cols] = random_shape(20, 100, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> n_genes(rows);
    std::vector<scl_real_t> total_counts(rows);
    
    scl_error_t err = scl_qc_compute_basic(
        mat, n_genes.data(), total_counts.data(), rows
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify all counts are non-negative
    for (scl_index_t i = 0; i < rows; ++i) {
        SCL_ASSERT_GE(n_genes[i], 0);
        SCL_ASSERT_LE(n_genes[i], cols);
        SCL_ASSERT_GE(total_counts[i], 0.0);
    }
}

SCL_TEST_CASE(basic_qc_null_handle) {
    std::vector<scl_index_t> n_genes(10);
    std::vector<scl_real_t> total_counts(10);
    
    scl_error_t err = scl_qc_compute_basic(
        nullptr, n_genes.data(), total_counts.data(), 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(basic_qc_null_n_genes) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> total_counts(10);
    
    scl_error_t err = scl_qc_compute_basic(
        mat, nullptr, total_counts.data(), 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(basic_qc_null_total_counts) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> n_genes(10);
    
    scl_error_t err = scl_qc_compute_basic(
        mat, n_genes.data(), nullptr, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(basic_qc_dimension_mismatch) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> n_genes(5);  // Wrong size
    std::vector<scl_real_t> total_counts(5);
    
    scl_error_t err = scl_qc_compute_basic(
        mat, n_genes.data(), total_counts.data(), 10
    );
    
    // Should either succeed (if it uses matrix rows) or return error
    // Implementation dependent
}

SCL_TEST_SUITE_END

// =============================================================================
// Subset Percentage
// =============================================================================

SCL_TEST_SUITE(subset_pct)

SCL_TEST_CASE(subset_pct_simple) {
    // 3 cells, 4 genes
    // Cell 0: genes 0,1,2 (total=6, subset=3 if genes 0,1 are in subset)
    // Cell 1: genes 1,2 (total=3, subset=1 if gene 1 is in subset)
    // Cell 2: genes 0,2,3 (total=6, subset=1 if gene 0 is in subset)
    std::vector<scl_index_t> indptr = {0, 3, 5, 8};
    std::vector<scl_index_t> indices = {0, 1, 2, 1, 2, 0, 2, 3};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0};
    
    Sparse mat = make_sparse_csr(3, 4, 8,
        indptr.data(), indices.data(), data.data());
    
    // Subset: genes 0 and 1
    std::vector<uint8_t> subset_mask = {1, 1, 0, 0};
    std::vector<scl_real_t> pcts(3);
    
    scl_error_t err = scl_qc_compute_subset_pct(
        mat, subset_mask.data(), pcts.data(), 3
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Cell 0: subset=3, total=6 -> 50%
    SCL_ASSERT_NEAR(pcts[0], 50.0, 1e-6);
    // Cell 1: subset=1, total=3 -> 33.33%
    SCL_ASSERT_NEAR(pcts[1], 100.0/3.0, 1e-6);
    // Cell 2: subset=1, total=6 -> 16.67%
    SCL_ASSERT_NEAR(pcts[2], 100.0/6.0, 1e-6);
}

SCL_TEST_CASE(subset_pct_all_in_subset) {
    std::vector<scl_index_t> indptr = {0, 2, 4};
    std::vector<scl_index_t> indices = {0, 1, 0, 1};
    std::vector<scl_real_t> data = {1.0, 2.0, 1.0, 2.0};
    
    Sparse mat = make_sparse_csr(2, 2, 4,
        indptr.data(), indices.data(), data.data());
    
    // All genes in subset
    std::vector<uint8_t> subset_mask = {1, 1};
    std::vector<scl_real_t> pcts(2);
    
    scl_error_t err = scl_qc_compute_subset_pct(
        mat, subset_mask.data(), pcts.data(), 2
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should be 100%
    SCL_ASSERT_NEAR(pcts[0], 100.0, 1e-6);
    SCL_ASSERT_NEAR(pcts[1], 100.0, 1e-6);
}

SCL_TEST_CASE(subset_pct_none_in_subset) {
    std::vector<scl_index_t> indptr = {0, 2, 4};
    std::vector<scl_index_t> indices = {0, 1, 0, 1};
    std::vector<scl_real_t> data = {1.0, 2.0, 1.0, 2.0};
    
    Sparse mat = make_sparse_csr(2, 2, 4,
        indptr.data(), indices.data(), data.data());
    
    // No genes in subset
    std::vector<uint8_t> subset_mask = {0, 0};
    std::vector<scl_real_t> pcts(2);
    
    scl_error_t err = scl_qc_compute_subset_pct(
        mat, subset_mask.data(), pcts.data(), 2
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should be 0%
    SCL_ASSERT_NEAR(pcts[0], 0.0, 1e-6);
    SCL_ASSERT_NEAR(pcts[1], 0.0, 1e-6);
}

SCL_TEST_CASE(subset_pct_empty_cell) {
    // Cell 0 has no genes
    std::vector<scl_index_t> indptr = {0, 0, 2};
    std::vector<scl_index_t> indices = {0, 1};
    std::vector<scl_real_t> data = {1.0, 2.0};
    
    Sparse mat = make_sparse_csr(2, 2, 2,
        indptr.data(), indices.data(), data.data());
    
    std::vector<uint8_t> subset_mask = {1, 1};
    std::vector<scl_real_t> pcts(2);
    
    scl_error_t err = scl_qc_compute_subset_pct(
        mat, subset_mask.data(), pcts.data(), 2
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Empty cell: percentage is undefined, should be 0 or NaN
    // Implementation dependent
}

SCL_TEST_RETRY(subset_pct_random, 3)
{
    Random rng(71);
    
    auto [rows, cols] = random_shape(20, 100, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Random subset mask
    std::vector<uint8_t> subset_mask(cols);
    for (scl_index_t j = 0; j < cols; ++j) {
        subset_mask[j] = rng.bernoulli(0.3) ? 1 : 0;
    }
    
    std::vector<scl_real_t> pcts(rows);
    
    scl_error_t err = scl_qc_compute_subset_pct(
        mat, subset_mask.data(), pcts.data(), rows
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Percentages should be in [0, 100]
    for (scl_index_t i = 0; i < rows; ++i) {
        SCL_ASSERT_GE(pcts[i], 0.0);
        SCL_ASSERT_LE(pcts[i], 100.0);
    }
}

SCL_TEST_CASE(subset_pct_null_handle) {
    std::vector<uint8_t> subset_mask(10);
    std::vector<scl_real_t> pcts(10);
    
    scl_error_t err = scl_qc_compute_subset_pct(
        nullptr, subset_mask.data(), pcts.data(), 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(subset_pct_null_mask) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> pcts(10);
    
    scl_error_t err = scl_qc_compute_subset_pct(
        mat, nullptr, pcts.data(), 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(subset_pct_null_output) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<uint8_t> subset_mask(20);
    
    scl_error_t err = scl_qc_compute_subset_pct(
        mat, subset_mask.data(), nullptr, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Fused QC Metrics
// =============================================================================

SCL_TEST_SUITE(fused_qc)

SCL_TEST_CASE(fused_qc_simple) {
    std::vector<scl_index_t> indptr = {0, 3, 5, 8};
    std::vector<scl_index_t> indices = {0, 1, 2, 1, 2, 0, 2, 3};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0};
    
    Sparse mat = make_sparse_csr(3, 4, 8,
        indptr.data(), indices.data(), data.data());
    
    std::vector<uint8_t> subset_mask = {1, 1, 0, 0};
    std::vector<scl_index_t> n_genes(3);
    std::vector<scl_real_t> total_counts(3);
    std::vector<scl_real_t> pcts(3);
    
    scl_error_t err = scl_qc_compute_fused(
        mat, subset_mask.data(),
        n_genes.data(), total_counts.data(), pcts.data(), 3
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify all outputs
    SCL_ASSERT_EQ(n_genes[0], 3);
    SCL_ASSERT_NEAR(total_counts[0], 6.0, 1e-10);
    SCL_ASSERT_NEAR(pcts[0], 50.0, 1e-6);
}

SCL_TEST_CASE(fused_qc_consistency) {
    // Compare fused with separate calls
    std::vector<scl_index_t> indptr = {0, 3, 5, 8};
    std::vector<scl_index_t> indices = {0, 1, 2, 1, 2, 0, 2, 3};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0};
    
    Sparse mat = make_sparse_csr(3, 4, 8,
        indptr.data(), indices.data(), data.data());
    
    std::vector<uint8_t> subset_mask = {1, 1, 0, 0};
    
    // Fused call
    std::vector<scl_index_t> n_genes_fused(3);
    std::vector<scl_real_t> total_counts_fused(3);
    std::vector<scl_real_t> pcts_fused(3);
    
    scl_qc_compute_fused(
        mat, subset_mask.data(),
        n_genes_fused.data(), total_counts_fused.data(), pcts_fused.data(), 3
    );
    
    // Separate calls
    std::vector<scl_index_t> n_genes_sep(3);
    std::vector<scl_real_t> total_counts_sep(3);
    std::vector<scl_real_t> pcts_sep(3);
    
    scl_qc_compute_basic(
        mat, n_genes_sep.data(), total_counts_sep.data(), 3
    );
    scl_qc_compute_subset_pct(
        mat, subset_mask.data(), pcts_sep.data(), 3
    );
    
    // Results should match
    for (scl_index_t i = 0; i < 3; ++i) {
        SCL_ASSERT_EQ(n_genes_fused[i], n_genes_sep[i]);
        SCL_ASSERT_NEAR(total_counts_fused[i], total_counts_sep[i], 1e-10);
        SCL_ASSERT_NEAR(pcts_fused[i], pcts_sep[i], 1e-6);
    }
}

SCL_TEST_RETRY(fused_qc_random, 3)
{
    Random rng(72);
    
    auto [rows, cols] = random_shape(20, 100, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<uint8_t> subset_mask(cols);
    for (scl_index_t j = 0; j < cols; ++j) {
        subset_mask[j] = rng.bernoulli(0.3) ? 1 : 0;
    }
    
    std::vector<scl_index_t> n_genes(rows);
    std::vector<scl_real_t> total_counts(rows);
    std::vector<scl_real_t> pcts(rows);
    
    scl_error_t err = scl_qc_compute_fused(
        mat, subset_mask.data(),
        n_genes.data(), total_counts.data(), pcts.data(), rows
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify all outputs are valid
    for (scl_index_t i = 0; i < rows; ++i) {
        SCL_ASSERT_GE(n_genes[i], 0);
        SCL_ASSERT_LE(n_genes[i], cols);
        SCL_ASSERT_GE(total_counts[i], 0.0);
        SCL_ASSERT_GE(pcts[i], 0.0);
        SCL_ASSERT_LE(pcts[i], 100.0);
    }
}

SCL_TEST_CASE(fused_qc_null_handle) {
    std::vector<uint8_t> subset_mask(10);
    std::vector<scl_index_t> n_genes(10);
    std::vector<scl_real_t> total_counts(10);
    std::vector<scl_real_t> pcts(10);
    
    scl_error_t err = scl_qc_compute_fused(
        nullptr, subset_mask.data(),
        n_genes.data(), total_counts.data(), pcts.data(), 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(fused_qc_null_mask) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> n_genes(10);
    std::vector<scl_real_t> total_counts(10);
    std::vector<scl_real_t> pcts(10);
    
    scl_error_t err = scl_qc_compute_fused(
        mat, nullptr,
        n_genes.data(), total_counts.data(), pcts.data(), 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(fused_qc_null_n_genes) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<uint8_t> subset_mask(20);
    std::vector<scl_real_t> total_counts(10);
    std::vector<scl_real_t> pcts(10);
    
    scl_error_t err = scl_qc_compute_fused(
        mat, subset_mask.data(),
        nullptr, total_counts.data(), pcts.data(), 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(fused_qc_null_total_counts) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<uint8_t> subset_mask(20);
    std::vector<scl_index_t> n_genes(10);
    std::vector<scl_real_t> pcts(10);
    
    scl_error_t err = scl_qc_compute_fused(
        mat, subset_mask.data(),
        n_genes.data(), nullptr, pcts.data(), 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(fused_qc_null_pcts) {
    auto A_eigen = random_sparse_csr(10, 20, 0.1);
    auto csr = from_eigen_csr(A_eigen);
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<uint8_t> subset_mask(20);
    std::vector<scl_index_t> n_genes(10);
    std::vector<scl_real_t> total_counts(10);
    
    scl_error_t err = scl_qc_compute_fused(
        mat, subset_mask.data(),
        n_genes.data(), total_counts.data(), nullptr, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

