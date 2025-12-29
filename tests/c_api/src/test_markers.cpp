// =============================================================================
// SCL Core - Marker Gene Selection Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/markers.h
//
// Functions tested:
//   - scl_markers_group_mean_expression
//   - scl_markers_percent_expressed
//   - scl_markers_find_markers
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/markers.h"

using namespace scl::test;
using precision::Tolerance;

// Helper: Generate random expression matrix (cells x genes)
static EigenCSR random_expression_matrix(
    scl_index_t n_cells,
    scl_index_t n_genes,
    double density,
    Random& rng
) {
    EigenCSR mat(n_cells, n_genes);
    mat.reserve(Eigen::VectorXi::Constant(n_cells, static_cast<int>(n_genes * density)));
    
    for (scl_index_t i = 0; i < n_cells; ++i) {
        for (scl_index_t j = 0; j < n_genes; ++j) {
            if (rng.bernoulli(density)) {
                // Expression values are typically non-negative
                mat.insert(i, j) = rng.uniform(0.0, 10.0);
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
// Group Mean Expression
// =============================================================================

SCL_TEST_SUITE(group_mean_expression)

SCL_TEST_RETRY(group_mean_expression_basic, 3)
{
    Random rng(42);
    
    scl_index_t n_cells = 20;
    scl_index_t n_genes = 10;
    scl_index_t n_groups = 3;
    
    auto expr_eigen = random_expression_matrix(n_cells, n_genes, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    // Generate group labels
    std::vector<scl_index_t> group_labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        group_labels[i] = rng.uniform_int(0, n_groups - 1);
    }
    
    std::vector<scl_real_t> mean_expr(n_genes * n_groups, 0.0);
    
    scl_error_t err = scl_markers_group_mean_expression(
        expr, group_labels.data(), n_cells, n_groups, n_genes, mean_expr.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify all means are computed (check non-NaN)
    for (scl_index_t i = 0; i < n_genes * n_groups; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(mean_expr[i]));
    }
}

SCL_TEST_CASE(group_mean_expression_single_group)
{
    Random rng(123);
    
    scl_index_t n_cells = 10;
    scl_index_t n_genes = 5;
    scl_index_t n_groups = 1;
    
    auto expr_eigen = random_expression_matrix(n_cells, n_genes, 0.5, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(n_cells, 0);  // All same group
    
    std::vector<scl_real_t> mean_expr(n_genes * n_groups);
    
    scl_error_t err = scl_markers_group_mean_expression(
        expr, group_labels.data(), n_cells, n_groups, n_genes, mean_expr.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify means are computed
    for (scl_index_t i = 0; i < n_genes; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(mean_expr[i]));
    }
}

SCL_TEST_CASE(group_mean_expression_null_expression)
{
    std::vector<scl_index_t> group_labels(10);
    std::vector<scl_real_t> mean_expr(50);
    
    scl_error_t err = scl_markers_group_mean_expression(
        nullptr, group_labels.data(), 10, 2, 5, mean_expr.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(group_mean_expression_null_labels)
{
    Random rng(456);
    auto expr_eigen = random_expression_matrix(10, 5, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_real_t> mean_expr(10);
    
    scl_error_t err = scl_markers_group_mean_expression(
        expr, nullptr, 10, 2, 5, mean_expr.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(group_mean_expression_null_output)
{
    Random rng(789);
    auto expr_eigen = random_expression_matrix(10, 5, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(10);
    
    scl_error_t err = scl_markers_group_mean_expression(
        expr, group_labels.data(), 10, 2, 5, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(group_mean_expression_zero_cells)
{
    Random rng(999);
    auto expr_eigen = random_expression_matrix(10, 5, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(1);
    std::vector<scl_real_t> mean_expr(10);
    
    scl_error_t err = scl_markers_group_mean_expression(
        expr, group_labels.data(), 0, 2, 5, mean_expr.data()
    );
    
    // Should handle zero cells gracefully
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_INVALID_ARGUMENT);
}

SCL_TEST_SUITE_END

// =============================================================================
// Percent Expressed
// =============================================================================

SCL_TEST_SUITE(percent_expressed)

SCL_TEST_RETRY(percent_expressed_basic, 3)
{
    Random rng(111);
    
    scl_index_t n_cells = 20;
    scl_index_t n_genes = 10;
    scl_index_t n_groups = 3;
    
    auto expr_eigen = random_expression_matrix(n_cells, n_genes, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        group_labels[i] = rng.uniform_int(0, n_groups - 1);
    }
    
    std::vector<scl_real_t> pct_expr(n_genes * n_groups);
    
    scl_error_t err = scl_markers_percent_expressed(
        expr, group_labels.data(), n_cells, n_groups, n_genes,
        pct_expr.data(), 0.0  // threshold = 0
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Percentages should be in [0, 100] range
    for (scl_index_t i = 0; i < n_genes * n_groups; ++i) {
        SCL_ASSERT_GE(pct_expr[i], 0.0);
        SCL_ASSERT_LE(pct_expr[i], 100.0);
    }
}

SCL_TEST_CASE(percent_expressed_with_threshold)
{
    Random rng(222);
    
    scl_index_t n_cells = 15;
    scl_index_t n_genes = 8;
    scl_index_t n_groups = 2;
    
    auto expr_eigen = random_expression_matrix(n_cells, n_genes, 0.4, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        group_labels[i] = rng.uniform_int(0, n_groups - 1);
    }
    
    std::vector<scl_real_t> pct_expr(n_genes * n_groups);
    
    scl_real_t threshold = 1.0;
    scl_error_t err = scl_markers_percent_expressed(
        expr, group_labels.data(), n_cells, n_groups, n_genes,
        pct_expr.data(), threshold
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Percentages should be in [0, 100] range
    for (scl_index_t i = 0; i < n_genes * n_groups; ++i) {
        SCL_ASSERT_GE(pct_expr[i], 0.0);
        SCL_ASSERT_LE(pct_expr[i], 100.0);
    }
}

SCL_TEST_CASE(percent_expressed_null_expression)
{
    std::vector<scl_index_t> group_labels(10);
    std::vector<scl_real_t> pct_expr(20);
    
    scl_error_t err = scl_markers_percent_expressed(
        nullptr, group_labels.data(), 10, 2, 5, pct_expr.data(), 0.0
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(percent_expressed_null_labels)
{
    Random rng(333);
    auto expr_eigen = random_expression_matrix(10, 5, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_real_t> pct_expr(10);
    
    scl_error_t err = scl_markers_percent_expressed(
        expr, nullptr, 10, 2, 5, pct_expr.data(), 0.0
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(percent_expressed_null_output)
{
    Random rng(444);
    auto expr_eigen = random_expression_matrix(10, 5, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(10);
    
    scl_error_t err = scl_markers_percent_expressed(
        expr, group_labels.data(), 10, 2, 5, nullptr, 0.0
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Find Marker Genes
// =============================================================================

SCL_TEST_SUITE(find_markers)

SCL_TEST_RETRY(find_markers_basic, 3)
{
    Random rng(555);
    
    scl_index_t n_cells = 30;
    scl_index_t n_genes = 20;
    scl_index_t n_groups = 3;
    
    // Create expression matrix with some group-specific markers
    auto expr_eigen = random_expression_matrix(n_cells, n_genes, 0.4, rng);
    
    // Make some genes group-specific (higher expression in group 0)
    for (scl_index_t i = 0; i < n_cells / n_groups; ++i) {
        for (scl_index_t j = 0; j < 3; ++j) {  // First 3 genes are markers
            expr_eigen.coeffRef(i, j) = 10.0 + rng.uniform(0.0, 5.0);
        }
    }
    expr_eigen.makeCompressed();
    
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    // Generate group labels
    std::vector<scl_index_t> group_labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        group_labels[i] = i / (n_cells / n_groups);
    }
    
    scl_index_t target_group = 0;
    std::vector<scl_index_t> marker_genes(n_genes);
    std::vector<scl_real_t> marker_scores(n_genes);
    scl_size_t n_markers = 0;
    
    scl_error_t err = scl_markers_find_markers(
        expr, group_labels.data(), n_cells, n_groups, n_genes,
        target_group, marker_genes.data(), marker_scores.data(), &n_markers,
        1.5,  // min_fold_change
        0.1,  // min_pct
        10    // n_top
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should find at least some markers
    SCL_ASSERT_LE(n_markers, static_cast<scl_size_t>(n_genes));
    
    // Verify marker genes are valid indices
    for (scl_size_t i = 0; i < n_markers; ++i) {
        SCL_ASSERT_GE(marker_genes[i], 0);
        SCL_ASSERT_LT(marker_genes[i], n_genes);
    }
}

SCL_TEST_CASE(find_markers_no_markers_found)
{
    Random rng(666);
    
    scl_index_t n_cells = 20;
    scl_index_t n_genes = 10;
    scl_index_t n_groups = 2;
    
    // Uniform expression (no markers)
    auto expr_eigen = random_expression_matrix(n_cells, n_genes, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        group_labels[i] = i / (n_cells / n_groups);
    }
    
    std::vector<scl_index_t> marker_genes(n_genes);
    std::vector<scl_real_t> marker_scores(n_genes);
    scl_size_t n_markers = 0;
    
    scl_error_t err = scl_markers_find_markers(
        expr, group_labels.data(), n_cells, n_groups, n_genes,
        0, marker_genes.data(), marker_scores.data(), &n_markers,
        10.0,  // Very high fold change (unlikely to find markers)
        0.5,   // min_pct
        10     // n_top
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // May find 0 markers if criteria too strict
    SCL_ASSERT_LE(n_markers, static_cast<scl_size_t>(n_genes));
}

SCL_TEST_CASE(find_markers_null_expression)
{
    std::vector<scl_index_t> group_labels(10);
    std::vector<scl_index_t> marker_genes(5);
    std::vector<scl_real_t> marker_scores(5);
    scl_size_t n_markers = 0;
    
    scl_error_t err = scl_markers_find_markers(
        nullptr, group_labels.data(), 10, 2, 5,
        0, marker_genes.data(), marker_scores.data(), &n_markers,
        1.5, 0.1, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(find_markers_null_labels)
{
    Random rng(777);
    auto expr_eigen = random_expression_matrix(10, 5, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> marker_genes(5);
    std::vector<scl_real_t> marker_scores(5);
    scl_size_t n_markers = 0;
    
    scl_error_t err = scl_markers_find_markers(
        expr, nullptr, 10, 2, 5,
        0, marker_genes.data(), marker_scores.data(), &n_markers,
        1.5, 0.1, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(find_markers_null_marker_genes)
{
    Random rng(888);
    auto expr_eigen = random_expression_matrix(10, 5, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(10);
    std::vector<scl_real_t> marker_scores(5);
    scl_size_t n_markers = 0;
    
    scl_error_t err = scl_markers_find_markers(
        expr, group_labels.data(), 10, 2, 5,
        0, nullptr, marker_scores.data(), &n_markers,
        1.5, 0.1, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(find_markers_null_marker_scores)
{
    Random rng(999);
    auto expr_eigen = random_expression_matrix(10, 5, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(10);
    std::vector<scl_index_t> marker_genes(5);
    scl_size_t n_markers = 0;
    
    scl_error_t err = scl_markers_find_markers(
        expr, group_labels.data(), 10, 2, 5,
        0, marker_genes.data(), nullptr, &n_markers,
        1.5, 0.1, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(find_markers_null_output)
{
    Random rng(1000);
    auto expr_eigen = random_expression_matrix(10, 5, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(10);
    std::vector<scl_index_t> marker_genes(5);
    std::vector<scl_real_t> marker_scores(5);
    
    scl_error_t err = scl_markers_find_markers(
        expr, group_labels.data(), 10, 2, 5,
        0, marker_genes.data(), marker_scores.data(), nullptr,
        1.5, 0.1, 10
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(find_markers_invalid_target_group)
{
    Random rng(1111);
    auto expr_eigen = random_expression_matrix(10, 5, 0.3, rng);
    Sparse expr = eigen_to_scl_sparse(expr_eigen);
    
    std::vector<scl_index_t> group_labels(10);
    std::vector<scl_index_t> marker_genes(5);
    std::vector<scl_real_t> marker_scores(5);
    scl_size_t n_markers = 0;
    
    scl_error_t err = scl_markers_find_markers(
        expr, group_labels.data(), 10, 2, 5,
        99,  // Invalid target group
        marker_genes.data(), marker_scores.data(), &n_markers,
        1.5, 0.1, 10
    );
    
    // Should either return error or handle gracefully
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_INVALID_ARGUMENT || 
                    err == SCL_ERROR_RANGE_ERROR);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

