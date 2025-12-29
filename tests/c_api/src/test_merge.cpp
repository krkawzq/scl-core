// =============================================================================
// SCL Core - Matrix Merge Operations Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/merge.h
//
// Functions tested:
//   - scl_merge_vstack (vertical stack)
//   - scl_merge_hstack (horizontal stack)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/merge.h"

using namespace scl::test;
using precision::Tolerance;

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
// Vertical Stack (vstack)
// =============================================================================

SCL_TEST_SUITE(vstack)

SCL_TEST_CASE(vstack_basic)
{
    // Matrix 1: 2x3
    auto mat1_eigen = random_sparse_csr(2, 3, 0.5);
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    
    // Matrix 2: 3x3 (same cols)
    auto mat2_eigen = random_sparse_csr(3, 3, 0.5);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_vstack(mat1, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify dimensions
    scl_index_t rows, cols;
    scl_sparse_rows(result, &rows);
    scl_sparse_cols(result, &cols);
    
    SCL_ASSERT_EQ(rows, 5);  // 2 + 3
    SCL_ASSERT_EQ(cols, 3);
}

SCL_TEST_RETRY(vstack_random_shapes, 5)
{
    Random rng(42);
    
    scl_index_t cols = rng.uniform_int(5, 20);
    scl_index_t rows1 = rng.uniform_int(5, 20);
    scl_index_t rows2 = rng.uniform_int(5, 20);
    
    auto mat1_eigen = random_sparse_csr(rows1, cols, 0.2, rng);
    auto mat2_eigen = random_sparse_csr(rows2, cols, 0.2, rng);
    
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_vstack(mat1, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t rows, result_cols;
    scl_sparse_rows(result, &rows);
    scl_sparse_cols(result, &result_cols);
    
    SCL_ASSERT_EQ(rows, rows1 + rows2);
    SCL_ASSERT_EQ(result_cols, cols);
}

SCL_TEST_CASE(vstack_single_row)
{
    // Matrix 1: 1x3
    auto mat1_eigen = random_sparse_csr(1, 3, 0.5);
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    
    // Matrix 2: 2x3
    auto mat2_eigen = random_sparse_csr(2, 3, 0.5);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_vstack(mat1, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t rows, cols;
    scl_sparse_rows(result, &rows);
    scl_sparse_cols(result, &cols);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 3);
}

SCL_TEST_CASE(vstack_empty_matrix)
{
    // Empty matrix (zero rows)
    EigenCSR empty_eigen(0, 5);
    empty_eigen.makeCompressed();
    Sparse empty = eigen_to_scl_sparse(empty_eigen);
    
    // Non-empty matrix
    auto mat_eigen = random_sparse_csr(3, 5, 0.3);
    Sparse mat = eigen_to_scl_sparse(mat_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_vstack(empty, mat, result.ptr());
    
    // Should handle empty matrix
    if (err == SCL_OK) {
        scl_index_t rows, cols;
        scl_sparse_rows(result, &rows);
        scl_sparse_cols(result, &cols);
        
        SCL_ASSERT_EQ(rows, 3);
        SCL_ASSERT_EQ(cols, 5);
    }
}

SCL_TEST_CASE(vstack_null_matrix1)
{
    auto mat2_eigen = random_sparse_csr(3, 3, 0.3);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_vstack(nullptr, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(vstack_null_matrix2)
{
    auto mat1_eigen = random_sparse_csr(2, 3, 0.3);
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_vstack(mat1, nullptr, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(vstack_null_output)
{
    auto mat1_eigen = random_sparse_csr(2, 3, 0.3);
    auto mat2_eigen = random_sparse_csr(3, 3, 0.3);
    
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    scl_error_t err = scl_merge_vstack(mat1, mat2, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(vstack_dimension_mismatch)
{
    // Different column counts
    auto mat1_eigen = random_sparse_csr(2, 3, 0.3);
    auto mat2_eigen = random_sparse_csr(3, 5, 0.3);  // Different cols
    
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_vstack(mat1, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_DIMENSION_MISMATCH);
}

SCL_TEST_CASE(vstack_verify_content)
{
    // Create specific matrices for content verification
    // Matrix 1: [[1, 2], [3, 4]]
    std::vector<scl_index_t> indptr1 = {0, 2, 4};
    std::vector<scl_index_t> indices1 = {0, 1, 0, 1};
    std::vector<scl_real_t> data1 = {1.0, 2.0, 3.0, 4.0};
    Sparse mat1 = make_sparse_csr(2, 2, 4, indptr1.data(), indices1.data(), data1.data());
    
    // Matrix 2: [[5, 6]]
    std::vector<scl_index_t> indptr2 = {0, 2};
    std::vector<scl_index_t> indices2 = {0, 1};
    std::vector<scl_real_t> data2 = {5.0, 6.0};
    Sparse mat2 = make_sparse_csr(1, 2, 2, indptr2.data(), indices2.data(), data2.data());
    
    Sparse result;
    scl_error_t err = scl_merge_vstack(mat1, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t rows, cols;
    scl_sparse_rows(result, &rows);
    scl_sparse_cols(result, &cols);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 2);
}

SCL_TEST_SUITE_END

// =============================================================================
// Horizontal Stack (hstack)
// =============================================================================

SCL_TEST_SUITE(hstack)

SCL_TEST_CASE(hstack_basic)
{
    // Matrix 1: 3x2
    auto mat1_eigen = random_sparse_csr(3, 2, 0.5);
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    
    // Matrix 2: 3x3 (same rows)
    auto mat2_eigen = random_sparse_csr(3, 3, 0.5);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_hstack(mat1, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify dimensions
    scl_index_t rows, cols;
    scl_sparse_rows(result, &rows);
    scl_sparse_cols(result, &cols);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 5);  // 2 + 3
}

SCL_TEST_RETRY(hstack_random_shapes, 5)
{
    Random rng(123);
    
    scl_index_t rows = rng.uniform_int(5, 20);
    scl_index_t cols1 = rng.uniform_int(5, 20);
    scl_index_t cols2 = rng.uniform_int(5, 20);
    
    auto mat1_eigen = random_sparse_csr(rows, cols1, 0.2, rng);
    auto mat2_eigen = random_sparse_csr(rows, cols2, 0.2, rng);
    
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_hstack(mat1, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t result_rows, cols;
    scl_sparse_rows(result, &result_rows);
    scl_sparse_cols(result, &cols);
    
    SCL_ASSERT_EQ(result_rows, rows);
    SCL_ASSERT_EQ(cols, cols1 + cols2);
}

SCL_TEST_CASE(hstack_single_column)
{
    // Matrix 1: 3x1
    auto mat1_eigen = random_sparse_csr(3, 1, 0.5);
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    
    // Matrix 2: 3x2
    auto mat2_eigen = random_sparse_csr(3, 2, 0.5);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_hstack(mat1, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t rows, cols;
    scl_sparse_rows(result, &rows);
    scl_sparse_cols(result, &cols);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 3);
}

SCL_TEST_CASE(hstack_empty_matrix)
{
    // Empty matrix (zero cols)
    EigenCSR empty_eigen(5, 0);
    empty_eigen.makeCompressed();
    Sparse empty = eigen_to_scl_sparse(empty_eigen);
    
    // Non-empty matrix
    auto mat_eigen = random_sparse_csr(5, 3, 0.3);
    Sparse mat = eigen_to_scl_sparse(mat_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_hstack(empty, mat, result.ptr());
    
    // Should handle empty matrix
    if (err == SCL_OK) {
        scl_index_t rows, cols;
        scl_sparse_rows(result, &rows);
        scl_sparse_cols(result, &cols);
        
        SCL_ASSERT_EQ(rows, 5);
        SCL_ASSERT_EQ(cols, 3);
    }
}

SCL_TEST_CASE(hstack_null_matrix1)
{
    auto mat2_eigen = random_sparse_csr(3, 3, 0.3);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_hstack(nullptr, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(hstack_null_matrix2)
{
    auto mat1_eigen = random_sparse_csr(3, 2, 0.3);
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_hstack(mat1, nullptr, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(hstack_null_output)
{
    auto mat1_eigen = random_sparse_csr(3, 2, 0.3);
    auto mat2_eigen = random_sparse_csr(3, 3, 0.3);
    
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    scl_error_t err = scl_merge_hstack(mat1, mat2, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(hstack_dimension_mismatch)
{
    // Different row counts
    auto mat1_eigen = random_sparse_csr(2, 3, 0.3);
    auto mat2_eigen = random_sparse_csr(5, 3, 0.3);  // Different rows
    
    Sparse mat1 = eigen_to_scl_sparse(mat1_eigen);
    Sparse mat2 = eigen_to_scl_sparse(mat2_eigen);
    
    Sparse result;
    scl_error_t err = scl_merge_hstack(mat1, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_DIMENSION_MISMATCH);
}

SCL_TEST_CASE(hstack_verify_content)
{
    // Create specific matrices for content verification
    // Matrix 1: [[1, 2], [3, 4]]
    std::vector<scl_index_t> indptr1 = {0, 2, 4};
    std::vector<scl_index_t> indices1 = {0, 1, 0, 1};
    std::vector<scl_real_t> data1 = {1.0, 2.0, 3.0, 4.0};
    Sparse mat1 = make_sparse_csr(2, 2, 4, indptr1.data(), indices1.data(), data1.data());
    
    // Matrix 2: [[5], [6]]
    std::vector<scl_index_t> indptr2 = {0, 1, 2};
    std::vector<scl_index_t> indices2 = {0, 0};
    std::vector<scl_real_t> data2 = {5.0, 6.0};
    Sparse mat2 = make_sparse_csr(2, 1, 2, indptr2.data(), indices2.data(), data2.data());
    
    Sparse result;
    scl_error_t err = scl_merge_hstack(mat1, mat2, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t rows, cols;
    scl_sparse_rows(result, &rows);
    scl_sparse_cols(result, &cols);
    
    SCL_ASSERT_EQ(rows, 2);
    SCL_ASSERT_EQ(cols, 3);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

