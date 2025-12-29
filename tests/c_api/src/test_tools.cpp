// =============================================================================
// SCL Core - Test Tools Verification (Simplified)
// =============================================================================

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

SCL_TEST_UNIT(random_generator_works) {
    Random rng(42);
    
    double val = rng.uniform(0.0, 1.0);
    SCL_ASSERT_GE(val, 0.0);
    SCL_ASSERT_LE(val, 1.0);
}

SCL_TEST_UNIT(random_sparse_matrix_generation) {
    Random rng(12345);
    auto mat = random_sparse_csr(10, 10, 0.1, rng);
    
    SCL_ASSERT_EQ(mat.rows(), 10);
    SCL_ASSERT_EQ(mat.cols(), 10);
    SCL_ASSERT_GT(mat.nonZeros(), 0);
}

SCL_TEST_UNIT(identity_matrix_generation) {
    auto mat = identity_csr(5);
    
    SCL_ASSERT_EQ(mat.rows(), 5);
    SCL_ASSERT_EQ(mat.cols(), 5);
    SCL_ASSERT_EQ(mat.nonZeros(), 5);
}

SCL_TEST_UNIT(eigen_conversion_works) {
    Random rng(777);
    auto eigen_mat = random_sparse_csr(10, 10, 0.1, rng);
    auto csr_arrays = from_eigen_csr(eigen_mat);
    
    Sparse scl_mat = make_sparse_csr(
        csr_arrays.rows, csr_arrays.cols, csr_arrays.nnz,
        csr_arrays.indptr.data(),
        csr_arrays.indices.data(),
        csr_arrays.data.data()
    );
    
    SCL_ASSERT_NOT_NULL(scl_mat.get());
    
    scl_index_t rows, cols, nnz;
    scl_sparse_rows(scl_mat, &rows);
    scl_sparse_cols(scl_mat, &cols);
    scl_sparse_nnz(scl_mat, &nnz);
    
    SCL_ASSERT_EQ(rows, eigen_mat.rows());
    SCL_ASSERT_EQ(cols, eigen_mat.cols());
    SCL_ASSERT_EQ(nnz, eigen_mat.nonZeros());
}

SCL_TEST_END

SCL_TEST_MAIN()
