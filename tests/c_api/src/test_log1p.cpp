// =============================================================================
// SCL Core - Logarithmic Transform Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/log1p.h
//
// Functions tested:
//   ✓ scl_log1p_inplace - log(1 + x) in-place
//   ✓ scl_log2p1_inplace - log2(1 + x) in-place
//   ✓ scl_expm1_inplace - exp(x) - 1 in-place
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

using namespace scl::test;
using precision::Tolerance;

SCL_TEST_BEGIN

// =============================================================================
// Helper: Create test sparse matrix
// =============================================================================

static Sparse create_test_matrix(scl_index_t rows, scl_index_t cols, 
                                 const std::vector<scl_real_t>& values) {
    std::vector<scl_index_t> indptr(rows + 1);
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    
    scl_index_t nnz = 0;
    for (scl_index_t i = 0; i < rows; ++i) {
        indptr[i] = nnz;
        for (scl_index_t j = 0; j < cols; ++j) {
            scl_index_t idx = i * cols + j;
            if (idx < static_cast<scl_index_t>(values.size()) && values[idx] != 0.0) {
                indices.push_back(j);
                data.push_back(values[idx]);
                ++nnz;
            }
        }
    }
    indptr[rows] = nnz;
    
    return make_sparse_csr(rows, cols, nnz, indptr.data(), indices.data(), data.data());
}

// =============================================================================
// log1p_inplace Tests
// =============================================================================

SCL_TEST_SUITE(log1p_inplace)

SCL_TEST_CASE(log1p_basic) {
    // Matrix: [1.0, 2.0, 3.0]
    std::vector<scl_real_t> values = {1.0, 2.0, 3.0};
    Sparse mat = create_test_matrix(1, 3, values);
    
    // Get original data
    std::vector<scl_index_t> indptr(2);
    std::vector<scl_index_t> indices(3);
    std::vector<scl_real_t> original_data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), original_data.data());
    
    // Apply log1p
    scl_error_t err = scl_log1p_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Get transformed data
    std::vector<scl_real_t> transformed_data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), transformed_data.data());
    
    // Verify: log1p(x) = log(1 + x)
    for (size_t i = 0; i < 3; ++i) {
        scl_real_t expected = std::log1p(original_data[i]);
        SCL_ASSERT_NEAR(transformed_data[i], expected, 1e-10);
    }
}

SCL_TEST_CASE(log1p_small_values) {
    // Small values where log1p is more accurate than log(1+x)
    std::vector<scl_real_t> values = {1e-10, 1e-15, 1e-20};
    Sparse mat = create_test_matrix(1, 3, values);
    
    std::vector<scl_index_t> indptr(2);
    std::vector<scl_index_t> indices(3);
    std::vector<scl_real_t> original_data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), original_data.data());
    
    scl_error_t err = scl_log1p_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    std::vector<scl_real_t> transformed_data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), transformed_data.data());
    
    for (size_t i = 0; i < 3; ++i) {
        scl_real_t expected = std::log1p(original_data[i]);
        SCL_ASSERT_NEAR(transformed_data[i], expected, 1e-12);
    }
}

SCL_TEST_CASE(log1p_zero_values) {
    // Matrix with zeros (should remain zero or become -inf for log1p(-1))
    std::vector<scl_real_t> values = {0.0, 1.0, -1.0};
    Sparse mat = create_test_matrix(1, 3, values);
    
    scl_error_t err = scl_log1p_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    std::vector<scl_index_t> indptr(2);
    std::vector<scl_index_t> indices(3);
    std::vector<scl_real_t> data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), data.data());
    
    // log1p(0) = 0
    SCL_ASSERT_NEAR(data[0], 0.0, 1e-12);
    // log1p(1) = log(2)
    SCL_ASSERT_NEAR(data[1], std::log(2.0), 1e-10);
    // log1p(-1) = -inf (or very large negative)
    SCL_ASSERT_TRUE(std::isinf(data[2]) || data[2] < -1e10);
}

SCL_TEST_RETRY(log1p_random_matrix, 3)
{
    Random rng(42);
    auto eigen_mat = random_sparse_csr(20, 20, 0.1, rng);
    auto csr = from_eigen_csr(eigen_mat);
    
    // Ensure all values are positive for log1p
    for (auto& val : csr.data) {
        val = std::abs(val) + 0.1;  // Make positive
    }
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Get original data
    std::vector<scl_index_t> indptr(csr.rows + 1);
    std::vector<scl_index_t> indices(csr.nnz);
    std::vector<scl_real_t> original_data(csr.nnz);
    scl_sparse_export(mat, indptr.data(), indices.data(), original_data.data());
    
    // Apply log1p
    scl_error_t err = scl_log1p_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Get transformed data
    std::vector<scl_real_t> transformed_data(csr.nnz);
    scl_sparse_export(mat, indptr.data(), indices.data(), transformed_data.data());
    
    // Verify each element
    for (size_t i = 0; i < csr.nnz; ++i) {
        scl_real_t expected = std::log1p(original_data[i]);
        SCL_ASSERT_NEAR(transformed_data[i], expected, Tolerance::normal());
    }
}

SCL_TEST_CASE(log1p_null_handle) {
    scl_error_t err = scl_log1p_inplace(nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(log1p_empty_matrix) {
    std::vector<scl_index_t> indptr = {0, 0, 0};
    std::vector<scl_index_t> indices(1);
    std::vector<scl_real_t> data(1);
    
    Sparse mat = make_sparse_csr(2, 2, 0, indptr.data(), indices.data(), data.data());
    
    scl_error_t err = scl_log1p_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// log2p1_inplace Tests
// =============================================================================

SCL_TEST_SUITE(log2p1_inplace)

SCL_TEST_CASE(log2p1_basic) {
    std::vector<scl_real_t> values = {1.0, 3.0, 7.0};
    Sparse mat = create_test_matrix(1, 3, values);
    
    std::vector<scl_index_t> indptr(2);
    std::vector<scl_index_t> indices(3);
    std::vector<scl_real_t> original_data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), original_data.data());
    
    scl_error_t err = scl_log2p1_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    std::vector<scl_real_t> transformed_data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), transformed_data.data());
    
    // Verify: log2p1(x) = log2(1 + x)
    for (size_t i = 0; i < 3; ++i) {
        scl_real_t expected = std::log2(1.0 + original_data[i]);
        SCL_ASSERT_NEAR(transformed_data[i], expected, 1e-10);
    }
}

SCL_TEST_CASE(log2p1_zero) {
    std::vector<scl_real_t> values = {0.0};
    Sparse mat = create_test_matrix(1, 1, values);
    
    scl_error_t err = scl_log2p1_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    std::vector<scl_index_t> indptr(2);
    std::vector<scl_index_t> indices(1);
    std::vector<scl_real_t> data(1);
    scl_sparse_export(mat, indptr.data(), indices.data(), data.data());
    
    // log2p1(0) = 0
    SCL_ASSERT_NEAR(data[0], 0.0, 1e-12);
}

SCL_TEST_RETRY(log2p1_random_matrix, 3)
{
    Random rng(123);
    auto eigen_mat = random_sparse_csr(15, 15, 0.15, rng);
    auto csr = from_eigen_csr(eigen_mat);
    
    // Make positive
    for (auto& val : csr.data) {
        val = std::abs(val) + 0.1;
    }
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indptr(csr.rows + 1);
    std::vector<scl_index_t> indices(csr.nnz);
    std::vector<scl_real_t> original_data(csr.nnz);
    scl_sparse_export(mat, indptr.data(), indices.data(), original_data.data());
    
    scl_error_t err = scl_log2p1_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    std::vector<scl_real_t> transformed_data(csr.nnz);
    scl_sparse_export(mat, indptr.data(), indices.data(), transformed_data.data());
    
    for (size_t i = 0; i < csr.nnz; ++i) {
        scl_real_t expected = std::log2(1.0 + original_data[i]);
        SCL_ASSERT_NEAR(transformed_data[i], expected, Tolerance::normal());
    }
}

SCL_TEST_CASE(log2p1_null_handle) {
    scl_error_t err = scl_log2p1_inplace(nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// expm1_inplace Tests
// =============================================================================

SCL_TEST_SUITE(expm1_inplace)

SCL_TEST_CASE(expm1_basic) {
    std::vector<scl_real_t> values = {0.0, 1.0, -1.0};
    Sparse mat = create_test_matrix(1, 3, values);
    
    std::vector<scl_index_t> indptr(2);
    std::vector<scl_index_t> indices(3);
    std::vector<scl_real_t> original_data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), original_data.data());
    
    scl_error_t err = scl_expm1_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    std::vector<scl_real_t> transformed_data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), transformed_data.data());
    
    // Verify: expm1(x) = exp(x) - 1
    for (size_t i = 0; i < 3; ++i) {
        scl_real_t expected = std::expm1(original_data[i]);
        SCL_ASSERT_NEAR(transformed_data[i], expected, 1e-10);
    }
}

SCL_TEST_CASE(expm1_small_values) {
    // Small values where expm1 is more accurate than exp(x) - 1
    std::vector<scl_real_t> values = {1e-10, 1e-15, -1e-10};
    Sparse mat = create_test_matrix(1, 3, values);
    
    std::vector<scl_index_t> indptr(2);
    std::vector<scl_index_t> indices(3);
    std::vector<scl_real_t> original_data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), original_data.data());
    
    scl_error_t err = scl_expm1_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    std::vector<scl_real_t> transformed_data(3);
    scl_sparse_export(mat, indptr.data(), indices.data(), transformed_data.data());
    
    for (size_t i = 0; i < 3; ++i) {
        scl_real_t expected = std::expm1(original_data[i]);
        SCL_ASSERT_NEAR(transformed_data[i], expected, 1e-12);
    }
}

SCL_TEST_CASE(expm1_zero) {
    std::vector<scl_real_t> values = {0.0};
    Sparse mat = create_test_matrix(1, 1, values);
    
    scl_error_t err = scl_expm1_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    std::vector<scl_index_t> indptr(2);
    std::vector<scl_index_t> indices(1);
    std::vector<scl_real_t> data(1);
    scl_sparse_export(mat, indptr.data(), indices.data(), data.data());
    
    // expm1(0) = 0
    SCL_ASSERT_NEAR(data[0], 0.0, 1e-12);
}

SCL_TEST_RETRY(expm1_random_matrix, 3)
{
    Random rng(456);
    auto eigen_mat = random_sparse_csr(20, 20, 0.1, rng);
    auto csr = from_eigen_csr(eigen_mat);
    
    // Scale values to reasonable range for exp
    for (auto& val : csr.data) {
        val = val * 0.1;  // Keep in [-1, 1] range
    }
    
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> indptr(csr.rows + 1);
    std::vector<scl_index_t> indices(csr.nnz);
    std::vector<scl_real_t> original_data(csr.nnz);
    scl_sparse_export(mat, indptr.data(), indices.data(), original_data.data());
    
    scl_error_t err = scl_expm1_inplace(mat.ptr());
    SCL_ASSERT_EQ(err, SCL_OK);
    
    std::vector<scl_real_t> transformed_data(csr.nnz);
    scl_sparse_export(mat, indptr.data(), indices.data(), transformed_data.data());
    
    for (size_t i = 0; i < csr.nnz; ++i) {
        scl_real_t expected = std::expm1(original_data[i]);
        SCL_ASSERT_NEAR(transformed_data[i], expected, Tolerance::normal());
    }
}

SCL_TEST_CASE(expm1_null_handle) {
    scl_error_t err = scl_expm1_inplace(nullptr);
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

