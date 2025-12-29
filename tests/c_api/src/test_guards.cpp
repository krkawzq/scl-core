// =============================================================================
// SCL Core - RAII Guards Test
// =============================================================================
//
// Test RAII wrappers without Eigen dependency.
//
// =============================================================================

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// Tiny Test Data (no Eigen needed)
// =============================================================================

static std::tuple<std::vector<scl_index_t>, std::vector<scl_index_t>, std::vector<scl_real_t>>
get_tiny_3x3() {
    std::vector<scl_index_t> indptr = {0, 2, 3, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    return {indptr, indices, data};
}

// =============================================================================
// Sparse Guard Tests
// =============================================================================

SCL_TEST_UNIT(sparse_guard_create_destroy) {
    auto [indptr, indices, data] = get_tiny_3x3();
    
    {
        Sparse mat = make_sparse_csr(
            3, 3, 6,
            indptr.data(), indices.data(), data.data()
        );
        
        SCL_ASSERT_NOT_NULL(mat.get());
        SCL_ASSERT_TRUE(mat.valid());
        SCL_ASSERT_TRUE(static_cast<bool>(mat));
    } // mat destroyed automatically
    
    // No leak or crash
    SCL_ASSERT_TRUE(true);
}

SCL_TEST_UNIT(sparse_guard_move_semantics) {
    auto [indptr, indices, data] = get_tiny_3x3();
    
    Sparse mat1 = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    scl_sparse_t handle1 = mat1.get();
    SCL_ASSERT_NOT_NULL(handle1);
    
    // Move construction
    Sparse mat2 = std::move(mat1);
    SCL_ASSERT_NULL(mat1.get());
    SCL_ASSERT_EQ(mat2.get(), handle1);
    
    // Move assignment
    Sparse mat3;
    mat3 = std::move(mat2);
    SCL_ASSERT_NULL(mat2.get());
    SCL_ASSERT_EQ(mat3.get(), handle1);
}

SCL_TEST_UNIT(sparse_guard_reset) {
    auto [indptr, indices, data] = get_tiny_3x3();
    
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    SCL_ASSERT_NOT_NULL(mat.get());
    
    mat.reset();
    SCL_ASSERT_NULL(mat.get());
    SCL_ASSERT_FALSE(mat.valid());
}

SCL_TEST_UNIT(sparse_guard_release) {
    auto [indptr, indices, data] = get_tiny_3x3();
    
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    scl_sparse_t handle = mat.release();
    
    SCL_ASSERT_NOT_NULL(handle);
    SCL_ASSERT_NULL(mat.get());
    
    // Manual cleanup
    scl_sparse_destroy(&handle);
}

SCL_TEST_UNIT(sparse_guard_implicit_conversion) {
    auto [indptr, indices, data] = get_tiny_3x3();
    
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    // Can use directly in C API calls
    scl_index_t rows, cols, nnz;
    SCL_ASSERT_EQ(scl_sparse_rows(mat, &rows), SCL_OK);
    SCL_ASSERT_EQ(scl_sparse_cols(mat, &cols), SCL_OK);
    SCL_ASSERT_EQ(scl_sparse_nnz(mat, &nnz), SCL_OK);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 3);
    SCL_ASSERT_EQ(nnz, 6);
}

SCL_TEST_UNIT(sparse_guard_wrap) {
    auto [indptr, indices, data] = get_tiny_3x3();
    
    Sparse mat = wrap_sparse_csr(
        3, 3, 6,
        indptr.data(), indices.data(), data.data()
    );
    
    SCL_ASSERT_NOT_NULL(mat.get());
    
    scl_index_t nnz;
    scl_sparse_nnz(mat, &nnz);
    SCL_ASSERT_EQ(nnz, 6);
}

// =============================================================================
// Dense Guard Tests
// =============================================================================

SCL_TEST_UNIT(dense_guard_wrap_destroy) {
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    {
        Dense mat = wrap_dense(2, 3, data.data());
        SCL_ASSERT_NOT_NULL(mat.get());
        SCL_ASSERT_TRUE(mat.valid());
    }
    
    SCL_ASSERT_TRUE(true);
}

SCL_TEST_UNIT(dense_guard_move_semantics) {
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0};
    
    Dense mat1 = wrap_dense(2, 2, data.data());
    scl_dense_t handle1 = mat1.get();
    
    Dense mat2 = std::move(mat1);
    SCL_ASSERT_NULL(mat1.get());
    SCL_ASSERT_EQ(mat2.get(), handle1);
}

SCL_TEST_UNIT(dense_guard_properties) {
    std::vector<scl_real_t> data(12);
    std::iota(data.begin(), data.end(), 1.0);
    
    Dense mat = wrap_dense(3, 4, data.data());
    
    scl_index_t rows, cols, stride;
    SCL_ASSERT_EQ(scl_dense_rows(mat, &rows), SCL_OK);
    SCL_ASSERT_EQ(scl_dense_cols(mat, &cols), SCL_OK);
    SCL_ASSERT_EQ(scl_dense_stride(mat, &stride), SCL_OK);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 4);
    SCL_ASSERT_EQ(stride, 4);
}

// =============================================================================
// Multiple Guards Interaction
// =============================================================================

SCL_TEST_UNIT(multiple_guards_no_interference) {
    auto [indptr, indices, data] = get_tiny_3x3();
    
    Sparse mat1 = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    Sparse mat2 = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    Sparse mat3 = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    SCL_ASSERT_NOT_NULL(mat1.get());
    SCL_ASSERT_NOT_NULL(mat2.get());
    SCL_ASSERT_NOT_NULL(mat3.get());
    
    SCL_ASSERT_NE(mat1.get(), mat2.get());
    SCL_ASSERT_NE(mat2.get(), mat3.get());
}

SCL_TEST_END

SCL_TEST_MAIN()

