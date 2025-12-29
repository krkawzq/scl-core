// =============================================================================
// SCL Core - Sparse Matrix Operations Tests
// =============================================================================

#include "test.hpp"

using namespace scl::test;

// Helper
static std::tuple<std::vector<scl_index_t>, std::vector<scl_index_t>, std::vector<scl_real_t>>
get_tiny_3x3() {
    std::vector<scl_index_t> indptr = {0, 2, 3, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    return {indptr, indices, data};
}

SCL_TEST_BEGIN

SCL_TEST_UNIT(transpose_works) {
    auto [indptr, indices, data] = get_tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    Sparse transposed;
    SCL_ASSERT_EQ(scl_sparse_transpose(mat, transposed.ptr()), SCL_OK);
    SCL_ASSERT_NOT_NULL(transposed.get());
}

SCL_TEST_UNIT(clone_works) {
    auto [indptr, indices, data] = get_tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    Sparse cloned;
    SCL_ASSERT_EQ(scl_sparse_clone(mat, cloned.ptr()), SCL_OK);
    SCL_ASSERT_NOT_NULL(cloned.get());
}

SCL_TEST_END

SCL_TEST_MAIN()
