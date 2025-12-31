// =============================================================================
// SCL Core v0.5 - Simple Smoke Tests
// =============================================================================

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

SCL_TEST_SUITE(smoke)

SCL_TEST_CASE(library_loads) {
    const char* version = scl_get_version();
    SCL_ASSERT_NOT_NULL(version);
    printf("    Version: %s\n", version);
}

SCL_TEST_CASE(zeros_works) {
    auto handle = scl_sparse_zeros(5, 5, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    
    SCL_ASSERT_NOT_NULL(handle);
    printf("    Created zeros matrix: %p\n", (void*)handle);
    
    auto rows = scl_sparse_rows(handle);
    printf("    Rows: %lld\n", (long long)rows);
    SCL_ASSERT_EQ(rows, 5);
    
    auto cols = scl_sparse_cols(handle);
    printf("    Cols: %lld\n", (long long)cols);
    SCL_ASSERT_EQ(cols, 5);
    
    auto nnz = scl_sparse_nnz(handle);
    printf("    NNZ: %lld\n", (long long)nnz);
    SCL_ASSERT_EQ(nnz, 0);
    
    scl_sparse_destroy(handle);
    printf("    Destroyed\n");
}

SCL_TEST_CASE(identity_works) {
    auto handle = scl_sparse_identity(3, SCL_REAL64, SCL_INDEX64);
    
    SCL_ASSERT_NOT_NULL(handle);
    printf("    Created identity matrix: %p\n", (void*)handle);
    
    SCL_ASSERT_EQ(scl_sparse_rows(handle), 3);
    SCL_ASSERT_EQ(scl_sparse_cols(handle), 3);
    
    auto nnz = scl_sparse_nnz(handle);
    printf("    NNZ: %lld\n", (long long)nnz);
    SCL_ASSERT_EQ(nnz, 3);
    
    scl_sparse_destroy(handle);
    printf("    Destroyed\n");
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

