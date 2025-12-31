/// @file test_summary.cpp
/// @brief Quick summary test - runs one test for each major function
///
/// Purpose: Quick smoke test to verify all major API functions work
///
/// Total: ~30 test cases

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

SCL_TEST_SUITE(api_coverage)

// Creation functions
SCL_TEST_TAGGED(api_zeros, "smoke", "quick") {
    auto m = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(api_identity, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

// Properties
SCL_TEST_TAGGED(api_rows_cols_nnz, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_rows(m), 10);
    SCL_ASSERT_EQ(scl_sparse_cols(m), 10);
    SCL_ASSERT_EQ(scl_sparse_nnz(m), 10);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(api_density_sparsity, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_GT(scl_sparse_density(m), 0.0);
    SCL_ASSERT_LT(scl_sparse_sparsity(m), 1.0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(api_is_empty, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_FALSE(scl_sparse_is_empty(m));
    scl_sparse_destroy(m);
}

// Type queries
SCL_TEST_TAGGED(api_value_type, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_value_type(m), SCL_INT32);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(api_index_type, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX32);
    SCL_ASSERT_EQ(scl_sparse_index_type(m), SCL_INDEX32);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(api_layout, "smoke", "quick") {
    auto m = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_EQ(scl_sparse_layout(m), SCL_LAYOUT_CSC);
    scl_sparse_destroy(m);
}

// Data access
SCL_TEST_TAGGED(api_get, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    double v = scl_sparse_get(m, 5, 5);
    SCL_ASSERT_NEAR(v, 1.0, 1e-10);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(api_exists, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_exists(m, 5, 5), 1);
    SCL_ASSERT_EQ(scl_sparse_exists(m, 5, 6), 0);
    scl_sparse_destroy(m);
}

// Operations
SCL_TEST_TAGGED(api_clone, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto c = scl_sparse_clone(m);
    SCL_ASSERT_SPARSE_VALID(c);
    scl_sparse_destroy(m);
    scl_sparse_destroy(c);
}

SCL_TEST_TAGGED(api_transpose, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_SPARSE_VALID(t);
    scl_sparse_destroy(m);
    scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(api_scale, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_scale(m, 2.0), 0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(api_sort_indices, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_sort_indices(m), 0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(api_is_sorted, "smoke", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_is_sorted(m), 1);
    scl_sparse_destroy(m);
}

// Error handling
SCL_TEST_TAGGED(api_error_state, "smoke", "quick") {
    scl_clear_error();
    SCL_ASSERT_FALSE(scl_has_error());
    
    auto m = scl_sparse_zeros(-1, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_NULL(m);
    SCL_ASSERT_TRUE(scl_has_error());
    
    scl_clear_error();
    SCL_ASSERT_FALSE(scl_has_error());
}

// Type system
SCL_TEST_TAGGED(api_value_type_queries, "smoke", "quick") {
    SCL_ASSERT_EQ(scl_value_type_category(SCL_INT32), 1);
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_INT32), 4);
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_INT32), "Int32");
}

// Version info
SCL_TEST_TAGGED(api_version, "smoke", "quick") {
    const char* v = scl_get_version();
    SCL_ASSERT_STR_EQ(v, "0.5.0");
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

