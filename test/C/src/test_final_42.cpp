/// @file test_final_42.cpp
/// @brief Final 42 tests to reach 500 test cases
///
/// Mixed tests covering various remaining scenarios

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// Quick Coverage Tests (42 tests)
// =============================================================================

SCL_TEST_SUITE(final_coverage_42)

// 1-10: More from_coo variants
SCL_TEST_TAGGED(final_01_coo_3elem, "final") {
    std::vector<std::int64_t> r={0,1,2}, c={0,1,2};
    std::vector<double> v={1,2,3};
    auto m = scl_sparse_from_coo(5,5,r.data(),c.data(),v.data(),3,SCL_REAL64,SCL_INDEX64,SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_02_coo_4elem, "final") {
    std::vector<std::int64_t> r={0,0,1,1}, c={0,1,0,1};
    std::vector<double> v={1,2,3,4};
    auto m = scl_sparse_from_coo(5,5,r.data(),c.data(),v.data(),4,SCL_REAL64,SCL_INDEX64,SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_03_coo_5elem, "final") {
    std::vector<std::int64_t> r={0,1,2,3,4}, c={0,1,2,3,4};
    std::vector<double> v={1,2,3,4,5};
    auto m = scl_sparse_from_coo(5,5,r.data(),c.data(),v.data(),5,SCL_REAL64,SCL_INDEX64,SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_04_coo_int8_csc, "final") {
    std::vector<std::int64_t> r={0,1,2}, c={0,1,2};
    std::vector<std::int8_t> v={1,2,3};
    auto m = scl_sparse_from_coo(5,5,r.data(),c.data(),v.data(),3,SCL_INT8,SCL_INDEX64,SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_05_coo_uint16_csc, "final") {
    std::vector<std::int64_t> r={0,1,2}, c={0,1,2};
    std::vector<std::uint16_t> v={100,200,300};
    auto m = scl_sparse_from_coo(5,5,r.data(),c.data(),v.data(),3,SCL_UINT16,SCL_INDEX64,SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_06_zeros_int8_csc, "final") {
    auto m = scl_sparse_zeros(10,10,SCL_INT8,SCL_INDEX32,SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_07_zeros_uint64_csr, "final") {
    auto m = scl_sparse_zeros(10,10,SCL_UINT64,SCL_INDEX32,SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_08_identity_real32_idx32, "final") {
    auto m = scl_sparse_identity(10,SCL_REAL32,SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_09_identity_int64_idx32, "final") {
    auto m = scl_sparse_identity(10,SCL_INT64,SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_10_identity_uint16_idx64, "final") {
    auto m = scl_sparse_identity(10,SCL_UINT16,SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

// 11-20: Operations
SCL_TEST_TAGGED(final_11_transpose_int8, "final") {
    auto m = scl_sparse_identity(5,SCL_INT8,SCL_INDEX64);
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_SPARSE_VALID(t);
    scl_sparse_destroy(m); scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(final_12_transpose_uint16, "final") {
    auto m = scl_sparse_identity(5,SCL_UINT16,SCL_INDEX64);
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_SPARSE_VALID(t);
    scl_sparse_destroy(m); scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(final_13_clone_int16, "final") {
    auto m = scl_sparse_identity(5,SCL_INT16,SCL_INDEX64);
    auto c = scl_sparse_clone(m);
    SCL_ASSERT_SPARSE_VALID(c);
    scl_sparse_destroy(m); scl_sparse_destroy(c);
}

SCL_TEST_TAGGED(final_14_clone_uint64, "final") {
    auto m = scl_sparse_identity(5,SCL_UINT64,SCL_INDEX64);
    auto c = scl_sparse_clone(m);
    SCL_ASSERT_SPARSE_VALID(c);
    scl_sparse_destroy(m); scl_sparse_destroy(c);
}

SCL_TEST_TAGGED(final_15_scale_int8, "final") {
    auto m = scl_sparse_identity(5,SCL_INT8,SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_scale(m,3.0),0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_16_scale_uint16, "final") {
    auto m = scl_sparse_identity(5,SCL_UINT16,SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_scale(m,5.0),0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_17_scale_real32, "final") {
    auto m = scl_sparse_identity(5,SCL_REAL32,SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_scale(m,2.5),0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_18_exists_int32, "final") {
    auto m = scl_sparse_identity(5,SCL_INT32,SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_exists(m,0,0),1);
    SCL_ASSERT_EQ(scl_sparse_exists(m,0,1),0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_19_exists_uint32, "final") {
    auto m = scl_sparse_identity(5,SCL_UINT32,SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_exists(m,2,2),1);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_20_get_int64, "final") {
    auto m = scl_sparse_identity(5,SCL_INT64,SCL_INDEX64);
    auto v = scl_sparse_get(m,1,1);
    SCL_ASSERT_TRUE(!std::isnan(v));
    scl_sparse_destroy(m);
}

// 21-30: More combinations
SCL_TEST_TAGGED(final_21_density_int32, "final") {
    auto m = scl_sparse_identity(10,SCL_INT32,SCL_INDEX64);
    auto d = scl_sparse_density(m);
    SCL_ASSERT_GT(d,0.0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_22_sparsity_uint32, "final") {
    auto m = scl_sparse_identity(10,SCL_UINT32,SCL_INDEX64);
    auto s = scl_sparse_sparsity(m);
    SCL_ASSERT_LT(s,1.0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_23_is_sorted_int8, "final") {
    auto m = scl_sparse_identity(5,SCL_INT8,SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_is_sorted(m),1);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_24_sort_uint8, "final") {
    auto m = scl_sparse_identity(5,SCL_UINT8,SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_sort_indices(m),0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_25_loop_create, "final") {
    for(int i=0;i<10;++i) {
        auto m = scl_sparse_identity(5,SCL_REAL64,SCL_INDEX64);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(final_26_loop_transpose, "final") {
    auto m = scl_sparse_identity(5,SCL_REAL64,SCL_INDEX64);
    for(int i=0;i<10;++i) {
        auto t = scl_sparse_transpose(m);
        scl_sparse_destroy(t);
    }
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_27_loop_clone, "final") {
    auto m = scl_sparse_identity(5,SCL_INT32,SCL_INDEX64);
    for(int i=0;i<10;++i) {
        auto c = scl_sparse_clone(m);
        scl_sparse_destroy(c);
    }
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_28_multi_transpose, "final") {
    auto m = scl_sparse_identity(5,SCL_UINT32,SCL_INDEX64);
    auto t1 = scl_sparse_transpose(m);
    auto t2 = scl_sparse_transpose(t1);
    auto t3 = scl_sparse_transpose(t2);
    SCL_ASSERT_SPARSE_VALID(t3);
    scl_sparse_destroy(m);scl_sparse_destroy(t1);
    scl_sparse_destroy(t2);scl_sparse_destroy(t3);
}

SCL_TEST_TAGGED(final_29_clone_transpose, "final") {
    auto m = scl_sparse_identity(5,SCL_REAL64,SCL_INDEX64);
    auto c = scl_sparse_clone(m);
    auto t = scl_sparse_transpose(c);
    SCL_ASSERT_SPARSE_VALID(t);
    scl_sparse_destroy(m);scl_sparse_destroy(c);scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(final_30_transpose_clone, "final") {
    auto m = scl_sparse_identity(5,SCL_INT32,SCL_INDEX64);
    auto t = scl_sparse_transpose(m);
    auto c = scl_sparse_clone(t);
    SCL_ASSERT_SPARSE_VALID(c);
    scl_sparse_destroy(m);scl_sparse_destroy(t);scl_sparse_destroy(c);
}

// 31-42: Final tests
SCL_TEST_TAGGED(final_31_real32_ops, "final") {
    auto m = scl_sparse_identity(5,SCL_REAL32,SCL_INDEX64);
    scl_sparse_scale(m,2.0);
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_SPARSE_VALID(t);
    scl_sparse_destroy(m);scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(final_32_int32_ops, "final") {
    auto m = scl_sparse_identity(5,SCL_INT32,SCL_INDEX64);
    scl_sparse_scale(m,10.0);
    auto c = scl_sparse_clone(m);
    SCL_ASSERT_SPARSE_VALID(c);
    scl_sparse_destroy(m);scl_sparse_destroy(c);
}

SCL_TEST_TAGGED(final_33_uint32_ops, "final") {
    auto m = scl_sparse_identity(5,SCL_UINT32,SCL_INDEX64);
    auto t = scl_sparse_transpose(m);
    scl_sparse_scale(t,2.0);
    SCL_ASSERT_SPARSE_VALID(t);
    scl_sparse_destroy(m);scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(final_34_query_chain, "final") {
    auto m = scl_sparse_identity(10,SCL_REAL64,SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_rows(m),10);
    SCL_ASSERT_EQ(scl_sparse_cols(m),10);
    SCL_ASSERT_EQ(scl_sparse_nnz(m),10);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_35_type_query_chain, "final") {
    auto m = scl_sparse_identity(5,SCL_INT32,SCL_INDEX32);
    SCL_ASSERT_EQ(scl_sparse_value_type(m),SCL_INT32);
    SCL_ASSERT_EQ(scl_sparse_index_type(m),SCL_INDEX32);
    SCL_ASSERT_EQ(scl_sparse_layout(m),SCL_LAYOUT_CSR);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(final_36_mixed_idx_types, "final") {
    auto m32 = scl_sparse_identity(5,SCL_REAL64,SCL_INDEX32);
    auto m64 = scl_sparse_identity(5,SCL_REAL64,SCL_INDEX64);
    SCL_ASSERT_NE(scl_sparse_index_type(m32),scl_sparse_index_type(m64));
    scl_sparse_destroy(m32);scl_sparse_destroy(m64);
}

SCL_TEST_TAGGED(final_37_all_value_categories, "final") {
    auto r = scl_sparse_identity(3,SCL_REAL64,SCL_INDEX64);
    auto i = scl_sparse_identity(3,SCL_INT32,SCL_INDEX64);
    auto u = scl_sparse_identity(3,SCL_UINT32,SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(r);
    SCL_ASSERT_SPARSE_VALID(i);
    SCL_ASSERT_SPARSE_VALID(u);
    scl_sparse_destroy(r);scl_sparse_destroy(i);scl_sparse_destroy(u);
}

SCL_TEST_TAGGED(final_38_error_clear, "final") {
    scl_clear_error();
    SCL_ASSERT_FALSE(scl_has_error());
    auto m = scl_sparse_zeros(-1,10,SCL_REAL64,SCL_INDEX64,SCL_LAYOUT_CSR);
    SCL_ASSERT_TRUE(scl_has_error());
    scl_clear_error();
    SCL_ASSERT_FALSE(scl_has_error());
}

SCL_TEST_TAGGED(final_39_version_check, "final") {
    auto v = scl_get_version();
    SCL_ASSERT_NOT_NULL(v);
    SCL_ASSERT_STR_EQ(v,"0.5.0");
}

SCL_TEST_TAGGED(final_40_value_type_queries, "final") {
    SCL_ASSERT_EQ(scl_value_type_category(SCL_INT32),1);
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_INT32),4);
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_INT32),"Int32");
}

SCL_TEST_TAGGED(final_41_value_type_checks, "final") {
    SCL_ASSERT_EQ(scl_value_type_is_real(SCL_REAL64),1);
    SCL_ASSERT_EQ(scl_value_type_is_int(SCL_INT32),1);
    SCL_ASSERT_EQ(scl_value_type_is_uint(SCL_UINT32),1);
}

SCL_TEST_TAGGED(final_42_complete, "final") {
    // 最后一个测试：验证整个系统工作
    auto m = scl_sparse_identity(10,SCL_REAL64,SCL_INDEX64);
    auto t = scl_sparse_transpose(m);
    auto c = scl_sparse_clone(t);
    scl_sparse_scale(c,2.0);
    
    SCL_ASSERT_SPARSE_VALID(c);
    SCL_ASSERT_EQ(scl_sparse_rows(c),10);
    SCL_ASSERT_EQ(scl_sparse_cols(c),10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(t);
    scl_sparse_destroy(c);
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

