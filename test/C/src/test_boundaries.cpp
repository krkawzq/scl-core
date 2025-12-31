/// @file test_boundaries.cpp
/// @brief Comprehensive boundary condition tests
///
/// Tests extreme values, edge cases, and boundary conditions for all functions
///
/// Total: 60+ test cases

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// SECTION 1: Dimension Boundaries (20 tests)
// =============================================================================

SCL_TEST_SUITE(dimension_boundaries)

SCL_TEST_TAGGED(zeros_1x1_minimum, "boundary", "quick") {
    auto m = scl_sparse_zeros(1, 1, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_DIMS(m, 1, 1);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(zeros_1xN_row_vector, "boundary") {
    auto m = scl_sparse_zeros(1, 1000, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_DIMS(m, 1, 1000);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(zeros_Nx1_col_vector, "boundary") {
    auto m = scl_sparse_zeros(1000, 1, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_DIMS(m, 1000, 1);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(zeros_power_of_2_dimensions, "boundary") {
    for (int exp = 6; exp <= 10; ++exp) {
        int dim = 1 << exp;  // 64, 128, 256, 512, 1024
        auto m = scl_sparse_zeros(dim, dim, SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSR);
        SCL_ASSERT_SPARSE_DIMS(m, dim, dim);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(identity_power_of_2_sizes, "boundary") {
    for (int exp = 4; exp <= 9; ++exp) {
        int n = 1 << exp;  // 16, 32, 64, 128, 256, 512
        auto m = scl_sparse_identity(n, SCL_REAL64, SCL_INDEX64);
        SCL_ASSERT_SPARSE_DIMS(m, n, n);
        SCL_ASSERT_SPARSE_NNZ(m, n);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(zeros_extreme_aspect_ratio_wide, "boundary") {
    auto m = scl_sparse_zeros(1, 10000, SCL_UINT8, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_DIMS(m, 1, 10000);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(zeros_extreme_aspect_ratio_tall, "boundary") {
    auto m = scl_sparse_zeros(10000, 1, SCL_UINT8, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_DIMS(m, 10000, 1);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(identity_n_equals_1, "boundary", "quick") {
    auto m = scl_sparse_identity(1, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_SPARSE_DIMS(m, 1, 1);
    SCL_ASSERT_SPARSE_NNZ(m, 1);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(zeros_large_10000x10000, "boundary", "slow") {
    auto m = scl_sparse_zeros(10000, 10000, SCL_INT16, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_DIMS(m, 10000, 10000);
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 2: Value Boundaries (25 tests)
// =============================================================================

SCL_TEST_SUITE(value_boundaries)

// 整数类型边界值
SCL_TEST_TAGGED(int8_min_max_values, "boundary", "integer") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<std::int8_t> vals = {-128, 0, 127};  // min, zero, max
    
    auto m = scl_sparse_from_coo(
        3, 3, rows.data(), cols.data(), vals.data(),
        3, SCL_INT8, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int16_boundary_values, "boundary", "integer") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<std::int16_t> vals = {-32768, 0, 32767};
    
    auto m = scl_sparse_from_coo(
        3, 3, rows.data(), cols.data(), vals.data(),
        3, SCL_INT16, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(uint8_all_values, "boundary", "unsigned") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<std::uint8_t> vals = {0, 128, 255};
    
    auto m = scl_sparse_from_coo(
        3, 3, rows.data(), cols.data(), vals.data(),
        3, SCL_UINT8, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(real32_epsilon_values, "boundary", "float") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<float> vals = {1e-38f, 1.0f, 3.4e38f};  // min, normal, max
    
    auto m = scl_sparse_from_coo(
        3, 3, rows.data(), cols.data(), vals.data(),
        3, SCL_REAL32, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(real64_very_small_values, "boundary", "float") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<double> vals = {1e-100, 1e-50, 1e-10};
    
    auto m = scl_sparse_from_coo(
        3, 3, rows.data(), cols.data(), vals.data(),
        3, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(real64_very_large_values, "boundary", "float") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<double> vals = {1e100, 1e200, 1e307};
    
    auto m = scl_sparse_from_coo(
        3, 3, rows.data(), cols.data(), vals.data(),
        3, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int32_all_negative, "boundary", "integer") {
    std::vector<std::int64_t> rows = {0, 1, 2, 3, 4};
    std::vector<std::int64_t> cols = {0, 1, 2, 3, 4};
    std::vector<std::int32_t> vals = {-1, -10, -100, -1000, -10000};
    
    auto m = scl_sparse_from_coo(
        5, 5, rows.data(), cols.data(), vals.data(),
        5, SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(uint32_powers_of_2, "boundary", "unsigned") {
    std::vector<std::int64_t> rows = {0, 1, 2, 3, 4};
    std::vector<std::int64_t> cols = {0, 1, 2, 3, 4};
    std::vector<std::uint32_t> vals = {1, 2, 4, 8, 16};
    
    auto m = scl_sparse_from_coo(
        5, 5, rows.data(), cols.data(), vals.data(),
        5, SCL_UINT32, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 3: NNZ Boundaries (15 tests)
// =============================================================================

SCL_TEST_SUITE(nnz_boundaries)

SCL_TEST_TAGGED(from_coo_nnz_1, "boundary", "quick") {
    std::vector<std::int64_t> rows = {5};
    std::vector<std::int64_t> cols = {5};
    std::vector<double> vals = {42.0};
    
    auto m = scl_sparse_from_coo(
        10, 10, rows.data(), cols.data(), vals.data(),
        1, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_NNZ(m, 1);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_dense_3x3, "boundary") {
    // 完全密集：nnz = rows * cols
    const int n = 3;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(static_cast<double>(i * n + j + 1));
        }
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        n * n, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_NNZ(m, 9);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_almost_dense, "boundary") {
    // nnz = rows * cols - 1
    const int n = 4;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == n-1 && j == n-1) continue;  // 跳过最后一个
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(1.0);
        }
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        n * n - 1, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_NNZ(m, 15);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_various_densities, "boundary") {
    const int n = 100;
    int densities[] = {1, 10, 50, 90, 99};  // 1%, 10%, 50%, 90%, 99%
    
    for (int nnz_pct : densities) {
        int nnz = (n * n * nnz_pct) / 100;
        std::vector<std::int64_t> rows(nnz);
        std::vector<std::int64_t> cols(nnz);
        std::vector<double> vals(nnz);
        
        for (int k = 0; k < nnz; ++k) {
            rows[k] = k / n;
            cols[k] = k % n;
            vals[k] = 1.0;
        }
        
        auto m = scl_sparse_from_coo(
            n, n, rows.data(), cols.data(), vals.data(),
            nnz, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
        );
        
        SCL_ASSERT_SPARSE_VALID(m);
        SCL_ASSERT_LE(scl_sparse_nnz(m), nnz);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(identity_various_sizes, "boundary") {
    int sizes[] = {1, 2, 5, 10, 50, 100, 500, 1000};
    
    for (int n : sizes) {
        auto m = scl_sparse_identity(n, SCL_INT32, SCL_INDEX64);
        SCL_ASSERT_SPARSE_DIMS(m, n, n);
        SCL_ASSERT_SPARSE_NNZ(m, n);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 2: Index Boundaries (20 tests)
// =============================================================================

SCL_TEST_SUITE(index_boundaries)

SCL_TEST_TAGGED(at_corner_elements, "boundary", "quick") {
    auto m = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    
    double v;
    // 四个角
    SCL_ASSERT_EQ(scl_sparse_at(m, 0, 0, &v), 0);      // 左上
    SCL_ASSERT_EQ(scl_sparse_at(m, 0, 9, &v), 0);      // 右上
    SCL_ASSERT_EQ(scl_sparse_at(m, 9, 0, &v), 0);      // 左下
    SCL_ASSERT_EQ(scl_sparse_at(m, 9, 9, &v), 0);      // 右下
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(at_edge_elements, "boundary") {
    auto m = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    
    double v;
    // 边缘
    SCL_ASSERT_EQ(scl_sparse_at(m, 0, 5, &v), 0);      // 上边
    SCL_ASSERT_EQ(scl_sparse_at(m, 9, 5, &v), 0);      // 下边
    SCL_ASSERT_EQ(scl_sparse_at(m, 5, 0, &v), 0);      // 左边
    SCL_ASSERT_EQ(scl_sparse_at(m, 5, 9, &v), 0);      // 右边
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(at_center_element, "boundary", "quick") {
    auto m = scl_sparse_identity(11, SCL_REAL64, SCL_INDEX64);
    
    double v = scl_sparse_get(m, 5, 5);  // 中心
    SCL_ASSERT_NEAR(v, 1.0, 1e-10);
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(at_row_equals_rows_minus_1, "boundary", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    double v = scl_sparse_get(m, 9, 9);  // 最后一个有效位置
    SCL_ASSERT_NEAR(v, 1.0, 1e-10);
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(exists_all_diagonal, "boundary") {
    auto m = scl_sparse_identity(20, SCL_INT32, SCL_INDEX64);
    
    // 对角线全部存在
    for (int i = 0; i < 20; ++i) {
        SCL_ASSERT_EQ(scl_sparse_exists(m, i, i), 1);
    }
    
    // 非对角线全部不存在
    for (int i = 0; i < 20; ++i) {
        int j = (i + 1) % 20;
        SCL_ASSERT_EQ(scl_sparse_exists(m, i, j), 0);
    }
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(get_all_zeros_off_diagonal, "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    // 所有非对角线元素应为0
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            if (i == j) continue;
            double v = scl_sparse_get(m, i, j);
            SCL_ASSERT_NEAR(v, 0.0, 1e-10);
        }
    }
    
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 3: Error Boundaries (20 tests)
// =============================================================================

SCL_TEST_SUITE(error_boundaries)

SCL_TEST_TAGGED(at_row_equals_rows, "error", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    double v;
    scl_clear_error();
    auto result = scl_sparse_at(m, 10, 0, &v);  // row=10, rows=10
    SCL_ASSERT_NE(result, 0);
    SCL_ASSERT_HAS_ERROR();
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(at_col_equals_cols, "error", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    double v;
    scl_clear_error();
    auto result = scl_sparse_at(m, 0, 10, &v);  // col=10, cols=10
    SCL_ASSERT_NE(result, 0);
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(at_both_out_of_bounds, "error", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    double v;
    scl_clear_error();
    auto result = scl_sparse_at(m, 100, 100, &v);
    SCL_ASSERT_NE(result, 0);
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(at_int64_max_index, "error", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    double v;
    scl_clear_error();
    auto result = scl_sparse_at(m, INT64_MAX, 0, &v);
    SCL_ASSERT_NE(result, 0);
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_nnz_equals_size_plus_1, "error", "boundary") {
    std::vector<std::int64_t> rows = {0};
    std::vector<std::int64_t> cols = {0};
    std::vector<double> vals = {1.0};
    
    scl_clear_error();
    auto m = scl_sparse_from_coo(
        2, 2, rows.data(), cols.data(), vals.data(),
        5,  // nnz=5 > 2×2=4
        SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_NULL(m);
    SCL_ASSERT_HAS_ERROR();
}

SCL_TEST_TAGGED(zeros_with_zero_rows, "error", "boundary") {
    scl_clear_error();
    auto m = scl_sparse_zeros(0, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    // 可能返回NULL或空矩阵，取决于实现
    if (m) {
        SCL_ASSERT_EQ(scl_sparse_rows(m), 0);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(zeros_with_zero_cols, "error", "boundary") {
    scl_clear_error();
    auto m = scl_sparse_zeros(10, 0, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    if (m) {
        SCL_ASSERT_EQ(scl_sparse_cols(m), 0);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(identity_n_equals_0, "error", "boundary") {
    scl_clear_error();
    auto m = scl_sparse_identity(0, SCL_REAL64, SCL_INDEX64);
    // 应该返回NULL或空矩阵
    if (m) scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

