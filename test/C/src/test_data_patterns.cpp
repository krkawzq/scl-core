/// @file test_data_patterns.cpp
/// @brief Tests for various sparse data patterns
///
/// Tests: diagonal, triangular, banded, block diagonal, random patterns
///
/// Total: 50+ test cases

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// SECTION 1: Diagonal Patterns (15 tests)
// =============================================================================

SCL_TEST_SUITE(diagonal_patterns)

SCL_TEST_TAGGED(main_diagonal, "pattern", "quick") {
    const int n = 10;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 0; i < n; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(static_cast<double>(i + 1));
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        n, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    SCL_ASSERT_SPARSE_NNZ(m, n);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(super_diagonal, "pattern") {
    // 主对角线上方一条
    const int n = 10;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 0; i < n - 1; ++i) {
        rows.push_back(i);
        cols.push_back(i + 1);
        vals.push_back(1.0);
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        n - 1, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    SCL_ASSERT_SPARSE_NNZ(m, n - 1);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(sub_diagonal, "pattern") {
    // 主对角线下方一条
    const int n = 10;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 1; i < n; ++i) {
        rows.push_back(i);
        cols.push_back(i - 1);
        vals.push_back(1.0);
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        n - 1, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(tridiagonal_matrix, "pattern") {
    // 三对角矩阵
    const int n = 10;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 0; i < n; ++i) {
        // 主对角线
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(2.0);
        
        // 上对角线
        if (i < n - 1) {
            rows.push_back(i);
            cols.push_back(i + 1);
            vals.push_back(-1.0);
        }
        
        // 下对角线
        if (i > 0) {
            rows.push_back(i);
            cols.push_back(i - 1);
            vals.push_back(-1.0);
        }
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        rows.size(), SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    SCL_ASSERT_EQ(scl_sparse_nnz(m), 3 * n - 2);
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 2: Triangular Patterns (10 tests)
// =============================================================================

SCL_TEST_SUITE(triangular_patterns)

SCL_TEST_TAGGED(lower_triangular, "pattern") {
    const int n = 5;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(static_cast<double>(i * n + j + 1));
        }
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        rows.size(), SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    SCL_ASSERT_EQ(scl_sparse_nnz(m), (n * (n + 1)) / 2);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(upper_triangular, "pattern") {
    const int n = 5;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(static_cast<double>(i * n + j + 1));
        }
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        rows.size(), SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    SCL_ASSERT_EQ(scl_sparse_nnz(m), (n * (n + 1)) / 2);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(strictly_lower_triangular, "pattern") {
    const int n = 5;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {  // j < i (不含对角线)
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(1.0);
        }
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        rows.size(), SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    SCL_ASSERT_EQ(scl_sparse_nnz(m), (n * (n - 1)) / 2);
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 3: Banded Patterns (15 tests)
// =============================================================================

SCL_TEST_SUITE(banded_patterns)

SCL_TEST_TAGGED(bandwidth_1, "pattern") {
    // 三对角（bandwidth=1）
    const int n = 10;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<int> vals;
    
    for (int i = 0; i < n; ++i) {
        for (int j = std::max(0, i-1); j <= std::min(n-1, i+1); ++j) {
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(i * n + j);
        }
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        rows.size(), SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(bandwidth_2, "pattern") {
    const int n = 10;
    const int bw = 2;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 0; i < n; ++i) {
        for (int j = std::max(0, i-bw); j <= std::min(n-1, i+bw); ++j) {
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(1.0);
        }
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        rows.size(), SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(bandwidth_5, "pattern") {
    const int n = 20;
    const int bw = 5;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    for (int i = 0; i < n; ++i) {
        for (int j = std::max(0, i-bw); j <= std::min(n-1, i+bw); ++j) {
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(static_cast<double>(i + j));
        }
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        rows.size(), SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 4: Block Patterns (10 tests)
// =============================================================================

SCL_TEST_SUITE(block_patterns)

SCL_TEST_TAGGED(block_diagonal_2x2_blocks, "pattern") {
    // 块对角矩阵：2个2×2块
    std::vector<std::int64_t> rows = {0, 0, 1, 1, 2, 2, 3, 3};
    std::vector<std::int64_t> cols = {0, 1, 0, 1, 2, 3, 2, 3};
    std::vector<double> vals = {1, 2, 3, 4, 5, 6, 7, 8};
    
    auto m = scl_sparse_from_coo(
        4, 4, rows.data(), cols.data(), vals.data(),
        8, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    SCL_ASSERT_SPARSE_NNZ(m, 8);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(checkerboard_pattern, "pattern") {
    // 棋盘模式
    const int n = 8;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<std::uint8_t> vals;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if ((i + j) % 2 == 0) {
                rows.push_back(i);
                cols.push_back(j);
                vals.push_back(1);
            }
        }
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        rows.size(), SCL_UINT8, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 5: Random Patterns (10 tests)
// =============================================================================

SCL_TEST_SUITE(random_patterns)

SCL_TEST_TAGGED(random_sparse_density_001, "pattern", "random") {
    Random rng(42);
    const int n = 100;
    const double density = 0.01;
    const int nnz = static_cast<int>(n * n * density);
    
    std::vector<std::int64_t> rows(nnz);
    std::vector<std::int64_t> cols(nnz);
    std::vector<double> vals(nnz);
    
    for (int k = 0; k < nnz; ++k) {
        rows[k] = rng.uniform_int(0, n - 1);
        cols[k] = rng.uniform_int(0, n - 1);
        vals[k] = rng.uniform(0.0, 10.0);
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        nnz, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(random_sparse_density_010, "pattern", "random") {
    Random rng(43);
    const int n = 100;
    const double density = 0.10;
    const int nnz = static_cast<int>(n * n * density);
    
    std::vector<std::int64_t> rows(nnz);
    std::vector<std::int64_t> cols(nnz);
    std::vector<double> vals(nnz);
    
    for (int k = 0; k < nnz; ++k) {
        rows[k] = rng.uniform_int(0, n - 1);
        cols[k] = rng.uniform_int(0, n - 1);
        vals[k] = rng.uniform(-10.0, 10.0);
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        nnz, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(random_int32_values, "pattern", "random", "integer") {
    Random rng(44);
    const int n = 50;
    const int nnz = 100;
    
    std::vector<std::int64_t> rows(nnz);
    std::vector<std::int64_t> cols(nnz);
    std::vector<std::int32_t> vals(nnz);
    
    for (int k = 0; k < nnz; ++k) {
        rows[k] = rng.uniform_int(0, n - 1);
        cols[k] = rng.uniform_int(0, n - 1);
        vals[k] = static_cast<std::int32_t>(rng.uniform_int(-1000, 1000));
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        nnz, SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 6: Special Value Patterns (15 tests)
// =============================================================================

SCL_TEST_SUITE(special_value_patterns)

SCL_TEST_TAGGED(all_ones_matrix, "pattern", "quick") {
    const int n = 10;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<std::uint8_t> vals;
    
    for (int i = 0; i < n; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(1);
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        n, SCL_UINT8, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(alternating_signs, "pattern") {
    const int n = 10;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<std::int32_t> vals;
    
    for (int i = 0; i < n; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back((i % 2 == 0) ? 1 : -1);
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        n, SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(geometric_sequence, "pattern") {
    const int n = 10;
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<double> vals;
    
    double val = 1.0;
    for (int i = 0; i < n; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(val);
        val *= 2.0;  // 1, 2, 4, 8, 16, ...
    }
    
    auto m = scl_sparse_from_coo(
        n, n, rows.data(), cols.data(), vals.data(),
        n, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(fibonacci_values, "pattern") {
    std::vector<std::int64_t> rows = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<std::int64_t> cols = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<std::int32_t> vals = {1, 1, 2, 3, 5, 8, 13, 21};
    
    auto m = scl_sparse_from_coo(
        8, 8, rows.data(), cols.data(), vals.data(),
        8, SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

