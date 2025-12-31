// =============================================================================
// SCL Core v0.5 - Log1p Kernel Tests
// =============================================================================
//
// Tests for log1p transformation kernels:
//   - log1p(x) = log(1 + x)
//   - log2p1(x) = log2(1 + x)
//   - expm1(x) = exp(x) - 1
//
// Coverage:
//   1. Dense array operations (Real32, Real64)
//   2. Sparse matrix operations (CSR, CSC)
//   3. Numerical accuracy verification
//   4. Edge cases (zeros, small values, large values)
//   5. Error handling (NULL pointers, invalid types)
//   6. In-place vs out-of-place operations
//
// =============================================================================

#include "test.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

using namespace scl::test;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

/// Reference implementation: log1p
template<typename T>
T ref_log1p(T x) {
    return std::log1p(x);
}

/// Reference implementation: log2p1
template<typename T>
T ref_log2p1(T x) {
    return std::log1p(x) / std::log(T(2));
}

/// Reference implementation: expm1
template<typename T>
T ref_expm1(T x) {
    return std::expm1(x);
}

/// Check if two floating point values are close
template<typename T>
bool is_close(T a, T b, T rtol = 1e-5, T atol = 1e-8) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0)) return true;
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

} // anonymous namespace

// =============================================================================
// Test Suite Begin
// =============================================================================

SCL_TEST_BEGIN

SCL_TEST_SUITE(log1p_dense_array)

// -----------------------------------------------------------------------------
// Test 1: Basic log1p on dense array (Real64)
// -----------------------------------------------------------------------------

SCL_TEST_CASE(log1p_array_real64_basic) {
    printf("  Testing log1p on Real64 dense array...\n");

    // Input data: [0, 1, 2, 3, 4]
    std::vector<double> input = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> output(input.size());

    // Apply log1p
    int32_t err = scl_log1p_array(
        input.data(),
        static_cast<int64_t>(input.size()),
        output.data(),
        SCL_REAL64
    );

    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    // Verify results
    for (size_t i = 0; i < input.size(); ++i) {
        double expected = ref_log1p(input[i]);
        printf("    [%zu] input=%.6f, output=%.6f, expected=%.6f\n",
               i, input[i], output[i], expected);
        SCL_ASSERT_TRUE(is_close(output[i], expected, 1e-10, 1e-12));
    }
}

// -----------------------------------------------------------------------------
// Test 2: Basic log1p on dense array (Real32)
// -----------------------------------------------------------------------------

SCL_TEST_CASE(log1p_array_real32_basic) {
    printf("  Testing log1p on Real32 dense array...\n");

    std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(input.size());

    int32_t err = scl_log1p_array(
        input.data(),
        static_cast<int64_t>(input.size()),
        output.data(),
        SCL_REAL32
    );

    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    for (size_t i = 0; i < input.size(); ++i) {
        float expected = ref_log1p(input[i]);
        printf("    [%zu] input=%.6f, output=%.6f, expected=%.6f\n",
               i, input[i], output[i], expected);
        SCL_ASSERT_TRUE(is_close(output[i], expected, 1e-5f, 1e-7f));
    }
}

// -----------------------------------------------------------------------------
// Test 3: In-place log1p
// -----------------------------------------------------------------------------

SCL_TEST_CASE(log1p_array_inplace) {
    printf("  Testing in-place log1p...\n");

    std::vector<double> data = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> expected(data.size());

    // Compute expected values
    for (size_t i = 0; i < data.size(); ++i) {
        expected[i] = ref_log1p(data[i]);
    }

    // Apply in-place (input == output)
    int32_t err = scl_log1p_array(
        data.data(),
        static_cast<int64_t>(data.size()),
        data.data(),  // Same pointer for in-place
        SCL_REAL64
    );

    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    // Verify
    for (size_t i = 0; i < data.size(); ++i) {
        printf("    [%zu] result=%.6f, expected=%.6f\n", i, data[i], expected[i]);
        SCL_ASSERT_TRUE(is_close(data[i], expected[i], 1e-10, 1e-12));
    }
}

// -----------------------------------------------------------------------------
// Test 4: Small values (numerical stability)
// -----------------------------------------------------------------------------

SCL_TEST_CASE(log1p_array_small_values) {
    printf("  Testing log1p with small values (numerical stability)...\n");

    // Small values where log1p(x) ≈ x for x << 1
    std::vector<double> input = {
        1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
        -1e-10, -1e-8, -1e-6, -1e-4, -1e-2
    };
    std::vector<double> output(input.size());

    int32_t err = scl_log1p_array(
        input.data(),
        static_cast<int64_t>(input.size()),
        output.data(),
        SCL_REAL64
    );

    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    for (size_t i = 0; i < input.size(); ++i) {
        double expected = ref_log1p(input[i]);
        printf("    [%zu] input=%.3e, output=%.3e, expected=%.3e\n",
               i, input[i], output[i], expected);
        SCL_ASSERT_TRUE(is_close(output[i], expected, 1e-10, 1e-15));
    }
}

// -----------------------------------------------------------------------------
// Test 5: Large array (parallel processing)
// -----------------------------------------------------------------------------

SCL_TEST_CASE(log1p_array_large) {
    printf("  Testing log1p on large array (parallel processing)...\n");

    const size_t N = 10000;
    std::vector<double> input(N);
    std::vector<double> output(N);

    // Fill with test data
    for (size_t i = 0; i < N; ++i) {
        input[i] = static_cast<double>(i) / 1000.0;
    }

    int32_t err = scl_log1p_array(
        input.data(),
        static_cast<int64_t>(N),
        output.data(),
        SCL_REAL64
    );

    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    // Verify first, middle, and last elements
    std::vector<size_t> check_indices = {0, N/4, N/2, 3*N/4, N-1};
    for (size_t idx : check_indices) {
        double expected = ref_log1p(input[idx]);
        printf("    [%zu] input=%.6f, output=%.6f, expected=%.6f\n",
               idx, input[idx], output[idx], expected);
        SCL_ASSERT_TRUE(is_close(output[idx], expected, 1e-10, 1e-12));
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Test Suite: log2p1 transformation
// =============================================================================

SCL_TEST_SUITE(log2p1_dense_array)

SCL_TEST_CASE(log2p1_array_real64_basic) {
    printf("  Testing log2p1 on Real64 dense array...\n");

    std::vector<double> input = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> output(input.size());

    int32_t err = scl_log2p1_array(
        input.data(),
        static_cast<int64_t>(input.size()),
        output.data(),
        SCL_REAL64
    );

    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    for (size_t i = 0; i < input.size(); ++i) {
        double expected = ref_log2p1(input[i]);
        printf("    [%zu] input=%.6f, output=%.6f, expected=%.6f\n",
               i, input[i], output[i], expected);
        SCL_ASSERT_TRUE(is_close(output[i], expected, 1e-10, 1e-12));
    }
}

SCL_TEST_CASE(log2p1_array_real32_basic) {
    printf("  Testing log2p1 on Real32 dense array...\n");

    std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(input.size());

    int32_t err = scl_log2p1_array(
        input.data(),
        static_cast<int64_t>(input.size()),
        output.data(),
        SCL_REAL32
    );

    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    for (size_t i = 0; i < input.size(); ++i) {
        float expected = ref_log2p1(input[i]);
        printf("    [%zu] input=%.6f, output=%.6f, expected=%.6f\n",
               i, input[i], output[i], expected);
        SCL_ASSERT_TRUE(is_close(output[i], expected, 1e-5f, 1e-7f));
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Test Suite: expm1 transformation
// =============================================================================

SCL_TEST_SUITE(expm1_dense_array)

SCL_TEST_CASE(expm1_array_real64_basic) {
    printf("  Testing expm1 on Real64 dense array...\n");

    std::vector<double> input = {0.0, 0.5, 1.0, 1.5, 2.0};
    std::vector<double> output(input.size());

    int32_t err = scl_expm1_array(
        input.data(),
        static_cast<int64_t>(input.size()),
        output.data(),
        SCL_REAL64
    );

    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    for (size_t i = 0; i < input.size(); ++i) {
        double expected = ref_expm1(input[i]);
        printf("    [%zu] input=%.6f, output=%.6f, expected=%.6f\n",
               i, input[i], output[i], expected);
        SCL_ASSERT_TRUE(is_close(output[i], expected, 1e-10, 1e-12));
    }
}

SCL_TEST_CASE(expm1_array_small_values) {
    printf("  Testing expm1 with small values (numerical stability)...\n");

    // For small x, expm1(x) ≈ x
    std::vector<double> input = {
        1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
        -1e-10, -1e-8, -1e-6, -1e-4, -1e-2
    };
    std::vector<double> output(input.size());

    int32_t err = scl_expm1_array(
        input.data(),
        static_cast<int64_t>(input.size()),
        output.data(),
        SCL_REAL64
    );

    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    for (size_t i = 0; i < input.size(); ++i) {
        double expected = ref_expm1(input[i]);
        printf("    [%zu] input=%.3e, output=%.3e, expected=%.3e\n",
               i, input[i], output[i], expected);
        SCL_ASSERT_TRUE(is_close(output[i], expected, 1e-10, 1e-15));
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Test Suite: Sparse Matrix Operations
// =============================================================================

SCL_TEST_SUITE(log1p_sparse_matrix)

SCL_TEST_CASE(log1p_sparse_csr_real64) {
    printf("  Testing log1p on CSR Real64 sparse matrix...\n");

    // Create a simple sparse matrix using COO
    // Matrix:
    //   [1  0  2]
    //   [0  3  0]
    //   [4  0  5]
    int64_t rows[] = {0, 0, 1, 2, 2};
    int64_t cols[] = {0, 2, 1, 0, 2};
    double vals[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int64_t nnz = 5;

    auto mat = scl_sparse_from_coo(
        3, 3,
        rows, cols, vals, nnz,
        SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );

    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_DIMS(mat, 3, 3);
    SCL_ASSERT_SPARSE_NNZ(mat, 5);

    printf("    Created matrix: %s\n", sparse_info(mat).c_str());

    // Apply log1p transformation
    int32_t err = scl_log1p_sparse(mat);
    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    // Extract values and verify
    auto exported = scl_sparse_to_coo(mat);
    SCL_ASSERT_SPARSE_VALID(exported);

    // Get data
    std::vector<double> result_vals(nnz);
    std::vector<int64_t> result_rows(nnz);
    std::vector<int64_t> result_cols(nnz);

    int32_t extract_err = scl_sparse_extract_coo(
        exported,
        result_rows.data(), result_cols.data(), result_vals.data(),
        nnz
    );
    SCL_ASSERT_EQ(extract_err, 0);

    // Verify transformed values
    for (int64_t i = 0; i < nnz; ++i) {
        double expected = ref_log1p(vals[i]);
        printf("    [%lld] original=%.6f, transformed=%.6f, expected=%.6f\n",
               i, vals[i], result_vals[i], expected);
        SCL_ASSERT_TRUE(is_close(result_vals[i], expected, 1e-10, 1e-12));
    }

    scl_sparse_destroy(exported);
    scl_sparse_destroy(mat);
}

SCL_TEST_CASE(log1p_sparse_csc_real32) {
    printf("  Testing log1p on CSC Real32 sparse matrix...\n");

    int64_t rows[] = {0, 1, 2, 0, 1};
    int64_t cols[] = {0, 0, 0, 1, 2};
    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int64_t nnz = 5;

    auto mat = scl_sparse_from_coo(
        3, 3,
        rows, cols, vals, nnz,
        SCL_REAL32, SCL_INDEX64, SCL_LAYOUT_CSC
    );

    SCL_ASSERT_SPARSE_VALID(mat);
    printf("    Created matrix: %s\n", sparse_info(mat).c_str());

    // Apply log1p
    int32_t err = scl_log1p_sparse(mat);
    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    // Export and verify
    auto exported = scl_sparse_to_coo(mat);
    SCL_ASSERT_SPARSE_VALID(exported);

    std::vector<float> result_vals(nnz);
    std::vector<int64_t> result_rows(nnz);
    std::vector<int64_t> result_cols(nnz);

    scl_sparse_extract_coo(
        exported,
        result_rows.data(), result_cols.data(), result_vals.data(),
        nnz
    );

    for (int64_t i = 0; i < nnz; ++i) {
        float expected = ref_log1p(vals[i]);
        printf("    [%lld] original=%.6f, transformed=%.6f, expected=%.6f\n",
               i, vals[i], result_vals[i], expected);
        SCL_ASSERT_TRUE(is_close(result_vals[i], expected, 1e-5f, 1e-7f));
    }

    scl_sparse_destroy(exported);
    scl_sparse_destroy(mat);
}

SCL_TEST_CASE(log2p1_sparse_csr) {
    printf("  Testing log2p1 on CSR sparse matrix...\n");

    int64_t rows[] = {0, 0, 1, 2, 2};
    int64_t cols[] = {0, 2, 1, 0, 2};
    double vals[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    int64_t nnz = 5;

    auto mat = scl_sparse_from_coo(
        3, 3,
        rows, cols, vals, nnz,
        SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );

    SCL_ASSERT_SPARSE_VALID(mat);

    int32_t err = scl_log2p1_sparse(mat);
    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    auto exported = scl_sparse_to_coo(mat);
    std::vector<double> result_vals(nnz);
    std::vector<int64_t> result_rows(nnz);
    std::vector<int64_t> result_cols(nnz);

    scl_sparse_extract_coo(exported, result_rows.data(), result_cols.data(), result_vals.data(), nnz);

    for (int64_t i = 0; i < nnz; ++i) {
        double expected = ref_log2p1(vals[i]);
        printf("    [%lld] original=%.6f, transformed=%.6f, expected=%.6f\n",
               i, vals[i], result_vals[i], expected);
        SCL_ASSERT_TRUE(is_close(result_vals[i], expected, 1e-10, 1e-12));
    }

    scl_sparse_destroy(exported);
    scl_sparse_destroy(mat);
}

SCL_TEST_CASE(expm1_sparse_csr) {
    printf("  Testing expm1 on CSR sparse matrix...\n");

    int64_t rows[] = {0, 0, 1, 2, 2};
    int64_t cols[] = {0, 2, 1, 0, 2};
    double vals[] = {0.0, 0.5, 1.0, 1.5, 2.0};
    int64_t nnz = 5;

    auto mat = scl_sparse_from_coo(
        3, 3,
        rows, cols, vals, nnz,
        SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
    );

    SCL_ASSERT_SPARSE_VALID(mat);

    int32_t err = scl_expm1_sparse(mat);
    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();

    auto exported = scl_sparse_to_coo(mat);
    std::vector<double> result_vals(nnz);
    std::vector<int64_t> result_rows(nnz);
    std::vector<int64_t> result_cols(nnz);

    scl_sparse_extract_coo(exported, result_rows.data(), result_cols.data(), result_vals.data(), nnz);

    for (int64_t i = 0; i < nnz; ++i) {
        double expected = ref_expm1(vals[i]);
        printf("    [%lld] original=%.6f, transformed=%.6f, expected=%.6f\n",
               i, vals[i], result_vals[i], expected);
        SCL_ASSERT_TRUE(is_close(result_vals[i], expected, 1e-10, 1e-12));
    }

    scl_sparse_destroy(exported);
    scl_sparse_destroy(mat);
}

SCL_TEST_SUITE_END

// =============================================================================
// Test Suite: Error Handling
// =============================================================================

SCL_TEST_SUITE(log1p_error_handling)

SCL_TEST_CASE(log1p_array_null_input) {
    printf("  Testing log1p with NULL input pointer...\n");

    std::vector<double> output(5);
    
    scl_clear_error();
    int32_t err = scl_log1p_array(
        nullptr,  // NULL input
        5,
        output.data(),
        SCL_REAL64
    );

    SCL_ASSERT_NE(err, 0);
    SCL_ASSERT_HAS_ERROR();
    printf("    Error (expected): %s\n", get_last_error().c_str());
}

SCL_TEST_CASE(log1p_array_null_output) {
    printf("  Testing log1p with NULL output pointer...\n");

    std::vector<double> input = {1.0, 2.0, 3.0};
    
    scl_clear_error();
    int32_t err = scl_log1p_array(
        input.data(),
        static_cast<int64_t>(input.size()),
        nullptr,  // NULL output
        SCL_REAL64
    );

    SCL_ASSERT_NE(err, 0);
    SCL_ASSERT_HAS_ERROR();
    printf("    Error (expected): %s\n", get_last_error().c_str());
}

SCL_TEST_CASE(log1p_array_negative_size) {
    printf("  Testing log1p with negative size...\n");

    std::vector<double> input = {1.0, 2.0};
    std::vector<double> output(2);
    
    scl_clear_error();
    int32_t err = scl_log1p_array(
        input.data(),
        -1,  // Negative size
        output.data(),
        SCL_REAL64
    );

    SCL_ASSERT_NE(err, 0);
    SCL_ASSERT_HAS_ERROR();
    printf("    Error (expected): %s\n", get_last_error().c_str());
}

SCL_TEST_CASE(log1p_array_zero_size) {
    printf("  Testing log1p with zero size (should succeed)...\n");

    std::vector<double> input = {1.0};
    std::vector<double> output(1);
    
    scl_clear_error();
    int32_t err = scl_log1p_array(
        input.data(),
        0,  // Zero size - should succeed without doing anything
        output.data(),
        SCL_REAL64
    );

    SCL_ASSERT_EQ(err, 0);
    SCL_ASSERT_NO_ERROR();
    printf("    Success (expected)\n");
}

SCL_TEST_CASE(log1p_sparse_null_matrix) {
    printf("  Testing log1p_sparse with NULL matrix...\n");

    scl_clear_error();
    int32_t err = scl_log1p_sparse(nullptr);

    SCL_ASSERT_NE(err, 0);
    SCL_ASSERT_HAS_ERROR();
    printf("    Error (expected): %s\n", get_last_error().c_str());
}

SCL_TEST_SUITE_END

// =============================================================================
// Test End and Main
// =============================================================================

SCL_TEST_END

SCL_TEST_MAIN()
