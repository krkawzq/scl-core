// =============================================================================
// SCL Core - Comprehensive algebra.h Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/algebra.h
//
// Functions tested (15 total):
//   ✓ scl_algebra_spmv
//   ✓ scl_algebra_spmv_simple
//   ✓ scl_algebra_spmv_scaled
//   ✓ scl_algebra_spmv_add
//   ✓ scl_algebra_spmv_transpose
//   ✓ scl_algebra_spmv_transpose_simple
//   ✓ scl_algebra_spmm
//   ✓ scl_algebra_spmv_fused_linear
//   ✓ scl_algebra_row_norms
//   ✓ scl_algebra_row_sums
//   ✓ scl_algebra_extract_diagonal
//   ✓ scl_algebra_scale_rows
//
// =============================================================================

#include "test.hpp"

extern "C" {
#include "scl/binding/c_api/algebra.h"
}

using namespace scl::test;

// Helper: Create 3x3 test matrix (CSR)
static Sparse make_test_matrix_3x3() {
    // [1.0, 0.0, 2.0]
    // [0.0, 3.0, 0.0]
    // [4.0, 5.0, 6.0]
    std::vector<scl_index_t> indptr = {0, 2, 3, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    return make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
}

// Helper: Create 4x3 test matrix (CSR)
static Sparse make_test_matrix_4x3() {
    // [1.0, 2.0, 0.0]
    // [0.0, 3.0, 4.0]
    // [5.0, 0.0, 6.0]
    // [0.0, 7.0, 0.0]
    std::vector<scl_index_t> indptr = {0, 2, 4, 6, 7};
    std::vector<scl_index_t> indices = {0, 1, 1, 2, 0, 2, 1};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    return make_sparse_csr(4, 3, 7, indptr.data(), indices.data(), data.data());
}

SCL_TEST_BEGIN

// =============================================================================
// SpMV Tests
// =============================================================================

SCL_TEST_SUITE(spmv_basic)

SCL_TEST_CASE(spmv_simple_basic) {
    auto mat = make_test_matrix_3x3();

    std::vector<scl_real_t> x = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> y(3, 0.0);

    scl_error_t err = scl_algebra_spmv_simple(
        mat, x.data(), x.size(), y.data(), y.size()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Expected: [1*1 + 0*2 + 2*3, 0*1 + 3*2 + 0*3, 4*1 + 5*2 + 6*3]
    //         = [7.0, 6.0, 32.0]
    SCL_ASSERT_NEAR(y[0], 7.0, 1e-10);
    SCL_ASSERT_NEAR(y[1], 6.0, 1e-10);
    SCL_ASSERT_NEAR(y[2], 32.0, 1e-10);
}

SCL_TEST_CASE(spmv_general_with_alpha_beta) {
    auto mat = make_test_matrix_3x3();

    std::vector<scl_real_t> x = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> y = {1.0, 1.0, 1.0};

    // y = 2.0 * A * x + 3.0 * y
    scl_error_t err = scl_algebra_spmv(
        mat, x.data(), x.size(), y.data(), y.size(), 2.0, 3.0
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Expected: 2*[7, 6, 32] + 3*[1, 1, 1] = [17, 15, 67]
    SCL_ASSERT_NEAR(y[0], 17.0, 1e-10);
    SCL_ASSERT_NEAR(y[1], 15.0, 1e-10);
    SCL_ASSERT_NEAR(y[2], 67.0, 1e-10);
}

SCL_TEST_CASE(spmv_scaled) {
    auto mat = make_test_matrix_3x3();

    std::vector<scl_real_t> x = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> y(3, 0.0);

    scl_error_t err = scl_algebra_spmv_scaled(
        mat, x.data(), x.size(), y.data(), y.size(), 2.5
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Expected: 2.5 * [7, 6, 32]
    SCL_ASSERT_NEAR(y[0], 17.5, 1e-10);
    SCL_ASSERT_NEAR(y[1], 15.0, 1e-10);
    SCL_ASSERT_NEAR(y[2], 80.0, 1e-10);
}

SCL_TEST_CASE(spmv_add) {
    auto mat = make_test_matrix_3x3();

    std::vector<scl_real_t> x = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> y = {10.0, 20.0, 30.0};

    scl_error_t err = scl_algebra_spmv_add(
        mat, x.data(), x.size(), y.data(), y.size()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Expected: [10, 20, 30] + [7, 6, 32]
    SCL_ASSERT_NEAR(y[0], 17.0, 1e-10);
    SCL_ASSERT_NEAR(y[1], 26.0, 1e-10);
    SCL_ASSERT_NEAR(y[2], 62.0, 1e-10);
}

SCL_TEST_CASE(spmv_null_pointer_checks) {
    auto mat = make_test_matrix_3x3();
    std::vector<scl_real_t> x = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> y(3, 0.0);

    // NULL matrix
    SCL_ASSERT_EQ(
        scl_algebra_spmv_simple(nullptr, x.data(), x.size(), y.data(), y.size()),
        SCL_ERROR_NULL_POINTER
    );

    // NULL x
    SCL_ASSERT_EQ(
        scl_algebra_spmv_simple(mat, nullptr, x.size(), y.data(), y.size()),
        SCL_ERROR_NULL_POINTER
    );

    // NULL y
    SCL_ASSERT_EQ(
        scl_algebra_spmv_simple(mat, x.data(), x.size(), nullptr, y.size()),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(spmv_dimension_mismatch) {
    auto mat = make_test_matrix_3x3();

    std::vector<scl_real_t> x_small = {1.0, 2.0};  // Too small
    std::vector<scl_real_t> y(3, 0.0);

    scl_error_t err = scl_algebra_spmv_simple(
        mat, x_small.data(), x_small.size(), y.data(), y.size()
    );

    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(spmv_zero_matrix) {
    std::vector<scl_index_t> indptr = {0, 0, 0, 0};
    std::vector<scl_index_t> indices(1);
    std::vector<scl_real_t> data(1);

    auto mat = make_sparse_csr(3, 3, 0, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> x = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> y(3, 5.0);

    scl_error_t err = scl_algebra_spmv_simple(mat, x.data(), x.size(), y.data(), y.size());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Result should be zero
    SCL_ASSERT_NEAR(y[0], 0.0, 1e-10);
    SCL_ASSERT_NEAR(y[1], 0.0, 1e-10);
    SCL_ASSERT_NEAR(y[2], 0.0, 1e-10);
}

SCL_TEST_SUITE_END

// =============================================================================
// Transposed SpMV Tests
// =============================================================================

SCL_TEST_SUITE(spmv_transpose)

SCL_TEST_CASE(spmv_transpose_simple_basic) {
    auto mat = make_test_matrix_4x3();

    std::vector<scl_real_t> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<scl_real_t> y(3, 0.0);

    scl_error_t err = scl_algebra_spmv_transpose_simple(
        mat, x.data(), x.size(), y.data(), y.size()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // A^T * x where A is 4x3, x is 4-vector, result is 3-vector
    // Column 0: [1, 0, 5, 0] dot [1, 2, 3, 4] = 1 + 0 + 15 + 0 = 16
    // Column 1: [2, 3, 0, 7] dot [1, 2, 3, 4] = 2 + 6 + 0 + 28 = 36
    // Column 2: [0, 4, 6, 0] dot [1, 2, 3, 4] = 0 + 8 + 18 + 0 = 26
    SCL_ASSERT_NEAR(y[0], 16.0, 1e-10);
    SCL_ASSERT_NEAR(y[1], 36.0, 1e-10);
    SCL_ASSERT_NEAR(y[2], 26.0, 1e-10);
}

SCL_TEST_CASE(spmv_transpose_general) {
    auto mat = make_test_matrix_4x3();

    std::vector<scl_real_t> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<scl_real_t> y = {1.0, 1.0, 1.0};

    // y = 2.0 * A^T * x + 3.0 * y
    scl_error_t err = scl_algebra_spmv_transpose(
        mat, x.data(), x.size(), y.data(), y.size(), 2.0, 3.0
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Expected: 2*[16, 36, 26] + 3*[1, 1, 1] = [35, 75, 55]
    SCL_ASSERT_NEAR(y[0], 35.0, 1e-10);
    SCL_ASSERT_NEAR(y[1], 75.0, 1e-10);
    SCL_ASSERT_NEAR(y[2], 55.0, 1e-10);
}

SCL_TEST_CASE(spmv_transpose_null_checks) {
    auto mat = make_test_matrix_4x3();
    std::vector<scl_real_t> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<scl_real_t> y(3, 0.0);

    SCL_ASSERT_EQ(
        scl_algebra_spmv_transpose_simple(nullptr, x.data(), x.size(), y.data(), y.size()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_spmv_transpose_simple(mat, nullptr, x.size(), y.data(), y.size()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_spmv_transpose_simple(mat, x.data(), x.size(), nullptr, y.size()),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// SpMM Tests
// =============================================================================

SCL_TEST_SUITE(spmm)

SCL_TEST_CASE(spmm_basic) {
    auto A = make_test_matrix_3x3();

    // X: 3x2 dense matrix (row-major)
    std::vector<scl_real_t> X = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };

    std::vector<scl_real_t> Y(6, 0.0);  // 3x2 result

    scl_error_t err = scl_algebra_spmm(A, X.data(), 2, Y.data(), 1.0, 0.0);

    SCL_ASSERT_EQ(err, SCL_OK);

    // A * X where A is 3x3, X is 3x2
    // Row 0: [1, 0, 2] * [[1, 2], [3, 4], [5, 6]] = [11, 14]
    // Row 1: [0, 3, 0] * [[1, 2], [3, 4], [5, 6]] = [9, 12]
    // Row 2: [4, 5, 6] * [[1, 2], [3, 4], [5, 6]] = [49, 64]
    SCL_ASSERT_NEAR(Y[0], 11.0, 1e-10);
    SCL_ASSERT_NEAR(Y[1], 14.0, 1e-10);
    SCL_ASSERT_NEAR(Y[2], 9.0, 1e-10);
    SCL_ASSERT_NEAR(Y[3], 12.0, 1e-10);
    SCL_ASSERT_NEAR(Y[4], 49.0, 1e-10);
    SCL_ASSERT_NEAR(Y[5], 64.0, 1e-10);
}

SCL_TEST_CASE(spmm_with_alpha_beta) {
    auto A = make_test_matrix_3x3();

    std::vector<scl_real_t> X = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> Y = {10.0, 20.0, 30.0};

    // Y = 2.0 * A * X + 3.0 * Y
    scl_error_t err = scl_algebra_spmm(A, X.data(), 1, Y.data(), 2.0, 3.0);

    SCL_ASSERT_EQ(err, SCL_OK);

    // A * [1, 2, 3]^T = [7, 6, 32]
    // Result: 2*[7, 6, 32] + 3*[10, 20, 30] = [44, 72, 154]
    SCL_ASSERT_NEAR(Y[0], 44.0, 1e-10);
    SCL_ASSERT_NEAR(Y[1], 72.0, 1e-10);
    SCL_ASSERT_NEAR(Y[2], 154.0, 1e-10);
}

SCL_TEST_CASE(spmm_null_checks) {
    auto A = make_test_matrix_3x3();
    std::vector<scl_real_t> X = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> Y(3, 0.0);

    SCL_ASSERT_EQ(
        scl_algebra_spmm(nullptr, X.data(), 1, Y.data(), 1.0, 0.0),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_spmm(A, nullptr, 1, Y.data(), 1.0, 0.0),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_spmm(A, X.data(), 1, nullptr, 1.0, 0.0),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Fused Operations
// =============================================================================

SCL_TEST_SUITE(fused_operations)

SCL_TEST_CASE(spmv_fused_linear_basic) {
    auto A = make_test_matrix_3x3();

    std::vector<scl_real_t> x = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> z = {2.0, 3.0, 4.0};
    std::vector<scl_real_t> y = {1.0, 1.0, 1.0};

    // y = 1.0 * A * x + 2.0 * A * z + 3.0 * y
    scl_error_t err = scl_algebra_spmv_fused_linear(
        A,
        x.data(), x.size(),
        z.data(), z.size(),
        y.data(), y.size(),
        1.0, 2.0, 3.0
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // A * x = [7, 6, 32]
    // A * z = [10, 9, 47]
    // Result: 1*[7, 6, 32] + 2*[10, 9, 47] + 3*[1, 1, 1]
    //       = [7, 6, 32] + [20, 18, 94] + [3, 3, 3]
    //       = [30, 27, 129]
    SCL_ASSERT_NEAR(y[0], 30.0, 1e-10);
    SCL_ASSERT_NEAR(y[1], 27.0, 1e-10);
    SCL_ASSERT_NEAR(y[2], 129.0, 1e-10);
}

SCL_TEST_CASE(spmv_fused_null_checks) {
    auto A = make_test_matrix_3x3();
    std::vector<scl_real_t> x = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> z = {2.0, 3.0, 4.0};
    std::vector<scl_real_t> y(3, 1.0);

    SCL_ASSERT_EQ(
        scl_algebra_spmv_fused_linear(
            nullptr, x.data(), x.size(), z.data(), z.size(), y.data(), y.size(), 1.0, 2.0, 3.0
        ),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_spmv_fused_linear(
            A, nullptr, x.size(), z.data(), z.size(), y.data(), y.size(), 1.0, 2.0, 3.0
        ),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_spmv_fused_linear(
            A, x.data(), x.size(), nullptr, z.size(), y.data(), y.size(), 1.0, 2.0, 3.0
        ),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_spmv_fused_linear(
            A, x.data(), x.size(), z.data(), z.size(), nullptr, y.size(), 1.0, 2.0, 3.0
        ),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Row Operations
// =============================================================================

SCL_TEST_SUITE(row_operations)

SCL_TEST_CASE(row_norms_basic) {
    auto mat = make_test_matrix_3x3();

    std::vector<scl_real_t> norms(3, 0.0);

    scl_error_t err = scl_algebra_row_norms(mat, norms.data(), norms.size());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Row 0: [1, 0, 2] -> sqrt(1 + 4) = sqrt(5)
    // Row 1: [0, 3, 0] -> sqrt(9) = 3
    // Row 2: [4, 5, 6] -> sqrt(16 + 25 + 36) = sqrt(77)
    SCL_ASSERT_NEAR(norms[0], std::sqrt(5.0), 1e-10);
    SCL_ASSERT_NEAR(norms[1], 3.0, 1e-10);
    SCL_ASSERT_NEAR(norms[2], std::sqrt(77.0), 1e-10);
}

SCL_TEST_CASE(row_sums_basic) {
    auto mat = make_test_matrix_3x3();

    std::vector<scl_real_t> sums(3, 0.0);

    scl_error_t err = scl_algebra_row_sums(mat, sums.data(), sums.size());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Row 0: 1 + 0 + 2 = 3
    // Row 1: 0 + 3 + 0 = 3
    // Row 2: 4 + 5 + 6 = 15
    SCL_ASSERT_NEAR(sums[0], 3.0, 1e-10);
    SCL_ASSERT_NEAR(sums[1], 3.0, 1e-10);
    SCL_ASSERT_NEAR(sums[2], 15.0, 1e-10);
}

SCL_TEST_CASE(extract_diagonal_basic) {
    auto mat = make_test_matrix_3x3();

    std::vector<scl_real_t> diag(3, 0.0);

    scl_error_t err = scl_algebra_extract_diagonal(mat, diag.data(), diag.size());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Diagonal: [1, 3, 6]
    SCL_ASSERT_NEAR(diag[0], 1.0, 1e-10);
    SCL_ASSERT_NEAR(diag[1], 3.0, 1e-10);
    SCL_ASSERT_NEAR(diag[2], 6.0, 1e-10);
}

SCL_TEST_CASE(extract_diagonal_rectangular) {
    auto mat = make_test_matrix_4x3();

    std::vector<scl_real_t> diag(3, 0.0);  // min(4, 3) = 3

    scl_error_t err = scl_algebra_extract_diagonal(mat, diag.data(), diag.size());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Diagonal: [1, 3, 6]
    SCL_ASSERT_NEAR(diag[0], 1.0, 1e-10);
    SCL_ASSERT_NEAR(diag[1], 3.0, 1e-10);
    SCL_ASSERT_NEAR(diag[2], 6.0, 1e-10);
}

SCL_TEST_CASE(scale_rows_basic) {
    // Create a modifiable copy
    std::vector<scl_index_t> indptr = {0, 2, 3, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    auto mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> scale_factors = {2.0, 3.0, 0.5};

    scl_error_t err = scl_algebra_scale_rows(mat, scale_factors.data(), scale_factors.size());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Export and verify
    scl_index_t nnz;
    scl_sparse_nnz(mat, &nnz);

    std::vector<scl_real_t> result_data(nnz);
    std::vector<scl_index_t> result_indices(nnz);
    std::vector<scl_index_t> result_indptr(4);

    scl_bool_t is_csr;
    scl_sparse_export(
        mat,
        result_indptr.data(), result_indices.data(), result_data.data(),
        &is_csr
    );

    // Row 0 scaled by 2.0: [2.0, 4.0]
    // Row 1 scaled by 3.0: [9.0]
    // Row 2 scaled by 0.5: [2.0, 2.5, 3.0]
    SCL_ASSERT_NEAR(result_data[0], 2.0, 1e-10);
    SCL_ASSERT_NEAR(result_data[1], 4.0, 1e-10);
    SCL_ASSERT_NEAR(result_data[2], 9.0, 1e-10);
    SCL_ASSERT_NEAR(result_data[3], 2.0, 1e-10);
    SCL_ASSERT_NEAR(result_data[4], 2.5, 1e-10);
    SCL_ASSERT_NEAR(result_data[5], 3.0, 1e-10);
}

SCL_TEST_CASE(row_operations_null_checks) {
    auto mat = make_test_matrix_3x3();
    std::vector<scl_real_t> output(3, 0.0);

    SCL_ASSERT_EQ(
        scl_algebra_row_norms(nullptr, output.data(), output.size()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_row_norms(mat, nullptr, output.size()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_row_sums(nullptr, output.data(), output.size()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_extract_diagonal(nullptr, output.data(), output.size()),
        SCL_ERROR_NULL_POINTER
    );

    std::vector<scl_real_t> scales = {1.0, 1.0, 1.0};
    SCL_ASSERT_EQ(
        scl_algebra_scale_rows(nullptr, scales.data(), scales.size()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_algebra_scale_rows(mat, nullptr, scales.size()),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(row_operations_size_mismatch) {
    auto mat = make_test_matrix_3x3();

    std::vector<scl_real_t> output_small(2, 0.0);  // Too small

    scl_error_t err = scl_algebra_row_norms(mat, output_small.data(), output_small.size());
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Random Matrix Tests
// =============================================================================

SCL_TEST_SUITE(random_matrices)

SCL_TEST_RETRY(spmv_random_test, 3) {
    Random rng(42);

    auto [rows, cols] = random_shape(10, 50, rng);
    auto mat = random_sparse_csr(rows, cols, 0.1, rng);

    std::vector<scl_real_t> x(cols);
    std::vector<scl_real_t> y(rows, 0.0);

    for (auto& v : x) v = rng.uniform(-1.0, 1.0);

    scl_error_t err = scl_algebra_spmv_simple(mat, x.data(), x.size(), y.data(), y.size());

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify result is finite
    for (scl_index_t i = 0; i < rows; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(y[i]));
    }
}

SCL_TEST_RETRY(row_operations_random, 3) {
    Random rng(123);

    auto [rows, cols] = random_shape(20, 100, rng);
    auto mat = random_sparse_csr(rows, cols, 0.05, rng);

    std::vector<scl_real_t> norms(rows);
    std::vector<scl_real_t> sums(rows);

    scl_error_t err1 = scl_algebra_row_norms(mat, norms.data(), norms.size());
    scl_error_t err2 = scl_algebra_row_sums(mat, sums.data(), sums.size());

    SCL_ASSERT_EQ(err1, SCL_OK);
    SCL_ASSERT_EQ(err2, SCL_OK);

    // All should be non-negative and finite
    for (scl_index_t i = 0; i < rows; ++i) {
        SCL_ASSERT_GE(norms[i], 0.0);
        SCL_ASSERT_TRUE(std::isfinite(norms[i]));
        SCL_ASSERT_TRUE(std::isfinite(sums[i]));
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
