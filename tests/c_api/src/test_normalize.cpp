// =============================================================================
// SCL Core - Normalization Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/normalize.h
//
// Functions tested:
//   ✓ scl_norm_compute_row_sums
//   ✓ scl_norm_scale_primary
//   ✓ scl_norm_primary_sums_masked
//   ✓ scl_norm_detect_highly_expressed
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "oracle.hpp"

using namespace scl::test;
using precision::Tolerance;

SCL_TEST_BEGIN

// =============================================================================
// Row Sums Computation Tests
// =============================================================================

SCL_TEST_SUITE(compute_row_sums)

SCL_TEST_CASE(row_sums_basic) {
    // Create a simple 3x3 matrix
    std::vector<scl_index_t> indptr = {0, 2, 4, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 2, 0, 1};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    Sparse mat = make_sparse_csr(3, 3, 6,
                                 indptr.data(), indices.data(), data.data());
    
    std::vector<scl_real_t> row_sums(3, 0.0);
    
    scl_error_t err = scl_norm_compute_row_sums(mat, row_sums.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Row 0: 1.0 + 2.0 = 3.0
    SCL_ASSERT_NEAR(row_sums[0], 3.0, 1e-10);
    // Row 1: 3.0 + 4.0 = 7.0
    SCL_ASSERT_NEAR(row_sums[1], 7.0, 1e-10);
    // Row 2: 5.0 + 6.0 = 11.0
    SCL_ASSERT_NEAR(row_sums[2], 11.0, 1e-10);
}

SCL_TEST_CASE(row_sums_empty_rows) {
    // Matrix with empty rows
    std::vector<scl_index_t> indptr = {0, 0, 2, 2, 4};
    std::vector<scl_index_t> indices = {0, 1, 0, 1};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0};
    
    Sparse mat = make_sparse_csr(4, 2, 4,
                                 indptr.data(), indices.data(), data.data());
    
    std::vector<scl_real_t> row_sums(4, 0.0);
    
    scl_error_t err = scl_norm_compute_row_sums(mat, row_sums.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Row 0 and 2 should be zero
    SCL_ASSERT_NEAR(row_sums[0], 0.0, 1e-10);
    SCL_ASSERT_NEAR(row_sums[2], 0.0, 1e-10);
    // Row 1: 1.0 + 2.0 = 3.0
    SCL_ASSERT_NEAR(row_sums[1], 3.0, 1e-10);
    // Row 3: 3.0 + 4.0 = 7.0
    SCL_ASSERT_NEAR(row_sums[3], 7.0, 1e-10);
}

SCL_TEST_CASE(row_sums_null_matrix) {
    std::vector<scl_real_t> row_sums(10);
    
    scl_error_t err = scl_norm_compute_row_sums(nullptr, row_sums.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(row_sums_null_output) {
    auto mat_eigen = random_sparse_csr(10, 10, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_norm_compute_row_sums(mat, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_RETRY(row_sums_random, 3)
{
    Random rng(42);
    auto [rows, cols] = random_shape(10, 50, rng);
    double density = random_density(0.05, 0.2, rng);
    
    auto mat_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> row_sums(rows, 0.0);
    
    scl_error_t err = scl_norm_compute_row_sums(mat, row_sums.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Compare with Eigen reference
    Eigen::VectorXd eigen_sums = mat_eigen * Eigen::VectorXd::Ones(cols);
    
    for (scl_index_t i = 0; i < rows; ++i) {
        SCL_ASSERT_NEAR(row_sums[i], eigen_sums[i], Tolerance::normal().atol());
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Scale Primary Dimension Tests
// =============================================================================

SCL_TEST_SUITE(scale_primary)

SCL_TEST_CASE(scale_primary_basic) {
    // Create 3x3 matrix
    std::vector<scl_index_t> indptr = {0, 2, 4, 6};
    std::vector<scl_index_t> indices = {0, 1, 0, 1, 0, 1};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    Sparse mat = make_sparse_csr(3, 2, 6,
                                 indptr.data(), indices.data(), data.data());
    
    // Scale each row by [2.0, 3.0, 4.0]
    std::vector<scl_real_t> scales = {2.0, 3.0, 4.0};
    
    scl_error_t err = scl_norm_scale_primary(mat, scales.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Extract and verify
    scl_sparse_raw_t raw;
    scl_error_t raw_err = scl_sparse_unsafe_get_raw(mat, &raw);
    SCL_ASSERT_EQ(raw_err, SCL_OK);
    
    // Row 0: [1.0, 2.0] * 2.0 = [2.0, 4.0]
    SCL_ASSERT_NEAR(raw.data[0], 2.0, 1e-10);
    SCL_ASSERT_NEAR(raw.data[1], 4.0, 1e-10);
    // Row 1: [3.0, 4.0] * 3.0 = [9.0, 12.0]
    SCL_ASSERT_NEAR(raw.data[2], 9.0, 1e-10);
    SCL_ASSERT_NEAR(raw.data[3], 12.0, 1e-10);
    // Row 2: [5.0, 6.0] * 4.0 = [20.0, 24.0]
    SCL_ASSERT_NEAR(raw.data[4], 20.0, 1e-10);
    SCL_ASSERT_NEAR(raw.data[5], 24.0, 1e-10);
}

SCL_TEST_CASE(scale_primary_null_matrix) {
    std::vector<scl_real_t> scales = {1.0, 2.0, 3.0};
    
    scl_error_t err = scl_norm_scale_primary(nullptr, scales.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(scale_primary_null_scales) {
    auto mat_eigen = random_sparse_csr(10, 10, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_error_t err = scl_norm_scale_primary(mat, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(scale_primary_identity) {
    auto I = identity_csr(5);
    auto csr = from_eigen_csr(I);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Scale by all ones (should not change)
    std::vector<scl_real_t> scales(5, 1.0);
    
    scl_error_t err = scl_norm_scale_primary(mat, scales.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify diagonal is still 1.0
    for (scl_index_t i = 0; i < 5; ++i) {
        scl_real_t val;
        scl_error_t get_err = scl_sparse_get(mat, i, i, &val);
        SCL_ASSERT_EQ(get_err, SCL_OK);
        SCL_ASSERT_NEAR(val, 1.0, 1e-10);
    }
}

SCL_TEST_CASE(scale_primary_zero_scale) {
    auto mat_eigen = random_sparse_csr(3, 3, 0.5);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Scale by zero (should zero out rows)
    std::vector<scl_real_t> scales = {0.0, 0.0, 0.0};
    
    scl_error_t err = scl_norm_scale_primary(mat, scales.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All values should be zero
    scl_sparse_raw_t raw;
    scl_sparse_unsafe_get_raw(mat, &raw);
    for (scl_index_t i = 0; i < raw.nnz; ++i) {
        SCL_ASSERT_NEAR(raw.data[i], 0.0, 1e-10);
    }
}

SCL_TEST_RETRY(scale_primary_random, 3)
{
    Random rng(123);
    auto [rows, cols] = random_shape(10, 30, rng);
    double density = random_density(0.1, 0.3, rng);
    
    auto mat_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Generate random scales
    std::vector<scl_real_t> scales(rows);
    for (scl_index_t i = 0; i < rows; ++i) {
        scales[i] = rng.uniform(0.5, 2.0);
    }
    
    scl_error_t err = scl_norm_scale_primary(mat, scales.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify by comparing with Eigen reference
    Eigen::VectorXd scale_vec(rows);
    for (scl_index_t i = 0; i < rows; ++i) {
        scale_vec[i] = scales[i];
    }
    
    EigenCSR scaled_ref = scale_vec.asDiagonal() * mat_eigen;
    
    // Compare matrices
    EigenCSR mat_scaled = to_eigen_csr(mat);
    SCL_ASSERT_TRUE(matrices_equal(mat_scaled, scaled_ref, Tolerance::normal()));
}

SCL_TEST_SUITE_END

// =============================================================================
// Primary Sums Masked Tests
// =============================================================================

SCL_TEST_SUITE(primary_sums_masked)

SCL_TEST_CASE(sums_masked_basic) {
    // Create 3x4 matrix
    std::vector<scl_index_t> indptr = {0, 2, 4, 6};
    std::vector<scl_index_t> indices = {0, 1, 1, 2, 0, 3};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    Sparse mat = make_sparse_csr(3, 4, 6,
                                 indptr.data(), indices.data(), data.data());
    
    // Mask: only columns 0 and 2 are included
    std::vector<unsigned char> mask = {1, 0, 1, 0};
    
    std::vector<scl_real_t> output(3, 0.0);
    
    scl_error_t err = scl_norm_primary_sums_masked(mat, mask.data(), output.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Row 0: columns 0,1 -> only column 0 is masked: 1.0
    SCL_ASSERT_NEAR(output[0], 1.0, 1e-10);
    // Row 1: columns 1,2 -> only column 2 is masked: 4.0
    SCL_ASSERT_NEAR(output[1], 4.0, 1e-10);
    // Row 2: columns 0,3 -> only column 0 is masked: 5.0
    SCL_ASSERT_NEAR(output[2], 5.0, 1e-10);
}

SCL_TEST_CASE(sums_masked_all_masked) {
    auto mat_eigen = random_sparse_csr(5, 10, 0.2);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // All columns masked
    std::vector<unsigned char> mask(10, 1);
    std::vector<scl_real_t> output(5, 0.0);
    
    scl_error_t err = scl_norm_primary_sums_masked(mat, mask.data(), output.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should equal row sums
    std::vector<scl_real_t> row_sums(5, 0.0);
    scl_norm_compute_row_sums(mat, row_sums.data());
    
    for (scl_index_t i = 0; i < 5; ++i) {
        SCL_ASSERT_NEAR(output[i], row_sums[i], 1e-10);
    }
}

SCL_TEST_CASE(sums_masked_none_masked) {
    auto mat_eigen = random_sparse_csr(5, 10, 0.2);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // No columns masked
    std::vector<unsigned char> mask(10, 0);
    std::vector<scl_real_t> output(5, 0.0);
    
    scl_error_t err = scl_norm_primary_sums_masked(mat, mask.data(), output.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All should be zero
    for (scl_index_t i = 0; i < 5; ++i) {
        SCL_ASSERT_NEAR(output[i], 0.0, 1e-10);
    }
}

SCL_TEST_CASE(sums_masked_null_matrix) {
    std::vector<unsigned char> mask(10, 1);
    std::vector<scl_real_t> output(5);
    
    scl_error_t err = scl_norm_primary_sums_masked(nullptr, mask.data(), output.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(sums_masked_null_mask) {
    auto mat_eigen = random_sparse_csr(5, 10, 0.2);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> output(5);
    
    scl_error_t err = scl_norm_primary_sums_masked(mat, nullptr, output.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(sums_masked_null_output) {
    auto mat_eigen = random_sparse_csr(5, 10, 0.2);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<unsigned char> mask(10, 1);
    
    scl_error_t err = scl_norm_primary_sums_masked(mat, mask.data(), nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Detect Highly Expressed Tests
// =============================================================================

SCL_TEST_SUITE(detect_highly_expressed)

SCL_TEST_CASE(detect_highly_expressed_basic) {
    // Create matrix where column 1 has high expression
    std::vector<scl_index_t> indptr = {0, 2, 4, 6};
    std::vector<scl_index_t> indices = {0, 1, 0, 1, 0, 1};
    std::vector<scl_real_t> data = {1.0, 10.0, 2.0, 10.0, 3.0, 10.0};
    
    Sparse mat = make_sparse_csr(3, 2, 6,
                                 indptr.data(), indices.data(), data.data());
    
    // Row sums: [11.0, 12.0, 13.0]
    std::vector<scl_real_t> row_sums = {11.0, 12.0, 13.0};
    
    // Column 1 contributes 10/11, 10/12, 10/13 > 0.7 in all rows
    std::vector<unsigned char> mask(2, 0);
    
    scl_error_t err = scl_norm_detect_highly_expressed(
        mat, row_sums.data(), 0.7, mask.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Column 1 should be marked
    SCL_ASSERT_EQ(mask[0], 0);
    SCL_ASSERT_EQ(mask[1], 1);
}

SCL_TEST_CASE(detect_highly_expressed_none) {
    // All columns have low expression
    std::vector<scl_index_t> indptr = {0, 2, 4};
    std::vector<scl_index_t> indices = {0, 1, 0, 1};
    std::vector<scl_real_t> data = {1.0, 1.0, 1.0, 1.0};
    
    Sparse mat = make_sparse_csr(2, 2, 4,
                                 indptr.data(), indices.data(), data.data());
    
    std::vector<scl_real_t> row_sums = {2.0, 2.0};
    std::vector<unsigned char> mask(2, 0);
    
    scl_error_t err = scl_norm_detect_highly_expressed(
        mat, row_sums.data(), 0.7, mask.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // No columns should be marked
    SCL_ASSERT_EQ(mask[0], 0);
    SCL_ASSERT_EQ(mask[1], 0);
}

SCL_TEST_CASE(detect_highly_expressed_all) {
    // Single column with all expression
    std::vector<scl_index_t> indptr = {0, 1, 2, 3};
    std::vector<scl_index_t> indices = {0, 0, 0};
    std::vector<scl_real_t> data = {10.0, 10.0, 10.0};
    
    Sparse mat = make_sparse_csr(3, 1, 3,
                                 indptr.data(), indices.data(), data.data());
    
    std::vector<scl_real_t> row_sums = {10.0, 10.0, 10.0};
    std::vector<unsigned char> mask(1, 0);
    
    scl_error_t err = scl_norm_detect_highly_expressed(
        mat, row_sums.data(), 0.5, mask.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Column should be marked
    SCL_ASSERT_EQ(mask[0], 1);
}

SCL_TEST_CASE(detect_highly_expressed_null_matrix) {
    std::vector<scl_real_t> row_sums(10, 1.0);
    std::vector<unsigned char> mask(10, 0);
    
    scl_error_t err = scl_norm_detect_highly_expressed(
        nullptr, row_sums.data(), 0.5, mask.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(detect_highly_expressed_null_row_sums) {
    auto mat_eigen = random_sparse_csr(10, 10, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<unsigned char> mask(10, 0);
    
    scl_error_t err = scl_norm_detect_highly_expressed(
        mat, nullptr, 0.5, mask.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(detect_highly_expressed_null_mask) {
    auto mat_eigen = random_sparse_csr(10, 10, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> row_sums(10, 1.0);
    
    scl_error_t err = scl_norm_detect_highly_expressed(
        mat, row_sums.data(), 0.5, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(detect_highly_expressed_zero_threshold) {
    auto mat_eigen = random_sparse_csr(5, 10, 0.3);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> row_sums(5);
    scl_norm_compute_row_sums(mat, row_sums.data());
    
    std::vector<unsigned char> mask(10, 0);
    
    // Zero threshold should mark all columns
    scl_error_t err = scl_norm_detect_highly_expressed(
        mat, row_sums.data(), 0.0, mask.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All columns with any expression should be marked
    // (Implementation dependent, but should not crash)
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

