// =============================================================================
// SCL Core - Highly Variable Genes (HVG) Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/hvg.h
//
// Functions tested:
//   - scl_hvg_compute_moments
//   - scl_hvg_compute_clipped_moments
//   - scl_hvg_select_by_dispersion
//   - scl_hvg_select_by_vst
//
// Reference implementation: Manual computation
// Precision requirement: Tolerance::normal() (rtol=1e-9, atol=1e-12)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/hvg.h"

using namespace scl::test;
using precision::Tolerance;

// Reference implementation for moments
namespace reference {

void compute_moments(
    const EigenCSR& matrix,
    std::vector<scl_real_t>& out_means,
    std::vector<scl_real_t>& out_vars,
    int ddof
) {
    scl_index_t rows = matrix.rows();
    scl_index_t cols = matrix.cols();
    
    out_means.assign(rows, 0.0);
    out_vars.assign(rows, 0.0);
    
    for (scl_index_t i = 0; i < rows; ++i) {
        scl_real_t sum = 0.0;
        scl_real_t sum_sq = 0.0;
        scl_index_t count = 0;
        
        for (EigenCSR::InnerIterator it(matrix, i); it; ++it) {
            scl_real_t v = it.value();
            sum += v;
            sum_sq += v * v;
            count++;
        }
        
        scl_real_t N = static_cast<scl_real_t>(cols);
        if (N > static_cast<scl_real_t>(ddof)) {
            out_means[i] = sum / N;
            scl_real_t mean = out_means[i];
            scl_real_t variance = (sum_sq - N * mean * mean) / (N - static_cast<scl_real_t>(ddof));
            if (variance < 0.0) variance = 0.0;
            out_vars[i] = variance;
        }
    }
}

} // namespace reference

SCL_TEST_BEGIN

// =============================================================================
// Compute Moments Tests
// =============================================================================

SCL_TEST_SUITE(compute_moments)

SCL_TEST_CASE(compute_moments_basic) {
    // Simple 3x3 matrix
    std::vector<scl_index_t> indptr = {0, 2, 3, 5};
    std::vector<scl_index_t> indices = {0, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    Sparse mat = make_sparse_csr(3, 3, 5, indptr.data(), indices.data(), data.data());
    
    std::vector<scl_real_t> means(3);
    std::vector<scl_real_t> vars(3);
    
    scl_error_t err = scl_hvg_compute_moments(mat, means.data(), vars.data(), 1);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Row 0: [1.0, 2.0, 0.0] -> mean = 1.0, var = ((1-1)^2 + (2-1)^2 + (0-1)^2) / 2
    SCL_ASSERT_NEAR(means[0], 1.0, 1e-10);
    SCL_ASSERT_GT(vars[0], 0.0);
}

SCL_TEST_CASE(compute_moments_constant_rows) {
    // Matrix with constant rows
    std::vector<scl_index_t> indptr = {0, 3, 6, 9};
    std::vector<scl_index_t> indices = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<scl_real_t> data = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
    
    Sparse mat = make_sparse_csr(3, 3, 9, indptr.data(), indices.data(), data.data());
    
    std::vector<scl_real_t> means(3);
    std::vector<scl_real_t> vars(3);
    
    scl_error_t err = scl_hvg_compute_moments(mat, means.data(), vars.data(), 1);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All rows have constant values, variance should be 0
    for (scl_index_t i = 0; i < 3; ++i) {
        SCL_ASSERT_NEAR(means[i], 5.0, 1e-10);
        SCL_ASSERT_NEAR(vars[i], 0.0, 1e-10);
    }
}

SCL_TEST_RETRY(compute_moments_random, 5)
{
    Random rng(42);
    auto [rows, cols] = random_shape(10, 50, rng);
    double density = random_density(0.05, 0.15, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> means(rows);
    std::vector<scl_real_t> vars(rows);
    
    scl_error_t err = scl_hvg_compute_moments(mat, means.data(), vars.data(), 1);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Reference implementation
    std::vector<scl_real_t> ref_means(rows);
    std::vector<scl_real_t> ref_vars(rows);
    reference::compute_moments(A_eigen, ref_means, ref_vars, 1);
    
    // Compare
    SCL_ASSERT_TRUE(precision::vectors_equal(means, ref_means, Tolerance::normal()));
    SCL_ASSERT_TRUE(precision::vectors_equal(vars, ref_vars, Tolerance::normal()));
}

SCL_TEST_CASE(compute_moments_different_ddof) {
    Random rng(123);
    auto A_eigen = random_sparse_csr(10, 20, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> means_ddof0(10);
    std::vector<scl_real_t> vars_ddof0(10);
    std::vector<scl_real_t> means_ddof1(10);
    std::vector<scl_real_t> vars_ddof1(10);
    
    scl_error_t err1 = scl_hvg_compute_moments(mat, means_ddof0.data(), vars_ddof0.data(), 0);
    SCL_ASSERT_EQ(err1, SCL_OK);
    
    scl_error_t err2 = scl_hvg_compute_moments(mat, means_ddof1.data(), vars_ddof1.data(), 1);
    SCL_ASSERT_EQ(err2, SCL_OK);
    
    // Means should be the same
    for (scl_index_t i = 0; i < 10; ++i) {
        SCL_ASSERT_NEAR(means_ddof0[i], means_ddof1[i], 1e-10);
    }
    
    // Variances should differ
    for (scl_index_t i = 0; i < 10; ++i) {
        if (vars_ddof0[i] > 1e-10) {
            SCL_ASSERT_GE(vars_ddof1[i], vars_ddof0[i]);
        }
    }
}

SCL_TEST_CASE(compute_moments_null_inputs) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> means(10), vars(10);
    
    scl_error_t err1 = scl_hvg_compute_moments(nullptr, means.data(), vars.data(), 1);
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_hvg_compute_moments(mat, nullptr, vars.data(), 1);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err3 = scl_hvg_compute_moments(mat, means.data(), nullptr, 1);
    SCL_ASSERT_EQ(err3, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Compute Clipped Moments Tests
// =============================================================================

SCL_TEST_SUITE(compute_clipped_moments)

SCL_TEST_CASE(compute_clipped_moments_basic) {
    std::vector<scl_index_t> indptr = {0, 3, 6};
    std::vector<scl_index_t> indices = {0, 1, 2, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 5.0, 10.0, 2.0, 6.0, 11.0};
    
    Sparse mat = make_sparse_csr(2, 3, 6, indptr.data(), indices.data(), data.data());
    
    // Clip values at 5.0
    std::vector<scl_real_t> clip_vals = {5.0, 5.0, 5.0};
    std::vector<scl_real_t> means(2);
    std::vector<scl_real_t> vars(2);
    
    scl_error_t err = scl_hvg_compute_clipped_moments(
        mat, clip_vals.data(), means.data(), vars.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Values should be clipped: [1.0, 5.0, 5.0] for row 0
    SCL_ASSERT_GT(means[0], 0.0);
    SCL_ASSERT_LE(means[0], 5.0);
}

SCL_TEST_RETRY(compute_clipped_moments_random, 3)
{
    Random rng(456);
    auto [rows, cols] = random_shape(10, 30, rng);
    double density = random_density(0.1, 0.2, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Random clip values
    std::vector<scl_real_t> clip_vals(static_cast<size_t>(cols));
    for (scl_index_t j = 0; j < cols; ++j) {
        clip_vals[j] = rng.uniform(1.0, 10.0);
    }
    
    std::vector<scl_real_t> means(rows);
    std::vector<scl_real_t> vars(rows);
    
    scl_error_t err = scl_hvg_compute_clipped_moments(
        mat, clip_vals.data(), means.data(), vars.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All means and vars should be finite
    for (scl_index_t i = 0; i < rows; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(means[i]));
        SCL_ASSERT_TRUE(std::isfinite(vars[i]));
        SCL_ASSERT_GE(vars[i], 0.0);
    }
}

SCL_TEST_CASE(compute_clipped_moments_null_inputs) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> clip_vals(10, 5.0);
    std::vector<scl_real_t> means(10), vars(10);
    
    scl_error_t err1 = scl_hvg_compute_clipped_moments(
        nullptr, clip_vals.data(), means.data(), vars.data()
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_hvg_compute_clipped_moments(
        mat, nullptr, means.data(), vars.data()
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Select by Dispersion Tests
// =============================================================================

SCL_TEST_SUITE(select_by_dispersion)

SCL_TEST_CASE(select_by_dispersion_basic) {
    // Create matrix with varying variance
    std::vector<scl_index_t> indptr = {0, 3, 6, 9};
    std::vector<scl_index_t> indices = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    // Row 0: low variance [1, 1, 1]
    // Row 1: high variance [1, 5, 10]
    // Row 2: medium variance [2, 3, 4]
    std::vector<scl_real_t> data = {
        1.0, 1.0, 1.0,
        1.0, 5.0, 10.0,
        2.0, 3.0, 4.0
    };
    
    Sparse mat = make_sparse_csr(3, 3, 9, indptr.data(), indices.data(), data.data());
    
    scl_size_t n_top = 2;
    std::vector<scl_index_t> indices_out(n_top);
    std::vector<uint8_t> mask(3);
    std::vector<scl_real_t> dispersions(3);
    
    scl_error_t err = scl_hvg_select_by_dispersion(
        mat, n_top, indices_out.data(), mask.data(), dispersions.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Row 1 should be selected (highest variance)
    // Check that mask is set correctly
    int selected_count = 0;
    for (size_t i = 0; i < 3; ++i) {
        if (mask[i] == 1) selected_count++;
    }
    SCL_ASSERT_EQ(selected_count, static_cast<int>(n_top));
}

SCL_TEST_RETRY(select_by_dispersion_random, 3)
{
    Random rng(789);
    auto [rows, cols] = random_shape(20, 50, rng);
    double density = random_density(0.1, 0.2, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t n_top = rng.uniform_int(1, static_cast<int>(rows / 2));
    std::vector<scl_index_t> indices_out(n_top);
    std::vector<uint8_t> mask(rows);
    std::vector<scl_real_t> dispersions(rows);
    
    scl_error_t err = scl_hvg_select_by_dispersion(
        mat, n_top, indices_out.data(), mask.data(), dispersions.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify outputs
    int selected_count = 0;
    for (size_t i = 0; i < rows; ++i) {
        if (mask[i] == 1) {
            selected_count++;
            SCL_ASSERT_TRUE(std::isfinite(dispersions[i]));
        }
    }
    SCL_ASSERT_EQ(selected_count, static_cast<int>(n_top));
    
    // Check indices are valid
    for (size_t i = 0; i < n_top; ++i) {
        SCL_ASSERT_GE(indices_out[i], 0);
        SCL_ASSERT_LT(indices_out[i], static_cast<scl_index_t>(rows));
    }
}

SCL_TEST_CASE(select_by_dispersion_all_selected) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(5, 10, 0.2, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t n_top = 5;  // Select all
    std::vector<scl_index_t> indices_out(n_top);
    std::vector<uint8_t> mask(5);
    std::vector<scl_real_t> dispersions(5);
    
    scl_error_t err = scl_hvg_select_by_dispersion(
        mat, n_top, indices_out.data(), mask.data(), dispersions.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All should be selected
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_EQ(mask[i], 1);
    }
}

SCL_TEST_CASE(select_by_dispersion_null_inputs) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t n_top = 5;
    std::vector<scl_index_t> indices_out(n_top);
    std::vector<uint8_t> mask(10);
    std::vector<scl_real_t> dispersions(10);
    
    scl_error_t err1 = scl_hvg_select_by_dispersion(
        nullptr, n_top, indices_out.data(), mask.data(), dispersions.data()
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_hvg_select_by_dispersion(
        mat, n_top, nullptr, mask.data(), dispersions.data()
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(select_by_dispersion_invalid_n_top) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    scl_size_t n_top = 0;  // Invalid
    std::vector<scl_index_t> indices_out(1);
    std::vector<uint8_t> mask(10);
    std::vector<scl_real_t> dispersions(10);
    
    scl_error_t err = scl_hvg_select_by_dispersion(
        mat, n_top, indices_out.data(), mask.data(), dispersions.data()
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Select by VST Tests
// =============================================================================

SCL_TEST_SUITE(select_by_vst)

SCL_TEST_CASE(select_by_vst_basic) {
    Random rng(111);
    auto A_eigen = random_sparse_csr(10, 20, 0.15, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> clip_vals(20, 5.0);
    scl_size_t n_top = 5;
    std::vector<scl_index_t> indices_out(static_cast<size_t>(n_top));
    std::vector<uint8_t> mask(10);
    std::vector<scl_real_t> variances(10);
    
    scl_error_t err = scl_hvg_select_by_vst(
        mat, clip_vals.data(), n_top,
        indices_out.data(), mask.data(), variances.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify outputs
    int selected_count = 0;
    for (size_t i = 0; i < 10; ++i) {
        if (mask[i] == 1) {
            selected_count++;
            SCL_ASSERT_TRUE(std::isfinite(variances[i]));
        }
    }
    SCL_ASSERT_EQ(selected_count, static_cast<int>(n_top));
}

SCL_TEST_RETRY(select_by_vst_random, 3)
{
    Random rng(222);
    auto [rows, cols] = random_shape(15, 40, rng);
    double density = random_density(0.1, 0.2, rng);
    auto A_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> clip_vals(cols);
    for (size_t j = 0; j < cols; ++j) {
        clip_vals[j] = rng.uniform(1.0, 10.0);
    }
    
    scl_size_t n_top = rng.uniform_int(1, static_cast<int>(rows / 2));
    std::vector<scl_index_t> indices_out(n_top);
    std::vector<uint8_t> mask(rows);
    std::vector<scl_real_t> variances(rows);
    
    scl_error_t err = scl_hvg_select_by_vst(
        mat, clip_vals.data(), n_top,
        indices_out.data(), mask.data(), variances.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(select_by_vst_null_inputs) {
    Random rng(42);
    auto A_eigen = random_sparse_csr(10, 10, 0.1, rng);
    auto csr = from_eigen_csr(A_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> clip_vals(10, 5.0);
    scl_size_t n_top = 5;
    std::vector<scl_index_t> indices_out(n_top);
    std::vector<uint8_t> mask(10);
    std::vector<scl_real_t> variances(10);
    
    scl_error_t err1 = scl_hvg_select_by_vst(
        nullptr, clip_vals.data(), n_top,
        indices_out.data(), mask.data(), variances.data()
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_hvg_select_by_vst(
        mat, nullptr, n_top,
        indices_out.data(), mask.data(), variances.data()
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

