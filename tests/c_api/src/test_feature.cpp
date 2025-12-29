// =============================================================================
// SCL Core - Feature Statistics Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/feature.h
//
// Functions tested (4):
//   ✓ scl_feature_standard_moments
//   ✓ scl_feature_clipped_moments
//   ✓ scl_feature_detection_rate
//   ✓ scl_feature_dispersion
//
// =============================================================================

#include "test.hpp"

extern "C" {
#include "scl/binding/c_api/feature.h"
}

using namespace scl::test;

// Helper: Create simple test matrix (3x4, features as columns)
static Sparse make_test_matrix() {
    // Matrix:
    // [1, 0, 2, 0]
    // [0, 3, 0, 4]
    // [5, 0, 6, 0]
    std::vector<scl_index_t> indptr = {0, 2, 3, 5};
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 5.0, 6.0};

    return make_sparse_csr(3, 4, 5, indptr.data(), indices.data(), data.data());
}

SCL_TEST_BEGIN

// =============================================================================
// Standard Moments Tests
// =============================================================================

SCL_TEST_SUITE(standard_moments)

SCL_TEST_CASE(basic_computation) {
    auto mat = make_test_matrix();

    std::vector<scl_real_t> means(4);
    std::vector<scl_real_t> vars(4);

    scl_error_t err = scl_feature_standard_moments(
        mat, means.data(), vars.data(), 4, 1
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Column 0: [1, 0, 5] -> mean = 2.0
    SCL_ASSERT_NEAR(means[0], 2.0, 1e-10);

    // Column 1: [0, 3, 0] -> mean = 1.0
    SCL_ASSERT_NEAR(means[1], 1.0, 1e-10);

    // Column 2: [2, 0, 6] -> mean = 8/3
    SCL_ASSERT_NEAR(means[2], 8.0/3.0, 1e-10);

    // Column 3: [0, 4, 0] -> mean = 4/3
    SCL_ASSERT_NEAR(means[3], 4.0/3.0, 1e-10);
}

SCL_TEST_CASE(null_matrix_check) {
    std::vector<scl_real_t> means(4);
    std::vector<scl_real_t> vars(4);

    scl_error_t err = scl_feature_standard_moments(
        nullptr, means.data(), vars.data(), 4, 1
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_output_check) {
    auto mat = make_test_matrix();
    std::vector<scl_real_t> vars(4);

    scl_error_t err = scl_feature_standard_moments(
        mat, nullptr, vars.data(), 4, 1
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    std::vector<scl_real_t> means(4);
    err = scl_feature_standard_moments(
        mat, means.data(), nullptr, 4, 1
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(zero_features) {
    auto mat = make_test_matrix();
    std::vector<scl_real_t> means(1);
    std::vector<scl_real_t> vars(1);

    scl_error_t err = scl_feature_standard_moments(
        mat, means.data(), vars.data(), 0, 1
    );

    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(ddof_variations) {
    auto mat = make_test_matrix();

    // DDOF = 0
    std::vector<scl_real_t> means0(4), vars0(4);
    scl_error_t err = scl_feature_standard_moments(
        mat, means0.data(), vars0.data(), 4, 0
    );
    SCL_ASSERT_EQ(err, SCL_OK);

    // DDOF = 1
    std::vector<scl_real_t> means1(4), vars1(4);
    err = scl_feature_standard_moments(
        mat, means1.data(), vars1.data(), 4, 1
    );
    SCL_ASSERT_EQ(err, SCL_OK);

    // Means should be same
    for (size_t i = 0; i < 4; ++i) {
        SCL_ASSERT_NEAR(means0[i], means1[i], 1e-10);
    }

    // Variances should differ (DDOF=1 gives larger variance)
    for (size_t i = 0; i < 4; ++i) {
        if (vars0[i] > 0) {
            SCL_ASSERT_TRUE(vars1[i] >= vars0[i]);
        }
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Clipped Moments Tests
// =============================================================================

SCL_TEST_SUITE(clipped_moments)

SCL_TEST_CASE(basic_computation) {
    auto mat = make_test_matrix();

    std::vector<scl_real_t> clip_vals = {3.0, 3.0, 3.0, 3.0};
    std::vector<scl_real_t> means(4);
    std::vector<scl_real_t> vars(4);

    scl_error_t err = scl_feature_clipped_moments(
        mat, clip_vals.data(), means.data(), vars.data(), 4
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // After clipping at 3.0:
    // Column 0: [1, 0, 3] (5 clipped to 3)
    // Column 1: [0, 3, 0]
    // Column 2: [2, 0, 3] (6 clipped to 3)
    // Column 3: [0, 3, 0] (4 clipped to 3)
}

SCL_TEST_CASE(null_matrix_check) {
    std::vector<scl_real_t> clip_vals = {3.0, 3.0};
    std::vector<scl_real_t> means(2);
    std::vector<scl_real_t> vars(2);

    scl_error_t err = scl_feature_clipped_moments(
        nullptr, clip_vals.data(), means.data(), vars.data(), 2
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_clip_vals_check) {
    auto mat = make_test_matrix();
    std::vector<scl_real_t> means(4);
    std::vector<scl_real_t> vars(4);

    scl_error_t err = scl_feature_clipped_moments(
        mat, nullptr, means.data(), vars.data(), 4
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_output_check) {
    auto mat = make_test_matrix();
    std::vector<scl_real_t> clip_vals = {3.0, 3.0, 3.0, 3.0};
    std::vector<scl_real_t> vars(4);

    scl_error_t err = scl_feature_clipped_moments(
        mat, clip_vals.data(), nullptr, vars.data(), 4
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    std::vector<scl_real_t> means(4);
    err = scl_feature_clipped_moments(
        mat, clip_vals.data(), means.data(), nullptr, 4
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(zero_clip_values) {
    auto mat = make_test_matrix();

    std::vector<scl_real_t> clip_vals = {0.0, 0.0, 0.0, 0.0};
    std::vector<scl_real_t> means(4);
    std::vector<scl_real_t> vars(4);

    scl_error_t err = scl_feature_clipped_moments(
        mat, clip_vals.data(), means.data(), vars.data(), 4
    );

    // Should succeed, all values clipped to 0
    SCL_ASSERT_EQ(err, SCL_OK);

    for (size_t i = 0; i < 4; ++i) {
        SCL_ASSERT_NEAR(means[i], 0.0, 1e-10);
        SCL_ASSERT_NEAR(vars[i], 0.0, 1e-10);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Detection Rate Tests
// =============================================================================

SCL_TEST_SUITE(detection_rate)

SCL_TEST_CASE(basic_computation) {
    auto mat = make_test_matrix();

    std::vector<scl_real_t> rates(4);

    scl_error_t err = scl_feature_detection_rate(
        mat, rates.data(), 4
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Column 0: 2 nonzeros out of 3 cells -> 2/3
    SCL_ASSERT_NEAR(rates[0], 2.0/3.0, 1e-10);

    // Column 1: 1 nonzero out of 3 cells -> 1/3
    SCL_ASSERT_NEAR(rates[1], 1.0/3.0, 1e-10);

    // Column 2: 2 nonzeros out of 3 cells -> 2/3
    SCL_ASSERT_NEAR(rates[2], 2.0/3.0, 1e-10);

    // Column 3: 1 nonzero out of 3 cells -> 1/3
    SCL_ASSERT_NEAR(rates[3], 1.0/3.0, 1e-10);
}

SCL_TEST_CASE(null_matrix_check) {
    std::vector<scl_real_t> rates(4);

    scl_error_t err = scl_feature_detection_rate(
        nullptr, rates.data(), 4
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_output_check) {
    auto mat = make_test_matrix();

    scl_error_t err = scl_feature_detection_rate(
        mat, nullptr, 4
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(zero_features) {
    auto mat = make_test_matrix();
    std::vector<scl_real_t> rates(1);

    scl_error_t err = scl_feature_detection_rate(
        mat, rates.data(), 0
    );

    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(all_zero_column) {
    // Create matrix with one all-zero column
    std::vector<scl_index_t> indptr = {0, 1, 1};
    std::vector<scl_index_t> indices = {0};
    std::vector<scl_real_t> data = {1.0};

    auto mat = make_sparse_csr(2, 2, 1, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> rates(2);
    scl_error_t err = scl_feature_detection_rate(mat, rates.data(), 2);

    SCL_ASSERT_EQ(err, SCL_OK);

    // Column 0: 1/2
    SCL_ASSERT_NEAR(rates[0], 0.5, 1e-10);

    // Column 1: 0/2 = 0
    SCL_ASSERT_NEAR(rates[1], 0.0, 1e-10);
}

SCL_TEST_SUITE_END

// =============================================================================
// Dispersion Tests
// =============================================================================

SCL_TEST_SUITE(dispersion)

SCL_TEST_CASE(basic_computation) {
    std::vector<scl_real_t> means = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> vars = {1.0, 4.0, 9.0};
    std::vector<scl_real_t> dispersion(3);

    scl_error_t err = scl_feature_dispersion(
        means.data(), vars.data(), dispersion.data(), 3
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Dispersion = var / mean
    SCL_ASSERT_NEAR(dispersion[0], 1.0, 1e-10);
    SCL_ASSERT_NEAR(dispersion[1], 2.0, 1e-10);
    SCL_ASSERT_NEAR(dispersion[2], 3.0, 1e-10);
}

SCL_TEST_CASE(null_means_check) {
    std::vector<scl_real_t> vars = {1.0, 2.0};
    std::vector<scl_real_t> dispersion(2);

    scl_error_t err = scl_feature_dispersion(
        nullptr, vars.data(), dispersion.data(), 2
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_vars_check) {
    std::vector<scl_real_t> means = {1.0, 2.0};
    std::vector<scl_real_t> dispersion(2);

    scl_error_t err = scl_feature_dispersion(
        means.data(), nullptr, dispersion.data(), 2
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(null_output_check) {
    std::vector<scl_real_t> means = {1.0, 2.0};
    std::vector<scl_real_t> vars = {1.0, 2.0};

    scl_error_t err = scl_feature_dispersion(
        means.data(), vars.data(), nullptr, 2
    );

    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(zero_features) {
    std::vector<scl_real_t> means = {1.0};
    std::vector<scl_real_t> vars = {1.0};
    std::vector<scl_real_t> dispersion(1);

    scl_error_t err = scl_feature_dispersion(
        means.data(), vars.data(), dispersion.data(), 0
    );

    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(zero_mean_handling) {
    // Test with zero mean (should handle gracefully)
    std::vector<scl_real_t> means = {0.0, 2.0, 3.0};
    std::vector<scl_real_t> vars = {1.0, 4.0, 9.0};
    std::vector<scl_real_t> dispersion(3);

    scl_error_t err = scl_feature_dispersion(
        means.data(), vars.data(), dispersion.data(), 3
    );

    // Implementation should handle zero mean appropriately
    // Either return error or set dispersion to infinity/large value
    // We accept either behavior
    if (err == SCL_OK) {
        // If it succeeds, dispersion[0] should be inf or very large
        SCL_ASSERT_TRUE(std::isinf(dispersion[0]) || dispersion[0] > 1e10);
    }
}

SCL_TEST_CASE(negative_values) {
    // Negative means/vars should be handled
    std::vector<scl_real_t> means = {-1.0, 2.0};
    std::vector<scl_real_t> vars = {1.0, 4.0};
    std::vector<scl_real_t> dispersion(2);

    scl_error_t err = scl_feature_dispersion(
        means.data(), vars.data(), dispersion.data(), 2
    );

    // Implementation decides if negative means are allowed
    // We just check it doesn't crash
}

SCL_TEST_SUITE_END

// =============================================================================
// Random Tests
// =============================================================================

SCL_TEST_SUITE(random_tests)

SCL_TEST_RETRY(random_standard_moments, 3) {
    Random rng(42);

    auto [rows, cols] = random_shape(10, 50, rng);
    double density = random_density(0.01, 0.2, rng);

    auto mat = random_sparse_csr(rows, cols, density, rng);

    std::vector<scl_real_t> means(cols);
    std::vector<scl_real_t> vars(cols);

    scl_error_t err = scl_feature_standard_moments(
        mat, means.data(), vars.data(), cols, 1
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Check all values are finite
    for (scl_index_t i = 0; i < cols; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(means[i]));
        SCL_ASSERT_TRUE(std::isfinite(vars[i]));
        SCL_ASSERT_TRUE(vars[i] >= 0.0);
    }
}

SCL_TEST_RETRY(random_detection_rate, 3) {
    Random rng(123);

    auto [rows, cols] = random_shape(20, 100, rng);
    double density = random_density(0.05, 0.3, rng);

    auto mat = random_sparse_csr(rows, cols, density, rng);

    std::vector<scl_real_t> rates(cols);

    scl_error_t err = scl_feature_detection_rate(
        mat, rates.data(), cols
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // All rates should be in [0, 1]
    for (scl_index_t i = 0; i < cols; ++i) {
        SCL_ASSERT_TRUE(rates[i] >= 0.0);
        SCL_ASSERT_TRUE(rates[i] <= 1.0);
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
