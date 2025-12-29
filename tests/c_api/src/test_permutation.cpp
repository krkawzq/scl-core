// =============================================================================
// SCL Core - Permutation Testing Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/permutation.h
//
// Functions tested:
//   ✓ scl_perm_correlation_test
//   ✓ scl_perm_fdr_correction_bh
//   ✓ scl_perm_fdr_correction_by
//   ✓ scl_perm_bonferroni_correction
//   ✓ scl_perm_holm_correction
//   ✓ scl_perm_count_significant
//   ✓ scl_perm_get_significant_indices
//   ✓ scl_perm_batch_test
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/permutation.h"

using namespace scl::test;
using precision::Tolerance;

// Reference implementation for correlation
static double reference_correlation(const std::vector<scl_real_t>& x,
                                     const std::vector<scl_real_t>& y) {
    size_t n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    
    for (size_t i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }
    
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    
    double cov = sum_xy / n - mean_x * mean_y;
    double var_x = sum_x2 / n - mean_x * mean_x;
    double var_y = sum_y2 / n - mean_y * mean_y;
    
    if (var_x < 1e-15 || var_y < 1e-15) return 0.0;
    
    return cov / std::sqrt(var_x * var_y);
}

SCL_TEST_BEGIN

// =============================================================================
// Correlation Test Tests
// =============================================================================

SCL_TEST_SUITE(correlation_test)

SCL_TEST_CASE(correlation_test_perfect_positive) {
    // Perfect positive correlation
    std::vector<scl_real_t> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<scl_real_t> y = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    double observed = reference_correlation(x, y);
    scl_real_t p_value = 0.0;
    
    scl_error_t err = scl_perm_correlation_test(
        x.data(), y.data(), x.size(),
        observed, 1000, &p_value, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(p_value, 0.0);
    SCL_ASSERT_LE(p_value, 1.0);
    
    // Perfect correlation should have very low p-value
    SCL_ASSERT_LT(p_value, 0.05);
}

SCL_TEST_CASE(correlation_test_perfect_negative) {
    // Perfect negative correlation
    std::vector<scl_real_t> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<scl_real_t> y = {5.0, 4.0, 3.0, 2.0, 1.0};
    
    double observed = reference_correlation(x, y);
    scl_real_t p_value = 0.0;
    
    scl_error_t err = scl_perm_correlation_test(
        x.data(), y.data(), x.size(),
        observed, 1000, &p_value, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(p_value, 0.0);
    SCL_ASSERT_LE(p_value, 1.0);
    
    // Perfect negative correlation should have very low p-value
    SCL_ASSERT_LT(p_value, 0.05);
}

SCL_TEST_CASE(correlation_test_no_correlation) {
    // Independent random vectors
    Random rng(42);
    std::vector<scl_real_t> x = random_vector(20, rng);
    std::vector<scl_real_t> y = random_vector(20, rng);
    
    double observed = reference_correlation(x, y);
    scl_real_t p_value = 0.0;
    
    scl_error_t err = scl_perm_correlation_test(
        x.data(), y.data(), x.size(),
        observed, 1000, &p_value, 123
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(p_value, 0.0);
    SCL_ASSERT_LE(p_value, 1.0);
    
    // No correlation should have high p-value (usually > 0.05)
    // But this is probabilistic, so we just check it's valid
}

SCL_TEST_CASE(correlation_test_null_x) {
    std::vector<scl_real_t> y(10, 1.0);
    scl_real_t p_value = 0.0;
    
    scl_error_t err = scl_perm_correlation_test(
        nullptr, y.data(), y.size(),
        0.5, 1000, &p_value, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(correlation_test_null_y) {
    std::vector<scl_real_t> x(10, 1.0);
    scl_real_t p_value = 0.0;
    
    scl_error_t err = scl_perm_correlation_test(
        x.data(), nullptr, x.size(),
        0.5, 1000, &p_value, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(correlation_test_null_pvalue) {
    std::vector<scl_real_t> x(10, 1.0);
    std::vector<scl_real_t> y(10, 1.0);
    
    scl_error_t err = scl_perm_correlation_test(
        x.data(), y.data(), x.size(),
        0.5, 1000, nullptr, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(correlation_test_zero_permutations) {
    std::vector<scl_real_t> x = {1.0, 2.0, 3.0};
    std::vector<scl_real_t> y = {1.0, 2.0, 3.0};
    scl_real_t p_value = 0.0;
    
    scl_error_t err = scl_perm_correlation_test(
        x.data(), y.data(), x.size(),
        1.0, 0, &p_value, 42
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(correlation_test_empty_vectors) {
    std::vector<scl_real_t> x;
    std::vector<scl_real_t> y;
    scl_real_t p_value = 0.0;
    
    scl_error_t err = scl_perm_correlation_test(
        x.data(), y.data(), 0,
        0.0, 1000, &p_value, 42
    );
    
    // Empty vectors might be handled differently
    // Just verify no crash
    (void)err;
}

SCL_TEST_RETRY(correlation_test_random, 3)
{
    Random rng(42);
    size_t n = rng.uniform_int(10, 50);
    
    std::vector<scl_real_t> x = random_vector(n, rng);
    std::vector<scl_real_t> y = random_vector(n, rng);
    
    double observed = reference_correlation(x, y);
    scl_real_t p_value = 0.0;
    
    scl_error_t err = scl_perm_correlation_test(
        x.data(), y.data(), n,
        observed, 1000, &p_value, rng.uniform_int(1, 10000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(p_value, 0.0);
    SCL_ASSERT_LE(p_value, 1.0);
}

SCL_TEST_SUITE_END

// =============================================================================
// FDR Correction (Benjamini-Hochberg) Tests
// =============================================================================

SCL_TEST_SUITE(fdr_correction_bh)

SCL_TEST_CASE(fdr_bh_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> q_values(p_values.size(), 0.0);
    
    scl_error_t err = scl_perm_fdr_correction_bh(
        p_values.data(), p_values.size(), q_values.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Q-values should be >= p-values
    for (size_t i = 0; i < p_values.size(); ++i) {
        SCL_ASSERT_GE(q_values[i], p_values[i]);
        SCL_ASSERT_GE(q_values[i], 0.0);
        SCL_ASSERT_LE(q_values[i], 1.0);
    }
    
    // Q-values should be monotonic (non-decreasing)
    for (size_t i = 1; i < p_values.size(); ++i) {
        SCL_ASSERT_GE(q_values[i], q_values[i - 1]);
    }
}

SCL_TEST_CASE(fdr_bh_all_significant) {
    std::vector<scl_real_t> p_values = {0.0001, 0.0002, 0.0003};
    std::vector<scl_real_t> q_values(p_values.size(), 0.0);
    
    scl_error_t err = scl_perm_fdr_correction_bh(
        p_values.data(), p_values.size(), q_values.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All should have low q-values
    for (size_t i = 0; i < p_values.size(); ++i) {
        SCL_ASSERT_LE(q_values[i], 0.05);
    }
}

SCL_TEST_CASE(fdr_bh_none_significant) {
    std::vector<scl_real_t> p_values = {0.5, 0.6, 0.7, 0.8, 0.9};
    std::vector<scl_real_t> q_values(p_values.size(), 0.0);
    
    scl_error_t err = scl_perm_fdr_correction_bh(
        p_values.data(), p_values.size(), q_values.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All should have high q-values
    for (size_t i = 0; i < p_values.size(); ++i) {
        SCL_ASSERT_GE(q_values[i], p_values[i]);
    }
}

SCL_TEST_CASE(fdr_bh_null_pvalues) {
    std::vector<scl_real_t> q_values(10);
    
    scl_error_t err = scl_perm_fdr_correction_bh(
        nullptr, 10, q_values.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(fdr_bh_null_output) {
    std::vector<scl_real_t> p_values = {0.01, 0.02, 0.03};
    
    scl_error_t err = scl_perm_fdr_correction_bh(
        p_values.data(), p_values.size(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(fdr_bh_empty) {
    std::vector<scl_real_t> p_values;
    std::vector<scl_real_t> q_values;
    
    scl_error_t err = scl_perm_fdr_correction_bh(
        p_values.data(), 0, q_values.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// FDR Correction (Benjamini-Yekutieli) Tests
// =============================================================================

SCL_TEST_SUITE(fdr_correction_by)

SCL_TEST_CASE(fdr_by_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> q_values(p_values.size(), 0.0);
    
    scl_error_t err = scl_perm_fdr_correction_by(
        p_values.data(), p_values.size(), q_values.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Q-values should be >= p-values
    for (size_t i = 0; i < p_values.size(); ++i) {
        SCL_ASSERT_GE(q_values[i], p_values[i]);
        SCL_ASSERT_GE(q_values[i], 0.0);
        SCL_ASSERT_LE(q_values[i], 1.0);
    }
    
    // BY correction is more conservative than BH
    // Q-values should be >= BH q-values
}

SCL_TEST_CASE(fdr_by_null_pvalues) {
    std::vector<scl_real_t> q_values(10);
    
    scl_error_t err = scl_perm_fdr_correction_by(
        nullptr, 10, q_values.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(fdr_by_null_output) {
    std::vector<scl_real_t> p_values = {0.01, 0.02, 0.03};
    
    scl_error_t err = scl_perm_fdr_correction_by(
        p_values.data(), p_values.size(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Bonferroni Correction Tests
// =============================================================================

SCL_TEST_SUITE(bonferroni_correction)

SCL_TEST_CASE(bonferroni_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1};
    std::vector<scl_real_t> adjusted(p_values.size(), 0.0);
    
    scl_error_t err = scl_perm_bonferroni_correction(
        p_values.data(), p_values.size(), adjusted.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Adjusted p-values should be >= original
    for (size_t i = 0; i < p_values.size(); ++i) {
        SCL_ASSERT_GE(adjusted[i], p_values[i]);
        SCL_ASSERT_GE(adjusted[i], 0.0);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
    
    // Bonferroni: p_adj = min(p * n, 1.0)
    for (size_t i = 0; i < p_values.size(); ++i) {
        double expected = std::min(p_values[i] * p_values.size(), 1.0);
        SCL_ASSERT_NEAR(adjusted[i], expected, 1e-6);
    }
}

SCL_TEST_CASE(bonferroni_caps_at_one) {
    std::vector<scl_real_t> p_values = {0.5, 0.6, 0.7};
    std::vector<scl_real_t> adjusted(p_values.size(), 0.0);
    
    scl_error_t err = scl_perm_bonferroni_correction(
        p_values.data(), p_values.size(), adjusted.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All should be capped at 1.0
    for (size_t i = 0; i < p_values.size(); ++i) {
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
}

SCL_TEST_CASE(bonferroni_null_pvalues) {
    std::vector<scl_real_t> adjusted(10);
    
    scl_error_t err = scl_perm_bonferroni_correction(
        nullptr, 10, adjusted.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(bonferroni_null_output) {
    std::vector<scl_real_t> p_values = {0.01, 0.02, 0.03};
    
    scl_error_t err = scl_perm_bonferroni_correction(
        p_values.data(), p_values.size(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Holm Correction Tests
// =============================================================================

SCL_TEST_SUITE(holm_correction)

SCL_TEST_CASE(holm_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1};
    std::vector<scl_real_t> adjusted(p_values.size(), 0.0);
    
    scl_error_t err = scl_perm_holm_correction(
        p_values.data(), p_values.size(), adjusted.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Adjusted p-values should be >= original
    for (size_t i = 0; i < p_values.size(); ++i) {
        SCL_ASSERT_GE(adjusted[i], p_values[i]);
        SCL_ASSERT_GE(adjusted[i], 0.0);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
    
    // Holm is less conservative than Bonferroni
    // Adjusted values should be <= Bonferroni
}

SCL_TEST_CASE(holm_null_pvalues) {
    std::vector<scl_real_t> adjusted(10);
    
    scl_error_t err = scl_perm_holm_correction(
        nullptr, 10, adjusted.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(holm_null_output) {
    std::vector<scl_real_t> p_values = {0.01, 0.02, 0.03};
    
    scl_error_t err = scl_perm_holm_correction(
        p_values.data(), p_values.size(), nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Count Significant Tests
// =============================================================================

SCL_TEST_SUITE(count_significant)

SCL_TEST_CASE(count_significant_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    scl_size_t n_sig = 0;
    
    scl_error_t err = scl_perm_count_significant(
        p_values.data(), p_values.size(), 0.05, &n_sig
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should count p-values <= 0.05
    SCL_ASSERT_GE(n_sig, 3);  // At least 0.001, 0.01, 0.05
    SCL_ASSERT_LE(n_sig, 5);
}

SCL_TEST_CASE(count_significant_strict) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    scl_size_t n_sig = 0;
    
    scl_error_t err = scl_perm_count_significant(
        p_values.data(), p_values.size(), 0.001, &n_sig
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Only 0.001 should be significant
    SCL_ASSERT_EQ(n_sig, 1);
}

SCL_TEST_CASE(count_significant_lenient) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    scl_size_t n_sig = 0;
    
    scl_error_t err = scl_perm_count_significant(
        p_values.data(), p_values.size(), 1.0, &n_sig
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All should be significant
    SCL_ASSERT_EQ(n_sig, 5);
}

SCL_TEST_CASE(count_significant_null_pvalues) {
    scl_size_t n_sig = 0;
    
    scl_error_t err = scl_perm_count_significant(
        nullptr, 10, 0.05, &n_sig
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(count_significant_null_output) {
    std::vector<scl_real_t> p_values = {0.01, 0.02, 0.03};
    
    scl_error_t err = scl_perm_count_significant(
        p_values.data(), p_values.size(), 0.05, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Get Significant Indices Tests
// =============================================================================

SCL_TEST_SUITE(get_significant_indices)

SCL_TEST_CASE(get_significant_indices_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_index_t> indices(10, 0);
    scl_size_t n_results = 0;
    
    scl_error_t err = scl_perm_get_significant_indices(
        p_values.data(), p_values.size(), 0.05,
        indices.data(), 10, &n_results
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should find at least 3 significant
    SCL_ASSERT_GE(n_results, 3);
    SCL_ASSERT_LE(n_results, 5);
    
    // Indices should be valid
    for (scl_size_t i = 0; i < n_results; ++i) {
        SCL_ASSERT_GE(indices[i], 0);
        SCL_ASSERT_LT(indices[i], 5);
        SCL_ASSERT_LE(p_values[indices[i]], 0.05);
    }
}

SCL_TEST_CASE(get_significant_indices_limited) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_index_t> indices(2, 0);
    scl_size_t n_results = 0;
    
    scl_error_t err = scl_perm_get_significant_indices(
        p_values.data(), p_values.size(), 0.05,
        indices.data(), 2, &n_results
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should be limited to max_results
    SCL_ASSERT_LE(n_results, 2);
}

SCL_TEST_CASE(get_significant_indices_null_pvalues) {
    std::vector<scl_index_t> indices(10);
    scl_size_t n_results = 0;
    
    scl_error_t err = scl_perm_get_significant_indices(
        nullptr, 10, 0.05,
        indices.data(), 10, &n_results
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(get_significant_indices_null_indices) {
    std::vector<scl_real_t> p_values = {0.01, 0.02, 0.03};
    scl_size_t n_results = 0;
    
    scl_error_t err = scl_perm_get_significant_indices(
        p_values.data(), p_values.size(), 0.05,
        nullptr, 10, &n_results
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(get_significant_indices_null_count) {
    std::vector<scl_real_t> p_values = {0.01, 0.02, 0.03};
    std::vector<scl_index_t> indices(10);
    
    scl_error_t err = scl_perm_get_significant_indices(
        p_values.data(), p_values.size(), 0.05,
        indices.data(), 10, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Batch Permutation Test Tests
// =============================================================================

SCL_TEST_SUITE(batch_test)

SCL_TEST_CASE(batch_test_basic) {
    // Create expression matrix
    auto mat_eigen = random_sparse_csr(50, 20, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Two groups: first 10 cells group 0, rest group 1
    std::vector<scl_index_t> labels(20);
    for (scl_index_t i = 0; i < 20; ++i) {
        labels[i] = (i < 10) ? 0 : 1;
    }
    
    std::vector<scl_real_t> p_values(50, 0.0);
    
    scl_error_t err = scl_perm_batch_test(
        mat, labels.data(), 100, p_values.data(), 42
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // P-values should be in [0, 1]
    for (scl_index_t i = 0; i < 50; ++i) {
        SCL_ASSERT_GE(p_values[i], 0.0);
        SCL_ASSERT_LE(p_values[i], 1.0);
    }
}

SCL_TEST_CASE(batch_test_null_matrix) {
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_real_t> p_values(20);
    
    scl_error_t err = scl_perm_batch_test(
        nullptr, labels.data(), 100, p_values.data(), 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(batch_test_null_labels) {
    auto mat_eigen = random_sparse_csr(20, 10, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_real_t> p_values(20);
    
    scl_error_t err = scl_perm_batch_test(
        mat, nullptr, 100, p_values.data(), 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(batch_test_null_output) {
    auto mat_eigen = random_sparse_csr(20, 10, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> labels(10, 0);
    
    scl_error_t err = scl_perm_batch_test(
        mat, labels.data(), 100, nullptr, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(batch_test_zero_permutations) {
    auto mat_eigen = random_sparse_csr(20, 10, 0.1);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_real_t> p_values(20);
    
    scl_error_t err = scl_perm_batch_test(
        mat, labels.data(), 0, p_values.data(), 42
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_RETRY(batch_test_random, 3)
{
    Random rng(42);
    auto [rows, cols] = random_shape(30, 50, rng);
    double density = random_density(0.05, 0.15, rng);
    
    auto mat_eigen = random_sparse_csr(rows, cols, density, rng);
    auto csr = from_eigen_csr(mat_eigen);
    Sparse mat = make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
    
    // Random group labels
    std::vector<scl_index_t> labels(cols);
    for (scl_index_t i = 0; i < cols; ++i) {
        labels[i] = rng.uniform_int(0, 1);
    }
    
    std::vector<scl_real_t> p_values(rows, 0.0);
    
    scl_error_t err = scl_perm_batch_test(
        mat, labels.data(), 100, p_values.data(), rng.uniform_int(1, 10000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify p-values
    for (scl_index_t i = 0; i < rows; ++i) {
        SCL_ASSERT_GE(p_values[i], 0.0);
        SCL_ASSERT_LE(p_values[i], 1.0);
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

