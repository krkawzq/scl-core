// =============================================================================
// SCL Kernel - Multiple Testing Correction Tests
// =============================================================================
//
// 测试 scl/binding/c_api/multiple_testing.h 中的多重检验校正函数
//
// 函数:
//   ✓ scl_benjamini_hochberg (BH FDR correction)
//   ✓ scl_bonferroni (Bonferroni correction)
//   ✓ scl_benjamini_yekutieli (BY FDR correction)
//   ✓ scl_holm_bonferroni (Holm-Bonferroni step-down)
//   ✓ scl_hochberg (Hochberg step-up)
//   ✓ scl_storey_qvalue (Storey q-value estimation)
//   ✓ scl_local_fdr (Local FDR estimation)
//   ✓ scl_count_significant (Count significant tests)
//   ✓ scl_significant_indices (Get significant indices)
//   ✓ scl_neglog10_pvalues (Negative log10 p-values)
//   ✓ scl_fisher_combine (Fisher's method)
//   ✓ scl_stouffer_combine (Stouffer's method)
//
// 参考实现: 标准统计公式
// 精度要求: Tolerance::statistical() (rtol=1e-4, atol=1e-6)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

extern "C" {
#include "scl/binding/c_api/multiple_testing.h"
}

using namespace scl::test;
using precision::Tolerance;

// Reference implementations
namespace reference {

inline void benjamini_hochberg(const std::vector<scl_real_t>& p_values,
                                std::vector<scl_real_t>& adjusted,
                                double fdr_level = 0.05) {
    size_t n = p_values.size();
    adjusted.resize(n);
    
    // Create index-value pairs and sort by p-value
    std::vector<std::pair<scl_real_t, size_t>> indexed;
    for (size_t i = 0; i < n; ++i) {
        indexed.push_back({p_values[i], i});
    }
    std::sort(indexed.begin(), indexed.end());
    
    // Apply BH procedure
    for (size_t i = 0; i < n; ++i) {
        size_t rank = i + 1;
        double adjusted_p = indexed[i].first * n / rank;
        adjusted[indexed[i].second] = adjusted_p;
    }
    
    // Step-up: ensure monotonicity
    for (size_t i = n - 1; i > 0; --i) {
        if (adjusted[i] < adjusted[i - 1]) {
            adjusted[i - 1] = adjusted[i];
        }
    }
    
    // Cap at 1.0
    for (size_t i = 0; i < n; ++i) {
        if (adjusted[i] > 1.0) adjusted[i] = 1.0;
    }
}

inline void bonferroni(const std::vector<scl_real_t>& p_values,
                      std::vector<scl_real_t>& adjusted) {
    size_t n = p_values.size();
    adjusted.resize(n);
    
    for (size_t i = 0; i < n; ++i) {
        adjusted[i] = std::min(p_values[i] * n, 1.0);
    }
}

inline double fisher_combine(const std::vector<scl_real_t>& p_values) {
    double chi2 = 0.0;
    for (auto p : p_values) {
        if (p > 0.0 && p < 1.0) {
            chi2 += -2.0 * std::log(p);
        }
    }
    return chi2;
}

inline void neglog10_pvalues(const std::vector<scl_real_t>& p_values,
                             std::vector<scl_real_t>& neglog) {
    neglog.resize(p_values.size());
    for (size_t i = 0; i < p_values.size(); ++i) {
        if (p_values[i] > 0.0) {
            neglog[i] = -std::log10(p_values[i]);
        } else {
            neglog[i] = std::numeric_limits<scl_real_t>::infinity();
        }
    }
}

} // namespace reference

SCL_TEST_BEGIN

// =============================================================================
// Benjamini-Hochberg FDR Correction
// =============================================================================

SCL_TEST_SUITE(benjamini_hochberg)

SCL_TEST_CASE(bh_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> adjusted(5);
    
    scl_error_t err = scl_benjamini_hochberg(
        p_values.data(), 5, adjusted.data(), 0.05
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Adjusted p-values should be >= original p-values
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(adjusted[i], p_values[i]);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
    
    // Compare with reference
    std::vector<scl_real_t> ref_adjusted;
    reference::benjamini_hochberg(p_values, ref_adjusted, 0.05);
    
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_TRUE(precision::approx_equal(adjusted[i], ref_adjusted[i], Tolerance::statistical()));
    }
}

SCL_TEST_RETRY(bh_random_pvalues, 3)
{
    Random rng(42);
    size_t n = rng.uniform_int(10, 50);
    
    std::vector<scl_real_t> p_values(n);
    for (size_t i = 0; i < n; ++i) {
        p_values[i] = rng.uniform(0.0, 1.0);
    }
    
    std::vector<scl_real_t> adjusted(n);
    double fdr_level = rng.uniform(0.01, 0.1);
    
    scl_error_t err = scl_benjamini_hochberg(
        p_values.data(), n, adjusted.data(), fdr_level
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify properties
    for (size_t i = 0; i < n; ++i) {
        SCL_ASSERT_GE(adjusted[i], 0.0);
        SCL_ASSERT_LE(adjusted[i], 1.0);
        SCL_ASSERT_GE(adjusted[i], p_values[i]);
    }
}

SCL_TEST_CASE(bh_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    std::vector<scl_real_t> adjusted(5);
    
    // NULL p_values
    scl_error_t err1 = scl_benjamini_hochberg(nullptr, 5, adjusted.data(), 0.05);
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    // NULL output
    scl_error_t err2 = scl_benjamini_hochberg(p_values.data(), 5, nullptr, 0.05);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(bh_zero_tests) {
    std::vector<scl_real_t> p_values;
    std::vector<scl_real_t> adjusted;
    
    scl_error_t err = scl_benjamini_hochberg(
        p_values.data(), 0, adjusted.data(), 0.05
    );
    
    // Should handle gracefully
    (void)err;
}

SCL_TEST_SUITE_END

// =============================================================================
// Bonferroni Correction
// =============================================================================

SCL_TEST_SUITE(bonferroni)

SCL_TEST_CASE(bonferroni_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> adjusted(5);
    
    scl_error_t err = scl_bonferroni(p_values.data(), 5, adjusted.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Compare with reference
    std::vector<scl_real_t> ref_adjusted;
    reference::bonferroni(p_values, ref_adjusted);
    
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_TRUE(precision::approx_equal(adjusted[i], ref_adjusted[i], Tolerance::statistical()));
        SCL_ASSERT_GE(adjusted[i], p_values[i]);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
}

SCL_TEST_RETRY(bonferroni_random, 3)
{
    Random rng(123);
    size_t n = rng.uniform_int(10, 50);
    
    std::vector<scl_real_t> p_values(n);
    for (size_t i = 0; i < n; ++i) {
        p_values[i] = rng.uniform(0.0, 1.0);
    }
    
    std::vector<scl_real_t> adjusted(n);
    scl_error_t err = scl_bonferroni(p_values.data(), n, adjusted.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < n; ++i) {
        SCL_ASSERT_GE(adjusted[i], p_values[i]);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
}

SCL_TEST_CASE(bonferroni_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    std::vector<scl_real_t> adjusted(5);
    
    scl_error_t err1 = scl_bonferroni(nullptr, 5, adjusted.data());
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_bonferroni(p_values.data(), 5, nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Benjamini-Yekutieli FDR Correction
// =============================================================================

SCL_TEST_SUITE(benjamini_yekutieli)

SCL_TEST_CASE(by_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> adjusted(5);
    
    scl_error_t err = scl_benjamini_yekutieli(p_values.data(), 5, adjusted.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Adjusted p-values should be >= original
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(adjusted[i], p_values[i]);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
}

SCL_TEST_RETRY(by_random, 3)
{
    Random rng(456);
    size_t n = rng.uniform_int(10, 50);
    
    std::vector<scl_real_t> p_values(n);
    for (size_t i = 0; i < n; ++i) {
        p_values[i] = rng.uniform(0.0, 1.0);
    }
    
    std::vector<scl_real_t> adjusted(n);
    scl_error_t err = scl_benjamini_yekutieli(p_values.data(), n, adjusted.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < n; ++i) {
        SCL_ASSERT_GE(adjusted[i], 0.0);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
}

SCL_TEST_CASE(by_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    std::vector<scl_real_t> adjusted(5);
    
    scl_error_t err1 = scl_benjamini_yekutieli(nullptr, 5, adjusted.data());
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_benjamini_yekutieli(p_values.data(), 5, nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Holm-Bonferroni Step-Down
// =============================================================================

SCL_TEST_SUITE(holm_bonferroni)

SCL_TEST_CASE(holm_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> adjusted(5);
    
    scl_error_t err = scl_holm_bonferroni(p_values.data(), 5, adjusted.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Adjusted p-values should be >= original
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(adjusted[i], p_values[i]);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
}

SCL_TEST_RETRY(holm_random, 3)
{
    Random rng(789);
    size_t n = rng.uniform_int(10, 50);
    
    std::vector<scl_real_t> p_values(n);
    for (size_t i = 0; i < n; ++i) {
        p_values[i] = rng.uniform(0.0, 1.0);
    }
    
    std::vector<scl_real_t> adjusted(n);
    scl_error_t err = scl_holm_bonferroni(p_values.data(), n, adjusted.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < n; ++i) {
        SCL_ASSERT_GE(adjusted[i], 0.0);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
}

SCL_TEST_CASE(holm_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    std::vector<scl_real_t> adjusted(5);
    
    scl_error_t err1 = scl_holm_bonferroni(nullptr, 5, adjusted.data());
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_holm_bonferroni(p_values.data(), 5, nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Hochberg Step-Up
// =============================================================================

SCL_TEST_SUITE(hochberg)

SCL_TEST_CASE(hochberg_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> adjusted(5);
    
    scl_error_t err = scl_hochberg(p_values.data(), 5, adjusted.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(adjusted[i], p_values[i]);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
}

SCL_TEST_RETRY(hochberg_random, 3)
{
    Random rng(111);
    size_t n = rng.uniform_int(10, 50);
    
    std::vector<scl_real_t> p_values(n);
    for (size_t i = 0; i < n; ++i) {
        p_values[i] = rng.uniform(0.0, 1.0);
    }
    
    std::vector<scl_real_t> adjusted(n);
    scl_error_t err = scl_hochberg(p_values.data(), n, adjusted.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < n; ++i) {
        SCL_ASSERT_GE(adjusted[i], 0.0);
        SCL_ASSERT_LE(adjusted[i], 1.0);
    }
}

SCL_TEST_CASE(hochberg_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    std::vector<scl_real_t> adjusted(5);
    
    scl_error_t err1 = scl_hochberg(nullptr, 5, adjusted.data());
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_hochberg(p_values.data(), 5, nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Storey Q-Value Estimation
// =============================================================================

SCL_TEST_SUITE(storey_qvalue)

SCL_TEST_CASE(storey_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> q_values(5);
    
    scl_error_t err = scl_storey_qvalue(p_values.data(), 5, q_values.data(), 0.5);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Q-values should be in [0, 1]
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(q_values[i], 0.0);
        SCL_ASSERT_LE(q_values[i], 1.0);
    }
}

SCL_TEST_RETRY(storey_random, 3)
{
    Random rng(222);
    size_t n = rng.uniform_int(10, 50);
    
    std::vector<scl_real_t> p_values(n);
    for (size_t i = 0; i < n; ++i) {
        p_values[i] = rng.uniform(0.0, 1.0);
    }
    
    std::vector<scl_real_t> q_values(n);
    double lambda = rng.uniform(0.1, 0.9);
    
    scl_error_t err = scl_storey_qvalue(p_values.data(), n, q_values.data(), lambda);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < n; ++i) {
        SCL_ASSERT_GE(q_values[i], 0.0);
        SCL_ASSERT_LE(q_values[i], 1.0);
    }
}

SCL_TEST_CASE(storey_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    std::vector<scl_real_t> q_values(5);
    
    scl_error_t err1 = scl_storey_qvalue(nullptr, 5, q_values.data(), 0.5);
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_storey_qvalue(p_values.data(), 5, nullptr, 0.5);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Local FDR Estimation
// =============================================================================

SCL_TEST_SUITE(local_fdr)

SCL_TEST_CASE(local_fdr_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> lfdr(5);
    
    scl_error_t err = scl_local_fdr(p_values.data(), 5, lfdr.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Local FDR should be in [0, 1]
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_GE(lfdr[i], 0.0);
        SCL_ASSERT_LE(lfdr[i], 1.0);
    }
}

SCL_TEST_RETRY(local_fdr_random, 3)
{
    Random rng(333);
    size_t n = rng.uniform_int(10, 50);
    
    std::vector<scl_real_t> p_values(n);
    for (size_t i = 0; i < n; ++i) {
        p_values[i] = rng.uniform(0.0, 1.0);
    }
    
    std::vector<scl_real_t> lfdr(n);
    scl_error_t err = scl_local_fdr(p_values.data(), n, lfdr.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < n; ++i) {
        SCL_ASSERT_GE(lfdr[i], 0.0);
        SCL_ASSERT_LE(lfdr[i], 1.0);
    }
}

SCL_TEST_CASE(local_fdr_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    std::vector<scl_real_t> lfdr(5);
    
    scl_error_t err1 = scl_local_fdr(nullptr, 5, lfdr.data());
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_local_fdr(p_values.data(), 5, nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Count Significant Tests
// =============================================================================

SCL_TEST_SUITE(count_significant)

SCL_TEST_CASE(count_significant_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    scl_size_t count = 0;
    
    scl_error_t err = scl_count_significant(p_values.data(), 5, 0.05, &count);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should count p-values <= 0.05
    SCL_ASSERT_GE(count, 0);
    SCL_ASSERT_LE(count, 5);
}

SCL_TEST_CASE(count_significant_all_significant) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.02, 0.03, 0.04};
    scl_size_t count = 0;
    
    scl_error_t err = scl_count_significant(p_values.data(), 5, 0.05, &count);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(count, 5);
}

SCL_TEST_CASE(count_significant_none_significant) {
    std::vector<scl_real_t> p_values = {0.1, 0.2, 0.3, 0.4, 0.5};
    scl_size_t count = 0;
    
    scl_error_t err = scl_count_significant(p_values.data(), 5, 0.05, &count);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(count, 0);
}

SCL_TEST_CASE(count_significant_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    scl_size_t count = 0;
    
    scl_error_t err1 = scl_count_significant(nullptr, 5, 0.05, &count);
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_count_significant(p_values.data(), 5, 0.05, nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Significant Indices
// =============================================================================

SCL_TEST_SUITE(significant_indices)

SCL_TEST_CASE(significant_indices_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_index_t> indices(5);
    scl_size_t count = 0;
    
    scl_error_t err = scl_significant_indices(
        p_values.data(), 5, 0.05,
        indices.data(), &count, 5
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_LE(count, 5);
    
    // Verify indices are valid
    for (scl_size_t i = 0; i < count; ++i) {
        SCL_ASSERT_LT(indices[i], 5);
        SCL_ASSERT_LE(p_values[indices[i]], 0.05);
    }
}

SCL_TEST_CASE(significant_indices_max_count) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_index_t> indices(2);  // Smaller buffer
    scl_size_t count = 0;
    
    scl_error_t err = scl_significant_indices(
        p_values.data(), 5, 0.05,
        indices.data(), &count, 2
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_LE(count, 2);
}

SCL_TEST_CASE(significant_indices_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    std::vector<scl_index_t> indices(5);
    scl_size_t count = 0;
    
    scl_error_t err1 = scl_significant_indices(
        nullptr, 5, 0.05, indices.data(), &count, 5
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_significant_indices(
        p_values.data(), 5, 0.05, nullptr, &count, 5
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err3 = scl_significant_indices(
        p_values.data(), 5, 0.05, indices.data(), nullptr, 5
    );
    SCL_ASSERT_EQ(err3, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Negative Log10 P-Values
// =============================================================================

SCL_TEST_SUITE(neglog10_pvalues)

SCL_TEST_CASE(neglog10_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> neglog(5);
    
    scl_error_t err = scl_neglog10_pvalues(p_values.data(), 5, neglog.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Compare with reference
    std::vector<scl_real_t> ref_neglog;
    reference::neglog10_pvalues(p_values, ref_neglog);
    
    for (size_t i = 0; i < 5; ++i) {
        SCL_ASSERT_TRUE(precision::approx_equal(neglog[i], ref_neglog[i], Tolerance::statistical()));
        SCL_ASSERT_GE(neglog[i], 0.0);
    }
}

SCL_TEST_RETRY(neglog10_random, 3)
{
    Random rng(444);
    size_t n = rng.uniform_int(10, 50);
    
    std::vector<scl_real_t> p_values(n);
    for (size_t i = 0; i < n; ++i) {
        p_values[i] = rng.uniform(1e-10, 1.0);
    }
    
    std::vector<scl_real_t> neglog(n);
    scl_error_t err = scl_neglog10_pvalues(p_values.data(), n, neglog.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (size_t i = 0; i < n; ++i) {
        SCL_ASSERT_GE(neglog[i], 0.0);
    }
}

SCL_TEST_CASE(neglog10_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    std::vector<scl_real_t> neglog(5);
    
    scl_error_t err1 = scl_neglog10_pvalues(nullptr, 5, neglog.data());
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_neglog10_pvalues(p_values.data(), 5, nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Fisher's Method
// =============================================================================

SCL_TEST_SUITE(fisher_combine)

SCL_TEST_CASE(fisher_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    scl_real_t chi2_stat = 0.0;
    
    scl_error_t err = scl_fisher_combine(p_values.data(), 5, &chi2_stat);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Compare with reference
    double ref_chi2 = reference::fisher_combine(p_values);
    SCL_ASSERT_TRUE(precision::approx_equal(chi2_stat, ref_chi2, Tolerance::statistical()));
    
    // Chi-squared statistic should be positive
    SCL_ASSERT_GT(chi2_stat, 0.0);
}

SCL_TEST_RETRY(fisher_random, 3)
{
    Random rng(555);
    size_t n = rng.uniform_int(5, 20);
    
    std::vector<scl_real_t> p_values(n);
    for (size_t i = 0; i < n; ++i) {
        p_values[i] = rng.uniform(1e-10, 1.0);
    }
    
    scl_real_t chi2_stat = 0.0;
    scl_error_t err = scl_fisher_combine(p_values.data(), n, &chi2_stat);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GT(chi2_stat, 0.0);
}

SCL_TEST_CASE(fisher_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    scl_real_t chi2_stat = 0.0;
    
    scl_error_t err1 = scl_fisher_combine(nullptr, 5, &chi2_stat);
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_fisher_combine(p_values.data(), 5, nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Stouffer's Method
// =============================================================================

SCL_TEST_SUITE(stouffer_combine)

SCL_TEST_CASE(stouffer_basic) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    scl_real_t z_score = 0.0;
    
    scl_error_t err = scl_stouffer_combine(p_values.data(), 5, nullptr, &z_score);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Z-score can be any real number
    (void)z_score;
}

SCL_TEST_CASE(stouffer_with_weights) {
    std::vector<scl_real_t> p_values = {0.001, 0.01, 0.05, 0.1, 0.5};
    std::vector<scl_real_t> weights = {1.0, 2.0, 1.0, 0.5, 1.0};
    scl_real_t z_score = 0.0;
    
    scl_error_t err = scl_stouffer_combine(p_values.data(), 5, weights.data(), &z_score);
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_RETRY(stouffer_random, 3)
{
    Random rng(666);
    size_t n = rng.uniform_int(5, 20);
    
    std::vector<scl_real_t> p_values(n);
    for (size_t i = 0; i < n; ++i) {
        p_values[i] = rng.uniform(1e-10, 1.0);
    }
    
    scl_real_t z_score = 0.0;
    scl_error_t err = scl_stouffer_combine(p_values.data(), n, nullptr, &z_score);
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(stouffer_null_inputs) {
    std::vector<scl_real_t> p_values(5);
    scl_real_t z_score = 0.0;
    
    scl_error_t err1 = scl_stouffer_combine(nullptr, 5, nullptr, &z_score);
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_stouffer_combine(p_values.data(), 5, nullptr, nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

