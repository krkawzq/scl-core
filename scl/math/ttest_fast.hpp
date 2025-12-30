#pragma once

/// @file scl/math/ttest_fast.hpp
/// @brief Fast approximate T-Test statistics
///
/// This header provides:
///   - Student's t-test (pooled variance)
///   - Welch's t-test (unequal variances)
///   - Fast p-value approximation
///   - Both scalar and SIMD implementations
///
/// @note DF > 30: normal approximation, DF <= 30: sigmoid heuristic
/// @note Uses fast approximate normal distribution (~1e-7 precision)

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/simd.hpp"
#include "scl/math/stats_fast.hpp"

#include <cmath>

namespace scl::math {

// =============================================================================
// SECTION 1: Standard Error Computations (Scalar)
// =============================================================================

/// @brief Pooled standard error (Student's t-test)
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Pooled standard error
/// @note SE = sqrt(pooled_var * (1/n1 + 1/n2))
[[nodiscard]]
SCL_FORCE_INLINE
auto ttest_se_pooled(
    double var1, double n1,
    double var2, double n2
) -> double {
    double df = n1 + n2 - 2.0;
    if (df <= 0) [[unlikely]] return 0.0;

    double v_pool = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / df;
    return std::sqrt(v_pool * (1.0 / n1 + 1.0 / n2));
}

/// @brief Welch's standard error
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Welch's standard error
/// @note SE = sqrt(var1/n1 + var2/n2)
[[nodiscard]]
SCL_FORCE_INLINE
auto ttest_se_welch(
    double var1, double n1,
    double var2, double n2
) -> double {
    return std::sqrt(var1 / n1 + var2 / n2);
}

/// @brief Welch-Satterthwaite degrees of freedom
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Approximate degrees of freedom
/// @note df = (v1/n1 + v2/n2)² / ((v1/n1)²/(n1-1) + (v2/n2)²/(n2-1))
[[nodiscard]]
SCL_FORCE_INLINE
auto ttest_df_welch(
    double var1, double n1,
    double var2, double n2
) -> double {
    double v1_n1 = var1 / n1;
    double v2_n2 = var2 / n2;
    double sum_v = v1_n1 + v2_n2;

    if (sum_v < 1e-12) [[unlikely]] return 1.0;

    double denom = (v1_n1 * v1_n1) / (n1 - 1.0) + (v2_n2 * v2_n2) / (n2 - 1.0);
    return (sum_v * sum_v) / denom;
}

// =============================================================================
// SECTION 2: P-Value Approximation (Scalar)
// =============================================================================

/// @brief Fast p-value approximation from t-statistic
/// @param[in] t_stat T-statistic
/// @param[in] df Degrees of freedom
/// @return Approximate two-sided p-value
/// @note For df > 30: uses normal approximation
/// @note For df <= 30: uses sigmoid heuristic
[[nodiscard]]
SCL_FORCE_INLINE
auto ttest_pvalue_fast(double t_stat, double df) -> double {
    if (df <= 0) [[unlikely]] return 1.0;

    double abs_t = std::abs(t_stat);

    // For large DF, t-distribution converges to normal
    if (df > 30.0) [[likely]] {
        return 2.0 * scl::math::normal_sf_fast(abs_t);
    }
    else [[unlikely]] {
        // Sigmoid heuristic for small DF
        double z = abs_t / std::sqrt(df + abs_t * abs_t);
        double cdf = 0.5 * (1.0 + z);
        return 2.0 * (1.0 - cdf);
    }
}

// =============================================================================
// SECTION 3: Complete T-Tests (Scalar)
// =============================================================================

/// @brief Complete Welch's t-test
/// @param[in] mean1 Mean of group 1
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] mean2 Mean of group 2
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Approximate two-sided p-value
/// @note Uses Welch-Satterthwaite degrees of freedom
[[nodiscard]]
SCL_FORCE_INLINE
auto welch_test_fast(
    double mean1, double var1, double n1,
    double mean2, double var2, double n2
) -> double {
    double se = ttest_se_welch(var1, n1, var2, n2);
    if (se < 1e-15) [[unlikely]] return 1.0;

    double t_stat = (mean1 - mean2) / se;
    double df = ttest_df_welch(var1, n1, var2, n2);

    return ttest_pvalue_fast(t_stat, df);
}

/// @brief Complete Student's t-test
/// @param[in] mean1 Mean of group 1
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] mean2 Mean of group 2
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Approximate two-sided p-value
/// @note Uses pooled variance estimate
[[nodiscard]]
SCL_FORCE_INLINE
auto student_test_fast(
    double mean1, double var1, double n1,
    double mean2, double var2, double n2
) -> double {
    double se = ttest_se_pooled(var1, n1, var2, n2);
    if (se < 1e-15) [[unlikely]] return 1.0;

    double t_stat = (mean1 - mean2) / se;
    double df = n1 + n2 - 2.0;

    return ttest_pvalue_fast(t_stat, df);
}

// =============================================================================
// SECTION 4: SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

/// @brief SIMD Welch's standard error
/// @tparam V SIMD vector type
/// @param[in] var1 Variances of group 1 (vector)
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] var2 Variances of group 2 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @return Standard errors (vector)
template<typename V>
[[nodiscard]]
SCL_FORCE_INLINE
auto ttest_se_welch(V var1, V n1, V var2, V n2) -> V {
    auto v1_n1 = s::Div(var1, n1);
    auto v2_n2 = s::Div(var2, n2);
    return s::Sqrt(s::Add(v1_n1, v2_n2));
}

/// @brief SIMD Welch-Satterthwaite degrees of freedom
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] var1 Variances of group 1 (vector)
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] var2 Variances of group 2 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @return Degrees of freedom (vector)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto ttest_df_welch(D d, V var1, V n1, V var2, V n2) -> V {
    auto one = s::Set(d, 1.0);
    auto v1_n1 = s::Div(var1, n1);
    auto v2_n2 = s::Div(var2, n2);
    auto sum_v = s::Add(v1_n1, v2_n2);

    auto term1 = s::Div(s::Mul(v1_n1, v1_n1), s::Sub(n1, one));
    auto term2 = s::Div(s::Mul(v2_n2, v2_n2), s::Sub(n2, one));

    return s::Div(s::Mul(sum_v, sum_v), s::Add(term1, term2));
}

/// @brief SIMD p-value using normal approximation (assumes large DF)
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] t_stat T-statistics (vector)
/// @return Approximate p-values (vector)
/// @note Assumes df > 30 for all lanes
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto ttest_pvalue_normal_fast(D d, V t_stat) -> V {
    auto abs_t = s::Abs(t_stat);
    auto sf = scl::math::simd::normal_sf_fast(d, abs_t);
    return s::Mul(s::Set(d, 2.0), sf);
}

/// @brief SIMD Welch's t-test (fast)
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] mean1 Means of group 1 (vector)
/// @param[in] var1 Variances of group 1 (vector)
/// @param[in] n1 Sample sizes of group 1 (vector)
/// @param[in] mean2 Means of group 2 (vector)
/// @param[in] var2 Variances of group 2 (vector)
/// @param[in] n2 Sample sizes of group 2 (vector)
/// @return Approximate p-values (vector)
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto welch_test_fast(
    D d,
    V mean1, V var1, V n1,
    V mean2, V var2, V n2
) -> V {
    const auto one = s::Set(d, 1.0);
    const auto zero = s::Zero(d);

    auto se = ttest_se_welch(var1, n1, var2, n2);

    // Guard against se ~ 0
    auto mask_valid = s::Gt(se, s::Set(d, 1e-15));
    auto t_stat = s::IfThenElse(mask_valid,
        s::Div(s::Sub(mean1, mean2), se),
        zero
    );

    auto p = ttest_pvalue_normal_fast(d, t_stat);

    // If SE too small, return p = 1.0
    return s::IfThenElse(mask_valid, p, one);
}

} // namespace simd

} // namespace scl::math

