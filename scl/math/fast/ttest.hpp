#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/math/fast/stats.hpp"

#include <cmath>

// =============================================================================
/// @file ttest.hpp
/// @brief Fast T-Test Statistics (Math Layer)
///
/// Provides high-performance implementations of Student's t and Welch's t
/// statistics with fast p-value approximation.
///
/// Implemented Tests:
///
/// 1. Student's t-test (pooled variance)
///    - Assumes equal variance in both groups
///    - SE = sqrt(v_pool * (1/n1 + 1/n2))
///
/// 2. Welch's t-test (separate variances)
///    - No equal variance assumption
///    - Uses Welch-Satterthwaite degrees of freedom
///
/// P-Value Approximation Strategy:
///
/// - DF > 30: Normal approximation (accurate for single-cell data)
/// - DF <= 30: Sigmoid heuristic (fast but approximate)
///
/// Use Cases:
///
/// - Differential expression analysis
/// - High-throughput statistical testing
/// - Real-time analysis pipelines
///
/// Note: For small sample sizes (n < 10), consider using exact t-distribution.
// =============================================================================

namespace scl::math::fast::ttest {

// =============================================================================
// SECTION 1: Scalar Implementations
// =============================================================================

/// @brief Calculate pooled standard error (Student's t-test).
///
/// Assumes equal variance in both groups.
/// SE = sqrt(v_pool * (1/n1 + 1/n2))
/// where v_pool = ((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)
///
/// @param var1 Variance of group 1
/// @param n1 Sample size of group 1
/// @param var2 Variance of group 2
/// @param n2 Sample size of group 2
/// @return Pooled standard error
SCL_FORCE_INLINE double se_pooled(
    double var1, double n1,
    double var2, double n2
) {
    double df = n1 + n2 - 2.0;
    if (df <= 0) return 0.0;

    double v_pool = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / df;
    return std::sqrt(v_pool * (1.0 / n1 + 1.0 / n2));
}

/// @brief Calculate Welch's standard error (Welch's t-test).
///
/// Does not assume equal variance.
/// SE = sqrt(v1/n1 + v2/n2)
///
/// @param var1 Variance of group 1
/// @param n1 Sample size of group 1
/// @param var2 Variance of group 2
/// @param n2 Sample size of group 2
/// @return Welch's standard error
SCL_FORCE_INLINE double se_welch(
    double var1, double n1,
    double var2, double n2
) {
    return std::sqrt(var1 / n1 + var2 / n2);
}

/// @brief Calculate Welch-Satterthwaite degrees of freedom.
///
/// df = (v1/n1 + v2/n2)^2 / ((v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1))
///
/// @param var1 Variance of group 1
/// @param n1 Sample size of group 1
/// @param var2 Variance of group 2
/// @param n2 Sample size of group 2
/// @return Effective degrees of freedom
SCL_FORCE_INLINE double df_welch(
    double var1, double n1,
    double var2, double n2
) {
    double v1_n1 = var1 / n1;
    double v2_n2 = var2 / n2;
    double sum_v = v1_n1 + v2_n2;

    if (sum_v < 1e-12) return 1.0;

    double denom = (v1_n1 * v1_n1) / (n1 - 1.0) + (v2_n2 * v2_n2) / (n2 - 1.0);
    return (sum_v * sum_v) / denom;
}

/// @brief Fast p-value approximation from t-statistic.
///
/// Strategy:
/// - DF > 30: Use normal approximation (Z-test)
/// - DF <= 30: Use sigmoid heuristic
///
/// For single-cell data with N > 100 per group, this is highly accurate.
///
/// @param t_stat T-statistic
/// @param df Degrees of freedom
/// @return Two-sided p-value
SCL_FORCE_INLINE double p_value_approx(double t_stat, double df) {
    if (df <= 0) return 1.0;

    double abs_t = std::abs(t_stat);

    // For large DF, t-distribution converges to normal
    if (df > 30.0) {
        return 2.0 * scl::math::fast::normal_sf(abs_t);
    }
    else {
        // Sigmoid heuristic for small DF
        // CDF ~ 0.5 * (1 + t / sqrt(df + t^2))
        double z = abs_t / std::sqrt(df + abs_t * abs_t);
        double cdf = 0.5 * (1.0 + z);
        return 2.0 * (1.0 - cdf);
    }
}

/// @brief Complete Welch's t-test (mean difference to p-value).
///
/// @param mean1 Mean of group 1
/// @param var1 Variance of group 1
/// @param n1 Sample size of group 1
/// @param mean2 Mean of group 2
/// @param var2 Variance of group 2
/// @param n2 Sample size of group 2
/// @return Two-sided p-value
SCL_FORCE_INLINE double welch_test(
    double mean1, double var1, double n1,
    double mean2, double var2, double n2
) {
    double se = se_welch(var1, n1, var2, n2);
    if (se < 1e-15) return 1.0;

    double t_stat = (mean1 - mean2) / se;
    double df = df_welch(var1, n1, var2, n2);

    return p_value_approx(t_stat, df);
}

/// @brief Complete Student's t-test (mean difference to p-value).
///
/// @param mean1 Mean of group 1
/// @param var1 Variance of group 1
/// @param n1 Sample size of group 1
/// @param mean2 Mean of group 2
/// @param var2 Variance of group 2
/// @param n2 Sample size of group 2
/// @return Two-sided p-value
SCL_FORCE_INLINE double student_test(
    double mean1, double var1, double n1,
    double mean2, double var2, double n2
) {
    double se = se_pooled(var1, n1, var2, n2);
    if (se < 1e-15) return 1.0;

    double t_stat = (mean1 - mean2) / se;
    double df = n1 + n2 - 2.0;

    return p_value_approx(t_stat, df);
}

// =============================================================================
// SECTION 2: SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

/// @brief SIMD Welch standard error.
///
/// @tparam V SIMD vector type
/// @param var1 Variances of group 1 (vector)
/// @param n1 Sample sizes of group 1 (vector)
/// @param var2 Variances of group 2 (vector)
/// @param n2 Sample sizes of group 2 (vector)
/// @return Standard errors (vector)
template <class V>
SCL_FORCE_INLINE V se_welch(V var1, V n1, V var2, V n2) {
    auto v1_n1 = s::Div(var1, n1);
    auto v2_n2 = s::Div(var2, n2);
    return s::Sqrt(s::Add(v1_n1, v2_n2));
}

/// @brief SIMD Welch-Satterthwaite degrees of freedom.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param var1 Variances of group 1 (vector)
/// @param n1 Sample sizes of group 1 (vector)
/// @param var2 Variances of group 2 (vector)
/// @param n2 Sample sizes of group 2 (vector)
/// @return Degrees of freedom (vector)
template <class D, class V>
SCL_FORCE_INLINE V df_welch(D d, V var1, V n1, V var2, V n2) {
    auto one = s::Set(d, 1.0);
    auto v1_n1 = s::Div(var1, n1);
    auto v2_n2 = s::Div(var2, n2);
    auto sum_v = s::Add(v1_n1, v2_n2);

    auto term1 = s::Div(s::Mul(v1_n1, v1_n1), s::Sub(n1, one));
    auto term2 = s::Div(s::Mul(v2_n2, v2_n2), s::Sub(n2, one));

    return s::Div(s::Mul(sum_v, sum_v), s::Add(term1, term2));
}

/// @brief SIMD p-value (normal approximation).
///
/// Assumes large DF (> 30). For single-cell data this is typically valid.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param t_stat T-statistics (vector)
/// @return Two-sided p-values (vector)
template <class D, class V>
SCL_FORCE_INLINE V p_value_normal(D d, V t_stat) {
    auto abs_t = s::Abs(t_stat);
    auto sf = scl::math::fast::simd::normal_sf(d, abs_t);
    return s::Mul(s::Set(d, 2.0), sf);
}

/// @brief SIMD complete Welch's t-test.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param mean1 Means of group 1 (vector)
/// @param var1 Variances of group 1 (vector)
/// @param n1 Sample sizes of group 1 (vector)
/// @param mean2 Means of group 2 (vector)
/// @param var2 Variances of group 2 (vector)
/// @param n2 Sample sizes of group 2 (vector)
/// @return Two-sided p-values (vector)
template <class D, class V>
SCL_FORCE_INLINE V welch_test(
    D d,
    V mean1, V var1, V n1,
    V mean2, V var2, V n2
) {
    const auto one = s::Set(d, 1.0);
    const auto zero = s::Zero(d);

    auto se = se_welch(var1, n1, var2, n2);

    // Guard against se ~ 0
    auto mask_valid = s::Gt(se, s::Set(d, 1e-15));
    auto t_stat = s::IfThenElse(mask_valid,
        s::Div(s::Sub(mean1, mean2), se),
        zero
    );

    auto p = p_value_normal(d, t_stat);

    // If SE too small, return p = 1.0
    return s::IfThenElse(mask_valid, p, one);
}

} // namespace simd

} // namespace scl::math::fast::ttest
