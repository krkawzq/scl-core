#pragma once

#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/math/approx/stats.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/math/approx/ttest.hpp
// BRIEF: Approximate T-Test Statistics
// =============================================================================

// Student's t and Welch's t with fast p-value approximation
// DF > 30: normal approximation, DF <= 30: sigmoid heuristic

namespace scl::math::approx::ttest {

// =============================================================================
// Scalar Implementations
// =============================================================================

// Pooled standard error (Student's t-test)
SCL_FORCE_INLINE double se_pooled(
    double var1, double n1,
    double var2, double n2
) {
    double df = n1 + n2 - 2.0;
    if (df <= 0) return 0.0;

    double v_pool = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / df;
    return std::sqrt(v_pool * (1.0 / n1 + 1.0 / n2));
}

// Welch's standard error
SCL_FORCE_INLINE double se_welch(
    double var1, double n1,
    double var2, double n2
) {
    return std::sqrt(var1 / n1 + var2 / n2);
}

// Welch-Satterthwaite degrees of freedom
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

// Fast p-value approximation from t-statistic
SCL_FORCE_INLINE double p_value_approx(double t_stat, double df) {
    if (df <= 0) return 1.0;

    double abs_t = std::abs(t_stat);

    // For large DF, t-distribution converges to normal
    if (df > 30.0) {
        return 2.0 * scl::math::approx::normal_sf(abs_t);
    }
    else {
        // Sigmoid heuristic for small DF
        double z = abs_t / std::sqrt(df + abs_t * abs_t);
        double cdf = 0.5 * (1.0 + z);
        return 2.0 * (1.0 - cdf);
    }
}

// Complete Welch's t-test
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

// Complete Student's t-test
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
// SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

template <class V>
SCL_FORCE_INLINE V se_welch(V var1, V n1, V var2, V n2) {
    auto v1_n1 = s::Div(var1, n1);
    auto v2_n2 = s::Div(var2, n2);
    return s::Sqrt(s::Add(v1_n1, v2_n2));
}

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

// P-value using normal approximation (assumes large DF)
template <class D, class V>
SCL_FORCE_INLINE V p_value_normal(D d, V t_stat) {
    auto abs_t = s::Abs(t_stat);
    auto sf = scl::math::approx::simd::normal_sf(d, abs_t);
    return s::Mul(s::Set(d, 2.0), sf);
}

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

} // namespace scl::math::approx::ttest
