#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/math/fast/stats.hpp" // For erfc / normal_sf

#include <cmath>

// =============================================================================
/// @file ttest.hpp
/// @brief Fast T-Test Statistics (Math Layer)
///
/// Implements Student's t and Welch's t statistics and p-value approximations.
// =============================================================================

namespace scl::math::fast::ttest {

// =============================================================================
// 1. Scalar Implementations
// =============================================================================

/// @brief Calculate Pooled Standard Error (Student's t-test).
/// Assumes equal variance.
SCL_FORCE_INLINE double se_pooled(
    double var1, double n1,
    double var2, double n2
) {
    // Pooled Variance: ((n1-1)v1 + (n2-1)v2) / (n1+n2-2)
    double df = n1 + n2 - 2.0;
    if (df <= 0) return 0.0;
    
    double v_pool = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / df;
    return std::sqrt(v_pool * (1.0 / n1 + 1.0 / n2));
}

/// @brief Calculate Welch's Standard Error (Welch's t-test).
/// Does not assume equal variance.
SCL_FORCE_INLINE double se_welch(
    double var1, double n1,
    double var2, double n2
) {
    return std::sqrt(var1 / n1 + var2 / n2);
}

/// @brief Calculate Welch-Satterthwaite Degrees of Freedom.
SCL_FORCE_INLINE double df_welch(
    double var1, double n1,
    double var2, double n2
) {
    double v1_n1 = var1 / n1;
    double v2_n2 = var2 / n2;
    double sum_v = v1_n1 + v2_n2;
    
    if (sum_v < 1e-12) return 1.0; // Avoid division by zero
    
    // df = (v1/n1 + v2/n2)^2 / ( (v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1) )
    double denom = (v1_n1 * v1_n1) / (n1 - 1.0) + (v2_n2 * v2_n2) / (n2 - 1.0);
    return (sum_v * sum_v) / denom;
}

/// @brief Fast P-value approximation from T-statistic.
///
/// Strategy:
/// - DF > 30: Use Normal approximation (Z-test), which is accurate for SC data.
/// - DF <= 30: Use a sigmoid approximation (heuristic).
SCL_FORCE_INLINE double p_value_approx(double t_stat, double df) {
    if (df <= 0) return 1.0;
    
    double abs_t = std::abs(t_stat);

    // For large DF, T-distribution converges to Normal distribution.
    // Single-cell data usually has N > 100, so this is the dominant path.
    if (df > 30.0) {
        // Normal SF: 0.5 * erfc(z / sqrt(2))
        return 2.0 * scl::math::fast::normal_sf(abs_t);
    } 
    else {
        // Fallback heuristic for very small groups (from original implementation).
        // Note: This is a rough approximation.
        // CDF ~ 0.5 * (1 + t / sqrt(df + t^2))
        double z = abs_t / std::sqrt(df + abs_t * abs_t);
        double cdf = 0.5 * (1.0 + z);
        return 2.0 * (1.0 - cdf);
    }
}

// =============================================================================
// 2. SIMD Implementations
// =============================================================================

namespace simd {
    namespace s = scl::simd;

    /// @brief SIMD Welch Standard Error.
    template <class V>
    SCL_FORCE_INLINE V se_welch(V var1, V n1, V var2, V n2) {
        auto v1_n1 = s::Div(var1, n1);
        auto v2_n2 = s::Div(var2, n2);
        return s::Sqrt(s::Add(v1_n1, v2_n2));
    }

    /// @brief SIMD Welch Degrees of Freedom.
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
    
    /// @brief SIMD P-value (Normal Approximation only).
    /// Assuming N is large enough for SIMD path.
    template <class D, class V>
    SCL_FORCE_INLINE V p_value_normal(D d, V t_stat) {
        auto abs_t = s::Abs(t_stat);
        // 2 * SF(t)
        auto sf = scl::math::fast::simd::normal_sf(d, abs_t);
        return s::Mul(s::Set(d, 2.0), sf);
    }

} // namespace simd

} // namespace scl::math::fast::ttest
