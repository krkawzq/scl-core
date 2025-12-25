#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/math/stats.hpp" // For precise normal_sf

#include <cmath>

namespace scl::math::mwu {

// =============================================================================
// Scalar Implementation (Precise)
// =============================================================================

namespace detail {
    /// @brief Compute the Mean and Standard Deviation for U statistic (precise).
    ///
    /// Formulas:
    /// mu = (n1 * n2) / 2
    /// var = (n1 * n2 / 12) * ((N + 1) - tie_correction)
    /// 
    /// This implementation uses careful numerical handling to avoid overflow
    /// and maintain precision for large sample sizes.
    SCL_FORCE_INLINE void moments(
        double n1, double n2, double tie_sum,
        double& mu, double& sd
    ) {
        const double N = n1 + n2;
        mu = 0.5 * n1 * n2;

        // Variance calculation with tie correction
        const double denom = N * (N - 1.0);
        
        double var;
        if (denom > 0.0) {
            // Standard formula with tie correction
            const double tie_correction = tie_sum / denom;
            var = (n1 * n2 / 12.0) * (N + 1.0 - tie_correction);
        } else {
            // Fallback for degenerate cases
            var = (n1 * n2 / 12.0) * (N + 1.0);
        }

        // Ensure non-negative variance
        if (var < 0.0) var = 0.0;
        
        sd = std::sqrt(var);
    }

    /// @brief Compute z-score with continuity correction (precise).
    SCL_FORCE_INLINE double compute_z(
        double U, double mu, double sd, double cc
    ) {
        if (sd <= 0.0) {
            return 0.0;
        }
        
        // Apply continuity correction
        double diff = std::abs(U - mu) - cc;
        if (diff < 0.0) diff = 0.0;
        
        return diff / sd;
    }
}

/// @brief Compute two-sided P-value from U statistic (precise).
/// 
/// @param U The Mann-Whitney U statistic
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for all tied groups, where t is tie size
/// @param cc Continuity correction (default 0.5)
/// @return Two-sided p-value
SCL_FORCE_INLINE double p_value_two_sided(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) {
    double mu, sd;
    detail::moments(n1, n2, tie_sum, mu, sd);
    
    if (sd <= 0.0) {
        return 1.0;
    }
    
    const double z = detail::compute_z(U, mu, sd, cc);
    
    // Two-sided p-value: 2 * P(Z > |z|) = 2 * SF(z)
    return 2.0 * scl::math::normal_sf(z);
}

/// @brief Compute one-sided "greater" P-value from U statistic (precise).
/// Tests if group 1 tends to have larger values than group 2.
/// 
/// @param U The Mann-Whitney U statistic
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for all tied groups
/// @param cc Continuity correction (default 0.5)
/// @return One-sided p-value for "greater" alternative
SCL_FORCE_INLINE double p_value_greater(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) {
    double mu, sd;
    detail::moments(n1, n2, tie_sum, mu, sd);
    
    if (sd <= 0.0) {
        return (U > mu) ? 0.0 : 1.0;
    }
    
    // For one-sided, don't take absolute value
    double diff = U - mu - cc;
    const double z = diff / sd;
    
    return scl::math::normal_sf(z);
}

/// @brief Compute one-sided "less" P-value from U statistic (precise).
/// Tests if group 1 tends to have smaller values than group 2.
/// 
/// @param U The Mann-Whitney U statistic
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for all tied groups
/// @param cc Continuity correction (default 0.5)
/// @return One-sided p-value for "less" alternative
SCL_FORCE_INLINE double p_value_less(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) {
    double mu, sd;
    detail::moments(n1, n2, tie_sum, mu, sd);
    
    if (sd <= 0.0) {
        return (U < mu) ? 0.0 : 1.0;
    }
    
    // For one-sided, don't take absolute value
    double diff = mu - U - cc;
    const double z = diff / sd;
    
    return scl::math::normal_sf(z);
}

// =============================================================================
// SIMD Implementation (Precise)
// =============================================================================

namespace simd {
    namespace s = scl::simd;

    /// @brief SIMD Calculation of MWU Mean and Standard Deviation (precise).
    template <class D, class V>
    SCL_FORCE_INLINE void moments(
        D d, V n1, V n2, V tie_sum,
        V& mu, V& sd
    ) {
        const auto zero = s::Zero(d);
        const auto one  = s::Set(d, 1.0);
        const auto twelve = s::Set(d, 12.0);

        const auto N = s::Add(n1, n2);
        
        // mu = 0.5 * n1 * n2
        mu = s::Mul(s::Set(d, 0.5), s::Mul(n1, n2));

        // Variance Calculation
        const auto denom = s::Mul(N, s::Sub(N, one));
        const auto base  = s::Div(s::Mul(n1, n2), twelve);
        
        // Term: (N + 1 - tie_sum / denom)
        const auto term_N_plus_1 = s::Add(N, one);
        
        // Handle denom > 0 check
        const auto mask_denom = s::Gt(denom, zero);
        const auto correction = s::Div(tie_sum, denom);
        
        // var = base * (term - correction) or base * term
        auto var_with_correction = s::Mul(base, s::Sub(term_N_plus_1, correction));
        auto var_without_correction = s::Mul(base, term_N_plus_1);
        
        auto var = s::IfThenElse(mask_denom, var_with_correction, var_without_correction);

        // Ensure non-negative variance
        var = s::Max(var, zero);
        
        // sd = sqrt(var)
        sd = s::Sqrt(var);
    }

    /// @brief SIMD computation of z-score with continuity correction.
    template <class D, class V>
    SCL_FORCE_INLINE V compute_z(
        D d, V U, V mu, V sd, V cc
    ) {
        const auto zero = s::Zero(d);
        
        // diff = |U - mu| - cc
        auto abs_diff = s::Abs(s::Sub(U, mu));
        auto diff = s::Sub(abs_diff, cc);
        diff = s::Max(diff, zero); // Ensure non-negative
        
        // z = diff / sd, with check for sd > 0
        auto mask_sd = s::Gt(sd, zero);
        auto z = s::IfThenElse(mask_sd, s::Div(diff, sd), zero);
        
        return z;
    }

    /// @brief SIMD Two-Sided P-Value (precise).
    template <class D, class V>
    SCL_FORCE_INLINE V p_value_two_sided(
        D d, V U, V n1, V n2, V tie_sum, V cc
    ) {
        V mu, sd;
        moments(d, n1, n2, tie_sum, mu, sd);

        const auto zero = s::Zero(d);
        const auto one = s::Set(d, 1.0);
        const auto two = s::Set(d, 2.0);
        
        auto z = compute_z(d, U, mu, sd, cc);
        
        // P = 2 * SF(z)
        auto sf = scl::math::simd::normal_sf(d, z);
        auto p = s::Mul(two, sf);

        // Boundary check: if sd <= 0, p = 1.0
        auto mask_valid = s::Gt(sd, zero);
        return s::IfThenElse(mask_valid, p, one);
    }

    /// @brief SIMD One-Sided "Greater" P-Value (precise).
    template <class D, class V>
    SCL_FORCE_INLINE V p_value_greater(
        D d, V U, V n1, V n2, V tie_sum, V cc
    ) {
        V mu, sd;
        moments(d, n1, n2, tie_sum, mu, sd);

        const auto zero = s::Zero(d);
        const auto one = s::Set(d, 1.0);
        
        // diff = U - mu - cc
        auto diff = s::Sub(s::Sub(U, mu), cc);
        
        // z = diff / sd
        auto mask_sd = s::Gt(sd, zero);
        auto z = s::IfThenElse(mask_sd, s::Div(diff, sd), zero);
        
        // P = SF(z)
        auto p = scl::math::simd::normal_sf(d, z);
        
        // If sd <= 0: p = (U > mu) ? 0 : 1
        auto mask_greater = s::Gt(U, mu);
        auto p_degenerate = s::IfThenElse(mask_greater, zero, one);
        
        return s::IfThenElse(mask_sd, p, p_degenerate);
    }

    /// @brief SIMD One-Sided "Less" P-Value (precise).
    template <class D, class V>
    SCL_FORCE_INLINE V p_value_less(
        D d, V U, V n1, V n2, V tie_sum, V cc
    ) {
        V mu, sd;
        moments(d, n1, n2, tie_sum, mu, sd);

        const auto zero = s::Zero(d);
        const auto one = s::Set(d, 1.0);
        
        // diff = mu - U - cc
        auto diff = s::Sub(s::Sub(mu, U), cc);
        
        // z = diff / sd
        auto mask_sd = s::Gt(sd, zero);
        auto z = s::IfThenElse(mask_sd, s::Div(diff, sd), zero);
        
        // P = SF(z)
        auto p = scl::math::simd::normal_sf(d, z);
        
        // If sd <= 0: p = (U < mu) ? 0 : 1
        auto mask_less = s::Lt(U, mu);
        auto p_degenerate = s::IfThenElse(mask_less, zero, one);
        
        return s::IfThenElse(mask_sd, p, p_degenerate);
    }

} // namespace simd

} // namespace scl::math::mwu

