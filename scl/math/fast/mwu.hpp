#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/math/fast/stats.hpp" // For normal_sf

#include <cmath>

namespace scl::math::fast::mwu {

// =============================================================================
// Scalar Implementation
// =============================================================================

namespace detail {
    /// @brief Compute the Mean and Inverse Standard Deviation for U statistic.
    ///
    /// Formulas:
    /// mu = (n1 * n2) / 2
    /// var = (n1 * n2 / 12) * ((N + 1) - tie_correction)
    SCL_FORCE_INLINE void moments(
        double n1, double n2, double tie_sum,
        double& mu, double& inv_sd
    ) {
        const double N = n1 + n2;
        mu = 0.5 * n1 * n2;

        const double denom = N * (N - 1.0);
        const double base  = n1 * n2 / 12.0;
        
        // Tie correction logic
        double var;
        if (denom > 1e-9) { // Avoid division by zero
            var = base * (N + 1.0 - tie_sum / denom);
        } else {
            var = base * (N + 1.0);
        }

        // Return inverse standard deviation for faster multiplication later
        inv_sd = (var <= 1e-15) ? 0.0 : (1.0 / std::sqrt(var));
    }
}

/// @brief Compute two-sided P-value from U statistic.
SCL_FORCE_INLINE double p_value_two_sided(
    double U, double n1, double n2, double tie_sum, double cc = 0.5
) {
    double mu, inv_sd;
    detail::moments(n1, n2, tie_sum, mu, inv_sd);
    
    if (inv_sd == 0.0) return 1.0;
    
    // Z-score calculation with continuity correction (cc)
    const double z = (std::abs(U - mu) - cc) * inv_sd;
    
    // 2 * SurvivalFunction(z)
    return 2.0 * scl::math::fast::normal_sf(z);
}

// =============================================================================
// SIMD Implementation
// =============================================================================

namespace simd {
    namespace s = scl::simd;

    /// @brief SIMD Calculation of MWU Mean and InvSD.
    template <class D, class V>
    SCL_FORCE_INLINE void moments(
        D d, V n1, V n2, V tie_sum,
        V& mu, V& inv_sd
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
        const auto correction = s::Div(tie_sum, denom);
        
        // Handle denom > 0 check branchlessly
        const auto mask_denom = s::Gt(denom, zero);
        
        // var = base * (term - correction)
        auto var_normal = s::Mul(base, s::Sub(term_N_plus_1, correction));
        // Fallback (rare, mostly for tiny N): var = base * term
        auto var_fallback = s::Mul(base, term_N_plus_1);
        
        auto var = s::IfThenElse(mask_denom, var_normal, var_fallback);

        // inv_sd = 1.0 / sqrt(var)
        // Guard against var <= 0
        const auto mask_var = s::Gt(var, zero);
        inv_sd = s::IfThenElse(mask_var, 
            s::Div(one, s::Sqrt(var)), 
            zero
        );
    }

    /// @brief SIMD Two-Sided P-Value.
    template <class D, class V>
    SCL_FORCE_INLINE V p_value_two_sided(
        D d, V U, V n1, V n2, V tie_sum, V cc
    ) {
        V mu, inv_sd;
        moments(d, n1, n2, tie_sum, mu, inv_sd);

        // Z = (|U - mu| - cc) * inv_sd
        auto abs_diff = s::Abs(s::Sub(U, mu));
        auto z = s::Mul(s::Sub(abs_diff, cc), inv_sd);
        
        // P = 2 * SF(z)
        auto sf = scl::math::fast::simd::normal_sf(d, z);
        auto p = s::Mul(s::Set(d, 2.0), sf);

        // Boundary check: if inv_sd == 0, p = 1.0
        return s::IfThenElse(s::Eq(inv_sd, s::Zero(d)), s::Set(d, 1.0), p);
    }

} // namespace simd

} // namespace scl::math::fast::mwu
