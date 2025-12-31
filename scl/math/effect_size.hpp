#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/math/stat_base.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/math/effect_size.hpp
// DESCRIPTION: Effect size computation utilities (Cohen's d, Hedges' g, Glass' Δ, CLES)
//              Pure mathematical functions for quantifying effect magnitude
// =============================================================================

namespace scl::math {

// =============================================================================
// Effect Size Types
// =============================================================================

/// @brief Effect size measurement types
enum class EffectSizeType {
    CohensD,     ///< Cohen's d: (mean2 - mean1) / pooled_sd
    HedgesG,     ///< Hedges' g: Bias-corrected Cohen's d
    GlassDelta,  ///< Glass' Δ: (mean2 - mean1) / sd1 (control group SD)
    CLES         ///< Common Language Effect Size (from AUROC)
};

// =============================================================================
// Cohen's d - Standardized Mean Difference
// =============================================================================

/// @brief Computes Cohen's d effect size
/// @tparam Real Floating-point type
/// @param[in] mean1 Mean of group 1
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] mean2 Mean of group 2
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Cohen's d = (mean2 - mean1) / pooled_sd
/// @note Returns 0 if n1 <= 1 or n2 <= 1 (insufficient data)
/// @note Returns 0 if pooled_sd < SIGMA_MIN (degenerate case)
/// @note Formula: d = (μ₂ - μ₁) / √[(σ₁²(n₁-1) + σ₂²(n₂-1)) / (n₁+n₂-2)]
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_cohens_d(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2
) -> Real {
    if (n1 <= 1 || n2 <= 1) [[unlikely]] {
        return Real(0);
    }

    const auto df1 = static_cast<double>(n1 - 1);
    const auto df2 = static_cast<double>(n2 - 1);
    const double pooled_var = (df1 * var1 + df2 * var2) / (df1 + df2);
    const double pooled_sd = std::sqrt(pooled_var);

    if (pooled_sd < stat_constants::SIGMA_MIN) [[unlikely]] {
        return Real(0);
    }

    return static_cast<Real>((mean2 - mean1) / pooled_sd);
}

// =============================================================================
// Hedges' g - Bias-Corrected Cohen's d
// =============================================================================

/// @brief Computes Hedges' g effect size (bias-corrected Cohen's d)
/// @tparam Real Floating-point type
/// @param[in] mean1 Mean of group 1
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] mean2 Mean of group 2
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Hedges' g = Cohen's d × J
/// @note J = 1 - 3/(4*df - 1) is the bias correction factor
/// @note Returns uncorrected d if df < 2
/// @note Hedges' g corrects for small-sample bias in Cohen's d
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_hedges_g(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2
) -> Real {
    const Real d = compute_cohens_d<Real>(mean1, var1, n1, mean2, var2, n2);

    // Hedges' correction factor: J = 1 - 3/(4*df - 1)
    const Size df = n1 + n2 - 2;
    if (df < 2) [[unlikely]] {
        return d;
    }

    const double J = 1.0 - 3.0 / (4.0 * static_cast<double>(df) - 1.0);
    return static_cast<Real>(static_cast<double>(d) * J);
}

// =============================================================================
// Glass' Delta - Control Group Standardized Difference
// =============================================================================

/// @brief Computes Glass' Δ effect size
/// @tparam Real Floating-point type
/// @param[in] mean1 Mean of group 1 (control group)
/// @param[in] var1 Variance of group 1 (control group)
/// @param[in] n1 Sample size of group 1
/// @param[in] mean2 Mean of group 2 (treatment group)
/// @param[in] var2 Variance of group 2 (unused)
/// @param[in] n2 Sample size of group 2 (unused)
/// @return Glass' Δ = (mean2 - mean1) / sd1
/// @note Returns 0 if n1 <= 1 (insufficient data)
/// @note Returns 0 if sd1 < SIGMA_MIN (degenerate case)
/// @note Uses only control group SD, useful when treatment affects variance
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_glass_delta(
    double mean1, double var1, Size n1,
    double mean2, [[maybe_unused]] double var2, [[maybe_unused]] Size n2
) -> Real {
    if (n1 <= 1) [[unlikely]] {
        return Real(0);
    }

    const double sd1 = std::sqrt(var1);
    if (sd1 < stat_constants::SIGMA_MIN) [[unlikely]] {
        return Real(0);
    }

    return static_cast<Real>((mean2 - mean1) / sd1);
}

// =============================================================================
// CLES - Common Language Effect Size
// =============================================================================

/// @brief Converts AUROC to CLES (Common Language Effect Size)
/// @tparam Real Floating-point type
/// @param[in] auroc Area Under ROC Curve value
/// @return CLES = AUROC (probability that random X2 > random X1)
/// @note AUROC and CLES are mathematically equivalent
/// @note CLES = 0.5 indicates no effect (50% chance)
/// @note CLES > 0.5 indicates group 2 tends to have higher values
/// @note CLES < 0.5 indicates group 1 tends to have higher values
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto auroc_to_cles(Real auroc) -> Real {
    return auroc;  // AUROC and CLES are equivalent
}

// =============================================================================
// Generic Effect Size Computation
// =============================================================================

/// @brief Computes effect size using specified method
/// @tparam Real Floating-point type
/// @param[in] mean1 Mean of group 1
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] mean2 Mean of group 2
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @param[in] type Effect size type to compute
/// @return Computed effect size value
/// @note For CLES type, returns 0.5 (requires AUROC input, not computable from moments)
/// @note Dispatches to appropriate specialized function based on type
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_effect_size(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2,
    EffectSizeType type
) -> Real {
    switch (type) {
        case EffectSizeType::CohensD:
            return compute_cohens_d<Real>(mean1, var1, n1, mean2, var2, n2);
        case EffectSizeType::HedgesG:
            return compute_hedges_g<Real>(mean1, var1, n1, mean2, var2, n2);
        case EffectSizeType::GlassDelta:
            return compute_glass_delta<Real>(mean1, var1, n1, mean2, var2, n2);
        case EffectSizeType::CLES:
            // CLES requires AUROC, not directly computable from moments
            return Real(0.5);
        default:
            return Real(0);
    }
}

} // namespace scl::math
