#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/math/stats.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/math/stat_base.hpp
// DESCRIPTION: Statistical foundation module providing constants, helper types,
//              and p-value computation utilities for statistical tests
// =============================================================================

namespace scl::math {

// =============================================================================
// Statistical Constants
// =============================================================================

/// @brief Statistical computation constants
namespace stat_constants {

/// @brief Inverse of sqrt(2), used in normal distribution calculations
constexpr double INV_SQRT2 = 0.7071067811865475244;

/// @brief Small epsilon for numerical stability
constexpr double EPS = 1e-9;

/// @brief Minimum standard deviation threshold
constexpr double SIGMA_MIN = 1e-12;

/// @brief Prefetch distance for cache optimization
constexpr Size PREFETCH_DISTANCE = 16;

/// @brief Threshold for switching to binary search
constexpr Size BINARY_SEARCH_THRESHOLD = 32;

} // namespace stat_constants

// =============================================================================
// Group Statistics Helper Types
// =============================================================================

/// @brief Precomputed constants for two-group statistical comparisons
/// @details Stores group sizes and frequently used derived quantities to avoid
///          redundant computation in statistical tests
struct GroupConstants {
    /// @brief Group 1 size as double
    double n1d;

    /// @brief Group 2 size as double
    double n2d;

    /// @brief Total sample size (n1 + n2)
    double N;

    /// @brief Reciprocal of n1 (1/n1)
    double inv_n1;

    /// @brief Reciprocal of n2 (1/n2)
    double inv_n2;

    /// @brief Constructs group constants from group sizes
    /// @param[in] n1 Size of group 1
    /// @param[in] n2 Size of group 2
    [[nodiscard]]
    SCL_FORCE_INLINE
    constexpr
    GroupConstants(Size n1, Size n2) noexcept
        : n1d(static_cast<double>(n1))
        , n2d(static_cast<double>(n2))
        , N(n1d + n2d)
        , inv_n1((n1 > 0) ? (1.0 / n1d) : 0.0)
        , inv_n2((n2 > 0) ? (1.0 / n2d) : 0.0)
    {}
};

/// @brief Extended constants for Mann-Whitney U test
/// @details Inherits from GroupConstants and adds MWU-specific precomputed values
///          for efficient rank-sum test computation
struct MWUConstants : public GroupConstants {
    /// @brief 0.5 * n1 * (n1 + 1) - used in rank sum calculations
    double half_n1_n1p1;

    /// @brief 0.5 * n1 * n2 - expected value under null hypothesis
    double half_n1_n2;

    /// @brief Base variance term: n1 * n2 / 12
    double var_base;

    /// @brief N + 1 (total sample size plus one)
    double N_p1;

    /// @brief N * (N - 1) - used in tie correction
    double N_Nm1;

    /// @brief Reciprocal of N*(N-1) for tie correction
    double inv_N_Nm1;

    /// @brief Constructs MWU constants from group sizes
    /// @param[in] n1 Size of group 1
    /// @param[in] n2 Size of group 2
    [[nodiscard]]
    SCL_FORCE_INLINE
    constexpr
    MWUConstants(Size n1, Size n2) noexcept
        : GroupConstants(n1, n2)
        , half_n1_n1p1(0.5 * n1d * (n1d + 1.0))
        , half_n1_n2(0.5 * n1d * n2d)
        , var_base(n1d * n2d / 12.0)
        , N_p1(N + 1.0)
        , N_Nm1(N * (N - 1.0))
        , inv_N_Nm1((N_Nm1 > stat_constants::EPS) ? (1.0 / N_Nm1) : 0.0)
    {}
};

// =============================================================================
// P-Value Computation Functions
// =============================================================================

/// @brief P-value computation utilities for statistical tests
namespace pvalue {

/// @brief Computes two-sided p-value from z-score (standard normal distribution)
/// @tparam Real Floating-point type (float or double)
/// @param[in] z Z-score (standardized test statistic)
/// @return Two-sided p-value: P(|Z| > |z|)
/// @note Uses the complementary error function via scl::math::stats
/// @note Formula: p = erfc(|z| / sqrt(2))
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto normal_two_sided(Real z) -> Real {
    using std::abs;
    return normal_sf(abs(z)) * Real(2);
}

/// @brief Computes one-sided p-value from z-score
/// @tparam Real Floating-point type (float or double)
/// @param[in] z Z-score (standardized test statistic)
/// @param[in] greater If true, compute P(Z > z); if false, compute P(Z < z)
/// @return One-sided p-value
/// @note Uses survival function from scl::math::stats
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto normal_one_sided(Real z, bool greater) -> Real {
    return greater ? normal_sf(z) : normal_cdf(z);
}

/// @brief Computes two-sided p-value from t-statistic
/// @tparam Real Floating-point type (float or double)
/// @param[in] t_stat t-statistic value
/// @return Two-sided p-value: P(|T| > |t|)
/// @note For large degrees of freedom, t-distribution approximates normal distribution
/// @warning This uses normal approximation; use exact t-distribution for small samples
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto t_two_sided(Real t_stat) -> Real {
    return normal_two_sided(t_stat);
}

/// @brief Computes p-value from chi-squared statistic (Wilson-Hilferty approximation)
/// @tparam Real Floating-point type (float or double)
/// @param[in] chi2 Chi-squared statistic
/// @param[in] df Degrees of freedom
/// @return Upper tail p-value: P(χ² > chi2)
/// @note Uses Wilson-Hilferty cube-root transformation to normal approximation
/// @note Accurate for df >= 1; returns 1.0 for invalid inputs
/// @warning Approximation; use exact chi-squared distribution for small df if needed
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
auto chisq_pvalue(Real chi2, Size df) -> Real {
    using std::sqrt;
    using std::cbrt;

    if (df == 0 || chi2 <= Real(0)) [[unlikely]] {
        return Real(1);
    }

    const auto d = static_cast<double>(df);
    const auto x = static_cast<double>(chi2);

    // Wilson-Hilferty transformation: cube-root of χ²/df approximates normal
    const double cube_root = cbrt(x / d);
    const double mean_adj = 1.0 - 2.0 / (9.0 * d);
    const double sd = sqrt(2.0 / (9.0 * d));
    const double z = (cube_root - mean_adj) / sd;

    // Upper tail probability
    return static_cast<Real>(normal_sf(z));
}

/// @brief Computes p-value from F-statistic (approximation)
/// @tparam Real Floating-point type (float or double)
/// @param[in] F F-statistic value
/// @param[in] df1 Numerator degrees of freedom
/// @param[in] df2 Denominator degrees of freedom
/// @return Upper tail p-value: P(F > f)
/// @note Uses cube-root transformation to approximate F-distribution with normal
/// @note Returns 1.0 for invalid inputs (zero df or non-positive F)
/// @warning Approximation; use exact F-distribution for small df if needed
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
auto f_pvalue(Real F, Size df1, Size df2) -> Real {
    using std::pow;
    using std::sqrt;

    if (df1 == 0 || df2 == 0 || F <= Real(0)) [[unlikely]] {
        return Real(1);
    }

    const auto f = static_cast<double>(F);
    const auto d1 = static_cast<double>(df1);
    const auto d2 = static_cast<double>(df2);

    // Transformation using cube roots
    const double lambda = (d1 * f) / (d1 * f + d2);
    const double z_num = (pow(lambda, 1.0/3.0) - (1.0 - 2.0/(9.0*d1))) / sqrt(2.0/(9.0*d1));
    const double z_den = (pow(1.0 - lambda, 1.0/3.0) - (1.0 - 2.0/(9.0*d2))) / sqrt(2.0/(9.0*d2));
    const double z = z_num - z_den;

    return static_cast<Real>(normal_sf(z));
}

} // namespace pvalue

// =============================================================================
// Utility Functions
// =============================================================================

/// @brief Computes log2 fold change between two means
/// @tparam Real Floating-point type (float or double)
/// @param[in] mean1 Mean of group 1
/// @param[in] mean2 Mean of group 2
/// @return log2(mean2 / mean1) with pseudocount for numerical stability
/// @note Adds small epsilon (EPS) to avoid division by zero or log(0)
/// @note Formula: log2((mean2 + ε) / (mean1 + ε))
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_log2_fc(Real mean1, Real mean2) -> Real {
    using std::log2;
    constexpr Real eps = static_cast<Real>(stat_constants::EPS);
    return log2((mean2 + eps) / (mean1 + eps));
}

} // namespace scl::math
