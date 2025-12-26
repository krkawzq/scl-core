#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/math/fast/stats.hpp"

#include <cmath>

// =============================================================================
/// @file mwu.hpp
/// @brief Fast Mann-Whitney U Test Statistics (Approximate)
///
/// Provides high-performance approximations of Mann-Whitney U test statistics
/// optimized for throughput over precision.
///
/// Key Differences from Precise Version:
///
/// - Uses fast erfc approximation (~1e-7 precision)
/// - Returns inverse standard deviation for faster multiplication
/// - Optimized for high-throughput differential expression analysis
///
/// Use Cases:
///
/// - Large-scale differential expression (thousands of genes)
/// - Statistical filtering where exact p-values are not critical
/// - Real-time analysis pipelines
///
/// For full precision, see scl/math/mwu.hpp
// =============================================================================

namespace scl::math::fast::mwu {

// =============================================================================
// SECTION 1: Internal Helpers
// =============================================================================

namespace detail {

/// @brief Compute mean and inverse standard deviation of U statistic.
///
/// Returns inv_sd instead of sd for faster multiplication in z-score
/// computation.
///
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for all tied groups
/// @param mu Output mean of U
/// @param inv_sd Output inverse standard deviation (1/sd)
SCL_FORCE_INLINE void moments(
    double n1, double n2, double tie_sum,
    double& mu, double& inv_sd
) {
    const double N = n1 + n2;
    mu = 0.5 * n1 * n2;

    const double denom = N * (N - 1.0);
    const double base = n1 * n2 / 12.0;

    // Tie correction logic
    double var;
    if (denom > 1e-9) {
        var = base * (N + 1.0 - tie_sum / denom);
    } else {
        var = base * (N + 1.0);
    }

    // Return inverse sd for faster multiplication
    inv_sd = (var <= 1e-15) ? 0.0 : (1.0 / std::sqrt(var));
}

} // namespace detail

// =============================================================================
// SECTION 2: Scalar P-Value Functions
// =============================================================================

/// @brief Fast two-sided p-value from U statistic.
///
/// @param U Mann-Whitney U statistic
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for tied groups
/// @param cc Continuity correction (default 0.5)
/// @return Two-sided p-value
SCL_FORCE_INLINE double p_value_two_sided(
    double U, double n1, double n2, double tie_sum, double cc = 0.5
) {
    double mu, inv_sd;
    detail::moments(n1, n2, tie_sum, mu, inv_sd);

    if (inv_sd == 0.0) return 1.0;

    // Z-score with continuity correction
    const double z = (std::abs(U - mu) - cc) * inv_sd;

    // 2 * SF(z)
    return 2.0 * scl::math::fast::normal_sf(z);
}

/// @brief Fast one-sided "greater" p-value from U statistic.
///
/// @param U Mann-Whitney U statistic
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for tied groups
/// @param cc Continuity correction (default 0.5)
/// @return One-sided p-value for "greater" alternative
SCL_FORCE_INLINE double p_value_greater(
    double U, double n1, double n2, double tie_sum, double cc = 0.5
) {
    double mu, inv_sd;
    detail::moments(n1, n2, tie_sum, mu, inv_sd);

    if (inv_sd == 0.0) return (U > mu) ? 0.0 : 1.0;

    const double z = (U - mu - cc) * inv_sd;
    return scl::math::fast::normal_sf(z);
}

/// @brief Fast one-sided "less" p-value from U statistic.
///
/// @param U Mann-Whitney U statistic
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for tied groups
/// @param cc Continuity correction (default 0.5)
/// @return One-sided p-value for "less" alternative
SCL_FORCE_INLINE double p_value_less(
    double U, double n1, double n2, double tie_sum, double cc = 0.5
) {
    double mu, inv_sd;
    detail::moments(n1, n2, tie_sum, mu, inv_sd);

    if (inv_sd == 0.0) return (U < mu) ? 0.0 : 1.0;

    const double z = (mu - U - cc) * inv_sd;
    return scl::math::fast::normal_sf(z);
}

// =============================================================================
// SECTION 3: SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

/// @brief SIMD computation of U statistic moments (fast).
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param n1 Sample sizes of group 1 (vector)
/// @param n2 Sample sizes of group 2 (vector)
/// @param tie_sum Tie sums (vector)
/// @param mu Output means (vector)
/// @param inv_sd Output inverse standard deviations (vector)
template <class D, class V>
SCL_FORCE_INLINE void moments(
    D d, V n1, V n2, V tie_sum,
    V& mu, V& inv_sd
) {
    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);
    const auto twelve = s::Set(d, 12.0);

    const auto N = s::Add(n1, n2);

    // mu = 0.5 * n1 * n2
    mu = s::Mul(s::Set(d, 0.5), s::Mul(n1, n2));

    // Variance calculation
    const auto denom = s::Mul(N, s::Sub(N, one));
    const auto base = s::Div(s::Mul(n1, n2), twelve);

    // Term: (N + 1 - tie_sum / denom)
    const auto term_N_plus_1 = s::Add(N, one);
    const auto correction = s::Div(tie_sum, denom);

    // Handle denom > 0 branchlessly
    const auto mask_denom = s::Gt(denom, zero);

    auto var_normal = s::Mul(base, s::Sub(term_N_plus_1, correction));
    auto var_fallback = s::Mul(base, term_N_plus_1);

    auto var = s::IfThenElse(mask_denom, var_normal, var_fallback);

    // inv_sd = 1.0 / sqrt(var), guard against var <= 0
    const auto mask_var = s::Gt(var, zero);
    inv_sd = s::IfThenElse(mask_var,
        s::Div(one, s::Sqrt(var)),
        zero
    );
}

/// @brief SIMD fast two-sided p-value.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param U U statistics (vector)
/// @param n1 Sample sizes of group 1 (vector)
/// @param n2 Sample sizes of group 2 (vector)
/// @param tie_sum Tie sums (vector)
/// @param cc Continuity corrections (vector)
/// @return Two-sided p-values (vector)
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

/// @brief SIMD fast one-sided "greater" p-value.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param U U statistics (vector)
/// @param n1 Sample sizes of group 1 (vector)
/// @param n2 Sample sizes of group 2 (vector)
/// @param tie_sum Tie sums (vector)
/// @param cc Continuity corrections (vector)
/// @return One-sided p-values for "greater" alternative (vector)
template <class D, class V>
SCL_FORCE_INLINE V p_value_greater(
    D d, V U, V n1, V n2, V tie_sum, V cc
) {
    V mu, inv_sd;
    moments(d, n1, n2, tie_sum, mu, inv_sd);

    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);

    auto z = s::Mul(s::Sub(s::Sub(U, mu), cc), inv_sd);
    auto p = scl::math::fast::simd::normal_sf(d, z);

    // If inv_sd == 0: p = (U > mu) ? 0 : 1
    auto mask_greater = s::Gt(U, mu);
    auto p_degenerate = s::IfThenElse(mask_greater, zero, one);

    return s::IfThenElse(s::Eq(inv_sd, zero), p_degenerate, p);
}

/// @brief SIMD fast one-sided "less" p-value.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param U U statistics (vector)
/// @param n1 Sample sizes of group 1 (vector)
/// @param n2 Sample sizes of group 2 (vector)
/// @param tie_sum Tie sums (vector)
/// @param cc Continuity corrections (vector)
/// @return One-sided p-values for "less" alternative (vector)
template <class D, class V>
SCL_FORCE_INLINE V p_value_less(
    D d, V U, V n1, V n2, V tie_sum, V cc
) {
    V mu, inv_sd;
    moments(d, n1, n2, tie_sum, mu, inv_sd);

    const auto zero = s::Zero(d);
    const auto one = s::Set(d, 1.0);

    auto z = s::Mul(s::Sub(s::Sub(mu, U), cc), inv_sd);
    auto p = scl::math::fast::simd::normal_sf(d, z);

    // If inv_sd == 0: p = (U < mu) ? 0 : 1
    auto mask_less = s::Lt(U, mu);
    auto p_degenerate = s::IfThenElse(mask_less, zero, one);

    return s::IfThenElse(s::Eq(inv_sd, zero), p_degenerate, p);
}

} // namespace simd

} // namespace scl::math::fast::mwu
