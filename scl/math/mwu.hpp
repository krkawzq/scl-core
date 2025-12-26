#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/math/stats.hpp"

#include <cmath>

// =============================================================================
/// @file mwu.hpp
/// @brief Mann-Whitney U Test Statistics (Precise Implementation)
///
/// Provides high-precision implementations of the Mann-Whitney U test for
/// non-parametric comparison of two independent samples.
///
/// The Mann-Whitney U Test:
///
/// Tests whether two samples come from the same distribution by comparing
/// ranks. The U statistic counts the number of times an observation from
/// one sample precedes an observation from the other sample.
///
/// Statistical Model:
///
/// Given samples X1, ..., Xn1 from distribution F and Y1, ..., Yn2 from
/// distribution G, the null hypothesis is H0: F = G.
///
/// Formulas:
///
/// - Mean: mu = n1 * n2 / 2
/// - Variance: var = (n1 * n2 / 12) * ((N + 1) - tie_correction)
/// - Tie correction: sum(t^3 - t) / (N * (N - 1)) for all tied groups
/// - Z-score: z = (|U - mu| - cc) / sqrt(var)
///
/// where N = n1 + n2, t = tie group size, cc = continuity correction.
///
/// Implementation Notes:
///
/// - Uses normal approximation for p-values (accurate for n1, n2 > 10)
/// - Supports tie correction for repeated values
/// - Continuity correction (default 0.5) improves accuracy
/// - For fast approximation, see scl/math/fast/mwu.hpp
///
/// Reference: Mann, Whitney (1947) "On a Test of Whether One of Two Random
/// Variables is Stochastically Larger than the Other"
// =============================================================================

namespace scl::math::mwu {

// =============================================================================
// SECTION 1: Internal Helpers
// =============================================================================

namespace detail {

/// @brief Compute mean and standard deviation of U statistic.
///
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for all tied groups
/// @param mu Output mean of U
/// @param sd Output standard deviation of U
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

/// @brief Compute z-score with continuity correction.
///
/// @param U U statistic
/// @param mu Mean of U
/// @param sd Standard deviation of U
/// @param cc Continuity correction
/// @return Z-score
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

} // namespace detail

// =============================================================================
// SECTION 2: Scalar P-Value Functions
// =============================================================================

/// @brief Two-sided p-value from U statistic.
///
/// Tests H0: F = G vs H1: F != G (two-tailed).
///
/// @param U Mann-Whitney U statistic
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for tied groups (default 0)
/// @param cc Continuity correction (default 0.5)
/// @return Two-sided p-value in [0, 1]
SCL_FORCE_INLINE double p_value_two_sided(
    double U, double n1, double n2, double tie_sum = 0.0, double cc = 0.5
) {
    double mu, sd;
    detail::moments(n1, n2, tie_sum, mu, sd);

    if (sd <= 0.0) {
        return 1.0;
    }

    const double z = detail::compute_z(U, mu, sd, cc);

    // Two-sided: P = 2 * P(Z > |z|) = 2 * SF(z)
    return 2.0 * scl::math::normal_sf(z);
}

/// @brief One-sided "greater" p-value from U statistic.
///
/// Tests H0: F = G vs H1: X tends to be larger than Y.
///
/// @param U Mann-Whitney U statistic
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for tied groups (default 0)
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

    // For one-sided, use signed difference
    double diff = U - mu - cc;
    const double z = diff / sd;

    return scl::math::normal_sf(z);
}

/// @brief One-sided "less" p-value from U statistic.
///
/// Tests H0: F = G vs H1: X tends to be smaller than Y.
///
/// @param U Mann-Whitney U statistic
/// @param n1 Sample size of group 1
/// @param n2 Sample size of group 2
/// @param tie_sum Sum of (t^3 - t) for tied groups (default 0)
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

    // For "less" alternative, flip the sign
    double diff = mu - U - cc;
    const double z = diff / sd;

    return scl::math::normal_sf(z);
}

// =============================================================================
// SECTION 3: SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

/// @brief SIMD computation of U statistic moments.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param n1 Sample sizes of group 1 (vector)
/// @param n2 Sample sizes of group 2 (vector)
/// @param tie_sum Tie sums (vector)
/// @param mu Output means (vector)
/// @param sd Output standard deviations (vector)
template <class D, class V>
SCL_FORCE_INLINE void moments(
    D d, V n1, V n2, V tie_sum,
    V& mu, V& sd
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

/// @brief SIMD z-score computation.
///
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param d SIMD descriptor
/// @param U U statistics (vector)
/// @param mu Means (vector)
/// @param sd Standard deviations (vector)
/// @param cc Continuity corrections (vector)
/// @return Z-scores (vector)
template <class D, class V>
SCL_FORCE_INLINE V compute_z(
    D d, V U, V mu, V sd, V cc
) {
    const auto zero = s::Zero(d);

    // diff = |U - mu| - cc
    auto abs_diff = s::Abs(s::Sub(U, mu));
    auto diff = s::Sub(abs_diff, cc);
    diff = s::Max(diff, zero);

    // z = diff / sd, with check for sd > 0
    auto mask_sd = s::Gt(sd, zero);
    auto z = s::IfThenElse(mask_sd, s::Div(diff, sd), zero);

    return z;
}

/// @brief SIMD two-sided p-value.
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

/// @brief SIMD one-sided "greater" p-value.
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

/// @brief SIMD one-sided "less" p-value.
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
