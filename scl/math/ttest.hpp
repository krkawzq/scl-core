#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/math/stat_base.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/math/ttest.hpp
// DESCRIPTION: T-test statistics computation utilities (Welch's and Student's)
//              Pure mathematical functions for two-sample t-tests
// =============================================================================

namespace scl::math {

// =============================================================================
// Standard Error Computation
// =============================================================================

/// @brief Computes pooled standard error for Student's t-test
/// @tparam Real Floating-point type
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Pooled standard error: sqrt(pooled_var * (1/n1 + 1/n2))
/// @note Returns 0 if degrees of freedom <= 0
/// @note Formula: SE = sqrt([(n1-1)*var1 + (n2-1)*var2] / (n1+n2-2) * (1/n1 + 1/n2))
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_pooled_se(
    double var1, double n1,
    double var2, double n2
) -> Real {
    const double df = n1 + n2 - 2.0;
    if (df <= 0.0) [[unlikely]] {
        return Real(0);
    }

    const double pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / df;
    const double se_sq = pooled_var * (1.0 / n1 + 1.0 / n2);

    return static_cast<Real>(std::sqrt(se_sq));
}

/// @brief Computes Welch's standard error (unequal variance)
/// @tparam Real Floating-point type
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Welch's standard error: sqrt(var1/n1 + var2/n2)
/// @note Does not assume equal variances
/// @note More robust than pooled SE when variances differ
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_welch_se(
    double var1, double n1,
    double var2, double n2
) -> Real {
    const double se_sq = var1 / n1 + var2 / n2;
    return static_cast<Real>(std::sqrt(se_sq));
}

// =============================================================================
// Degrees of Freedom
// =============================================================================

/// @brief Computes Welch-Satterthwaite degrees of freedom
/// @tparam Real Floating-point type
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Approximate degrees of freedom for Welch's t-test
/// @note Returns 1.0 if denominator is too small (< SIGMA_MIN)
/// @note Formula: df = (v1/n1 + v2/n2)² / [(v1/n1)²/(n1-1) + (v2/n2)²/(n2-1)]
/// @note Always returns value in range [min(n1-1, n2-1), n1+n2-2]
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_welch_df(
    double var1, double n1,
    double var2, double n2
) -> Real {
    const double v1_n1 = var1 / n1;
    const double v2_n2 = var2 / n2;
    const double sum_v = v1_n1 + v2_n2;

    if (sum_v < stat_constants::SIGMA_MIN) [[unlikely]] {
        return Real(1);
    }

    const double term1 = (v1_n1 * v1_n1) / (n1 - 1.0);
    const double term2 = (v2_n2 * v2_n2) / (n2 - 1.0);
    const double denom = term1 + term2;

    if (denom < stat_constants::SIGMA_MIN) [[unlikely]] {
        return Real(1);
    }

    return static_cast<Real>((sum_v * sum_v) / denom);
}

// =============================================================================
// T-Statistic Computation
// =============================================================================

/// @brief Computes t-statistic from means and standard error
/// @tparam Real Floating-point type
/// @param[in] mean1 Mean of group 1
/// @param[in] mean2 Mean of group 2
/// @param[in] se Standard error
/// @return t-statistic: (mean1 - mean2) / se
/// @note Returns 0 if se < SIGMA_MIN (degenerate case)
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_t_statistic(
    double mean1,
    double mean2,
    double se
) -> Real {
    if (se < stat_constants::SIGMA_MIN) [[unlikely]] {
        return Real(0);
    }
    return static_cast<Real>((mean1 - mean2) / se);
}

// =============================================================================
// P-Value Computation (Approximation)
// =============================================================================

/// @brief Computes two-sided p-value from t-statistic (approximation)
/// @tparam Real Floating-point type
/// @param[in] t_stat t-statistic value
/// @param[in] df Degrees of freedom
/// @return Two-sided p-value: P(|T| > |t|)
/// @note Uses normal approximation for df > 30
/// @note Uses sigmoid heuristic for df <= 30 (less accurate)
/// @note Returns 1.0 if df <= 0
/// @warning This is an approximation; exact t-distribution CDF would be more accurate
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
auto compute_t_pvalue_approx(Real t_stat, double df) -> Real {
    using std::abs;
    using std::sqrt;

    if (df <= 0.0) [[unlikely]] {
        return Real(1);
    }

    const Real abs_t = abs(t_stat);

    // For large degrees of freedom, t-distribution converges to normal
    if (df > 30.0) [[likely]] {
        return pvalue::normal_two_sided(abs_t);
    } else {
        // Sigmoid heuristic for small df (less accurate approximation)
        const double z = static_cast<double>(abs_t) / sqrt(df + static_cast<double>(abs_t * abs_t));
        const double cdf = 0.5 * (1.0 + z);
        return static_cast<Real>(2.0 * (1.0 - cdf));
    }
}

// =============================================================================
// Complete T-Tests
// =============================================================================

/// @brief Performs Welch's t-test (unequal variance)
/// @tparam Real Floating-point type
/// @param[in] mean1 Mean of group 1
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] mean2 Mean of group 2
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @param[out] out_t_stat t-statistic
/// @param[out] out_df Welch-Satterthwaite degrees of freedom
/// @param[out] out_pval Two-sided p-value (approximation)
/// @note Robust to unequal variances (heteroscedasticity)
/// @note Uses Welch-Satterthwaite approximation for degrees of freedom
/// @note Returns p=1.0 if standard error is too small (< SIGMA_MIN)
template<Arithmetic Real>
SCL_FORCE_INLINE
auto welch_ttest(
    double mean1, double var1, double n1,
    double mean2, double var2, double n2,
    Real& out_t_stat,
    Real& out_df,
    Real& out_pval
) -> void {
    const Real se = compute_welch_se<Real>(var1, n1, var2, n2);

    if (static_cast<double>(se) < stat_constants::SIGMA_MIN) [[unlikely]] {
        out_t_stat = Real(0);
        out_df = Real(1);
        out_pval = Real(1);
        return;
    }

    out_t_stat = compute_t_statistic<Real>(mean1, mean2, static_cast<double>(se));
    out_df = compute_welch_df<Real>(var1, n1, var2, n2);
    out_pval = compute_t_pvalue_approx(out_t_stat, static_cast<double>(out_df));
}

/// @brief Performs Student's t-test (equal variance assumed)
/// @tparam Real Floating-point type
/// @param[in] mean1 Mean of group 1
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] mean2 Mean of group 2
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @param[out] out_t_stat t-statistic
/// @param[out] out_df Degrees of freedom (n1 + n2 - 2)
/// @param[out] out_pval Two-sided p-value (approximation)
/// @note Assumes equal population variances (homoscedasticity)
/// @note More powerful than Welch's test when variances are truly equal
/// @note Returns p=1.0 if standard error is too small (< SIGMA_MIN)
template<Arithmetic Real>
SCL_FORCE_INLINE
auto student_ttest(
    double mean1, double var1, double n1,
    double mean2, double var2, double n2,
    Real& out_t_stat,
    Real& out_df,
    Real& out_pval
) -> void {
    const Real se = compute_pooled_se<Real>(var1, n1, var2, n2);

    if (static_cast<double>(se) < stat_constants::SIGMA_MIN) [[unlikely]] {
        out_t_stat = Real(0);
        out_df = Real(n1 + n2 - 2.0);
        out_pval = Real(1);
        return;
    }

    out_t_stat = compute_t_statistic<Real>(mean1, mean2, static_cast<double>(se));
    out_df = static_cast<Real>(n1 + n2 - 2.0);
    out_pval = compute_t_pvalue_approx(out_t_stat, static_cast<double>(out_df));
}

// =============================================================================
// Simplified Interfaces (Returns P-Value Only)
// =============================================================================

/// @brief Performs Welch's t-test and returns p-value only
/// @tparam Real Floating-point type
/// @param[in] mean1 Mean of group 1
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] mean2 Mean of group 2
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Two-sided p-value (approximation)
/// @note Convenience wrapper for welch_ttest()
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
auto welch_ttest_pvalue(
    double mean1, double var1, double n1,
    double mean2, double var2, double n2
) -> Real {
    const Real se = compute_welch_se<Real>(var1, n1, var2, n2);

    if (static_cast<double>(se) < stat_constants::SIGMA_MIN) [[unlikely]] {
        return Real(1);
    }

    const Real t_stat = compute_t_statistic<Real>(mean1, mean2, static_cast<double>(se));
    const Real df = compute_welch_df<Real>(var1, n1, var2, n2);

    return compute_t_pvalue_approx(t_stat, static_cast<double>(df));
}

/// @brief Performs Student's t-test and returns p-value only
/// @tparam Real Floating-point type
/// @param[in] mean1 Mean of group 1
/// @param[in] var1 Variance of group 1
/// @param[in] n1 Sample size of group 1
/// @param[in] mean2 Mean of group 2
/// @param[in] var2 Variance of group 2
/// @param[in] n2 Sample size of group 2
/// @return Two-sided p-value (approximation)
/// @note Convenience wrapper for student_ttest()
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
auto student_ttest_pvalue(
    double mean1, double var1, double n1,
    double mean2, double var2, double n2
) -> Real {
    const Real se = compute_pooled_se<Real>(var1, n1, var2, n2);

    if (static_cast<double>(se) < stat_constants::SIGMA_MIN) [[unlikely]] {
        return Real(1);
    }

    const Real t_stat = compute_t_statistic<Real>(mean1, mean2, static_cast<double>(se));
    const double df = n1 + n2 - 2.0;

    return compute_t_pvalue_approx(t_stat, df);
}

} // namespace scl::math
