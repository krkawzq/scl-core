#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/stat/stat_base.hpp
// BRIEF: Common types, constants, and p-value computation for statistical kernels
// =============================================================================

namespace scl::kernel::stat {

// =============================================================================
// Configuration Constants
// =============================================================================

namespace config {
    constexpr double INV_SQRT2 = 0.7071067811865475244;
    constexpr double EPS = 1e-9;
    constexpr double SIGMA_MIN = 1e-12;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size BINARY_SEARCH_THRESHOLD = 32;
}

// =============================================================================
// Group Constants (Precomputed values for two-group comparisons)
// =============================================================================

struct GroupConstants {
    double n1d;
    double n2d;
    double N;
    double inv_n1;
    double inv_n2;

    SCL_FORCE_INLINE GroupConstants(Size n1, Size n2)
        : n1d(static_cast<double>(n1))
        , n2d(static_cast<double>(n2))
        , N(n1d + n2d)
        , inv_n1((n1 > 0) ? (1.0 / n1d) : 0.0)
        , inv_n2((n2 > 0) ? (1.0 / n2d) : 0.0)
    {}
};

// =============================================================================
// MWU-Specific Constants (Extended from GroupConstants)
// =============================================================================

struct MWUConstants : public GroupConstants {
    double half_n1_n1p1;
    double half_n1_n2;
    double var_base;
    double N_p1;
    double N_Nm1;
    double inv_N_Nm1;

    SCL_FORCE_INLINE MWUConstants(Size n1, Size n2)
        : GroupConstants(n1, n2)
        , half_n1_n1p1(0.5 * n1d * (n1d + 1.0))
        , half_n1_n2(0.5 * n1d * n2d)
        , var_base(n1d * n2d / 12.0)
        , N_p1(N + 1.0)
        , N_Nm1(N * (N - 1.0))
        , inv_N_Nm1((N_Nm1 > config::EPS) ? (1.0 / N_Nm1) : 0.0)
    {}
};

// =============================================================================
// P-Value Computation
// =============================================================================

namespace pvalue {

// Horner polynomial approximation for erfc (max error < 1.5e-7)
SCL_FORCE_INLINE Real fast_erfc(Real x) {
    Real sign = Real(1);
    if (x < Real(0)) {
        sign = Real(-1);
        x = -x;
    }

    Real t = Real(1) / (Real(1) + Real(0.3275911) * x);
    Real t2 = t * t;
    Real t3 = t2 * t;
    Real t4 = t3 * t;
    Real t5 = t4 * t;

    Real poly = Real(0.254829592) * t
              - Real(0.284496736) * t2
              + Real(1.421413741) * t3
              - Real(1.453152027) * t4
              + Real(1.061405429) * t5;

    Real result = poly * std::exp(-x * x);

    return (sign > Real(0)) ? result : (Real(2) - result);
}

// Two-sided p-value from z-score (normal approximation)
SCL_FORCE_INLINE Real normal_two_sided(Real z) {
    return fast_erfc(std::abs(z) * static_cast<Real>(config::INV_SQRT2));
}

// One-sided p-value from z-score
SCL_FORCE_INLINE Real normal_one_sided(Real z, bool greater) {
    Real p = Real(0.5) * fast_erfc(z * static_cast<Real>(config::INV_SQRT2));
    return greater ? p : (Real(1) - p);
}

// Two-sided p-value from t-statistic (using normal approximation for large df)
SCL_FORCE_INLINE Real t_two_sided(Real t_stat) {
    return normal_two_sided(t_stat);
}

// Chi-squared p-value approximation (Wilson-Hilferty)
SCL_FORCE_INLINE Real chisq_pvalue(Real chi2, Size df) {
    if (SCL_UNLIKELY(df == 0 || chi2 <= Real(0))) {
        return Real(1);
    }

    double d = static_cast<double>(df);
    double x = static_cast<double>(chi2);

    // Wilson-Hilferty transformation
    double cube_root = std::cbrt(x / d);
    double z = (cube_root - (1.0 - 2.0 / (9.0 * d))) / std::sqrt(2.0 / (9.0 * d));

    // Upper tail probability
    return static_cast<Real>(0.5 * std::erfc(z * config::INV_SQRT2));
}

// F-distribution p-value approximation
SCL_FORCE_INLINE Real f_pvalue(Real F, Size df1, Size df2) {
    if (SCL_UNLIKELY(df1 == 0 || df2 == 0 || F <= Real(0))) {
        return Real(1);
    }

    double f = static_cast<double>(F);
    double d1 = static_cast<double>(df1);
    double d2 = static_cast<double>(df2);

    // Approximation using normal distribution
    double lambda = (d1 * f) / (d1 * f + d2);
    double z = (std::pow(lambda, 1.0/3.0) - (1.0 - 2.0/(9.0*d1))) /
               std::sqrt(2.0/(9.0*d1)) -
               (std::pow(1.0 - lambda, 1.0/3.0) - (1.0 - 2.0/(9.0*d2))) /
               std::sqrt(2.0/(9.0*d2));

    return static_cast<Real>(0.5 * std::erfc(z * config::INV_SQRT2));
}

} // namespace pvalue

// =============================================================================
// Utility Functions
// =============================================================================

// Compute log2 fold change with pseudocount
SCL_FORCE_INLINE Real compute_log2_fc(double mean1, double mean2) {
    return static_cast<Real>(std::log2((mean2 + config::EPS) / (mean1 + config::EPS)));
}

} // namespace scl::kernel::stat
