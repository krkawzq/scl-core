// =============================================================================
// FILE: scl/kernel/stat/stat_base.h
// BRIEF: API reference for common statistical kernel infrastructure
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

namespace scl::kernel::stat {

/* -----------------------------------------------------------------------------
 * NAMESPACE: config
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Common configuration constants for statistical kernels.
 *
 * CONSTANTS:
 *     INV_SQRT2              - 1/sqrt(2), used in normal distribution calculations
 *     EPS                    - Small epsilon for numerical stability (1e-9)
 *     SIGMA_MIN              - Minimum sigma threshold (1e-12)
 *     PREFETCH_DISTANCE      - Cache prefetch distance (16 elements)
 *     BINARY_SEARCH_THRESHOLD - Threshold for switching to binary search (32)
 * -------------------------------------------------------------------------- */
namespace config {
    constexpr double INV_SQRT2 = 0.7071067811865475244;
    constexpr double EPS = 1e-9;
    constexpr double SIGMA_MIN = 1e-12;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size BINARY_SEARCH_THRESHOLD = 32;
}

/* -----------------------------------------------------------------------------
 * STRUCT: GroupConstants
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Precomputed constants for two-group statistical comparisons.
 *
 * MEMBERS:
 *     n1d      - Group 1 size as double
 *     n2d      - Group 2 size as double
 *     N        - Total sample size (n1d + n2d)
 *     inv_n1   - Precomputed 1/n1 (0 if n1 == 0)
 *     inv_n2   - Precomputed 1/n2 (0 if n2 == 0)
 *
 * CONSTRUCTOR:
 *     GroupConstants(Size n1, Size n2)
 *
 * USAGE:
 *     Inherit or compose with kernel-specific constants.
 *     Precomputed reciprocals avoid division in hot loops.
 * -------------------------------------------------------------------------- */
struct GroupConstants {
    double n1d;
    double n2d;
    double N;
    double inv_n1;
    double inv_n2;

    GroupConstants(Size n1, Size n2);
};

/* -----------------------------------------------------------------------------
 * STRUCT: MWUConstants
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Extended constants for Mann-Whitney U test computation.
 *
 * INHERITS:
 *     GroupConstants
 *
 * ADDITIONAL MEMBERS:
 *     half_n1_n1p1 - 0.5 * n1 * (n1 + 1), for U statistic
 *     half_n1_n2   - 0.5 * n1 * n2, expected U under null
 *     var_base     - n1 * n2 / 12, base variance term
 *     N_p1         - N + 1
 *     N_Nm1        - N * (N - 1)
 *     inv_N_Nm1    - Precomputed 1 / (N * (N - 1))
 *
 * CONSTRUCTOR:
 *     MWUConstants(Size n1, Size n2)
 * -------------------------------------------------------------------------- */
struct MWUConstants : public GroupConstants {
    double half_n1_n1p1;
    double half_n1_n2;
    double var_base;
    double N_p1;
    double N_Nm1;
    double inv_N_Nm1;

    MWUConstants(Size n1, Size n2);
};

/* -----------------------------------------------------------------------------
 * NAMESPACE: pvalue
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     P-value computation functions using fast approximations.
 * -------------------------------------------------------------------------- */
namespace pvalue {

/* -----------------------------------------------------------------------------
 * FUNCTION: fast_erfc
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Fast complementary error function approximation.
 *
 * PARAMETERS:
 *     x        [in]  Input value
 *
 * RETURNS:
 *     erfc(x) with maximum error < 1.5e-7
 *
 * ALGORITHM:
 *     Horner polynomial approximation with 5 coefficients.
 *     Handles negative x via symmetry.
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
Real fast_erfc(Real x);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_two_sided
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Two-sided p-value from z-score using normal approximation.
 *
 * PARAMETERS:
 *     z        [in]  Z-score (test statistic)
 *
 * RETURNS:
 *     P(|Z| >= |z|) under standard normal distribution
 *
 * ALGORITHM:
 *     erfc(|z| / sqrt(2))
 * -------------------------------------------------------------------------- */
Real normal_two_sided(Real z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_one_sided
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     One-sided p-value from z-score.
 *
 * PARAMETERS:
 *     z        [in]  Z-score
 *     greater  [in]  If true, compute P(Z >= z); else P(Z <= z)
 *
 * RETURNS:
 *     One-tailed probability
 * -------------------------------------------------------------------------- */
Real normal_one_sided(Real z, bool greater);

/* -----------------------------------------------------------------------------
 * FUNCTION: t_two_sided
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Two-sided p-value from t-statistic.
 *
 * PARAMETERS:
 *     t_stat   [in]  T-statistic
 *
 * RETURNS:
 *     Two-tailed p-value using normal approximation
 *
 * NOTE:
 *     Uses normal approximation, accurate for large degrees of freedom.
 * -------------------------------------------------------------------------- */
Real t_two_sided(Real t_stat);

/* -----------------------------------------------------------------------------
 * FUNCTION: chisq_pvalue
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Chi-squared distribution p-value approximation.
 *
 * PARAMETERS:
 *     chi2     [in]  Chi-squared statistic
 *     df       [in]  Degrees of freedom
 *
 * RETURNS:
 *     Upper tail probability P(X^2 >= chi2)
 *
 * ALGORITHM:
 *     Wilson-Hilferty cube root transformation to normal.
 * -------------------------------------------------------------------------- */
Real chisq_pvalue(Real chi2, Size df);

/* -----------------------------------------------------------------------------
 * FUNCTION: f_pvalue
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     F-distribution p-value approximation.
 *
 * PARAMETERS:
 *     F        [in]  F-statistic
 *     df1      [in]  Numerator degrees of freedom
 *     df2      [in]  Denominator degrees of freedom
 *
 * RETURNS:
 *     Upper tail probability P(F >= f)
 *
 * ALGORITHM:
 *     Normal approximation via transformation.
 * -------------------------------------------------------------------------- */
Real f_pvalue(Real F, Size df1, Size df2);

} // namespace pvalue

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_log2_fc
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute log2 fold change with pseudocount.
 *
 * PARAMETERS:
 *     mean1    [in]  Mean of group 1 (reference)
 *     mean2    [in]  Mean of group 2 (target)
 *
 * RETURNS:
 *     log2((mean2 + EPS) / (mean1 + EPS))
 *
 * NOTE:
 *     EPS = 1e-9 prevents division by zero and log of zero.
 * -------------------------------------------------------------------------- */
Real compute_log2_fc(double mean1, double mean2);

} // namespace scl::kernel::stat
