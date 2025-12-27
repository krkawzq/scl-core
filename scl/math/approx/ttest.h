// =============================================================================
// FILE: scl/math/approx/ttest.h
// BRIEF: API reference for approximate t-test statistics (approximate)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"

namespace scl::math::approx::ttest {

/* -----------------------------------------------------------------------------
 * FUNCTION: welch_test (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Approximate Welch's t-test with approximate p-value.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double welch_test(
 *         double mean1, double var1, double n1,
 *         double mean2, double var2, double n2
 *     )
 *
 * PARAMETERS:
 *     mean1   [in]  Mean of group 1
 *     var1    [in]  Variance of group 1
 *     n1      [in]  Sample size of group 1
 *     mean2   [in]  Mean of group 2
 *     var2    [in]  Variance of group 2
 *     n2      [in]  Sample size of group 2
 *
 * RETURNS:
 *     Approximate two-sided p-value in [0, 1]
 *
 * PRECONDITIONS:
 *     - n1 > 1 and n2 > 1
 *     - var1 >= 0 and var2 >= 0
 *
 * ALGORITHM:
 *     1. Compute Welch's standard error: se = sqrt(v1/n1 + v2/n2)
 *     2. Compute t-statistic: t = (mean1 - mean2) / se
 *     3. Compute Welch-Satterthwaite degrees of freedom
 *     4. Approximate p-value:
 *        - If df > 30: Use normal approximation (Z-test)
 *        - If df <= 30: Use sigmoid heuristic
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - pure function
 *
 * NUMERICAL NOTES:
 *     Does not assume equal variance (more robust than Student's t)
 *     Normal approximation is accurate for single-cell data (N > 100 per group)
 *     Precision: ~1e-7 due to approximate erfc
 *
 * REFERENCE:
 *     Welch, B. L. (1947). The generalization of Student's problem when several
 *     different population variances are involved. Biometrika.
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double welch_test(
    double mean1, double var1, double n1,
    double mean2, double var2, double n2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: student_test (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Approximate Student's t-test with approximate p-value (pooled variance).
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double student_test(
 *         double mean1, double var1, double n1,
 *         double mean2, double var2, double n2
 *     )
 *
 * PARAMETERS:
 *     mean1   [in]  Mean of group 1
 *     var1    [in]  Variance of group 1
 *     n1      [in]  Sample size of group 1
 *     mean2   [in]  Mean of group 2
 *     var2    [in]  Variance of group 2
 *     n2      [in]  Sample size of group 2
 *
 * RETURNS:
 *     Approximate two-sided p-value in [0, 1]
 *
 * PRECONDITIONS:
 *     - n1 > 1 and n2 > 1
 *     - var1 >= 0 and var2 >= 0
 *
 * ALGORITHM:
 *     1. Compute pooled variance: v_pool = ((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)
 *     2. Compute pooled standard error: se = sqrt(v_pool * (1/n1 + 1/n2))
 *     3. Compute t-statistic: t = (mean1 - mean2) / se
 *     4. Degrees of freedom: df = n1 + n2 - 2
 *     5. Approximate p-value (same strategy as Welch's test)
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 *
 * NUMERICAL NOTES:
 *     Assumes equal variance in both groups
 *     Use welch_test if equal variance assumption is questionable
 *     More powerful than Welch's test when variances are truly equal
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double student_test(
    double mean1, double var1, double n1,
    double mean2, double var2, double n2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: se_pooled (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute pooled standard error for Student's t-test.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double se_pooled(
 *         double var1, double n1,
 *         double var2, double n2
 *     )
 *
 * PARAMETERS:
 *     var1    [in]  Variance of group 1
 *     n1      [in]  Sample size of group 1
 *     var2    [in]  Variance of group 2
 *     n2      [in]  Sample size of group 2
 *
 * RETURNS:
 *     Pooled standard error
 *
 * ALGORITHM:
 *     v_pool = ((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)
 *     se = sqrt(v_pool * (1/n1 + 1/n2))
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double se_pooled(
    double var1, double n1,
    double var2, double n2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: se_welch (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Welch's standard error.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double se_welch(
 *         double var1, double n1,
 *         double var2, double n2
 *     )
 *
 * PARAMETERS:
 *     var1    [in]  Variance of group 1
 *     n1      [in]  Sample size of group 1
 *     var2    [in]  Variance of group 2
 *     n2      [in]  Sample size of group 2
 *
 * RETURNS:
 *     Welch's standard error
 *
 * ALGORITHM:
 *     se = sqrt(v1/n1 + v2/n2)
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double se_welch(
    double var1, double n1,
    double var2, double n2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: df_welch (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Welch-Satterthwaite degrees of freedom.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double df_welch(
 *         double var1, double n1,
 *         double var2, double n2
 *     )
 *
 * PARAMETERS:
 *     var1    [in]  Variance of group 1
 *     n1      [in]  Sample size of group 1
 *     var2    [in]  Variance of group 2
 *     n2      [in]  Sample size of group 2
 *
 * RETURNS:
 *     Effective degrees of freedom
 *
 * ALGORITHM:
 *     df = (v1/n1 + v2/n2)^2 / ((v1/n1)^2/(n1-1) + (v2/n2)^2/(n2-1))
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 *
 * REFERENCE:
 *     Satterthwaite, F. E. (1946). An approximate distribution of estimates of
 *     variance components. Biometrics Bulletin.
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double df_welch(
    double var1, double n1,
    double var2, double n2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_approx (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Approximate p-value approximation from t-statistic and degrees of freedom.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double p_value_approx(double t_stat, double df)
 *
 * PARAMETERS:
 *     t_stat  [in]  T-statistic
 *     df      [in]  Degrees of freedom
 *
 * RETURNS:
 *     Approximate two-sided p-value
 *
 * ALGORITHM:
 *     If df > 30:
 *         Use normal approximation: p = 2 * P(Z > |t|)
 *     Else:
 *         Use sigmoid heuristic: z = |t| / sqrt(df + t^2)
 *         cdf = 0.5 * (1 + z)
 *         p = 2 * (1 - cdf)
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 *
 * NUMERICAL NOTES:
 *     Normal approximation is highly accurate for df > 30
 *     Sigmoid heuristic provides approximate approximation for small df
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double p_value_approx(double t_stat, double df);

// =============================================================================
// SIMD Implementations
// =============================================================================

namespace simd {

/* -----------------------------------------------------------------------------
 * FUNCTION: welch_test (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized approximate Welch's t-test.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V welch_test(
 *         D d,
 *         V mean1, V var1, V n1,
 *         V mean2, V var2, V n2
 *     )
 *
 * PARAMETERS:
 *     d       [in]  SIMD descriptor
 *     mean1   [in]  Vector of means (group 1)
 *     var1    [in]  Vector of variances (group 1)
 *     n1      [in]  Vector of sample sizes (group 1)
 *     mean2   [in]  Vector of means (group 2)
 *     var2    [in]  Vector of variances (group 2)
 *     n2      [in]  Vector of sample sizes (group 2)
 *
 * RETURNS:
 *     Vector of approximate two-sided p-values
 *
 * ALGORITHM:
 *     Fully vectorized computation using normal approximation
 *     Assumes large degrees of freedom (typical for single-cell data)
 *     Handles degenerate case (se ~ 0) via masking
 *
 * COMPLEXITY:
 *     Time:  O(1) - true SIMD parallelism
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 *
 * NUMERICAL NOTES:
 *     Uses normal approximation for all samples (assumes df > 30)
 *     For small sample sizes, consider scalar version with full df calculation
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V welch_test(
    D d,
    V mean1, V var1, V n1,
    V mean2, V var2, V n2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: se_welch (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized Welch's standard error.
 *
 * SIGNATURE:
 *     template <class V>
 *     SCL_FORCE_INLINE V se_welch(V var1, V n1, V var2, V n2)
 *
 * PARAMETERS:
 *     var1    [in]  Vector of variances (group 1)
 *     n1      [in]  Vector of sample sizes (group 1)
 *     var2    [in]  Vector of variances (group 2)
 *     n2      [in]  Vector of sample sizes (group 2)
 *
 * RETURNS:
 *     Vector of standard errors
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class V>
SCL_FORCE_INLINE V se_welch(V var1, V n1, V var2, V n2);

/* -----------------------------------------------------------------------------
 * FUNCTION: df_welch (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized Welch-Satterthwaite degrees of freedom.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V df_welch(D d, V var1, V n1, V var2, V n2)
 *
 * PARAMETERS:
 *     d       [in]  SIMD descriptor
 *     var1    [in]  Vector of variances (group 1)
 *     n1      [in]  Vector of sample sizes (group 1)
 *     var2    [in]  Vector of variances (group 2)
 *     n2      [in]  Vector of sample sizes (group 2)
 *
 * RETURNS:
 *     Vector of effective degrees of freedom
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V df_welch(D d, V var1, V n1, V var2, V n2);

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_normal (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized p-value using normal approximation.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V p_value_normal(D d, V t_stat)
 *
 * PARAMETERS:
 *     d       [in]  SIMD descriptor
 *     t_stat  [in]  Vector of t-statistics
 *
 * RETURNS:
 *     Vector of two-sided p-values
 *
 * ALGORITHM:
 *     p = 2 * P(Z > |t|) using approximate normal_sf
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 *
 * NUMERICAL NOTES:
 *     Assumes large degrees of freedom (df > 30)
 *     Accurate for typical single-cell differential expression analysis
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V p_value_normal(D d, V t_stat);

} // namespace simd

} // namespace scl::math::approx::ttest
