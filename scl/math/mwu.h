// =============================================================================
// FILE: scl/math/mwu.h
// BRIEF: API reference for Mann-Whitney U test statistics (precise)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"

namespace scl::math::mwu {

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_two_sided (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Two-sided p-value for Mann-Whitney U test with full precision.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double p_value_two_sided(
 *         double U, double n1, double n2,
 *         double tie_sum = 0.0, double cc = 0.5
 *     )
 *
 * PARAMETERS:
 *     U        [in]  Mann-Whitney U statistic
 *     n1       [in]  Sample size of group 1
 *     n2       [in]  Sample size of group 2
 *     tie_sum  [in]  Sum of (t^3 - t) for all tied groups (default 0)
 *     cc       [in]  Continuity correction (default 0.5)
 *
 * RETURNS:
 *     Two-sided p-value in [0, 1]
 *     Tests H0: F = G vs H1: F != G
 *
 * PRECONDITIONS:
 *     - n1 > 0 and n2 > 0
 *     - U >= 0
 *     - tie_sum >= 0
 *
 * ALGORITHM:
 *     1. Compute mean: mu = n1 * n2 / 2
 *     2. Compute variance with tie correction:
 *        var = (n1 * n2 / 12) * (N + 1 - tie_sum / (N * (N - 1)))
 *        where N = n1 + n2
 *     3. Compute z-score: z = (|U - mu| - cc) / sqrt(var)
 *     4. Return p = 2 * P(Z > z) using precise normal_sf
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - pure function
 *
 * NUMERICAL NOTES:
 *     Uses precise std::erfc for full precision (~15 digits)
 *     For large datasets (n1, n2 > 10), normal approximation is accurate
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double p_value_two_sided(
    double U, double n1, double n2,
    double tie_sum = 0.0, double cc = 0.5
);

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_greater (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     One-sided p-value for "greater" alternative hypothesis.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double p_value_greater(
 *         double U, double n1, double n2,
 *         double tie_sum = 0.0, double cc = 0.5
 *     )
 *
 * PARAMETERS:
 *     U        [in]  Mann-Whitney U statistic
 *     n1       [in]  Sample size of group 1
 *     n2       [in]  Sample size of group 2
 *     tie_sum  [in]  Tie correction sum (default 0)
 *     cc       [in]  Continuity correction (default 0.5)
 *
 * RETURNS:
 *     One-sided p-value for H1: X tends to be larger than Y
 *
 * ALGORITHM:
 *     1. Compute moments (mu, sd) with tie correction
 *     2. Compute z = (U - mu - cc) / sd
 *     3. Return P(Z > z)
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double p_value_greater(
    double U, double n1, double n2,
    double tie_sum = 0.0, double cc = 0.5
);

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_less (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     One-sided p-value for "less" alternative hypothesis.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double p_value_less(
 *         double U, double n1, double n2,
 *         double tie_sum = 0.0, double cc = 0.5
 *     )
 *
 * PARAMETERS:
 *     U        [in]  Mann-Whitney U statistic
 *     n1       [in]  Sample size of group 1
 *     n2       [in]  Sample size of group 2
 *     tie_sum  [in]  Tie correction sum (default 0)
 *     cc       [in]  Continuity correction (default 0.5)
 *
 * RETURNS:
 *     One-sided p-value for H1: X tends to be smaller than Y
 *
 * ALGORITHM:
 *     1. Compute moments (mu, sd) with tie correction
 *     2. Compute z = (mu - U - cc) / sd
 *     3. Return P(Z > z)
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double p_value_less(
    double U, double n1, double n2,
    double tie_sum = 0.0, double cc = 0.5
);

// =============================================================================
// SIMD Implementations
// =============================================================================

namespace simd {

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_two_sided (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized two-sided Mann-Whitney U test p-values.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V p_value_two_sided(
 *         D d, V U, V n1, V n2, V tie_sum, V cc
 *     )
 *
 * PARAMETERS:
 *     d        [in]  SIMD descriptor
 *     U        [in]  Vector of U statistics
 *     n1       [in]  Vector of sample sizes (group 1)
 *     n2       [in]  Vector of sample sizes (group 2)
 *     tie_sum  [in]  Vector of tie correction sums
 *     cc       [in]  Vector of continuity corrections
 *
 * RETURNS:
 *     Vector of two-sided p-values
 *
 * ALGORITHM:
 *     Fully vectorized computation with branchless handling of edge cases:
 *     1. Compute moments (mu, sd) in SIMD
 *     2. Compute z-scores in SIMD
 *     3. Call vectorized normal_sf
 *     4. Handle sd <= 0 case via masking (returns p = 1.0)
 *
 * COMPLEXITY:
 *     Time:  O(1) - true SIMD parallelism
 *     Space: O(1) - register-only computation
 *
 * THREAD SAFETY:
 *     Safe
 *
 * NUMERICAL NOTES:
 *     Uses precise vectorized normal_sf (lane-wise std::erfc)
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V p_value_two_sided(
    D d, V U, V n1, V n2, V tie_sum, V cc
);

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_greater (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized one-sided "greater" p-values.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V p_value_greater(
 *         D d, V U, V n1, V n2, V tie_sum, V cc
 *     )
 *
 * PARAMETERS:
 *     d        [in]  SIMD descriptor
 *     U        [in]  Vector of U statistics
 *     n1       [in]  Vector of sample sizes (group 1)
 *     n2       [in]  Vector of sample sizes (group 2)
 *     tie_sum  [in]  Vector of tie sums
 *     cc       [in]  Vector of continuity corrections
 *
 * RETURNS:
 *     Vector of one-sided p-values for "greater" alternative
 *
 * ALGORITHM:
 *     Vectorized z-score computation: z = (U - mu - cc) / sd
 *     Handles degenerate case (sd <= 0) via masking
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V p_value_greater(
    D d, V U, V n1, V n2, V tie_sum, V cc
);

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_less (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized one-sided "less" p-values.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V p_value_less(
 *         D d, V U, V n1, V n2, V tie_sum, V cc
 *     )
 *
 * PARAMETERS:
 *     d        [in]  SIMD descriptor
 *     U        [in]  Vector of U statistics
 *     n1       [in]  Vector of sample sizes (group 1)
 *     n2       [in]  Vector of sample sizes (group 2)
 *     tie_sum  [in]  Vector of tie sums
 *     cc       [in]  Vector of continuity corrections
 *
 * RETURNS:
 *     Vector of one-sided p-values for "less" alternative
 *
 * ALGORITHM:
 *     Vectorized z-score computation: z = (mu - U - cc) / sd
 *     Handles degenerate case via masking
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V p_value_less(
    D d, V U, V n1, V n2, V tie_sum, V cc
);

} // namespace simd

} // namespace scl::math::mwu
