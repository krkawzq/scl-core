// =============================================================================
// FILE: scl/math/approx/mwu.h
// BRIEF: API reference for approximate Mann-Whitney U test statistics (approximate)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"

namespace scl::math::approx::mwu {

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_two_sided (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Approximate two-sided p-value for Mann-Whitney U test using approximate erfc.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double p_value_two_sided(
 *         double U, double n1, double n2, double tie_sum, double cc = 0.5
 *     )
 *
 * PARAMETERS:
 *     U        [in]  Mann-Whitney U statistic
 *     n1       [in]  Sample size of group 1
 *     n2       [in]  Sample size of group 2
 *     tie_sum  [in]  Sum of (t^3 - t) for all tied groups
 *     cc       [in]  Continuity correction (default 0.5)
 *
 * RETURNS:
 *     Approximate two-sided p-value in [0, 1]
 *
 * ALGORITHM:
 *     1. Compute mean and inverse standard deviation (for faster multiplication)
 *     2. Compute z = (|U - mu| - cc) * inv_sd
 *     3. Return 2 * P(Z > z) using approximate normal_sf
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 *
 * NUMERICAL NOTES:
 *     Precision: ~1e-7 due to approximate erfc
 *     Returns inverse SD instead of SD for faster z-score computation
 *     Ideal for high-throughput differential expression analysis
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double p_value_two_sided(
    double U, double n1, double n2, double tie_sum, double cc = 0.5
);

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_greater (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Approximate one-sided "greater" p-value using approximate erfc.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double p_value_greater(
 *         double U, double n1, double n2, double tie_sum, double cc = 0.5
 *     )
 *
 * PARAMETERS:
 *     U        [in]  Mann-Whitney U statistic
 *     n1       [in]  Sample size of group 1
 *     n2       [in]  Sample size of group 2
 *     tie_sum  [in]  Tie correction sum
 *     cc       [in]  Continuity correction (default 0.5)
 *
 * RETURNS:
 *     Approximate one-sided p-value for "greater" alternative
 *
 * ALGORITHM:
 *     z = (U - mu - cc) * inv_sd
 *     Return P(Z > z) using approximate normal_sf
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double p_value_greater(
    double U, double n1, double n2, double tie_sum, double cc = 0.5
);

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_less (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Approximate one-sided "less" p-value using approximate erfc.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double p_value_less(
 *         double U, double n1, double n2, double tie_sum, double cc = 0.5
 *     )
 *
 * PARAMETERS:
 *     U        [in]  Mann-Whitney U statistic
 *     n1       [in]  Sample size of group 1
 *     n2       [in]  Sample size of group 2
 *     tie_sum  [in]  Tie correction sum
 *     cc       [in]  Continuity correction (default 0.5)
 *
 * RETURNS:
 *     Approximate one-sided p-value for "less" alternative
 *
 * ALGORITHM:
 *     z = (mu - U - cc) * inv_sd
 *     Return P(Z > z) using approximate normal_sf
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double p_value_less(
    double U, double n1, double n2, double tie_sum, double cc = 0.5
);

// =============================================================================
// SIMD Implementations
// =============================================================================

namespace simd {

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_two_sided (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized approximate two-sided Mann-Whitney U test p-values.
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
 *     Vector of approximate two-sided p-values
 *
 * ALGORITHM:
 *     Fully vectorized with branchless handling:
 *     1. Compute moments (mu, inv_sd) in SIMD
 *     2. z = (|U - mu| - cc) * inv_sd
 *     3. Call vectorized approximate normal_sf
 *     4. Handle degenerate case (inv_sd == 0) via masking
 *
 * COMPLEXITY:
 *     Time:  O(1) - true SIMD parallelism
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 *
 * NUMERICAL NOTES:
 *     Uses approximate vectorized erfc (much faster than precise version)
 *     Precision: ~1e-7 per lane
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V p_value_two_sided(
    D d, V U, V n1, V n2, V tie_sum, V cc
);

/* -----------------------------------------------------------------------------
 * FUNCTION: p_value_greater (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized approximate one-sided "greater" p-values.
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
 *     Vector of approximate one-sided p-values for "greater" alternative
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
 *     Vectorized approximate one-sided "less" p-values.
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
 *     Vector of approximate one-sided p-values for "less" alternative
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

} // namespace scl::math::approx::mwu
