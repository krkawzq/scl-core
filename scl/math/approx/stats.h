// =============================================================================
// FILE: scl/math/approx/stats.h
// BRIEF: API reference for approximate statistical distribution functions (approximate)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"

namespace scl::math::approx {

/* -----------------------------------------------------------------------------
 * FUNCTION: erfc (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Approximate complementary error function using Abramowitz-Stegun approximation.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double erfc(double x)
 *
 * PARAMETERS:
 *     x   [in]  Input value
 *
 * RETURNS:
 *     Approximate erfc(x) in range [0, 2]
 *
 * ALGORITHM:
 *     Abramowitz and Stegun rational approximation (Formula 7.1.26)
 *     Uses Horner's method for polynomial evaluation
 *     Result selection based on sign of x
 *
 * COMPLEXITY:
 *     Time:  O(1) - significantly faster than std::erfc
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - pure function
 *
 * NUMERICAL NOTES:
 *     Precision: ~1e-7 relative error (7 significant digits)
 *     Suitable for high-throughput p-value computation
 *     For full precision, use scl::math::erfc
 *
 * REFERENCE:
 *     Abramowitz & Stegun, Handbook of Mathematical Functions, 1964
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double erfc(double x);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_sf (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Approximate normal survival function using approximate erfc.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double normal_sf(double z)
 *
 * PARAMETERS:
 *     z   [in]  Z-score (standard normal deviate)
 *
 * RETURNS:
 *     Approximate P(Z > z) in range [0, 1]
 *
 * ALGORITHM:
 *     SF(z) = 0.5 * erfc(z / sqrt(2))
 *     Uses approximate Abramowitz-Stegun erfc internally
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 *
 * NUMERICAL NOTES:
 *     Precision: ~1e-7 relative error
 *     Ideal for differential expression analysis with thousands of genes
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double normal_sf(double z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_cdf (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Approximate normal CDF using approximate erfc.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double normal_cdf(double z)
 *
 * PARAMETERS:
 *     z   [in]  Z-score (standard normal deviate)
 *
 * RETURNS:
 *     Approximate P(Z <= z) in range [0, 1]
 *
 * ALGORITHM:
 *     CDF(z) = 0.5 * erfc(-z / sqrt(2))
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double normal_cdf(double z);

// =============================================================================
// SIMD Implementations
// =============================================================================

namespace simd {

/* -----------------------------------------------------------------------------
 * FUNCTION: erfc (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized approximate complementary error function.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V erfc(D d, V x)
 *
 * PARAMETERS:
 *     d   [in]  SIMD descriptor
 *     x   [in]  Input vector
 *
 * RETURNS:
 *     Vector containing approximate erfc(x[i]) for each lane i
 *
 * ALGORITHM:
 *     Fully vectorized Abramowitz-Stegun approximation
 *     Uses SIMD polynomial evaluation and exponential
 *     Branchless sign handling via masking
 *
 * COMPLEXITY:
 *     Time:  O(1) - true SIMD parallelism
 *     Space: O(1) - register-only computation
 *
 * THREAD SAFETY:
 *     Safe
 *
 * NUMERICAL NOTES:
 *     Precision: ~1e-7 per lane
 *     Much faster than scl::math::simd::erfc (which uses scalar loops)
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V erfc(D d, V x);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_sf (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized approximate normal survival function.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V normal_sf(D d, V z)
 *
 * PARAMETERS:
 *     d   [in]  SIMD descriptor
 *     z   [in]  Z-score vector
 *
 * RETURNS:
 *     Vector containing approximate P(Z > z[i]) for each lane i
 *
 * ALGORITHM:
 *     SF(z) = 0.5 * erfc(z / sqrt(2))
 *     Fully vectorized arithmetic
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V normal_sf(D d, V z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_cdf (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vectorized approximate normal CDF.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V normal_cdf(D d, V z)
 *
 * PARAMETERS:
 *     d   [in]  SIMD descriptor
 *     z   [in]  Z-score vector
 *
 * RETURNS:
 *     Vector containing approximate P(Z <= z[i]) for each lane i
 *
 * ALGORITHM:
 *     CDF(z) = 0.5 * erfc(-z / sqrt(2))
 *     Fully vectorized
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V normal_cdf(D d, V z);

} // namespace simd

} // namespace scl::math::approx
