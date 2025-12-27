// =============================================================================
// FILE: scl/math/stats.h
// BRIEF: API reference for precise statistical distribution functions
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"

namespace scl::math {

/* -----------------------------------------------------------------------------
 * FUNCTION: erfc (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Complementary error function with full precision.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double erfc(double x)
 *
 * PARAMETERS:
 *     x   [in]  Input value
 *
 * RETURNS:
 *     erfc(x) = 1 - erf(x) = (2/sqrt(pi)) * integral from x to infinity of exp(-t^2) dt
 *     Result in range [0, 2]
 *
 * ALGORITHM:
 *     Delegates to std::erfc for full IEEE 754 double precision (~15 digits)
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - pure function with no shared state
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double erfc(double x);

/* -----------------------------------------------------------------------------
 * FUNCTION: erf (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Error function with full precision.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double erf(double x)
 *
 * PARAMETERS:
 *     x   [in]  Input value
 *
 * RETURNS:
 *     erf(x) = (2/sqrt(pi)) * integral from 0 to x of exp(-t^2) dt
 *     Result in range [-1, 1]
 *
 * ALGORITHM:
 *     Delegates to std::erf for full precision
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - pure function
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double erf(double x);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_cdf (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Standard normal cumulative distribution function.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double normal_cdf(double z)
 *
 * PARAMETERS:
 *     z   [in]  Z-score (standard normal deviate)
 *
 * RETURNS:
 *     P(Z <= z) for Z ~ N(0,1)
 *     Result in range [0, 1]
 *
 * ALGORITHM:
 *     CDF(z) = 0.5 * erfc(-z / sqrt(2))
 *     Uses precise std::erfc internally
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - pure function
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double normal_cdf(double z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_sf (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Standard normal survival function (complement of CDF).
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double normal_sf(double z)
 *
 * PARAMETERS:
 *     z   [in]  Z-score (standard normal deviate)
 *
 * RETURNS:
 *     P(Z > z) = 1 - CDF(z)
 *     Result in range [0, 1]
 *
 * ALGORITHM:
 *     SF(z) = 0.5 * erfc(z / sqrt(2))
 *     More numerically stable than 1 - CDF(z) for large z
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - pure function
 *
 * NUMERICAL NOTES:
 *     Preferred over 1 - normal_cdf(z) for large positive z
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double normal_sf(double z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_pdf (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Standard normal probability density function.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double normal_pdf(double z)
 *
 * PARAMETERS:
 *     z   [in]  Z-score (standard normal deviate)
 *
 * RETURNS:
 *     f(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
 *     Result is always non-negative
 *
 * ALGORITHM:
 *     Direct evaluation using exp and precomputed constant
 *     inv_sqrt_2pi = 0.3989422804014327
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - pure function
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double normal_pdf(double z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_logcdf (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Natural logarithm of standard normal CDF.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double normal_logcdf(double z)
 *
 * PARAMETERS:
 *     z   [in]  Z-score (standard normal deviate)
 *
 * RETURNS:
 *     log(P(Z <= z))
 *     Result is always non-positive
 *
 * ALGORITHM:
 *     For z < -20: Uses asymptotic expansion
 *         log(Phi(z)) ~ -z^2/2 - log(sqrt(2*pi)) - log(-z)
 *     Otherwise: log(normal_cdf(z))
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - pure function
 *
 * NUMERICAL NOTES:
 *     Provides improved stability for extreme negative values
 *     Avoids underflow in log(CDF(z)) for z < -37
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double normal_logcdf(double z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_logsf (scalar)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Natural logarithm of standard normal survival function.
 *
 * SIGNATURE:
 *     SCL_FORCE_INLINE double normal_logsf(double z)
 *
 * PARAMETERS:
 *     z   [in]  Z-score (standard normal deviate)
 *
 * RETURNS:
 *     log(P(Z > z))
 *     Result is always non-positive
 *
 * ALGORITHM:
 *     For z > 20: Uses asymptotic expansion
 *         log(1 - Phi(z)) ~ -z^2/2 - log(sqrt(2*pi)) - log(z)
 *     Otherwise: log(normal_sf(z))
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - pure function
 *
 * NUMERICAL NOTES:
 *     Provides improved stability for extreme positive values
 *     Avoids underflow in log(SF(z)) for z > 37
 * -------------------------------------------------------------------------- */
SCL_FORCE_INLINE double normal_logsf(double z);

// =============================================================================
// SIMD Implementations
// =============================================================================

namespace simd {

/* -----------------------------------------------------------------------------
 * FUNCTION: erfc (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Complementary error function, vectorized with full precision.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V erfc(D d, V x)
 *
 * PARAMETERS:
 *     d   [in]  SIMD descriptor (e.g., scl::simd::Tag)
 *     x   [in]  Input vector
 *
 * RETURNS:
 *     Vector containing erfc(x[i]) for each lane i
 *
 * ALGORITHM:
 *     Lane-wise scalar std::erfc calls via temporary buffers
 *     Not vectorized intrinsically, but provides uniform API
 *
 * COMPLEXITY:
 *     Time:  O(lanes) - serial scalar calls
 *     Space: O(lanes) - two aligned buffers
 *
 * THREAD SAFETY:
 *     Safe - uses stack-allocated buffers
 *
 * NUMERICAL NOTES:
 *     Slower than scl::math::fast::simd::erfc but maintains full precision
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V erfc(D d, V x);

/* -----------------------------------------------------------------------------
 * FUNCTION: erf (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Error function, vectorized with full precision.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V erf(D d, V x)
 *
 * PARAMETERS:
 *     d   [in]  SIMD descriptor
 *     x   [in]  Input vector
 *
 * RETURNS:
 *     Vector containing erf(x[i]) for each lane i
 *
 * ALGORITHM:
 *     Lane-wise scalar std::erf calls
 *
 * COMPLEXITY:
 *     Time:  O(lanes)
 *     Space: O(lanes)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V erf(D d, V x);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_cdf (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Standard normal CDF, vectorized.
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
 *     Vector containing P(Z <= z[i]) for each lane i
 *
 * ALGORITHM:
 *     CDF(z) = 0.5 * erfc(-z / sqrt(2))
 *     Fully vectorized arithmetic, calls vectorized erfc
 *
 * COMPLEXITY:
 *     Time:  O(lanes) - limited by erfc
 *     Space: O(lanes)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V normal_cdf(D d, V z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_sf (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Standard normal survival function, vectorized.
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
 *     Vector containing P(Z > z[i]) for each lane i
 *
 * ALGORITHM:
 *     SF(z) = 0.5 * erfc(z / sqrt(2))
 *     Fully vectorized arithmetic
 *
 * COMPLEXITY:
 *     Time:  O(lanes)
 *     Space: O(lanes)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V normal_sf(D d, V z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_pdf (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Standard normal PDF, vectorized.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V normal_pdf(D d, V z)
 *
 * PARAMETERS:
 *     d   [in]  SIMD descriptor
 *     z   [in]  Z-score vector
 *
 * RETURNS:
 *     Vector containing f(z[i]) for each lane i
 *
 * ALGORITHM:
 *     f(z) = inv_sqrt_2pi * exp(-z^2/2)
 *     Fully vectorized using SIMD multiply, exp
 *
 * COMPLEXITY:
 *     Time:  O(1) - true SIMD parallelism
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V normal_pdf(D d, V z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_logcdf (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Log of standard normal CDF, vectorized.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V normal_logcdf(D d, V z)
 *
 * PARAMETERS:
 *     d   [in]  SIMD descriptor
 *     z   [in]  Z-score vector
 *
 * RETURNS:
 *     Vector containing log(P(Z <= z[i])) for each lane i
 *
 * ALGORITHM:
 *     Lane-wise calls to scalar normal_logcdf (includes asymptotic branch)
 *
 * COMPLEXITY:
 *     Time:  O(lanes)
 *     Space: O(lanes)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V normal_logcdf(D d, V z);

/* -----------------------------------------------------------------------------
 * FUNCTION: normal_logsf (SIMD)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Log of standard normal survival function, vectorized.
 *
 * SIGNATURE:
 *     template <class D, class V>
 *     SCL_FORCE_INLINE V normal_logsf(D d, V z)
 *
 * PARAMETERS:
 *     d   [in]  SIMD descriptor
 *     z   [in]  Z-score vector
 *
 * RETURNS:
 *     Vector containing log(P(Z > z[i])) for each lane i
 *
 * ALGORITHM:
 *     Lane-wise calls to scalar normal_logsf
 *
 * COMPLEXITY:
 *     Time:  O(lanes)
 *     Space: O(lanes)
 *
 * THREAD SAFETY:
 *     Safe
 * -------------------------------------------------------------------------- */
template <class D, class V>
SCL_FORCE_INLINE V normal_logsf(D d, V z);

} // namespace simd

} // namespace scl::math
