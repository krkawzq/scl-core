#pragma once

/// @file scl/math/stats.hpp
/// @brief Precise statistical distribution functions
///
/// This header provides:
///   - Error functions (erf, erfc) using std::erf/erfc (~15 significant digits)
///   - Normal distribution functions (CDF, SF, PDF, log variants)
///   - Both scalar and SIMD implementations
///
/// @note For approximate fast versions (~1e-7 precision), see scl/math/stats_fast.hpp

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/simd.hpp"

#include <cmath>

namespace scl::math {

// =============================================================================
// SECTION 1: Error Functions (Scalar)
// =============================================================================

/// @brief Complementary error function (precise)
/// @param[in] x Input value
/// @return erfc(x) = 1 - erf(x)
/// @note Uses std::erfc for full precision (~15 significant digits)
[[nodiscard]]
SCL_FORCE_INLINE
auto erfc(double x) -> double {
    return std::erfc(x);
}

/// @brief Error function (precise)
/// @param[in] x Input value
/// @return erf(x)
/// @note Uses std::erf for full precision (~15 significant digits)
[[nodiscard]]
SCL_FORCE_INLINE
auto erf(double x) -> double {
    return std::erf(x);
}

// =============================================================================
// SECTION 2: Normal Distribution Functions (Scalar)
// =============================================================================

/// @brief Normal cumulative distribution function
/// @param[in] z Z-score (standardized value)
/// @return P(Z <= z) for standard normal distribution
/// @note Φ(z) = 0.5 * erfc(-z / √2)
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_cdf(double z) -> double {
    return 0.5 * std::erfc(-z * 0.7071067811865475);
}

/// @brief Normal survival function (complementary CDF)
/// @param[in] z Z-score (standardized value)
/// @return P(Z > z) for standard normal distribution
/// @note SF(z) = 1 - Φ(z) = 0.5 * erfc(z / √2)
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_sf(double z) -> double {
    return 0.5 * std::erfc(z * 0.7071067811865475);
}

/// @brief Normal probability density function
/// @param[in] z Z-score (standardized value)
/// @return PDF value at z
/// @note φ(z) = (1/√(2π)) * exp(-z²/2)
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_pdf(double z) -> double {
    constexpr double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * z * z);
}

/// @brief Log of normal CDF (numerically stable)
/// @param[in] z Z-score (standardized value)
/// @return log(Φ(z))
/// @note Uses asymptotic expansion for large negative z
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_logcdf(double z) -> double {
    if (z < -20.0) [[unlikely]] {
        // Asymptotic expansion for large negative z
        double z2 = z * z;
        return -0.5 * z2 - 0.9189385332046727 - std::log(-z);
    }
    return std::log(normal_cdf(z));
}

/// @brief Log of normal survival function (numerically stable)
/// @param[in] z Z-score (standardized value)
/// @return log(SF(z)) = log(1 - Φ(z))
/// @note Uses asymptotic expansion for large positive z
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_logsf(double z) -> double {
    if (z > 20.0) [[unlikely]] {
        // Asymptotic expansion for large positive z
        double z2 = z * z;
        return -0.5 * z2 - 0.9189385332046727 - std::log(z);
    }
    return std::log(normal_sf(z));
}

// =============================================================================
// SECTION 3: SIMD Implementations
// =============================================================================

namespace simd {

namespace s = scl::simd;

/// @brief Lane-wise precise erfc via std::erfc
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] x Input vector
/// @return erfc(x) for each lane
/// @note Falls back to scalar std::erfc for each lane
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto erfc(D d, V x) -> V {
    const Size lanes = s::Lanes(d);
    
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_in[32];
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_out[32];

    s::Store(x, d, buffer_in);

    for (Size i = 0; i < lanes; ++i) {
        buffer_out[i] = std::erfc(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

/// @brief Lane-wise precise erf via std::erf
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] x Input vector
/// @return erf(x) for each lane
/// @note Falls back to scalar std::erf for each lane
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto erf(D d, V x) -> V {
    const Size lanes = s::Lanes(d);
    
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_in[32];
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_out[32];

    s::Store(x, d, buffer_in);

    for (Size i = 0; i < lanes; ++i) {
        buffer_out[i] = std::erf(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

/// @brief SIMD normal CDF
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] z Z-score vector
/// @return Φ(z) for each lane
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_cdf(D d, V z) -> V {
    const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
    const auto half = s::Set(d, 0.5);

    auto arg = s::Mul(s::Neg(z), inv_sqrt2);
    return s::Mul(half, erfc(d, arg));
}

/// @brief SIMD normal survival function
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] z Z-score vector
/// @return SF(z) for each lane
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_sf(D d, V z) -> V {
    const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
    const auto half = s::Set(d, 0.5);

    auto arg = s::Mul(z, inv_sqrt2);
    return s::Mul(half, erfc(d, arg));
}

/// @brief SIMD normal PDF
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] z Z-score vector
/// @return φ(z) for each lane
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_pdf(D d, V z) -> V {
    const auto inv_sqrt_2pi = s::Set(d, 0.3989422804014327);
    const auto half = s::Set(d, 0.5);

    auto z2 = s::Mul(z, z);
    auto exp_term = s::Exp(d, s::Neg(s::Mul(half, z2)));
    return s::Mul(inv_sqrt_2pi, exp_term);
}

/// @brief SIMD log normal CDF
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] z Z-score vector
/// @return log(Φ(z)) for each lane
/// @note Falls back to scalar for asymptotic expansion
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_logcdf(D d, V z) -> V {
    const Size lanes = s::Lanes(d);
    
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_in[32];
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_out[32];

    s::Store(z, d, buffer_in);

    for (Size i = 0; i < lanes; ++i) {
        buffer_out[i] = scl::math::normal_logcdf(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

/// @brief SIMD log normal survival function
/// @tparam D SIMD descriptor type
/// @tparam V SIMD vector type
/// @param[in] d SIMD descriptor
/// @param[in] z Z-score vector
/// @return log(SF(z)) for each lane
/// @note Falls back to scalar for asymptotic expansion
template<class D, class V>
[[nodiscard]]
SCL_FORCE_INLINE
auto normal_logsf(D d, V z) -> V {
    const Size lanes = s::Lanes(d);
    
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_in[32];
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    alignas(64) double buffer_out[32];

    s::Store(z, d, buffer_in);

    for (Size i = 0; i < lanes; ++i) {
        buffer_out[i] = scl::math::normal_logsf(buffer_in[i]);
    }

    return s::Load(d, buffer_out);
}

} // namespace simd

} // namespace scl::math

