#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include <cmath>

namespace scl::math {

// =============================================================================
// Scalar Implementations (Precise/Exact)
// =============================================================================

/// @brief Precise complementary error function erfc(x).
/// Uses standard library implementation for full precision.
SCL_FORCE_INLINE double erfc(double x) {
    return std::erfc(x);
}

/// @brief Precise error function erf(x).
/// Uses standard library implementation for full precision.
SCL_FORCE_INLINE double erf(double x) {
    return std::erf(x);
}

/// @brief Normal cumulative distribution function CDF(z).
/// Precise implementation using standard library erfc.
SCL_FORCE_INLINE double normal_cdf(double z) {
    return 0.5 * std::erfc(-z * 0.7071067811865475); // -z/sqrt(2)
}

/// @brief Normal survival function SF(z) = 1 - CDF(z).
/// Precise implementation using standard library erfc.
SCL_FORCE_INLINE double normal_sf(double z) {
    return 0.5 * std::erfc(z * 0.7071067811865475); // z/sqrt(2)
}

/// @brief Normal probability density function PDF(z).
/// f(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
SCL_FORCE_INLINE double normal_pdf(double z) {
    constexpr double inv_sqrt_2pi = 0.3989422804014327; // 1/sqrt(2*pi)
    return inv_sqrt_2pi * std::exp(-0.5 * z * z);
}

/// @brief Natural logarithm of normal CDF.
/// More precise for extreme values.
SCL_FORCE_INLINE double normal_logcdf(double z) {
    if (z < -20.0) {
        // Use asymptotic expansion for large negative values
        // log(Phi(z)) ≈ log(phi(z)) - log(-z) - 1/(2z^2) - ...
        double z2 = z * z;
        return -0.5 * z2 - 0.9189385332046727 - std::log(-z); // -log(sqrt(2*pi)) = -0.9189385332046727
    }
    return std::log(normal_cdf(z));
}

/// @brief Natural logarithm of normal survival function.
/// More precise for extreme values.
SCL_FORCE_INLINE double normal_logsf(double z) {
    if (z > 20.0) {
        // Use asymptotic expansion for large positive values
        double z2 = z * z;
        return -0.5 * z2 - 0.9189385332046727 - std::log(z);
    }
    return std::log(normal_sf(z));
}

// =============================================================================
// SIMD Implementations (Precise)
// =============================================================================

namespace simd {
    
    namespace s = scl::simd;

    /// @brief SIMD version of precise erfc using standard library calls.
    /// Note: This performs lane-wise scalar calls, which is slower than 
    /// vectorized approximations but maintains full precision.
    template <class D, class V>
    SCL_FORCE_INLINE V erfc(D d, V x) {
        alignas(64) double buffer_in[s::Lanes(d)];
        alignas(64) double buffer_out[s::Lanes(d)];
        
        s::Store(x, d, buffer_in);
        
        for (size_t i = 0; i < s::Lanes(d); ++i) {
            buffer_out[i] = std::erfc(buffer_in[i]);
        }
        
        return s::Load(d, buffer_out);
    }

    /// @brief SIMD version of precise erf using standard library calls.
    template <class D, class V>
    SCL_FORCE_INLINE V erf(D d, V x) {
        alignas(64) double buffer_in[s::Lanes(d)];
        alignas(64) double buffer_out[s::Lanes(d)];
        
        s::Store(x, d, buffer_in);
        
        for (size_t i = 0; i < s::Lanes(d); ++i) {
            buffer_out[i] = std::erf(buffer_in[i]);
        }
        
        return s::Load(d, buffer_out);
    }

    /// @brief SIMD Normal CDF
    template <class D, class V>
    SCL_FORCE_INLINE V normal_cdf(D d, V z) {
        const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
        const auto half      = s::Set(d, 0.5);
        
        auto arg = s::Mul(s::Neg(z), inv_sqrt2);
        return s::Mul(half, erfc(d, arg));
    }

    /// @brief SIMD Normal Survival Function
    template <class D, class V>
    SCL_FORCE_INLINE V normal_sf(D d, V z) {
        const auto inv_sqrt2 = s::Set(d, 0.7071067811865475);
        const auto half      = s::Set(d, 0.5);
        
        auto arg = s::Mul(z, inv_sqrt2);
        return s::Mul(half, erfc(d, arg));
    }

    /// @brief SIMD Normal PDF
    template <class D, class V>
    SCL_FORCE_INLINE V normal_pdf(D d, V z) {
        const auto inv_sqrt_2pi = s::Set(d, 0.3989422804014327);
        const auto half         = s::Set(d, 0.5);
        
        auto z2 = s::Mul(z, z);
        auto exp_term = s::Exp(d, s::Neg(s::Mul(half, z2)));
        return s::Mul(inv_sqrt_2pi, exp_term);
    }

    /// @brief SIMD Normal log-CDF
    template <class D, class V>
    SCL_FORCE_INLINE V normal_logcdf(D d, V z) {
        alignas(64) double buffer_in[s::Lanes(d)];
        alignas(64) double buffer_out[s::Lanes(d)];
        
        s::Store(z, d, buffer_in);
        
        for (size_t i = 0; i < s::Lanes(d); ++i) {
            buffer_out[i] = normal_logcdf(buffer_in[i]);
        }
        
        return s::Load(d, buffer_out);
    }

    /// @brief SIMD Normal log-SF
    template <class D, class V>
    SCL_FORCE_INLINE V normal_logsf(D d, V z) {
        alignas(64) double buffer_in[s::Lanes(d)];
        alignas(64) double buffer_out[s::Lanes(d)];
        
        s::Store(z, d, buffer_in);
        
        for (size_t i = 0; i < s::Lanes(d); ++i) {
            buffer_out[i] = normal_logsf(buffer_in[i]);
        }
        
        return s::Load(d, buffer_out);
    }

} // namespace simd

} // namespace scl::math

