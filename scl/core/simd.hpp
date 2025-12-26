#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"

// =============================================================================
// Highway Configuration
// =============================================================================

// 1. Propagate Scalar-Only mode from SCL config to Highway
#if defined(SCL_ONLY_SCALAR) && !defined(HWY_COMPILE_ONLY_SCALAR)
    #define HWY_COMPILE_ONLY_SCALAR
#endif

// 2. Disable annoying console logs from Highway
#define HWY_DISABLED_TARGETS_LOG

// 3. Include the main Highway header
#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h> // Include highway math functions like Exp


// =============================================================================
/// @file simd.hpp
/// @brief SCL SIMD Wrapper (Google Highway)
///
/// This header exposes Google Highway intrinsics under the `scl::simd` namespace.
/// It enables writing architecture-agnostic vectorized code using SCL types.
///
/// Usage:
/// scl::simd::Tag d;                 // Auto-matches scl::Real (f32/f64)
/// auto v = scl::simd::Load(d, ptr); // Calls AVX2/AVX512/NEON Load internally
/// auto r = scl::simd::Add(v, v);
// =============================================================================

namespace scl::simd {

    // =========================================================================
    // 1. Core Namespace Injection
    // =========================================================================

    // [Magic Line]
    // Import all Highway functions (Load, Store, Add, Mul, etc.) from the 
    // arch-specific namespace (e.g., hwy::N_AVX2) into scl::simd.
    //
    // This allows you to write `scl::simd::Add(...)` directly.
    using namespace hwy::HWY_NAMESPACE;

    // =========================================================================
    // 2. Smart Tags (Type Inference)
    // =========================================================================

    /// @brief The primary SIMD descriptor tag for `scl::Real`.
    ///
    /// Instead of writing `ScalableTag<scl::Real>`, just use `Tag`.
    /// It automatically selects the vector width supported by the hardware.
    using Tag = ScalableTag<scl::Real>;

    /// @brief SIMD descriptor tag for `scl::Index` (int64_t).
    using IndexTag = ScalableTag<scl::Index>;

    /// @brief SIMD descriptor tag for integers of the same size as Real.
    /// Useful for bitwise operations or masks on floats.
    using ReinterpretTag = RebindToUnsigned<Tag>;

    // =========================================================================
    // 3. Helper Constants
    // =========================================================================

    /// @brief Get the number of lanes (elements) in a vector register.
    /// Wraps `Lanes()` to make it easier to call without instantiating a dummy tag.
    SCL_FORCE_INLINE size_t lanes() {
        return Lanes(Tag());
    }

} // namespace scl::simd
