#pragma once

#include "scl/core/type.hpp"

// =============================================================================
// Highway Configuration
// =============================================================================

#if defined(SCL_ONLY_SCALAR) && !defined(HWY_COMPILE_ONLY_SCALAR)
    #define HWY_COMPILE_ONLY_SCALAR
#endif

#define HWY_DISABLED_TARGETS_LOG

#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>

// =============================================================================
// FILE: scl/core/simd.hpp
// BRIEF: SCL SIMD Wrapper (Google Highway)
// =============================================================================

namespace scl::simd {

    // =========================================================================
    // Core Namespace Injection
    // =========================================================================

    // Import Highway functions into scl::simd namespace
    using namespace hwy::HWY_NAMESPACE;

    // =========================================================================
    // Smart Tags (Type Inference)
    // =========================================================================

    using RealTag = ScalableTag<scl::Real>;
    using IndexTag = ScalableTag<scl::Index>;
    using ReinterpretTag = RebindToUnsigned<RealTag>;

    // =========================================================================
    // Type-Based SIMD Tag Selection
    // =========================================================================

    template <typename T>
    using SimdTagFor = std::conditional_t<
        std::is_same_v<T, Real>, RealTag,
        std::conditional_t<std::is_same_v<T, Index>, IndexTag,
            ScalableTag<T>>>;

} // namespace scl::simd
