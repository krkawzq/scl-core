#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"

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

    using Tag = ScalableTag<scl::Real>;
    using IndexTag = ScalableTag<scl::Index>;
    using ReinterpretTag = RebindToUnsigned<Tag>;

    // =========================================================================
    // Helper Constants
    // =========================================================================

    SCL_FORCE_INLINE size_t lanes() {
        return Lanes(Tag());
    }

    // =========================================================================
    // Type-Based SIMD Tag Selection
    // =========================================================================

    template <typename T>
    SCL_FORCE_INLINE auto GetSimdTag() {
        if constexpr (std::is_same_v<T, Real>) {
            return Tag();
        } else if constexpr (std::is_same_v<T, Index>) {
            return IndexTag();
        } else {
            return ScalableTag<T>();
        }
    }

    template <typename T>
    using SimdTagFor = std::conditional_t<
        std::is_same_v<T, Real>, Tag,
        std::conditional_t<std::is_same_v<T, Index>, IndexTag,
            ScalableTag<T>>>;

} // namespace scl::simd
