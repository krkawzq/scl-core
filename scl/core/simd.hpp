#pragma once

/**
 * @file scl/core/simd.hpp
 * @brief SCL SIMD Wrapper - Pure Google Highway Integration
 *
 * This header provides a minimal, dependency-free Highway SIMD wrapper:
 *   - Highway configuration and header inclusion
 *   - Namespace injection for convenient access to Highway primitives
 *   - Generic SIMD type tag selection utilities
 *
 * @note This is a foundational header with minimal dependencies.
 *       Higher-level SIMD algorithms (sorting, searching, etc.) are
 *       provided in separate headers to avoid circular dependencies.
 *
 * @note Depends only on scl/config.hpp for platform detection.
 */

#include "scl/config.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// =============================================================================
// Highway Configuration
// =============================================================================

/// @brief Force scalar-only mode if requested
#if defined(SCL_ONLY_SCALAR) && !defined(HWY_COMPILE_ONLY_SCALAR)
  #define HWY_COMPILE_ONLY_SCALAR
#endif

/// @brief Disable target logging to reduce binary size
#define HWY_DISABLED_TARGETS_LOG

// Include Highway headers
#include <hwy/highway.h>

// Optional: Highway contrib headers (math, sort, etc.)
// These are included on-demand by higher-level algorithms
// #include <hwy/contrib/math/math-inl.h>
// #include <hwy/contrib/sort/vqsort-inl.h>

namespace scl::simd {

// =============================================================================
// SECTION 1: Namespace Injection
// =============================================================================

/// @brief Import Highway functions into scl::simd namespace
/// @note This provides convenient access to all Highway SIMD primitives
///       Users can call hwy::HWY_NAMESPACE::Load() as scl::simd::Load()
using namespace hwy::HWY_NAMESPACE;

// =============================================================================
// SECTION 2: Generic SIMD Type Tags
// =============================================================================

/// @brief Generic SIMD tag selection for any scalar type
/// @tparam T Scalar element type
/// @note Creates a scalable SIMD tag for the given type, allowing
///       the compiler to select the best vector width for the target
template <typename T>
using SimdTagFor = hwy::HWY_NAMESPACE::ScalableTag<T>;

/// @brief Fixed-size SIMD tag selection
/// @tparam T Scalar element type
/// @tparam kLanes Number of lanes (vector width)
template <typename T, std::size_t kLanes>
using FixedSimdTagFor = hwy::HWY_NAMESPACE::FixedTag<T, kLanes>;

// =============================================================================
// SECTION 3: SIMD Type Traits
// =============================================================================

/// @brief Check if a type is SIMD-compatible (trivially copyable arithmetic)
/// @tparam T Type to check
template <typename T>
inline constexpr bool is_simd_compatible_v =
    std::is_arithmetic_v<T> && std::is_trivially_copyable_v<T>;

/// @brief Concept for SIMD-compatible types
/// @tparam T Type to check
template <typename T>
concept SimdCompatible = is_simd_compatible_v<T>;

// =============================================================================
// SECTION 4: SIMD Configuration Query
// =============================================================================

/// @brief Get the number of SIMD lanes for a given type
/// @tparam T Scalar element type
/// @return Number of lanes (vector width) for the current target
template <typename T>
[[nodiscard]]
constexpr
auto simd_lanes() noexcept -> std::size_t {
  const SimdTagFor<T> d;
  return hwy::HWY_NAMESPACE::Lanes(d);
}

/// @brief Get the SIMD vector byte size for a given type
/// @tparam T Scalar element type
/// @return Size of SIMD vector in bytes for the current target
template <typename T>
[[nodiscard]]
constexpr
auto simd_bytes() noexcept -> std::size_t {
  return simd_lanes<T>() * sizeof(T);
}

/// @brief Check if SIMD is enabled (not scalar-only mode)
/// @return true if SIMD instructions are available
[[nodiscard]]
constexpr
auto simd_enabled() noexcept -> bool {
#if defined(HWY_COMPILE_ONLY_SCALAR)
  return false;
#else
  return true;
#endif
}

/// @brief Get the current SIMD target name
/// @return String describing the current SIMD target (e.g., "AVX2", "NEON")
[[nodiscard]]
inline
auto simd_target_name() noexcept -> const char* {
  return hwy::TargetName(hwy::HWY_TARGET);
}

}  // namespace scl::simd

