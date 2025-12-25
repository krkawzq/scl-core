#pragma once

#include "scl/config.hpp"
#include <cstddef>
#include <cstdint>

// =============================================================================
/// @file common.hpp
/// @brief Common Types and Utilities
///
/// This header provides common type definitions, constants, and utility
/// functions used throughout the SCL core library.
///
/// =============================================================================

namespace scl {

// =============================================================================
// Precision-Dependent Type Definitions
// =============================================================================

#if defined(SCL_USE_FLOAT32)
    /// @brief Floating-point type (float32)
    using Float = float;
    /// @brief Floating-point constant suffix
    #define SCL_FLOAT_SUFFIX f
#elif defined(SCL_USE_FLOAT64)
    /// @brief Floating-point type (float64)
    using Float = double;
    /// @brief Floating-point constant suffix
    #define SCL_FLOAT_SUFFIX
#elif defined(SCL_USE_FLOAT16)
    // Note: float16 support requires appropriate library (e.g., half.hpp)
    #include <half.hpp>
    /// @brief Floating-point type (float16)
    using Float = half_float::half;
    /// @brief Floating-point constant suffix
    #define SCL_FLOAT_SUFFIX
#else
    #error "No precision type defined! Check scl/config.hpp"
#endif

// =============================================================================
// Common Type Aliases
// =============================================================================

/// @brief Size type (platform-dependent)
using Size = std::size_t;

/// @brief Index type (signed, for loop indices that may be negative)
using Index = std::ptrdiff_t;

/// @brief Integer type for counts and dimensions
using Int = std::int32_t;

// =============================================================================
// Constants
// =============================================================================

/// @brief Invalid index constant
constexpr Index INVALID_INDEX = -1;

/// @brief Default alignment for SIMD operations (bytes)
constexpr Size SIMD_ALIGNMENT = 32;

// =============================================================================
// Utility Functions
// =============================================================================

/// @brief Check if a pointer is aligned to a given boundary
///
/// @param ptr Pointer to check
/// @param alignment Alignment in bytes (must be power of 2)
/// @return True if pointer is aligned
inline bool is_aligned(const void* ptr, Size alignment) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

/// @brief Align a size to a given boundary
///
/// @param size Size to align
/// @param alignment Alignment in bytes (must be power of 2)
/// @return Aligned size
inline Size align_size(Size size, Size alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

} // namespace scl

