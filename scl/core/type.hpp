#pragma once

/**
 * @file scl/core/type.hpp
 * @brief Core type system for SCL - Type definitions and precision configuration
 *
 * This header provides:
 *   - C-style type definitions (scl_int8_t, scl_float32_t, etc.)
 *   - Platform-specific extended type support (int128, float16, float128)
 *   - C++ style type aliases (Int8, Real32, Index32, etc.)
 *   - Default type configuration (Index, Real, Int, Uint)
 *   - Type traits and concepts for SCL types
 *   - Precision support detection and SFINAE utilities
 *
 * @note This is a foundational header with minimal dependencies.
 *       The library uses dynamic dispatch - Real and Index are just default types.
 *
 * @note Extended type detection macros from scl/config.hpp:
 *       - SCL_HAS_INT128, SCL_HAS_FLOAT16, SCL_HAS_FLOAT128
 *       - SCL_ENABLE_INT128, SCL_ENABLE_FLOAT16, SCL_ENABLE_FLOAT128
 */

#include "scl/config.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// =============================================================================
// SECTION 1: Extended Type Native Aliases
// =============================================================================

namespace scl::detail {

// Extended type native aliases (platform-dependent)
#if SCL_HAS_INT128
using int128_native_t = __int128;
using uint128_native_t = unsigned __int128;
#endif

#if SCL_HAS_NATIVE_FLOAT16
  #if defined(__FLT16_MAX__)
using float16_native_t = _Float16;
  #elif defined(__ARM_FP16_FORMAT_IEEE)
using float16_native_t = __fp16;
  #endif
#endif

#if SCL_HAS_FLOAT128
using float128_native_t = __float128;
#endif

}  // namespace scl::detail

// =============================================================================
// SECTION 2: C-Style Type Definitions (scl_xxx_t)
// =============================================================================

namespace scl {

// -----------------------------------------------------------------------------
// Null Type (for unsupported precisions)
// -----------------------------------------------------------------------------

/// @brief Null type for unsupported precision types
/// @note Used as placeholder when platform doesn't support certain precisions
using scl_null_t = std::nullptr_t;

// -----------------------------------------------------------------------------
// Integer Types (Signed)
// -----------------------------------------------------------------------------

/// @brief 8-bit signed integer (C-style)
using scl_int8_t = std::int8_t;

/// @brief 16-bit signed integer (C-style)
using scl_int16_t = std::int16_t;

/// @brief 32-bit signed integer (C-style)
using scl_int32_t = std::int32_t;

/// @brief 64-bit signed integer (C-style)
using scl_int64_t = std::int64_t;

#if SCL_ENABLE_INT128
/// @brief 128-bit signed integer (C-style, platform-dependent)
using scl_int128_t = detail::int128_native_t;
#endif

// -----------------------------------------------------------------------------
// Integer Types (Unsigned)
// -----------------------------------------------------------------------------

/// @brief 8-bit unsigned integer (C-style)
using scl_uint8_t = std::uint8_t;

/// @brief 16-bit unsigned integer (C-style)
using scl_uint16_t = std::uint16_t;

/// @brief 32-bit unsigned integer (C-style)
using scl_uint32_t = std::uint32_t;

/// @brief 64-bit unsigned integer (C-style)
using scl_uint64_t = std::uint64_t;

#if SCL_ENABLE_INT128
/// @brief 128-bit unsigned integer (C-style, platform-dependent)
using scl_uint128_t = detail::uint128_native_t;
#endif

// -----------------------------------------------------------------------------
// Floating-Point Types
// -----------------------------------------------------------------------------

#if SCL_ENABLE_FLOAT16
/// @brief 16-bit floating-point (C-style, IEEE 754 binary16, platform-dependent)
using scl_float16_t = detail::float16_native_t;
#endif

/// @brief 32-bit floating-point (C-style, IEEE 754 binary32)
using scl_float32_t = float;

/// @brief 64-bit floating-point (C-style, IEEE 754 binary64)
using scl_float64_t = double;

#if SCL_ENABLE_FLOAT128
/// @brief 128-bit floating-point (C-style, IEEE 754 binary128, platform-dependent)
using scl_float128_t = detail::float128_native_t;
#endif

// -----------------------------------------------------------------------------
// Size and Pointer Types
// -----------------------------------------------------------------------------

/// @brief Size type (C-style, unsigned integer for sizes)
using scl_size_t = std::size_t;

/// @brief Pointer difference type (C-style, signed integer)
using scl_ptrdiff_t = std::ptrdiff_t;

/// @brief Byte type (C-style)
using scl_byte_t = std::byte;

/// @brief Mask type (C-style, unsigned integer for bit masks)
using scl_mask_t = std::uint32_t;

}  // namespace scl

// =============================================================================
// SECTION 3: C++ Style Type Aliases
// =============================================================================

namespace scl {

// -----------------------------------------------------------------------------
// Integer Types (Signed)
// -----------------------------------------------------------------------------

/// @brief 8-bit signed integer
using Int8 = scl_int8_t;

/// @brief 16-bit signed integer
using Int16 = scl_int16_t;

/// @brief 32-bit signed integer
using Int32 = scl_int32_t;

/// @brief 64-bit signed integer
using Int64 = scl_int64_t;

#if SCL_ENABLE_INT128
/// @brief 128-bit signed integer (extended precision)
using Int128 = scl_int128_t;
#else
/// @brief 128-bit signed integer placeholder (platform not supported)
using Int128 = scl_null_t;
#endif

// -----------------------------------------------------------------------------
// Integer Types (Unsigned)
// -----------------------------------------------------------------------------

/// @brief 8-bit unsigned integer
using UInt8 = scl_uint8_t;

/// @brief 16-bit unsigned integer
using UInt16 = scl_uint16_t;

/// @brief 32-bit unsigned integer
using UInt32 = scl_uint32_t;

/// @brief 64-bit unsigned integer
using UInt64 = scl_uint64_t;

#if SCL_ENABLE_INT128
/// @brief 128-bit unsigned integer (extended precision)
using UInt128 = scl_uint128_t;
#else
/// @brief 128-bit unsigned integer placeholder (platform not supported)
using UInt128 = scl_null_t;
#endif

// -----------------------------------------------------------------------------
// Floating-Point Types
// -----------------------------------------------------------------------------

#if SCL_ENABLE_FLOAT16
/// @brief 16-bit floating-point (half precision, IEEE 754 binary16)
using Real16 = scl_float16_t;
#else
/// @brief 16-bit floating-point placeholder (platform not supported)
using Real16 = scl_null_t;
#endif

/// @brief 32-bit floating-point (single precision, IEEE 754 binary32)
using Real32 = scl_float32_t;

/// @brief 64-bit floating-point (double precision, IEEE 754 binary64)
using Real64 = scl_float64_t;

#if SCL_ENABLE_FLOAT128
/// @brief 128-bit floating-point (quadruple precision, IEEE 754 binary128)
using Real128 = scl_float128_t;
#else
/// @brief 128-bit floating-point placeholder (platform not supported)
using Real128 = scl_null_t;
#endif

// -----------------------------------------------------------------------------
// Index Types (for array indexing)
// -----------------------------------------------------------------------------

/// @brief 32-bit signed index type
using Index32 = scl_int32_t;

/// @brief 64-bit signed index type
using Index64 = scl_int64_t;

// Note: Index128 not supported - Index32/Index64 are sufficient for practical use

// -----------------------------------------------------------------------------
// Utility Types
// -----------------------------------------------------------------------------

/// @brief Size type (unsigned integer for sizes and counts)
using Size = scl_size_t;

/// @brief Stride type (signed integer for memory strides)
using Stride = scl_ptrdiff_t;

/// @brief Offset type (signed integer for pointer offsets)
using Offset = scl_ptrdiff_t;

/// @brief Raw byte type
using Byte = scl_byte_t;

/// @brief Mask type (unsigned integer for bit masks)
using Mask = scl_mask_t;

// -----------------------------------------------------------------------------
// Legacy Aliases (for compatibility)
// -----------------------------------------------------------------------------

/// @brief Single-precision floating-point (legacy alias)
using Float32 = Real32;

/// @brief Double-precision floating-point (legacy alias)
using Float64 = Real64;

}  // namespace scl

// =============================================================================
// SECTION 4: Default Type Configuration
// =============================================================================

namespace scl {

// -----------------------------------------------------------------------------
// Default Index Type
// -----------------------------------------------------------------------------

/// @brief Default index type configuration
/// @note Can be overridden with -DSCL_DEFAULT_INDEX32 or -DSCL_DEFAULT_INDEX64
#if defined(SCL_DEFAULT_INDEX64)
/// @brief Default index type (64-bit)
using Index = Index64;
inline constexpr int kDefaultIndexBits = 64;
#elif defined(SCL_DEFAULT_INDEX32)
/// @brief Default index type (32-bit)
using Index = Index32;
inline constexpr int kDefaultIndexBits = 32;
#else
/// @brief Default index type (32-bit by default)
using Index = Index32;
inline constexpr int kDefaultIndexBits = 32;
#endif

// -----------------------------------------------------------------------------
// Default Real Type
// -----------------------------------------------------------------------------

/// @brief Default floating-point type configuration
/// @note Can be overridden with -DSCL_DEFAULT_REAL32 or -DSCL_DEFAULT_REAL64
#if defined(SCL_DEFAULT_REAL32)
/// @brief Default floating-point type (32-bit)
using Real = Real32;
inline constexpr int kDefaultRealBits = 32;
#elif defined(SCL_DEFAULT_REAL64)
/// @brief Default floating-point type (64-bit)
using Real = Real64;
inline constexpr int kDefaultRealBits = 64;
#else
/// @brief Default floating-point type (64-bit by default)
using Real = Real64;
inline constexpr int kDefaultRealBits = 64;
#endif

// -----------------------------------------------------------------------------
// Default Int Type
// -----------------------------------------------------------------------------

/// @brief Default signed integer type configuration
/// @note Can be overridden with -DSCL_DEFAULT_INT32 or -DSCL_DEFAULT_INT64
#if defined(SCL_DEFAULT_INT64)
/// @brief Default signed integer type (64-bit)
using Int = Int64;
inline constexpr int kDefaultIntBits = 64;
#elif defined(SCL_DEFAULT_INT32)
/// @brief Default signed integer type (32-bit)
using Int = Int32;
inline constexpr int kDefaultIntBits = 32;
#else
/// @brief Default signed integer type (32-bit by default)
using Int = Int32;
inline constexpr int kDefaultIntBits = 32;
#endif

// -----------------------------------------------------------------------------
// Default Uint Type
// -----------------------------------------------------------------------------

/// @brief Default unsigned integer type configuration
/// @note Can be overridden with -DSCL_DEFAULT_UINT32 or -DSCL_DEFAULT_UINT64
#if defined(SCL_DEFAULT_UINT64)
/// @brief Default unsigned integer type (64-bit)
using Uint = UInt64;
inline constexpr int kDefaultUintBits = 64;
#elif defined(SCL_DEFAULT_UINT32)
/// @brief Default unsigned integer type (32-bit)
using Uint = UInt32;
inline constexpr int kDefaultUintBits = 32;
#else
/// @brief Default unsigned integer type (32-bit by default)
using Uint = UInt32;
inline constexpr int kDefaultUintBits = 32;
#endif

}  // namespace scl

// =============================================================================
// SECTION 5: Type Information (Compile-Time)
// =============================================================================

namespace scl {

/// @brief Compile-time type size information
struct TypeInfo {
  // Default type info
  static constexpr const char* kRealTypeName =
      (kDefaultRealBits == 32) ? "real32" : "real64";  // NOLINT
  static constexpr const char* kIndexTypeName =
      (kDefaultIndexBits == 32) ? "index32" : "index64";  // NOLINT

  static constexpr int kRealSize = sizeof(Real);
  static constexpr int kIndexSize = sizeof(Index);

  // Extended type availability
  static constexpr bool kHasFloat16 = SCL_ENABLE_FLOAT16;
  static constexpr bool kHasFloat128 = SCL_ENABLE_FLOAT128;
  static constexpr bool kHasInt128 = SCL_ENABLE_INT128;

  // Standard type sizes
  static constexpr int kInt8Size = sizeof(Int8);
  static constexpr int kInt16Size = sizeof(Int16);
  static constexpr int kInt32Size = sizeof(Int32);
  static constexpr int kInt64Size = sizeof(Int64);

  static constexpr int kReal32Size = sizeof(Real32);
  static constexpr int kReal64Size = sizeof(Real64);

#if SCL_ENABLE_FLOAT16
  static constexpr int kReal16Size = sizeof(Real16);
#endif

#if SCL_ENABLE_FLOAT128
  static constexpr int kReal128Size = sizeof(Real128);
#endif

#if SCL_ENABLE_INT128
  static constexpr int kInt128Size = sizeof(Int128);
#endif
};

}  // namespace scl

// =============================================================================
// SECTION 6: Type Traits
// =============================================================================

namespace scl {

// -----------------------------------------------------------------------------
// Basic Type Category Checks
// -----------------------------------------------------------------------------

/// @brief Check if type is a standard signed integer type
template <typename T>
inline constexpr bool is_int_v =
    std::is_same_v<T, Int8> || std::is_same_v<T, Int16> ||
    std::is_same_v<T, Int32> || std::is_same_v<T, Int64>;

/// @brief Check if type is a standard unsigned integer type
template <typename T>
inline constexpr bool is_uint_v =
    std::is_same_v<T, UInt8> || std::is_same_v<T, UInt16> ||
    std::is_same_v<T, UInt32> || std::is_same_v<T, UInt64>;

/// @brief Check if type is a standard real type
template <typename T>
inline constexpr bool is_real_v =
    std::is_same_v<T, Real32> || std::is_same_v<T, Real64>;

/// @brief Check if type is an index type
template <typename T>
inline constexpr bool is_index_v =
    std::is_same_v<T, Index32> || std::is_same_v<T, Index64>;

// -----------------------------------------------------------------------------
// Extended Type Checks
// -----------------------------------------------------------------------------

#if SCL_ENABLE_INT128
/// @brief Check if type is an extended integer type (128-bit)
template <typename T>
inline constexpr bool is_int_extended_v = std::is_same_v<T, Int128>;
#else
template <typename T>
inline constexpr bool is_int_extended_v = false;
#endif

#if SCL_ENABLE_FLOAT16 || SCL_ENABLE_FLOAT128
/// @brief Check if type is an extended real type (16-bit or 128-bit)
template <typename T>
inline constexpr bool is_real_extended_v =
#if SCL_ENABLE_FLOAT16
    std::is_same_v<T, Real16>
#else
    false
#endif
#if SCL_ENABLE_FLOAT128
    || std::is_same_v<T, Real128>
#endif
    ;
#else
template <typename T>
inline constexpr bool is_real_extended_v = false;
#endif

// -----------------------------------------------------------------------------
// Combined Type Checks
// -----------------------------------------------------------------------------

/// @brief Check if type is any signed integer type (including extended)
template <typename T>
inline constexpr bool is_int_any_v = is_int_v<T> || is_int_extended_v<T>;

/// @brief Check if type is any unsigned integer type (including extended)
template <typename T>
inline constexpr bool is_uint_any_v = is_uint_v<T>
#if SCL_ENABLE_INT128
    || std::is_same_v<T, UInt128>
#endif
    ;

/// @brief Check if type is any real type (including extended)
template <typename T>
inline constexpr bool is_real_any_v = is_real_v<T> || is_real_extended_v<T>;

/// @brief Check if type is any numeric type
template <typename T>
inline constexpr bool is_numeric_v =
    is_int_any_v<T> || is_uint_any_v<T> || is_real_any_v<T>;

/// @brief Check if type is a supported value type for operations
template <typename T>
inline constexpr bool is_supported_value_type_v =
    is_real_v<T> || is_int_v<T> || is_uint_v<T>;

}  // namespace scl

// =============================================================================
// SECTION 7: Type Concepts (C++20)
// =============================================================================

namespace scl {

// -----------------------------------------------------------------------------
// Standard Concepts
// -----------------------------------------------------------------------------

/// @brief Concept for arithmetic types
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/// @brief Concept for floating-point types
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

/// @brief Concept for integral types
template <typename T>
concept Integral = std::is_integral_v<T>;

/// @brief Concept for signed integral types
template <typename T>
concept SignedIntegral = std::is_integral_v<T> && std::is_signed_v<T>;

/// @brief Concept for unsigned integral types
template <typename T>
concept UnsignedIntegral = std::is_integral_v<T> && std::is_unsigned_v<T>;

// -----------------------------------------------------------------------------
// SCL Type Concepts
// -----------------------------------------------------------------------------

/// @brief Concept for standard signed integer types (Int8/16/32/64)
template <typename T>
concept Integer = is_int_v<T>;

/// @brief Concept for standard unsigned integer types (UInt8/16/32/64)
template <typename T>
concept Unsigned = is_uint_v<T>;

/// @brief Concept for standard real types (Real32/64)
template <typename T>
concept Floating = is_real_v<T>;

/// @brief Concept for index types (Index32/64)
template <typename T>
concept Indexing = is_index_v<T>;

/// @brief Concept for any signed integer type (including extended)
template <typename T>
concept IntegerAny = is_int_any_v<T>;

/// @brief Concept for any unsigned integer type (including extended)
template <typename T>
concept UnsignedAny = is_uint_any_v<T>;

/// @brief Concept for any real type (including extended)
template <typename T>
concept FloatingAny = is_real_any_v<T>;

/// @brief Concept for any numeric type
template <typename T>
concept Numeric = is_numeric_v<T>;

}  // namespace scl

// =============================================================================
// SECTION 8: Value Type Category
// =============================================================================

namespace scl {

/// @brief Value type category enumeration
enum class ValueCategory {
  kReal,  ///< Floating-point types
  kInt,   ///< Signed integer types
  kUint   ///< Unsigned integer types
};

/// @brief Get value type category at compile time
/// @tparam T Value type
/// @return ValueCategory enum
template <typename T>
[[nodiscard]]
constexpr
auto value_category() noexcept -> ValueCategory {
  if constexpr (is_real_v<T>) {
    return ValueCategory::kReal;
  } else if constexpr (is_int_v<T>) {
    return ValueCategory::kInt;
  } else if constexpr (is_uint_v<T>) {
    return ValueCategory::kUint;
  } else {
    static_assert(is_supported_value_type_v<T>,
                  "Unsupported value type - must be Real, Int, or Uint");
    return ValueCategory::kReal;  // Fallback (unreachable)
  }
}

}  // namespace scl

// =============================================================================
// SECTION 9: Type Utilities
// =============================================================================

namespace scl::type {

/// @brief Get the size of a type in bytes (compile-time)
/// @tparam T Type to query
/// @return Size in bytes
template <typename T>
[[nodiscard]]
constexpr
auto size_bytes() noexcept -> std::size_t {
  return sizeof(T);
}

/// @brief Get the alignment of a type in bytes (compile-time)
/// @tparam T Type to query
/// @return Alignment in bytes
template <typename T>
[[nodiscard]]
constexpr
auto alignment_bytes() noexcept -> std::size_t {
  return alignof(T);
}

/// @brief Check if two types have compatible precision (same size)
/// @tparam T First type
/// @tparam U Second type
/// @return true if types have same size
template <typename T, typename U>
[[nodiscard]]
constexpr
auto is_compatible_precision() noexcept -> bool {
  return sizeof(T) == sizeof(U);
}

/// @brief Get the wider of two numeric types
/// @tparam T First type
/// @tparam U Second type
template <typename T, typename U>
using wider_t = std::conditional_t<(sizeof(T) >= sizeof(U)), T, U>;

}  // namespace scl::type

// =============================================================================
// SECTION 10: Precision Support Detection (SFINAE)
// =============================================================================

namespace scl {

namespace detail {

/// @brief Check if a type is scl_null_t (unsupported precision)
/// @tparam T Type to check
template <typename T>
inline constexpr bool is_null_precision_v = std::is_same_v<T, scl_null_t>;

}  // namespace detail

// -----------------------------------------------------------------------------
// Precision Support Checks
// -----------------------------------------------------------------------------

/// @brief Check if Real16 is supported on this platform
inline constexpr bool kHasReal16Support = !detail::is_null_precision_v<Real16>;

/// @brief Check if Real128 is supported on this platform
inline constexpr bool kHasReal128Support =
    !detail::is_null_precision_v<Real128>;

/// @brief Check if Int128 is supported on this platform
inline constexpr bool kHasInt128Support = !detail::is_null_precision_v<Int128>;

/// @brief Check if UInt128 is supported on this platform
inline constexpr bool kHasUInt128Support =
    !detail::is_null_precision_v<UInt128>;

/// @brief Check if a specific precision type is supported
/// @tparam P Precision type to check
/// @return true if supported, false if scl_null_t
template <typename P>
[[nodiscard]]
constexpr
auto is_precision_supported() noexcept -> bool {
  return !detail::is_null_precision_v<P>;
}

// -----------------------------------------------------------------------------
// SFINAE Utilities
// -----------------------------------------------------------------------------

/// @brief Enable if precision is supported (not scl_null_t)
/// @tparam P Precision type to check
template <typename P>
using enable_if_supported_t =
    std::enable_if_t<!detail::is_null_precision_v<P>>;

/// @brief Enable if precision is unsupported (is scl_null_t)
/// @tparam P Precision type to check
template <typename P>
using enable_if_unsupported_t =
    std::enable_if_t<detail::is_null_precision_v<P>>;

// -----------------------------------------------------------------------------
// Precision Concepts
// -----------------------------------------------------------------------------

/// @brief Concept for supported precision types
template <typename P>
concept SupportedPrecision = !detail::is_null_precision_v<P>;

/// @brief Concept for unsupported precision types
template <typename P>
concept UnsupportedPrecision = detail::is_null_precision_v<P>;

}  // namespace scl

