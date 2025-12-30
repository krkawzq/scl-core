#pragma once

/// @file scl/core/type.hpp
/// @brief Core type system for SCL - C-style types and precision configuration
///
/// This header provides:
/// - C-style type definitions (scl_float32_t, scl_int64_t, etc.)
/// - Platform-specific extended type detection (128-bit support)
/// - SCL type aliases (Real16/32/64/128, Index32/64/128)
/// - Default type configuration (Index, Real)
/// - Type traits and concepts for SCL types
///
/// @note This is a foundational header with minimal dependencies.
///       The library uses dynamic dispatch - Real and Index are just default types.

#include "scl/core/macros.hpp"

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <concepts>

// =============================================================================
// SECTION 1: Platform Detection for Extended Types
// =============================================================================

namespace scl::detail {

// -----------------------------------------------------------------------------
// 128-bit Integer Support Detection
// -----------------------------------------------------------------------------

#if defined(__SIZEOF_INT128__) && !defined(SCL_NO_INT128)
    #define SCL_HAS_INT128 1
    using int128_native_t = __int128;
    using uint128_native_t = unsigned __int128;
#else
    #define SCL_HAS_INT128 0
#endif

// -----------------------------------------------------------------------------
// Float16 Support Detection (IEEE 754 binary16)
// -----------------------------------------------------------------------------

#if defined(__FLT16_MANT_DIG__) && !defined(SCL_NO_FLOAT16)
    #define SCL_HAS_FLOAT16 1
    using float16_native_t = _Float16;
#elif defined(__ARM_FP16_FORMAT_IEEE) && !defined(SCL_NO_FLOAT16)
    #define SCL_HAS_FLOAT16 1
    using float16_native_t = __fp16;
#else
    #define SCL_HAS_FLOAT16 0
#endif

// -----------------------------------------------------------------------------
// Float128 Support Detection (IEEE 754 binary128 / quadruple precision)
// -----------------------------------------------------------------------------

#if defined(__SIZEOF_FLOAT128__) && !defined(SCL_NO_FLOAT128)
    #define SCL_HAS_FLOAT128 1
    #if defined(__GNUC__) && !defined(__clang__)
        using float128_native_t = __float128;
    #elif defined(__clang__)
        using float128_native_t = __float128;
    #else
        using float128_native_t = long double;  // Fallback
    #endif
#else
    #define SCL_HAS_FLOAT128 0
#endif

// -----------------------------------------------------------------------------
// Extended Type Support Configuration
// -----------------------------------------------------------------------------

/// @brief Enable extended types if platform supports them and user requests them
#if defined(SCL_TYPE_EXTENDED)
    #define SCL_ENABLE_INT128   (SCL_HAS_INT128)
    #define SCL_ENABLE_FLOAT16  (SCL_HAS_FLOAT16)
    #define SCL_ENABLE_FLOAT128 (SCL_HAS_FLOAT128)
#else
    #define SCL_ENABLE_INT128   0
    #define SCL_ENABLE_FLOAT16  0
    #define SCL_ENABLE_FLOAT128 0
#endif

}  // namespace scl::detail

// =============================================================================
// SECTION 2: C-Style Type Definitions (scl_xxx_t)
// =============================================================================

namespace scl {

// -----------------------------------------------------------------------------
// Integer Types (Standard)
// -----------------------------------------------------------------------------

/// @brief 8-bit signed integer (C-style)
using scl_int8_t = std::int8_t;

/// @brief 16-bit signed integer (C-style)
using scl_int16_t = std::int16_t;

/// @brief 32-bit signed integer (C-style)
using scl_int32_t = std::int32_t;

/// @brief 64-bit signed integer (C-style)
using scl_int64_t = std::int64_t;

/// @brief 8-bit unsigned integer (C-style)
using scl_uint8_t = std::uint8_t;

/// @brief 16-bit unsigned integer (C-style)
using scl_uint16_t = std::uint16_t;

/// @brief 32-bit unsigned integer (C-style)
using scl_uint32_t = std::uint32_t;

/// @brief 64-bit unsigned integer (C-style)
using scl_uint64_t = std::uint64_t;

// -----------------------------------------------------------------------------
// Extended Integer Types (128-bit)
// -----------------------------------------------------------------------------

#if SCL_ENABLE_INT128
    /// @brief 128-bit signed integer (C-style, platform-dependent)
    using scl_int128_t = detail::int128_native_t;
    
    /// @brief 128-bit unsigned integer (C-style, platform-dependent)
    using scl_uint128_t = detail::uint128_native_t;
#endif

// -----------------------------------------------------------------------------
// Floating-Point Types (Standard)
// -----------------------------------------------------------------------------

/// @brief 32-bit floating-point (C-style, IEEE 754 binary32)
using scl_float32_t = float;

/// @brief 64-bit floating-point (C-style, IEEE 754 binary64)
using scl_float64_t = double;

// -----------------------------------------------------------------------------
// Extended Floating-Point Types
// -----------------------------------------------------------------------------

#if SCL_ENABLE_FLOAT16
    /// @brief 16-bit floating-point (C-style, IEEE 754 binary16, platform-dependent)
    using scl_float16_t = detail::float16_native_t;
#endif

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

}  // namespace scl

// =============================================================================
// SECTION 3: SCL Type Aliases (C++ Style)
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
#endif

// -----------------------------------------------------------------------------
// Floating-Point Types
// -----------------------------------------------------------------------------

#if SCL_ENABLE_FLOAT16
    /// @brief 16-bit floating-point (half precision, IEEE 754 binary16)
    using Real16 = scl_float16_t;
#endif

/// @brief 32-bit floating-point (single precision, IEEE 754 binary32)
using Real32 = scl_float32_t;

/// @brief 64-bit floating-point (double precision, IEEE 754 binary64)
using Real64 = scl_float64_t;

#if SCL_ENABLE_FLOAT128
    /// @brief 128-bit floating-point (quadruple precision, IEEE 754 binary128)
    using Real128 = scl_float128_t;
#endif

// -----------------------------------------------------------------------------
// Index Types (for array indexing)
// -----------------------------------------------------------------------------

/// @brief 32-bit signed index type
using Index32 = scl_int32_t;

/// @brief 64-bit signed index type
using Index64 = scl_int64_t;

#if SCL_ENABLE_INT128
    /// @brief 128-bit signed index type (extended precision)
    using Index128 = scl_int128_t;
#endif

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

}  // namespace scl

// =============================================================================
// SECTION 4: Default Type Configuration
// =============================================================================

namespace scl {

// -----------------------------------------------------------------------------
// Default Precision Selection
// -----------------------------------------------------------------------------

/// @brief Default index type configuration
/// @note Can be overridden with -DSCL_DEFAULT_INDEX32 or -DSCL_DEFAULT_INDEX64
#if defined(SCL_DEFAULT_INDEX64)
    /// @brief Default index type (64-bit)
    using Index = Index64;
    inline constexpr int default_index_bits = 64;
#elif defined(SCL_DEFAULT_INDEX32)
    /// @brief Default index type (32-bit)
    using Index = Index32;
    inline constexpr int default_index_bits = 32;
#else
    /// @brief Default index type (32-bit by default)
    using Index = Index32;
    inline constexpr int default_index_bits = 32;
#endif


/// @brief Default floating-point type configuration
/// @note Can be overridden with -DSCL_DEFAULT_REAL32 or -DSCL_DEFAULT_REAL64
#if defined(SCL_DEFAULT_REAL32)
    /// @brief Default floating-point type (32-bit)
    using Real = Real32;
    inline constexpr int default_real_bits = 32;
#elif defined(SCL_DEFAULT_REAL64)
    /// @brief Default floating-point type (64-bit)
    using Real = Real64;
    inline constexpr int default_real_bits = 64;
#else
    /// @brief Default floating-point type (64-bit by default)
    using Real = Real64;
    inline constexpr int default_real_bits = 64;
#endif

// -----------------------------------------------------------------------------
// Legacy Aliases (for compatibility)
// -----------------------------------------------------------------------------

/// @brief Single-precision floating-point (legacy alias)
using Float32 = Real32;

/// @brief Double-precision floating-point (legacy alias)
using Float64 = Real64;

}  // namespace scl

// =============================================================================
// SECTION 5: Type Information (Compile-Time)
// =============================================================================

namespace scl {

/// @brief Compile-time type size information
struct TypeInfo {
    // Default type info
    static constexpr const char* real_type_name = 
        (default_real_bits == 32) ? "real32" : "real64";
    static constexpr const char* index_type_name = 
        (default_index_bits == 32) ? "index32" : "index64";
    
    static constexpr int real_size = sizeof(Real);
    static constexpr int index_size = sizeof(Index);
    
    // Extended type availability
    static constexpr bool has_float16 = SCL_ENABLE_FLOAT16;
    static constexpr bool has_float128 = SCL_ENABLE_FLOAT128;
    static constexpr bool has_int128 = SCL_ENABLE_INT128;
    
    // Standard type sizes
    static constexpr int int8_size = sizeof(Int8);
    static constexpr int int16_size = sizeof(Int16);
    static constexpr int int32_size = sizeof(Int32);
    static constexpr int int64_size = sizeof(Int64);
    
    static constexpr int real32_size = sizeof(Real32);
    static constexpr int real64_size = sizeof(Real64);
    
#if SCL_ENABLE_FLOAT16
    static constexpr int real16_size = sizeof(Real16);
#endif
    
#if SCL_ENABLE_FLOAT128
    static constexpr int real128_size = sizeof(Real128);
#endif
    
#if SCL_ENABLE_INT128
    static constexpr int int128_size = sizeof(Int128);
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

/// @brief Check if type is a standard integer type
template<typename T>
inline constexpr bool is_int_v = 
    std::is_same_v<T, Int8> || 
    std::is_same_v<T, Int16> || 
    std::is_same_v<T, Int32> || 
    std::is_same_v<T, Int64>;

/// @brief Check if type is a standard unsigned integer type
template<typename T>
inline constexpr bool is_uint_v = 
    std::is_same_v<T, UInt8> || 
    std::is_same_v<T, UInt16> || 
    std::is_same_v<T, UInt32> || 
    std::is_same_v<T, UInt64>;

/// @brief Check if type is a standard real type
template<typename T>
inline constexpr bool is_real_v = 
    std::is_same_v<T, Real32> || 
    std::is_same_v<T, Real64>;

/// @brief Check if type is an index type
template<typename T>
inline constexpr bool is_index_v = 
    std::is_same_v<T, Index32> || 
    std::is_same_v<T, Index64>;

// -----------------------------------------------------------------------------
// Extended Type Checks
// -----------------------------------------------------------------------------

#if SCL_ENABLE_INT128
/// @brief Check if type is an extended integer type (128-bit)
template<typename T>
inline constexpr bool is_int_extended_v = std::is_same_v<T, Int128>;
#else
template<typename T>
inline constexpr bool is_int_extended_v = false;
#endif

#if SCL_ENABLE_FLOAT16 || SCL_ENABLE_FLOAT128
/// @brief Check if type is an extended real type (16-bit or 128-bit)
template<typename T>
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
template<typename T>
inline constexpr bool is_real_extended_v = false;
#endif

// -----------------------------------------------------------------------------
// Combined Type Checks
// -----------------------------------------------------------------------------

/// @brief Check if type is any integer type (including extended)
template<typename T>
inline constexpr bool is_int_any_v = 
    is_int_v<T> || is_int_extended_v<T>;

/// @brief Check if type is any real type (including extended)
template<typename T>
inline constexpr bool is_real_any_v = 
    is_real_v<T> || is_real_extended_v<T>;

/// @brief Check if type is any numeric type
template<typename T>
inline constexpr bool is_numeric_v = 
    is_int_any_v<T> || is_uint_v<T> || is_real_any_v<T>;

}  // namespace scl

// =============================================================================
// SECTION 7: Type Concepts (C++20)
// =============================================================================

namespace scl {

// -----------------------------------------------------------------------------
// Standard Concepts
// -----------------------------------------------------------------------------

/// @brief Concept for arithmetic types
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/// @brief Concept for floating-point types
template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

/// @brief Concept for integral types
template<typename T>
concept Integral = std::is_integral_v<T>;

/// @brief Concept for signed integral types
template<typename T>
concept SignedIntegral = std::is_integral_v<T> && std::is_signed_v<T>;

/// @brief Concept for unsigned integral types
template<typename T>
concept UnsignedIntegral = std::is_integral_v<T> && std::is_unsigned_v<T>;

// -----------------------------------------------------------------------------
// SCL Type Concepts
// -----------------------------------------------------------------------------

/// @brief Concept for standard integer types
template<typename T>
concept IntLike = is_int_v<T>;

/// @brief Concept for standard unsigned integer types
template<typename T>
concept UIntLike = is_uint_v<T>;

/// @brief Concept for standard real types
template<typename T>
concept RealLike = is_real_v<T>;

/// @brief Concept for index types
template<typename T>
concept IndexLike = is_index_v<T>;

/// @brief Concept for any integer type (including extended)
template<typename T>
concept IntAny = is_int_any_v<T>;

/// @brief Concept for any real type (including extended)
template<typename T>
concept RealAny = is_real_any_v<T>;

/// @brief Concept for any numeric type
template<typename T>
concept Numeric = is_numeric_v<T>;

}  // namespace scl

// =============================================================================
// SECTION 8: Type Utilities
// =============================================================================

namespace scl {

/// @brief Get the size of a type in bytes (compile-time)
/// @tparam T Type to query
/// @return Size in bytes
template<typename T>
[[nodiscard]] constexpr 
auto type_size_bytes() noexcept -> Size {
    return sizeof(T);
}

/// @brief Get the alignment of a type in bytes (compile-time)
/// @tparam T Type to query
/// @return Alignment in bytes
template<typename T>
[[nodiscard]] constexpr 
auto type_alignment_bytes() noexcept -> Size {
    return alignof(T);
}

/// @brief Check if two types have compatible precision
/// @tparam T First type
/// @tparam U Second type
/// @return true if types have same size, false otherwise
template<typename T, typename U>
[[nodiscard]] constexpr
auto is_compatible_precision() noexcept -> bool {
    return sizeof(T) == sizeof(U);
}

/// @brief Get the wider of two numeric types
/// @tparam T First type
/// @tparam U Second type
template<typename T, typename U>
using wider_type_t = std::conditional_t<(sizeof(T) >= sizeof(U)), T, U>;

}  // namespace scl
