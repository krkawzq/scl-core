#pragma once

/**
 * @file scl/core/macro.hpp
 * @brief Cross-platform compiler abstractions and optimization hints
 *
 * This header provides MACRO-BASED utilities:
 * - Function attributes (only those without C++20 standard equivalents)
 * - Prefetch and cache hints
 * - Optimizer hints
 * - Loop optimization hints
 * - Compile-time string utilities
 *
 * @note C++20 attributes should be used directly:
 *       - [[nodiscard]], [[maybe_unused]], [[deprecated]]
 *       - [[likely]], [[unlikely]] (on branches, not expressions)
 *       - [[no_unique_address]]
 *
 * @note Memory-related macros (alignment, stack arrays) are in:
 *       @see scl/core/memory.hpp
 *
 * @note Other utilities:
 * - Error handling: @see scl/core/error.hpp
 * - Bit manipulation: @see scl/core/bits.hpp
 * - Debug utilities: @see scl/core/debug.hpp
 * - Source location: @see scl/core/source_location.hpp
 * - IO types: @see scl/core/io.hpp
 */

#include <cstddef>
#include <cstdint>

#include "scl/config.hpp"

// =============================================================================
// SECTION 1: Function Attributes (Only Non-Standard)
// =============================================================================
//
// NOTE: Do NOT use macros for these C++20 standard attributes:
//   - [[nodiscard]]        - Use directly
//   - [[maybe_unused]]     - Use directly
//   - [[deprecated("x")]]  - Use directly
//   - [[likely]]           - Use on branch: if (...) [[likely]] { }
//   - [[unlikely]]         - Use on branch: if (...) [[unlikely]] { }
//   - [[no_unique_address]] - Use directly
//
// The macros below are ONLY for attributes without C++20 equivalents.
// =============================================================================

// Force inline (no standard equivalent)
#if SCL_CONFIG_COMPILER_MSVC
  #define SCL_FORCE_INLINE __forceinline
#elif SCL_CONFIG_COMPILER_GCC_LIKE
  #define SCL_FORCE_INLINE inline __attribute__((always_inline))
#else
  #define SCL_FORCE_INLINE inline
#endif

// Prevent inlining
#if SCL_CONFIG_COMPILER_MSVC
  #define SCL_NOINLINE __declspec(noinline)
#elif SCL_CONFIG_COMPILER_GCC_LIKE
  #define SCL_NOINLINE __attribute__((noinline))
#else
  #define SCL_NOINLINE
#endif

// Pointer aliasing hint (no standard equivalent)
#if SCL_CONFIG_COMPILER_MSVC
  #define SCL_RESTRICT __restrict
#elif SCL_CONFIG_COMPILER_GCC_LIKE
  #define SCL_RESTRICT __restrict__
#else
  #define SCL_RESTRICT
#endif

// DLL export/import
#if SCL_CONFIG_PLATFORM_WINDOWS
  #define SCL_EXPORT __declspec(dllexport)
  #define SCL_IMPORT __declspec(dllimport)
#elif SCL_CONFIG_COMPILER_GCC_LIKE
  #define SCL_EXPORT __attribute__((visibility("default")))
  #define SCL_IMPORT __attribute__((visibility("default")))
#else
  #define SCL_EXPORT
  #define SCL_IMPORT
#endif

// Hot/cold function hints
#if SCL_CONFIG_COMPILER_GCC_LIKE
  #define SCL_HOT __attribute__((hot))
  #define SCL_COLD __attribute__((cold))
#else
  #define SCL_HOT
  #define SCL_COLD
#endif

// Flatten: inline all calls within this function
#if SCL_CONFIG_COMPILER_GCC_LIKE
  #define SCL_FLATTEN __attribute__((flatten))
#else
  #define SCL_FLATTEN
#endif

// Pure function hints
#if SCL_CONFIG_COMPILER_GCC_LIKE
  /// @brief Function result depends only on arguments, no side effects
  #define SCL_PURE __attribute__((pure))
  /// @brief Function result depends only on arguments, no memory access
  #define SCL_CONST_FUNC __attribute__((const))
#else
  #define SCL_PURE
  #define SCL_CONST_FUNC
#endif

// No-throw for optimizer (use noexcept in code, this is for optimizer hints)
#if SCL_CONFIG_COMPILER_GCC_LIKE
  #define SCL_NOTHROW_ATTR __attribute__((nothrow))
#elif SCL_CONFIG_COMPILER_MSVC
  #define SCL_NOTHROW_ATTR __declspec(nothrow)
#else
  #define SCL_NOTHROW_ATTR
#endif

// =============================================================================
// SECTION 2: Prefetch and Cache Hints
// =============================================================================

/// @brief Prefetch locality levels
/// 0 = NTA (non-temporal, no cache)
/// 1 = T2 (low locality, L3)
/// 2 = T1 (medium locality, L2)
/// 3 = T0 (high locality, L1)

#if SCL_CONFIG_COMPILER_GCC_LIKE
  #define SCL_PREFETCH(addr, rw, locality) \
    __builtin_prefetch((addr), (rw), (locality))
  #define SCL_PREFETCH_READ(addr, locality) \
    __builtin_prefetch((addr), 0, (locality))
  #define SCL_PREFETCH_WRITE(addr, locality) \
    __builtin_prefetch((addr), 1, (locality))
#elif SCL_CONFIG_COMPILER_MSVC
  #include <xmmintrin.h>
  #define SCL_PREFETCH(addr, rw, locality)                             \
    _mm_prefetch(reinterpret_cast<const char*>(addr),                  \
                 (locality) == 0   ? _MM_HINT_NTA                      \
                 : (locality) == 1 ? _MM_HINT_T2                       \
                 : (locality) == 2 ? _MM_HINT_T1                       \
                                   : _MM_HINT_T0)
  #define SCL_PREFETCH_READ(addr, locality) SCL_PREFETCH(addr, 0, locality)
  #define SCL_PREFETCH_WRITE(addr, locality) SCL_PREFETCH(addr, 1, locality)
#else
  #define SCL_PREFETCH(addr, rw, locality) ((void)0)
  #define SCL_PREFETCH_READ(addr, locality) ((void)0)
  #define SCL_PREFETCH_WRITE(addr, locality) ((void)0)
#endif

// Prefetch for streaming (non-temporal) loads
#if SCL_CONFIG_COMPILER_GCC_LIKE
  #define SCL_PREFETCH_NTA(addr) __builtin_prefetch((addr), 0, 0)
#elif SCL_CONFIG_COMPILER_MSVC
  #define SCL_PREFETCH_NTA(addr) \
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA)
#else
  #define SCL_PREFETCH_NTA(addr) ((void)0)
#endif

// =============================================================================
// SECTION 3: Optimizer Hints
// =============================================================================

// Assume condition is true (UB if false)
#if SCL_CONFIG_COMPILER_CLANG
  #define SCL_ASSUME(cond) __builtin_assume(cond)
#elif SCL_CONFIG_COMPILER_GCC
  #define SCL_ASSUME(cond) \
    do {                   \
      if (!(cond))         \
        __builtin_unreachable(); \
    } while (0)
#elif SCL_CONFIG_COMPILER_MSVC
  #define SCL_ASSUME(cond) __assume(cond)
#else
  #define SCL_ASSUME(cond) ((void)0)
#endif

// Mark code as unreachable
#if SCL_CONFIG_COMPILER_GCC_LIKE
  #define SCL_UNREACHABLE() __builtin_unreachable()
#elif SCL_CONFIG_COMPILER_MSVC
  #define SCL_UNREACHABLE() __assume(0)
#else
  #define SCL_UNREACHABLE() ((void)0)
#endif

// =============================================================================
// SECTION 4: Loop Optimization Hints
// =============================================================================

// Helper for pragma stringification
#define SCL_PRAGMA_IMPL(x) _Pragma(#x)
#define SCL_PRAGMA(x) SCL_PRAGMA_IMPL(x)

#if SCL_CONFIG_COMPILER_CLANG
  #define SCL_UNROLL(n) SCL_PRAGMA(clang loop unroll_count(n))
  #define SCL_UNROLL_FULL SCL_PRAGMA(clang loop unroll(full))
  #define SCL_VECTORIZE SCL_PRAGMA(clang loop vectorize(enable))
  #define SCL_NO_VECTORIZE SCL_PRAGMA(clang loop vectorize(disable))
  #define SCL_INTERLEAVE(n) SCL_PRAGMA(clang loop interleave_count(n))
#elif SCL_CONFIG_COMPILER_GCC
  #define SCL_UNROLL(n) SCL_PRAGMA(GCC unroll n)
  #define SCL_UNROLL_FULL SCL_PRAGMA(GCC unroll 16)
  #define SCL_VECTORIZE SCL_PRAGMA(GCC ivdep)
  #define SCL_NO_VECTORIZE
  #define SCL_INTERLEAVE(n)
#elif SCL_CONFIG_COMPILER_MSVC
  #define SCL_UNROLL(n)
  #define SCL_UNROLL_FULL
  #define SCL_VECTORIZE SCL_PRAGMA(loop(ivdep))
  #define SCL_NO_VECTORIZE SCL_PRAGMA(loop(no_vector))
  #define SCL_INTERLEAVE(n)
#else
  #define SCL_UNROLL(n)
  #define SCL_UNROLL_FULL
  #define SCL_VECTORIZE
  #define SCL_NO_VECTORIZE
  #define SCL_INTERLEAVE(n)
#endif

// OpenMP SIMD hint
#if defined(_OPENMP) && _OPENMP >= 201307
  #define SCL_OMP_SIMD SCL_PRAGMA(omp simd)
  #define SCL_OMP_SIMD_ALIGNED(vars, align) \
    SCL_PRAGMA(omp simd aligned(vars : align))
#else
  #define SCL_OMP_SIMD
  #define SCL_OMP_SIMD_ALIGNED(vars, align)
#endif

// =============================================================================
// SECTION 5: Compile-Time String Utilities
// =============================================================================

namespace scl {

/// @brief Get array size at compile time
/// @tparam T Element type
/// @tparam N Array size
/// @param arr Reference to C-style array
/// @return Number of elements
template <typename T, std::size_t N>
[[nodiscard]]
constexpr
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
auto array_size(const T (&arr)[N]) noexcept -> std::size_t {
  (void)arr;  // Suppress unused warning
  return N;
}

/// @brief Mark variable as unused
/// @tparam T Type of the variable
/// @param value Value to mark as unused
template <typename T>
constexpr
auto unused(const T& value) noexcept -> void {
  (void)value;
}

}  // namespace scl

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

// Stringify (must be macro for preprocessor stringification)
#define SCL_STRINGIFY_IMPL(x) #x
#define SCL_STRINGIFY(x) SCL_STRINGIFY_IMPL(x)

// Concatenate tokens (must be macro for preprocessor token pasting)
#define SCL_CONCAT_IMPL(a, b) a##b
#define SCL_CONCAT(a, b) SCL_CONCAT_IMPL(a, b)

// Array size (backward compatibility)
#define SCL_ARRAY_SIZE(arr) ::scl::array_size(arr)

// Unique identifier generation (must be macro for __LINE__)
#define SCL_UNIQUE_ID(prefix) SCL_CONCAT(prefix, __LINE__)

// Suppress unused warnings (backward compatibility)
#define SCL_UNUSED(x) ::scl::unused(x)

// NOLINTEND(cppcoreguidelines-macro-usage)
