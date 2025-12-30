#pragma once

/// @file scl/core/bits.hpp
/// @brief Bit manipulation utilities and alignment helpers
///
/// This header provides:
/// - Count leading zeros (clz)
/// - Count trailing zeros (ctz)
/// - Population count (popcount)
/// - Power-of-2 checks and rounding
/// - Alignment utilities (align_up, align_down, is_aligned)
/// - Pointer alignment checks

#include "scl/core/macro.hpp"

#include <cstdint>
#include <cstddef>
#include <type_traits>

// MSVC intrinsics
#if SCL_COMPILER_MSVC
    #include <intrin.h>
#endif

// =============================================================================
// Bit Manipulation Utilities
// =============================================================================

namespace scl::bits {

/// @brief Count leading zeros
/// @param x Input value
/// @return Number of leading zero bits (32 if x == 0)
[[nodiscard]]
constexpr
auto clz(std::uint32_t x) noexcept -> int {
    if (x == 0) return 32;
    #if SCL_COMPILER_GCC_LIKE
        return __builtin_clz(x);
    #elif SCL_COMPILER_MSVC
        unsigned long index;
        _BitScanReverse(&index, x);
        return 31 - static_cast<int>(index);
    #else
        int count = 0;
        for (int i = 31; i >= 0; --i) {
            if (x & (1u << i)) break;
            ++count;
        }
        return count;
    #endif
}

/// @brief Count leading zeros (64-bit)
/// @param x Input value
/// @return Number of leading zero bits (64 if x == 0)
[[nodiscard]]
constexpr
auto clz(std::uint64_t x) noexcept -> int {
    if (x == 0) return 64;
    #if SCL_COMPILER_GCC_LIKE
        return __builtin_clzll(x);
    #elif SCL_COMPILER_MSVC
        unsigned long index;
        _BitScanReverse64(&index, x);
        return 63 - static_cast<int>(index);
    #else
        int count = 0;
        for (int i = 63; i >= 0; --i) {
            if (x & (1ULL << i)) break;
            ++count;
        }
        return count;
    #endif
}

/// @brief Count trailing zeros
/// @param x Input value
/// @return Number of trailing zero bits (32 if x == 0)
[[nodiscard]]
constexpr
auto ctz(std::uint32_t x) noexcept -> int {
    if (x == 0) return 32;
    #if SCL_COMPILER_GCC_LIKE
        return __builtin_ctz(x);
    #elif SCL_COMPILER_MSVC
        unsigned long index;
        _BitScanForward(&index, x);
        return static_cast<int>(index);
    #else
        int count = 0;
        while ((x & 1) == 0) { x >>= 1; ++count; }
        return count;
    #endif
}

/// @brief Count trailing zeros (64-bit)
/// @param x Input value
/// @return Number of trailing zero bits (64 if x == 0)
[[nodiscard]]
constexpr
auto ctz(std::uint64_t x) noexcept -> int {
    if (x == 0) return 64;
    #if SCL_COMPILER_GCC_LIKE
        return __builtin_ctzll(x);
    #elif SCL_COMPILER_MSVC
        unsigned long index;
        _BitScanForward64(&index, x);
        return static_cast<int>(index);
    #else
        int count = 0;
        while ((x & 1) == 0) { x >>= 1; ++count; }
        return count;
    #endif
}

/// @brief Population count (number of set bits)
/// @param x Input value
/// @return Number of 1 bits in x
[[nodiscard]]
constexpr
auto popcount(std::uint32_t x) noexcept -> int {
    #if SCL_COMPILER_GCC_LIKE
        return __builtin_popcount(x);
    #elif SCL_COMPILER_MSVC
        return static_cast<int>(__popcnt(x));
    #else
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        return ((x + (x >> 4) & 0x0F0F0F0F) * 0x01010101) >> 24;
    #endif
}

/// @brief Population count (64-bit)
/// @param x Input value
/// @return Number of 1 bits in x
[[nodiscard]]
constexpr
auto popcount(std::uint64_t x) noexcept -> int {
    #if SCL_COMPILER_GCC_LIKE
        return __builtin_popcountll(x);
    #elif SCL_COMPILER_MSVC
        return static_cast<int>(__popcnt64(x));
    #else
        x = x - ((x >> 1) & 0x5555555555555555ULL);
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        return ((x + (x >> 4) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56;
    #endif
}

/// @brief Check if value is power of 2
/// @tparam T Integral type
/// @param x Input value
/// @return true if x is a power of 2, false otherwise
template<typename T>
[[nodiscard]]
constexpr
auto is_power_of_2(T x) noexcept -> bool {
    static_assert(std::is_integral_v<T>, "Requires integral type");
    return x > 0 && (x & (x - 1)) == 0;
}

/// @brief Round up to next power of 2
/// @param x Input value
/// @return Smallest power of 2 that is >= x
[[nodiscard]]
constexpr
auto next_power_of_2(std::uint32_t x) noexcept -> std::uint32_t {
    if (x == 0) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

/// @brief Round up to next power of 2 (64-bit)
/// @param x Input value
/// @return Smallest power of 2 that is >= x
[[nodiscard]]
constexpr
auto next_power_of_2(std::uint64_t x) noexcept -> std::uint64_t {
    if (x == 0) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

/// @brief Align value up to alignment boundary
/// @tparam T Integral type
/// @param value Value to align
/// @param alignment Alignment boundary (must be power of 2)
/// @return Smallest multiple of alignment that is >= value
/// @pre alignment must be a power of 2
template<typename T>
[[nodiscard]]
constexpr
auto align_up(T value, T alignment) noexcept -> T {
    static_assert(std::is_integral_v<T>, "Requires integral type");
    return (value + alignment - 1) & ~(alignment - 1);
}

/// @brief Align value down to alignment boundary
/// @tparam T Integral type
/// @param value Value to align
/// @param alignment Alignment boundary (must be power of 2)
/// @return Largest multiple of alignment that is <= value
/// @pre alignment must be a power of 2
template<typename T>
[[nodiscard]]
constexpr
auto align_down(T value, T alignment) noexcept -> T {
    static_assert(std::is_integral_v<T>, "Requires integral type");
    return value & ~(alignment - 1);
}

/// @brief Check if value is aligned to boundary
/// @tparam T Integral type
/// @param value Value to check
/// @param alignment Alignment boundary (must be power of 2)
/// @return true if value is a multiple of alignment
/// @pre alignment must be a power of 2
template<typename T>
[[nodiscard]]
constexpr
auto is_aligned(T value, T alignment) noexcept -> bool {
    static_assert(std::is_integral_v<T>, "Requires integral type");
    return (value & (alignment - 1)) == 0;
}

/// @brief Check if pointer is aligned to boundary
/// @tparam T Pointer type
/// @param ptr Pointer to check
/// @param alignment Alignment boundary in bytes (must be power of 2)
/// @return true if pointer address is a multiple of alignment
/// @pre alignment must be a power of 2
template<typename T>
[[nodiscard]]
auto is_ptr_aligned(const T* ptr, std::size_t alignment) noexcept -> bool {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

}  // namespace scl::bits
