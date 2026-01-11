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
///
/// @note Pointer alignment checks are in @see scl/core/memory.hpp

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scl/config.hpp"

// MSVC intrinsics
#if SCL_CONFIG_COMPILER_MSVC
  #include <intrin.h>
#endif

// =============================================================================
// Bit Manipulation Utilities
// =============================================================================

namespace scl::bits {

// -----------------------------------------------------------------------------
// Count Leading Zeros
// -----------------------------------------------------------------------------

/// @brief Count leading zeros
/// @param[in] val Input value
/// @return Number of leading zero bits (32 if val == 0)
[[nodiscard]]
constexpr
auto clz(std::uint32_t val) noexcept -> int {
  if (val == 0) {
    return 32;
  }
#if SCL_CONFIG_COMPILER_GCC_LIKE
  return __builtin_clz(val);
#elif SCL_CONFIG_COMPILER_MSVC
  unsigned long index;
  _BitScanReverse(&index, val);
  return 31 - static_cast<int>(index);
#else
  int count = 0;
  for (int i = 31; i >= 0; --i) {
    if (val & (1U << i)) {
      break;
    }
    ++count;
  }
  return count;
#endif
}

/// @brief Count leading zeros (64-bit)
/// @param[in] val Input value
/// @return Number of leading zero bits (64 if val == 0)
[[nodiscard]]
constexpr
auto clz(std::uint64_t val) noexcept -> int {
  if (val == 0) {
    return 64;
  }
#if SCL_CONFIG_COMPILER_GCC_LIKE
  return __builtin_clzll(val);
#elif SCL_CONFIG_COMPILER_MSVC
  unsigned long index;
  _BitScanReverse64(&index, val);
  return 63 - static_cast<int>(index);
#else
  int count = 0;
  for (int i = 63; i >= 0; --i) {
    if (val & (1ULL << i)) {
      break;
    }
    ++count;
  }
  return count;
#endif
}

// -----------------------------------------------------------------------------
// Count Trailing Zeros
// -----------------------------------------------------------------------------

/// @brief Count trailing zeros
/// @param[in] val Input value
/// @return Number of trailing zero bits (32 if val == 0)
[[nodiscard]]
constexpr
auto ctz(std::uint32_t val) noexcept -> int {
  if (val == 0) {
    return 32;
  }
#if SCL_CONFIG_COMPILER_GCC_LIKE
  return __builtin_ctz(val);
#elif SCL_CONFIG_COMPILER_MSVC
  unsigned long index;
  _BitScanForward(&index, val);
  return static_cast<int>(index);
#else
  int count = 0;
  while ((val & 1) == 0) {
    val >>= 1;
    ++count;
  }
  return count;
#endif
}

/// @brief Count trailing zeros (64-bit)
/// @param[in] val Input value
/// @return Number of trailing zero bits (64 if val == 0)
[[nodiscard]]
constexpr
auto ctz(std::uint64_t val) noexcept -> int {
  if (val == 0) {
    return 64;
  }
#if SCL_CONFIG_COMPILER_GCC_LIKE
  return __builtin_ctzll(val);
#elif SCL_CONFIG_COMPILER_MSVC
  unsigned long index;
  _BitScanForward64(&index, val);
  return static_cast<int>(index);
#else
  int count = 0;
  while ((val & 1) == 0) {
    val >>= 1;
    ++count;
  }
  return count;
#endif
}

// -----------------------------------------------------------------------------
// Population Count
// -----------------------------------------------------------------------------

/// @brief Population count (number of set bits)
/// @param[in] val Input value
/// @return Number of 1 bits in val
[[nodiscard]]
constexpr
auto popcount(std::uint32_t val) noexcept -> int {
#if SCL_CONFIG_COMPILER_GCC_LIKE
  return __builtin_popcount(val);
#elif SCL_CONFIG_COMPILER_MSVC
  return static_cast<int>(__popcnt(val));
#else
  val = val - ((val >> 1) & 0x55555555U);
  val = (val & 0x33333333U) + ((val >> 2) & 0x33333333U);
  return static_cast<int>(((val + (val >> 4) & 0x0F0F0F0FU) * 0x01010101U) >> 24);
#endif
}

/// @brief Population count (64-bit)
/// @param[in] val Input value
/// @return Number of 1 bits in val
[[nodiscard]]
constexpr
auto popcount(std::uint64_t val) noexcept -> int {
#if SCL_CONFIG_COMPILER_GCC_LIKE
  return __builtin_popcountll(val);
#elif SCL_CONFIG_COMPILER_MSVC
  return static_cast<int>(__popcnt64(val));
#else
  val = val - ((val >> 1) & 0x5555555555555555ULL);
  val = (val & 0x3333333333333333ULL) + ((val >> 2) & 0x3333333333333333ULL);
  return static_cast<int>(
      ((val + (val >> 4) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#endif
}

// -----------------------------------------------------------------------------
// Power of 2 Utilities
// -----------------------------------------------------------------------------

/// @brief Check if value is power of 2
/// @tparam T Integral type
/// @param[in] val Input value
/// @return true if val is a power of 2, false otherwise
template <typename T>
[[nodiscard]]
constexpr
auto is_power_of_2(T val) noexcept -> bool {
  static_assert(std::is_integral_v<T>, "Requires integral type");
  return val > 0 && (val & (val - 1)) == 0;
}

/// @brief Round up to next power of 2
/// @param[in] val Input value
/// @return Smallest power of 2 that is >= val
[[nodiscard]]
constexpr
auto next_power_of_2(std::uint32_t val) noexcept -> std::uint32_t {
  if (val == 0) {
    return 1;
  }
  --val;
  val |= val >> 1;
  val |= val >> 2;
  val |= val >> 4;
  val |= val >> 8;
  val |= val >> 16;
  return val + 1;
}

/// @brief Round up to next power of 2 (64-bit)
/// @param[in] val Input value
/// @return Smallest power of 2 that is >= val
[[nodiscard]]
constexpr
auto next_power_of_2(std::uint64_t val) noexcept -> std::uint64_t {
  if (val == 0) {
    return 1;
  }
  --val;
  val |= val >> 1;
  val |= val >> 2;
  val |= val >> 4;
  val |= val >> 8;
  val |= val >> 16;
  val |= val >> 32;
  return val + 1;
}

// -----------------------------------------------------------------------------
// Alignment Utilities
// -----------------------------------------------------------------------------

/// @brief Align value up to alignment boundary
/// @tparam T Integral type
/// @param[in] value Value to align
/// @param[in] alignment Alignment boundary (must be power of 2)
/// @return Smallest multiple of alignment that is >= value
/// @pre alignment must be a power of 2
template <typename T>
[[nodiscard]]
constexpr
auto align_up(T value, T alignment) noexcept -> T {
  static_assert(std::is_integral_v<T>, "Requires integral type");
  return (value + alignment - 1) & ~(alignment - 1);
}

/// @brief Align value down to alignment boundary
/// @tparam T Integral type
/// @param[in] value Value to align
/// @param[in] alignment Alignment boundary (must be power of 2)
/// @return Largest multiple of alignment that is <= value
/// @pre alignment must be a power of 2
template <typename T>
[[nodiscard]]
constexpr
auto align_down(T value, T alignment) noexcept -> T {
  static_assert(std::is_integral_v<T>, "Requires integral type");
  return value & ~(alignment - 1);
}

/// @brief Check if value is aligned to boundary
/// @tparam T Integral type
/// @param[in] value Value to check
/// @param[in] alignment Alignment boundary (must be power of 2)
/// @return true if value is a multiple of alignment
/// @pre alignment must be a power of 2
template <typename T>
[[nodiscard]]
constexpr
auto is_aligned(T value, T alignment) noexcept -> bool {
  static_assert(std::is_integral_v<T>, "Requires integral type");
  return (value & (alignment - 1)) == 0;
}

// -----------------------------------------------------------------------------
// SIMD Alignment Partitioning
// -----------------------------------------------------------------------------

/// @brief Result of alignment partitioning for SIMD processing
struct AlignmentPartition {
  std::size_t head;    ///< Elements before first aligned block
  std::size_t body;    ///< Elements in aligned blocks
  std::size_t tail;    ///< Elements after last aligned block
  std::size_t blocks;  ///< Number of aligned blocks
};

/// @brief Compute alignment partition for SIMD processing
/// @param[in] base_ptr Pointer to array start (as uintptr_t)
/// @param[in] count Number of elements
/// @param[in] lane_count SIMD lane count (must be power of 2)
/// @param[in] elem_size Size of each element in bytes
/// @return AlignmentPartition with {head, body, tail, blocks}
/// @note Invariant: head + body + tail == count
/// @note Invariant: body == blocks * lane_count
[[nodiscard]]
constexpr
auto partition_for_alignment(std::uintptr_t base_ptr,
                             std::size_t count,
                             std::size_t lane_count,
                             std::size_t elem_size) noexcept -> AlignmentPartition {
  // Invalid lane_count (must be power of 2)
  if (lane_count == 0 || (lane_count & (lane_count - 1)) != 0) {
    return {count, 0, 0, 0};
  }

  const std::size_t align_bytes = lane_count * elem_size;
  const std::size_t misalign = base_ptr % align_bytes;
  const std::size_t head_bytes = (misalign != 0) ? (align_bytes - misalign) : 0;
  const std::size_t head = head_bytes / elem_size;

  // Array too small for aligned block
  if (head >= count) {
    return {count, 0, 0, 0};
  }

  const std::size_t remaining = count - head;
  const std::size_t blocks = remaining / lane_count;
  const std::size_t body = blocks * lane_count;
  const std::size_t tail = remaining - body;

  return {head, body, tail, blocks};
}

/// @brief Compute alignment partition for typed pointers
/// @tparam T Element type
/// @param[in] ptr Pointer to array start
/// @param[in] count Number of elements
/// @param[in] lane_count SIMD lane count (must be power of 2)
/// @return AlignmentPartition with {head, body, tail, blocks}
template <typename T>
[[nodiscard]]
constexpr
auto partition_for_alignment(const T* ptr,
                             std::size_t count,
                             std::size_t lane_count) noexcept -> AlignmentPartition {
  return partition_for_alignment(
      reinterpret_cast<std::uintptr_t>(ptr),
      count,
      lane_count,
      sizeof(T));
}

/// @brief Check if pointer is aligned for given SIMD lane count
/// @tparam T Element type
/// @param[in] ptr Pointer to check
/// @param[in] lane_count SIMD lane count (must be power of 2)
/// @return true if ptr is aligned for lane_count elements
template <typename T>
[[nodiscard]]
constexpr
auto is_simd_aligned(const T* ptr, std::size_t lane_count) noexcept -> bool {
  return (reinterpret_cast<std::uintptr_t>(ptr) % (lane_count * sizeof(T))) == 0;
}

}  // namespace scl::bits

