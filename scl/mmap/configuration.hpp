#pragma once

// =============================================================================
/// @file configuration.hpp
/// @brief SCL MMAP Module Configuration
///
/// Compile-time configuration for the memory-mapped sparse matrix module.
/// All page-related constants are constexpr for optimal code generation.
///
/// Key Constants:
///
/// 1. SCL_MMAP_PAGE_SIZE: Page size in bytes (must be power of 2)
/// 2. SCL_MMAP_PAGE_SHIFT: log2(PAGE_SIZE) for bit-shift operations
/// 3. SCL_MMAP_PAGE_MASK: Mask for extracting page offset
///
/// Performance Notes:
///
/// - Page size should be tuned based on workload:
///   - Small pages (64KB): Better for random access
///   - Large pages (1MB+): Better for sequential scan
/// - Default 1MB balances both patterns for typical bioinformatics workloads
// =============================================================================

#include "scl/core/macros.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace scl::mmap {

// =============================================================================
// Compile-Time Page Configuration
// =============================================================================

/// @brief Page size in bytes (must be power of 2)
#ifndef SCL_MMAP_PAGE_SIZE
#define SCL_MMAP_PAGE_SIZE (1024 * 1024)
#endif

/// @brief Default pool size in pages
#ifndef SCL_MMAP_DEFAULT_POOL_SIZE
#define SCL_MMAP_DEFAULT_POOL_SIZE 64
#endif

// =============================================================================
// Compile-Time Bit Operations
// =============================================================================

namespace detail {

/// @brief Compile-time log2 for power-of-2 values
constexpr std::size_t constexpr_log2(std::size_t n) {
    return (n <= 1) ? 0 : 1 + constexpr_log2(n >> 1);
}

/// @brief Check if value is power of 2
constexpr bool is_power_of_2(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

} // namespace detail

// =============================================================================
// Page Constants (Constexpr for Zero-Overhead)
// =============================================================================

/// @brief Page size constant
inline constexpr std::size_t kPageSize = SCL_MMAP_PAGE_SIZE;

/// @brief Page size bit shift (log2 of page size)
inline constexpr std::size_t kPageShift = detail::constexpr_log2(kPageSize);

/// @brief Page offset mask (page_size - 1)
inline constexpr std::size_t kPageMask = kPageSize - 1;

/// @brief Default pool capacity
inline constexpr std::size_t kDefaultPoolSize = SCL_MMAP_DEFAULT_POOL_SIZE;

// Compile-time validation
static_assert(detail::is_power_of_2(kPageSize), 
    "SCL_MMAP_PAGE_SIZE must be a power of 2 for efficient bit operations");
static_assert(kPageSize >= 4096, 
    "SCL_MMAP_PAGE_SIZE should be at least 4KB for efficiency");
static_assert((1ULL << kPageShift) == kPageSize, 
    "Page shift calculation error");

// =============================================================================
// Runtime Configuration
// =============================================================================

/// @brief Mmap runtime configuration
struct MmapConfig {
    /// Maximum resident pages (determines memory footprint)
    std::size_t max_resident_pages = kDefaultPoolSize;

    /// Prefetch depth for sequential scan (pages to prefetch ahead)
    std::size_t prefetch_depth = 4;

    /// Enable writeback for dirty pages
    bool enable_writeback = false;
    
    /// Use huge pages if available (Linux only)
    bool use_huge_pages = false;

    // -------------------------------------------------------------------------
    // Factory Methods
    // -------------------------------------------------------------------------
    
    /// @brief Configuration optimized for sequential scan
    static constexpr MmapConfig sequential(std::size_t pool_size = 32) {
        return MmapConfig{pool_size, 8, false, false};
    }
    
    /// @brief Configuration optimized for random access
    static constexpr MmapConfig random_access(std::size_t pool_size = 128) {
        return MmapConfig{pool_size, 1, false, false};
    }
    
    /// @brief Configuration for large dataset streaming
    static constexpr MmapConfig streaming(std::size_t pool_size = 16) {
        return MmapConfig{pool_size, 2, false, true};
    }
};

// =============================================================================
// Utility Functions
// =============================================================================

/// @brief Calculate number of pages needed for given byte count
SCL_FORCE_INLINE constexpr std::size_t bytes_to_pages(std::size_t bytes) {
    return (bytes + kPageMask) >> kPageShift;
}

/// @brief Convert byte offset to page index (bit shift)
SCL_FORCE_INLINE constexpr std::size_t byte_to_page_idx(std::size_t byte_offset) {
    return byte_offset >> kPageShift;
}

/// @brief Convert byte offset to offset within page (bit mask)
SCL_FORCE_INLINE constexpr std::size_t byte_to_page_offset(std::size_t byte_offset) {
    return byte_offset & kPageMask;
}

/// @brief Convert page index to byte offset
SCL_FORCE_INLINE constexpr std::size_t page_to_byte_offset(std::size_t page_idx) {
    return page_idx << kPageShift;
}

} // namespace scl::mmap
