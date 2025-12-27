#pragma once

#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <thread>

// =============================================================================
// FILE: scl/mmap/configuration.hpp
// BRIEF: Enhanced Configuration with Auto-tuning Support
// =============================================================================

namespace scl::mmap {

// =============================================================================
// Compile-Time Page Configuration
// =============================================================================

#ifndef SCL_MMAP_PAGE_SIZE
#define SCL_MMAP_PAGE_SIZE (1024 * 1024)
#endif

#ifndef SCL_MMAP_DEFAULT_POOL_SIZE
#define SCL_MMAP_DEFAULT_POOL_SIZE 64
#endif

#ifndef SCL_MMAP_MAX_PAGE_SIZE
#define SCL_MMAP_MAX_PAGE_SIZE (16 * 1024 * 1024)
#endif

// =============================================================================
// Compile-Time Bit Operations
// =============================================================================

namespace detail {

constexpr std::size_t constexpr_log2(std::size_t n) noexcept {
    return (n <= 1) ? 0 : 1 + constexpr_log2(n >> 1);
}

constexpr bool is_power_of_2(std::size_t n) noexcept {
    return n > 0 && (n & (n - 1)) == 0;
}

template <typename T>
constexpr T constexpr_min(T a, T b) noexcept {
    return (a < b) ? a : b;
}

template <typename T>
constexpr T constexpr_max(T a, T b) noexcept {
    return (a > b) ? a : b;
}

} // namespace detail

// =============================================================================
// Page Constants
// =============================================================================

inline constexpr std::size_t kPageSize = SCL_MMAP_PAGE_SIZE;
inline constexpr std::size_t kPageShift = detail::constexpr_log2(kPageSize);
inline constexpr std::size_t kPageMask = kPageSize - 1;
inline constexpr std::size_t kDefaultPoolSize = SCL_MMAP_DEFAULT_POOL_SIZE;
inline constexpr std::size_t kMaxPageSize = SCL_MMAP_MAX_PAGE_SIZE;
inline constexpr std::size_t kMinPageSize = 4096;

// =============================================================================
// Compile-Time Validation
// =============================================================================

static_assert(detail::is_power_of_2(kPageSize), 
    "SCL_MMAP_PAGE_SIZE must be a power of 2");

static_assert(kPageSize >= kMinPageSize, 
    "SCL_MMAP_PAGE_SIZE should be at least 4KB");

static_assert(kPageSize <= kMaxPageSize,
    "SCL_MMAP_PAGE_SIZE exceeds maximum");

static_assert((1ULL << kPageShift) == kPageSize, 
    "Page shift calculation error");

static_assert(kDefaultPoolSize >= 4,
    "SCL_MMAP_DEFAULT_POOL_SIZE should be at least 4");

// =============================================================================
// Access Pattern Hints
// =============================================================================

enum class AccessPattern {
    Sequential,
    Random,
    Strided,
    Adaptive,
    Unknown
};

// =============================================================================
// Runtime Configuration
// =============================================================================

struct MmapConfig {
    std::size_t max_resident_pages = kDefaultPoolSize;
    std::size_t prefetch_depth = 4;
    std::size_t num_prefetch_threads = 0;
    bool enable_writeback = false;
    bool use_huge_pages = false;
    bool auto_tune = true;
    AccessPattern access_hint = AccessPattern::Adaptive;
    
    // =========================================================================
    // Factory Methods
    // =========================================================================
    
    static constexpr MmapConfig sequential(std::size_t pool_size = 32) noexcept {
        return MmapConfig{
            pool_size,        // max_resident_pages
            8,                // prefetch_depth
            0,                // num_prefetch_threads (auto)
            false,            // enable_writeback
            false,            // use_huge_pages
            true,             // auto_tune
            AccessPattern::Sequential
        };
    }
    
    static constexpr MmapConfig random_access(std::size_t pool_size = 128) noexcept {
        return MmapConfig{
            pool_size,
            1,
            0,
            false,
            false,
            true,
            AccessPattern::Random
        };
    }
    
    static constexpr MmapConfig streaming(std::size_t pool_size = 16) noexcept {
        return MmapConfig{
            pool_size,
            2,
            0,
            false,
            true,
            false,
            AccessPattern::Sequential
        };
    }
    
    static constexpr MmapConfig read_write(std::size_t pool_size = 64) noexcept {
        return MmapConfig{
            pool_size,
            4,
            0,
            true,
            false,
            true,
            AccessPattern::Adaptive
        };
    }
    
    static constexpr MmapConfig strided(std::size_t pool_size = 64, 
                                       std::size_t prefetch = 2) noexcept {
        return MmapConfig{
            pool_size,
            prefetch,
            0,
            false,
            false,
            true,
            AccessPattern::Strided
        };
    }
    
    static MmapConfig auto_detect(std::size_t estimated_pages = 0) noexcept {
        std::size_t hw_threads = std::thread::hardware_concurrency();
        if (hw_threads == 0) hw_threads = 8;
        
        std::size_t pool_size = kDefaultPoolSize;
        if (estimated_pages > 0) {
            pool_size = detail::constexpr_min(estimated_pages / 2, 
                                             static_cast<std::size_t>(256));
            pool_size = detail::constexpr_max(pool_size, 
                                             static_cast<std::size_t>(16));
        }
        
        return MmapConfig{
            pool_size,
            4,
            hw_threads / 4,
            false,
            false,
            true,
            AccessPattern::Adaptive
        };
    }
    
    // =========================================================================
    // Validation
    // =========================================================================
    
    void validate() const {
        SCL_CHECK_ARG(max_resident_pages >= 2,
            "MmapConfig: max_resident_pages must be at least 2");
        SCL_CHECK_ARG(prefetch_depth <= max_resident_pages,
            "MmapConfig: prefetch_depth cannot exceed max_resident_pages");
        SCL_CHECK_ARG(num_prefetch_threads <= 64,
            "MmapConfig: num_prefetch_threads should not exceed 64");
    }
    
    SCL_NODISCARD bool is_valid() const noexcept {
        return max_resident_pages >= 2 && 
               prefetch_depth <= max_resident_pages &&
               num_prefetch_threads <= 64;
    }
    
    // =========================================================================
    // Computed Properties
    // =========================================================================
    
    SCL_NODISCARD constexpr std::size_t memory_bytes() const noexcept {
        return max_resident_pages * kPageSize;
    }
    
    SCL_NODISCARD constexpr double memory_mb() const noexcept {
        return static_cast<double>(memory_bytes()) / (1024.0 * 1024.0);
    }
    
    SCL_NODISCARD constexpr double memory_gb() const noexcept {
        return static_cast<double>(memory_bytes()) / (1024.0 * 1024.0 * 1024.0);
    }
    
    SCL_NODISCARD std::size_t get_prefetch_threads() const noexcept {
        if (num_prefetch_threads > 0) {
            return num_prefetch_threads;
        }
        
        std::size_t hw_threads = std::thread::hardware_concurrency();
        if (hw_threads == 0) hw_threads = 8;
        
        switch (access_hint) {
            case AccessPattern::Sequential:
                return detail::constexpr_max(static_cast<std::size_t>(1), hw_threads / 4);
            case AccessPattern::Random:
                return detail::constexpr_max(static_cast<std::size_t>(2), hw_threads / 2);
            case AccessPattern::Strided:
                return detail::constexpr_max(static_cast<std::size_t>(1), hw_threads / 4);
            case AccessPattern::Adaptive:
                return detail::constexpr_max(static_cast<std::size_t>(2), hw_threads / 3);
            default:
                return detail::constexpr_max(static_cast<std::size_t>(1), hw_threads / 4);
        }
    }
    
    SCL_NODISCARD std::size_t get_adaptive_prefetch_depth() const noexcept {
        if (!auto_tune) return prefetch_depth;
        
        switch (access_hint) {
            case AccessPattern::Sequential:
                return detail::constexpr_min(prefetch_depth * 2, max_resident_pages / 2);
            case AccessPattern::Random:
                return detail::constexpr_min(prefetch_depth / 2, static_cast<std::size_t>(2));
            case AccessPattern::Strided:
                return prefetch_depth;
            case AccessPattern::Adaptive:
                return detail::constexpr_min(prefetch_depth, max_resident_pages / 4);
            default:
                return prefetch_depth;
        }
    }
};

// =============================================================================
// Address Translation Utilities
// =============================================================================

SCL_NODISCARD SCL_FORCE_INLINE constexpr std::size_t bytes_to_pages(std::size_t bytes) noexcept {
    if (bytes == 0) return 0;
    
    constexpr std::size_t kMaxBytes = (SIZE_MAX >> kPageShift) << kPageShift;
    if (bytes > kMaxBytes) {
        return SIZE_MAX >> kPageShift;
    }
    
    return (bytes + kPageMask) >> kPageShift;
}

SCL_NODISCARD SCL_FORCE_INLINE constexpr std::size_t byte_to_page_idx(std::size_t byte_offset) noexcept {
    return byte_offset >> kPageShift;
}

SCL_NODISCARD SCL_FORCE_INLINE constexpr std::size_t byte_to_page_offset(std::size_t byte_offset) noexcept {
    return byte_offset & kPageMask;
}

SCL_NODISCARD SCL_FORCE_INLINE constexpr std::size_t page_to_byte_offset(std::size_t page_idx) noexcept {
    return page_idx << kPageShift;
}

template <typename T>
SCL_NODISCARD SCL_FORCE_INLINE constexpr std::size_t elements_per_page() noexcept {
    return kPageSize / sizeof(T);
}

template <typename T>
SCL_NODISCARD SCL_FORCE_INLINE constexpr std::size_t element_to_page_idx(std::size_t element_idx) noexcept {
    constexpr std::size_t kMaxElements = SIZE_MAX / sizeof(T);
    if (element_idx > kMaxElements) {
        return SIZE_MAX >> kPageShift;
    }
    return byte_to_page_idx(element_idx * sizeof(T));
}

template <typename T>
SCL_NODISCARD SCL_FORCE_INLINE constexpr std::size_t element_to_page_offset(std::size_t element_idx) noexcept {
    constexpr std::size_t kMaxElements = SIZE_MAX / sizeof(T);
    if (element_idx > kMaxElements) {
        return 0;
    }
    return byte_to_page_offset(element_idx * sizeof(T));
}

// =============================================================================
// Performance Hint Utilities
// =============================================================================

inline const char* access_pattern_name(AccessPattern pattern) noexcept {
    switch (pattern) {
        case AccessPattern::Sequential: return "Sequential";
        case AccessPattern::Random: return "Random";
        case AccessPattern::Strided: return "Strided";
        case AccessPattern::Adaptive: return "Adaptive";
        case AccessPattern::Unknown: return "Unknown";
        default: return "Invalid";
    }
}

inline AccessPattern detect_pattern_from_stride(std::size_t stride) noexcept {
    if (stride == 1) return AccessPattern::Sequential;
    if (stride > 1 && stride < 16) return AccessPattern::Strided;
    return AccessPattern::Random;
}

} // namespace scl::mmap
