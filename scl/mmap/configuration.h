// =============================================================================
// FILE: scl/mmap/configuration.h
// BRIEF: API reference for mmap module configuration and address translation
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>

namespace scl::mmap {

/* =============================================================================
 * COMPILE-TIME CONSTANTS
 * =============================================================================
 * SUMMARY:
 *     Page size and derived constants (all constexpr for zero-overhead).
 *
 * kPageSize:  Page size in bytes (default: 1MB, must be power of 2)
 * kPageShift: log2(kPageSize) for fast division via bit shift
 * kPageMask:  kPageSize - 1 for fast modulo via bit mask
 *
 * CONFIGURATION:
 *     Override at compile time:
 *         -DSCL_MMAP_PAGE_SIZE=262144  (256KB pages)
 *         -DSCL_MMAP_PAGE_SIZE=4194304 (4MB pages)
 *
 * PAGE SIZE SELECTION GUIDE:
 *     64KB:   Best for random access (many small reads)
 *     256KB:  Balanced for mixed workloads
 *     1MB:    Default, best for sequential scan (bioinformatics)
 *     4MB+:   Best for streaming large datasets
 *
 * DERIVED CONSTANTS:
 *     kDefaultPoolSize: Default cache capacity in pages (64)
 *     kMaxPageSize:     Maximum supported page size (16MB)
 *     kMinPageSize:     Minimum supported page size (4KB)
 *
 * COMPILE-TIME VALIDATION:
 *     - kPageSize must be power of 2
 *     - kPageSize >= 4KB
 *     - kPageSize <= 16MB
 *     - kPageShift correctly computed
 * -------------------------------------------------------------------------- */
inline constexpr std::size_t kPageSize;
inline constexpr std::size_t kPageShift;
inline constexpr std::size_t kPageMask;
inline constexpr std::size_t kDefaultPoolSize;
inline constexpr std::size_t kMaxPageSize;
inline constexpr std::size_t kMinPageSize;

/* =============================================================================
 * ENUM: AccessPattern
 * =============================================================================
 * SUMMARY:
 *     Access pattern hints for auto-tuning prefetch behavior.
 *
 * VALUES:
 *     Sequential: Linear scan (high prefetch depth)
 *     Random:     Sparse lookups (minimal prefetch)
 *     Strided:    Fixed stride access (moderate prefetch)
 *     Adaptive:   Unknown pattern (auto-detect)
 *     Unknown:    No hint available
 *
 * USAGE:
 *     MmapConfig config = MmapConfig::sequential();
 *     config.access_hint = AccessPattern::Sequential;
 *
 * AUTO-TUNING:
 *     Configuration adjusts prefetch_depth and num_threads based on hint:
 *     - Sequential: depth *= 2, threads = cores / 4
 *     - Random: depth /= 2, threads = cores / 2
 *     - Adaptive: Default tuning
 * -------------------------------------------------------------------------- */
enum class AccessPattern {
    Sequential,
    Random,
    Strided,
    Adaptive,
    Unknown
};

/* =============================================================================
 * STRUCT: MmapConfig
 * =============================================================================
 * SUMMARY:
 *     Runtime configuration for memory-mapped arrays.
 *
 * DESIGN PURPOSE:
 *     Centralized configuration for cache behavior:
 *     - Memory limits: max_resident_pages
 *     - Prefetch tuning: depth, threads, pattern hints
 *     - Feature flags: writeback, huge pages, auto-tuning
 *
 * MEMORY USAGE:
 *     Total memory = max_resident_pages * kPageSize
 *     Default: 64 pages * 1MB = 64MB resident memory
 *
 * THREAD SAFETY:
 *     Config objects are not thread-safe. Create before sharing CacheManager.
 * -------------------------------------------------------------------------- */
struct MmapConfig {
    std::size_t max_resident_pages;   // Cache capacity in pages
    std::size_t prefetch_depth;       // Pages to prefetch ahead
    std::size_t num_prefetch_threads; // Worker threads (0 = auto)
    bool enable_writeback;            // Enable dirty page writeback
    bool use_huge_pages;              // Use huge pages if available
    bool auto_tune;                   // Auto-adjust based on pattern
    AccessPattern access_hint;        // Access pattern hint

    /* -------------------------------------------------------------------------
     * FACTORY: sequential
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Configuration optimized for sequential scan.
     *
     * PARAMETERS:
     *     pool_size [in] - Cache capacity in pages (default: 32)
     *
     * TUNING:
     *     - High prefetch depth (8 pages)
     *     - Moderate pool size
     *     - Access hint: Sequential
     *
     * BEST FOR:
     *     Row-by-row iteration, file reading, streaming
     * ---------------------------------------------------------------------- */
    static constexpr MmapConfig sequential(std::size_t pool_size = 32) noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: random_access
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Configuration optimized for random access.
     *
     * PARAMETERS:
     *     pool_size [in] - Cache capacity in pages (default: 128)
     *
     * TUNING:
     *     - Large pool (hold more unique pages)
     *     - Minimal prefetch (depth = 1)
     *     - Access hint: Random
     *
     * BEST FOR:
     *     Sparse lookups, graph traversal, hash tables
     * ---------------------------------------------------------------------- */
    static constexpr MmapConfig random_access(std::size_t pool_size = 128) noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: streaming
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Configuration for datasets larger than RAM.
     *
     * PARAMETERS:
     *     pool_size [in] - Cache capacity in pages (default: 16)
     *
     * TUNING:
     *     - Small pool (limited memory)
     *     - Moderate prefetch (depth = 2)
     *     - Huge pages enabled (reduce TLB misses)
     *
     * BEST FOR:
     *     Datasets > available RAM, one-pass algorithms
     * ---------------------------------------------------------------------- */
    static constexpr MmapConfig streaming(std::size_t pool_size = 16) noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: read_write
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Configuration for read-write workloads.
     *
     * PARAMETERS:
     *     pool_size [in] - Cache capacity in pages (default: 64)
     *
     * TUNING:
     *     - Writeback enabled
     *     - Balanced pool and prefetch
     *
     * BEST FOR:
     *     Modifying data with persistence
     * ---------------------------------------------------------------------- */
    static constexpr MmapConfig read_write(std::size_t pool_size = 64) noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: strided
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Configuration for strided access patterns.
     *
     * PARAMETERS:
     *     pool_size [in] - Cache capacity (default: 64)
     *     prefetch  [in] - Prefetch depth (default: 2)
     *
     * BEST FOR:
     *     Column-major access, interleaved data
     * ---------------------------------------------------------------------- */
    static constexpr MmapConfig strided(
        std::size_t pool_size = 64,
        std::size_t prefetch = 2
    ) noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: auto_detect
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Auto-detect optimal configuration based on dataset size and hardware.
     *
     * ALGORITHM:
     *     1. Get hardware_concurrency() for thread count
     *     2. If estimated_pages provided:
     *        pool_size = min(estimated_pages / 2, 256)
     *        pool_size = max(pool_size, 16)
     *     3. Otherwise: Use default (64)
     *     4. num_threads = cores / 4
     *     5. access_hint = Adaptive
     *
     * PARAMETERS:
     *     estimated_pages [in] - Approximate total pages (0 = unknown)
     *
     * POSTCONDITIONS:
     *     Returns config tuned to hardware and dataset size.
     *
     * RECOMMENDED:
     *     Use this for unknown workloads, good default.
     * ---------------------------------------------------------------------- */
    static MmapConfig auto_detect(std::size_t estimated_pages = 0) noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: validate
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Validate configuration parameters.
     *
     * CHECKS:
     *     - max_resident_pages >= 2
     *     - prefetch_depth <= max_resident_pages
     *     - num_prefetch_threads <= 64
     *
     * THROWS:
     *     ValueError if validation fails.
     * ---------------------------------------------------------------------- */
    void validate() const;

    [[nodiscard]] bool is_valid() const noexcept;

    /* -------------------------------------------------------------------------
     * METHODS: Computed Properties
     * -------------------------------------------------------------------------
     * memory_bytes: Total memory footprint in bytes
     * memory_mb:    Memory in megabytes
     * memory_gb:    Memory in gigabytes
     *
     * FORMULA:
     *     memory_bytes = max_resident_pages * kPageSize
     *
     * COMPLEXITY:
     *     O(1) constexpr
     * ---------------------------------------------------------------------- */
    [[nodiscard]] constexpr std::size_t memory_bytes() const noexcept;
    [[nodiscard]] constexpr double memory_mb() const noexcept;
    [[nodiscard]] constexpr double memory_gb() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: get_prefetch_threads
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get effective prefetch thread count (with auto-sizing).
     *
     * ALGORITHM:
     *     If num_prefetch_threads > 0:
     *         Return user-specified value
     *     Otherwise (auto-sizing):
     *         Based on access_hint:
     *         - Sequential: max(1, cores / 4)
     *         - Random:     max(2, cores / 2)
     *         - Strided:    max(1, cores / 4)
     *         - Adaptive:   max(2, cores / 3)
     *
     * POSTCONDITIONS:
     *     Returns thread count >= 1.
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::size_t get_prefetch_threads() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: get_adaptive_prefetch_depth
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get effective prefetch depth (with auto-tuning).
     *
     * ALGORITHM:
     *     If auto_tune == false:
     *         Return prefetch_depth unchanged
     *     Otherwise:
     *         Based on access_hint:
     *         - Sequential: min(depth * 2, max_resident / 2)
     *         - Random:     min(depth / 2, 2)
     *         - Strided:    depth (unchanged)
     *         - Adaptive:   min(depth, max_resident / 4)
     *
     * POSTCONDITIONS:
     *     Returns adjusted depth <= max_resident_pages.
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::size_t get_adaptive_prefetch_depth() const noexcept;
};

/* =============================================================================
 * ADDRESS TRANSLATION FUNCTIONS
 * =============================================================================
 * SUMMARY:
 *     Fast address translation using bit operations (no division/modulo).
 *
 * PERFORMANCE:
 *     All functions are constexpr and inline:
 *     - Compiler optimizes to single instruction (shift/mask)
 *     - Zero function call overhead
 *     - Approximately 10x faster than division/modulo
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: bytes_to_pages
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute number of pages needed for byte count (ceiling division).
 *
 * ALGORITHM:
 *     pages = (bytes + kPageMask) >> kPageShift
 *     Equivalent to: ceiling(bytes / kPageSize)
 *
 * PRECONDITIONS:
 *     None (handles all inputs safely)
 *
 * POSTCONDITIONS:
 *     Returns 0 if bytes == 0.
 *     Returns max representable pages if bytes > kMaxBytes.
 *
 * COMPLEXITY:
 *     Time: O(1) constexpr
 * -------------------------------------------------------------------------- */
[[nodiscard]] constexpr std::size_t bytes_to_pages(std::size_t bytes) noexcept;

/* -----------------------------------------------------------------------------
 * FUNCTION: byte_to_page_idx
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert byte offset to page index (fast division).
 *
 * ALGORITHM:
 *     page_idx = byte_offset >> kPageShift
 *     Equivalent to: byte_offset / kPageSize
 *
 * COMPLEXITY:
 *     Time: O(1) constexpr (single shift instruction)
 * -------------------------------------------------------------------------- */
[[nodiscard]] constexpr std::size_t byte_to_page_idx(std::size_t byte_offset) noexcept;

/* -----------------------------------------------------------------------------
 * FUNCTION: byte_to_page_offset
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert byte offset to offset within page (fast modulo).
 *
 * ALGORITHM:
 *     page_offset = byte_offset & kPageMask
 *     Equivalent to: byte_offset % kPageSize
 *
 * POSTCONDITIONS:
 *     Returns value in range [0, kPageSize-1].
 *
 * COMPLEXITY:
 *     Time: O(1) constexpr (single AND instruction)
 * -------------------------------------------------------------------------- */
[[nodiscard]] constexpr std::size_t byte_to_page_offset(std::size_t byte_offset) noexcept;

/* -----------------------------------------------------------------------------
 * FUNCTION: page_to_byte_offset
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert page index to byte offset of page start.
 *
 * ALGORITHM:
 *     byte_offset = page_idx << kPageShift
 *     Equivalent to: page_idx * kPageSize
 *
 * COMPLEXITY:
 *     Time: O(1) constexpr
 * -------------------------------------------------------------------------- */
[[nodiscard]] constexpr std::size_t page_to_byte_offset(std::size_t page_idx) noexcept;

/* -----------------------------------------------------------------------------
 * FUNCTION: elements_per_page<T>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute number of T elements that fit in one page.
 *
 * ALGORITHM:
 *     elements = kPageSize / sizeof(T)
 *
 * PRECONDITIONS:
 *     kPageSize must be divisible by sizeof(T) (checked at compile time).
 *
 * POSTCONDITIONS:
 *     Returns number of complete elements per page.
 *
 * COMPLEXITY:
 *     Time: O(1) constexpr
 * -------------------------------------------------------------------------- */
template <typename T>
[[nodiscard]] constexpr std::size_t elements_per_page() noexcept;

/* -----------------------------------------------------------------------------
 * FUNCTION: element_to_page_idx<T>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert element index to page index.
 *
 * ALGORITHM:
 *     byte_offset = element_idx * sizeof(T)
 *     page_idx = byte_offset >> kPageShift
 *
 * PRECONDITIONS:
 *     None (overflow-safe)
 *
 * POSTCONDITIONS:
 *     If element_idx > SIZE_MAX / sizeof(T):
 *         Returns SIZE_MAX >> kPageShift (safe maximum)
 *     Otherwise:
 *         Returns correct page index
 *
 * COMPLEXITY:
 *     Time: O(1) constexpr
 *
 * OVERFLOW SAFETY:
 *     Checked before multiplication, returns safe maximum on overflow.
 * -------------------------------------------------------------------------- */
template <typename T>
[[nodiscard]] constexpr std::size_t element_to_page_idx(std::size_t element_idx) noexcept;

/* -----------------------------------------------------------------------------
 * FUNCTION: element_to_page_offset<T>
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert element index to byte offset within page.
 *
 * ALGORITHM:
 *     byte_offset = element_idx * sizeof(T)
 *     page_offset = byte_offset & kPageMask
 *
 * POSTCONDITIONS:
 *     Returns offset in range [0, kPageSize-1].
 *     Returns 0 on overflow (safe fallback).
 *
 * COMPLEXITY:
 *     Time: O(1) constexpr
 * -------------------------------------------------------------------------- */
template <typename T>
[[nodiscard]] constexpr std::size_t element_to_page_offset(std::size_t element_idx) noexcept;

/* =============================================================================
 * UTILITY FUNCTIONS
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: access_pattern_name
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert AccessPattern enum to human-readable string.
 *
 * POSTCONDITIONS:
 *     Returns string literal (never nullptr).
 *
 * USAGE:
 *     printf("Pattern: %s\n", access_pattern_name(config.access_hint));
 * -------------------------------------------------------------------------- */
const char* access_pattern_name(AccessPattern pattern) noexcept;

/* -----------------------------------------------------------------------------
 * FUNCTION: detect_pattern_from_stride
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Heuristic to detect access pattern from stride value.
 *
 * ALGORITHM:
 *     If stride == 1: Sequential
 *     Else if 1 < stride < 16: Strided
 *     Else: Random
 *
 * POSTCONDITIONS:
 *     Returns best-guess access pattern.
 *
 * USAGE:
 *     AccessPattern pattern = detect_pattern_from_stride(my_stride);
 *     config.access_hint = pattern;
 * -------------------------------------------------------------------------- */
AccessPattern detect_pattern_from_stride(std::size_t stride) noexcept;

} // namespace scl::mmap
