// =============================================================================
// FILE: scl/mmap/cache/tiered.h
// BRIEF: API reference for tiered cache (L1/L2) implementation
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "eviction.h"
#include "../backend/backend.h"
#include "../memory/numa.h"

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <functional>
#include <memory>
#include <span>

namespace scl::mmap::cache {

/* =============================================================================
 * STRUCT: CacheTier
 * =============================================================================
 * SUMMARY:
 *     Configuration for a single cache tier.
 *
 * FIELDS:
 *     capacity_pages - Maximum pages in this tier
 *     eviction       - Eviction policy configuration
 *     numa_node      - Preferred NUMA node (-1 = local)
 *     use_huge_pages - Allocate with huge pages
 * -------------------------------------------------------------------------- */
struct CacheTier {
    std::size_t capacity_pages = 256;
    EvictionConfig eviction = EvictionConfig::lru();
    int numa_node = -1;
    bool use_huge_pages = false;
};

/* =============================================================================
 * STRUCT: TieredCacheConfig
 * =============================================================================
 * SUMMARY:
 *     Configuration for tiered cache system.
 *
 * DESIGN:
 *     L1: Small, fast, low-latency eviction (LRU/Clock)
 *     L2: Large, slower, adaptive eviction (ARC/CostBased)
 *
 * FIELDS:
 *     l1                  - L1 (hot) tier configuration
 *     l2                  - L2 (warm) tier configuration
 *     enable_l2           - Enable L2 tier (disable for simple cache)
 *     writeback_on_evict  - Write dirty pages before eviction
 *     prefetch_to_l2      - Prefetched pages go to L2 first
 *     promote_threshold   - L2 accesses before promotion to L1
 *     stats_sample_rate   - Sample rate for statistics (1 = all)
 * -------------------------------------------------------------------------- */
struct TieredCacheConfig {
    CacheTier l1 = {
        .capacity_pages = 64,
        .eviction = EvictionConfig::lru(),
        .numa_node = -1,
        .use_huge_pages = false
    };

    CacheTier l2 = {
        .capacity_pages = 512,
        .eviction = EvictionConfig::arc(),
        .numa_node = -1,
        .use_huge_pages = false
    };

    bool enable_l2 = true;
    bool writeback_on_evict = true;
    bool prefetch_to_l2 = true;
    std::size_t promote_threshold = 2;
    std::size_t stats_sample_rate = 1;

    /* -------------------------------------------------------------------------
     * FACTORY: simple
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Single-tier LRU cache configuration.
     * ---------------------------------------------------------------------- */
    static constexpr TieredCacheConfig simple(std::size_t capacity = 256) noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: balanced
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Two-tier cache with balanced L1/L2 sizes.
     * ---------------------------------------------------------------------- */
    static constexpr TieredCacheConfig balanced() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: large_working_set
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for large working sets.
     * ---------------------------------------------------------------------- */
    static constexpr TieredCacheConfig large_working_set() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: low_latency
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration optimized for low latency access.
     * ---------------------------------------------------------------------- */
    static constexpr TieredCacheConfig low_latency() noexcept;
};

/* =============================================================================
 * STRUCT: CacheStats
 * =============================================================================
 * SUMMARY:
 *     Statistics for cache performance monitoring.
 * -------------------------------------------------------------------------- */
struct CacheStats {
    // Hit/miss counters
    std::size_t l1_hits;
    std::size_t l1_misses;
    std::size_t l2_hits;
    std::size_t l2_misses;
    std::size_t backend_reads;

    // Eviction counters
    std::size_t l1_evictions;
    std::size_t l2_evictions;
    std::size_t dirty_writebacks;

    // Promotion/demotion
    std::size_t l2_to_l1_promotions;
    std::size_t l1_to_l2_demotions;

    // Prefetch stats
    std::size_t prefetch_requests;
    std::size_t prefetch_hits;

    // Latency stats (accumulated)
    std::chrono::nanoseconds total_l1_latency;
    std::chrono::nanoseconds total_l2_latency;
    std::chrono::nanoseconds total_backend_latency;

    /* -------------------------------------------------------------------------
     * METHOD: hit_rate
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Overall cache hit rate (0.0 - 1.0).
     * ---------------------------------------------------------------------- */
    double hit_rate() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: l1_hit_rate
     * -------------------------------------------------------------------------
     * RETURNS:
     *     L1 hit rate among all accesses.
     * ---------------------------------------------------------------------- */
    double l1_hit_rate() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: l2_hit_rate
     * -------------------------------------------------------------------------
     * RETURNS:
     *     L2 hit rate among L1 misses.
     * ---------------------------------------------------------------------- */
    double l2_hit_rate() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: avg_latency
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Average access latency in nanoseconds.
     * ---------------------------------------------------------------------- */
    std::chrono::nanoseconds avg_latency() const noexcept;
};

/* =============================================================================
 * STRUCT: PageHandle
 * =============================================================================
 * SUMMARY:
 *     RAII handle to a cached page with automatic unpinning.
 *
 * DESIGN PURPOSE:
 *     Provides safe access to cached pages:
 *     - Page is pinned while handle exists
 *     - Automatic unpin on destruction
 *     - Zero-copy access to page data
 *
 * THREAD SAFETY:
 *     Handles are NOT thread-safe. Do not share across threads.
 *     Create separate handles per thread.
 * -------------------------------------------------------------------------- */
class PageHandle {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: PageHandle (default)
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     Creates invalid handle (data() returns nullptr).
     * ---------------------------------------------------------------------- */
    PageHandle() noexcept;

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: PageHandle (move)
     * ---------------------------------------------------------------------- */
    PageHandle(PageHandle&& other) noexcept;
    PageHandle& operator=(PageHandle&& other) noexcept;

    PageHandle(const PageHandle&) = delete;
    PageHandle& operator=(const PageHandle&) = delete;

    /* -------------------------------------------------------------------------
     * DESTRUCTOR
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     - Page unpinned in cache
     *     - Dirty flag respected (writeback may occur)
     * ---------------------------------------------------------------------- */
    ~PageHandle();

    /* -------------------------------------------------------------------------
     * METHOD: data
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Pointer to page data, or nullptr if invalid.
     *
     * LIFETIME:
     *     Valid until handle is destroyed or moved.
     * ---------------------------------------------------------------------- */
    const std::byte* data() const noexcept;
    std::byte* data() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: size
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Size of page in bytes (kPageSize or 0 if invalid).
     * ---------------------------------------------------------------------- */
    std::size_t size() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: page_idx
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Page index in backend, or SIZE_MAX if invalid.
     * ---------------------------------------------------------------------- */
    std::size_t page_idx() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: valid
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if handle points to valid page.
     * ---------------------------------------------------------------------- */
    bool valid() const noexcept;
    explicit operator bool() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: mark_dirty
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Mark page as modified (requires writeback on eviction).
     * ---------------------------------------------------------------------- */
    void mark_dirty() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: is_dirty
     * -------------------------------------------------------------------------
     * RETURNS:
     *     True if page has been modified.
     * ---------------------------------------------------------------------- */
    bool is_dirty() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: release
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Release page without waiting for writeback.
     *
     * POSTCONDITIONS:
     *     Handle becomes invalid.
     * ---------------------------------------------------------------------- */
    void release() noexcept;

private:
    friend class TieredCache;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    PageHandle(std::unique_ptr<Impl> impl);
};

/* =============================================================================
 * CLASS: TieredCache
 * =============================================================================
 * SUMMARY:
 *     Two-tier page cache with configurable eviction policies.
 *
 * DESIGN PURPOSE:
 *     Provides high-performance caching for mmap pages:
 *     - L1: Small, fast cache for hot pages (low eviction overhead)
 *     - L2: Large, adaptive cache for warm pages (ARC/cost-based)
 *     - NUMA-aware page allocation
 *     - Background writeback for dirty pages
 *
 * ARCHITECTURE:
 *     [Request] --> [L1 Lookup] --> HIT --> [Return PageHandle]
 *                       |
 *                      MISS
 *                       |
 *                       v
 *                  [L2 Lookup] --> HIT --> [Promote to L1] --> [Return]
 *                       |
 *                      MISS
 *                       |
 *                       v
 *                  [Backend Load] --> [Insert L1/L2] --> [Return]
 *
 * PAGE LIFECYCLE:
 *     1. Load from backend --> Insert to L2 (or L1 if small)
 *     2. Access in L2 --> Promote to L1 after threshold
 *     3. Evict from L1 --> Demote to L2
 *     4. Evict from L2 --> Writeback if dirty, discard
 *
 * THREAD SAFETY:
 *     Fully thread-safe. Uses fine-grained locking:
 *     - Separate locks for L1 and L2
 *     - Reader-writer lock for lookups vs modifications
 *     - Lock-free statistics updates
 * -------------------------------------------------------------------------- */
class TieredCache {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: TieredCache
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     config [in] - Cache configuration
     *
     * POSTCONDITIONS:
     *     - Cache initialized with configured capacity
     *     - NUMA allocator set up based on config
     *     - Ready for page requests
     * ---------------------------------------------------------------------- */
    explicit TieredCache(TieredCacheConfig config = {});

    ~TieredCache();

    TieredCache(const TieredCache&) = delete;
    TieredCache& operator=(const TieredCache&) = delete;
    TieredCache(TieredCache&&) noexcept;
    TieredCache& operator=(TieredCache&&) noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: get
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get a page from cache, loading from backend if needed.
     *
     * PARAMETERS:
     *     page_idx [in] - Page index to retrieve
     *     backend  [in] - Storage backend to load from on miss
     *
     * PRECONDITIONS:
     *     - backend != nullptr
     *     - page_idx < backend->num_pages()
     *
     * POSTCONDITIONS:
     *     On success:
     *     - Returns valid PageHandle
     *     - Page is pinned until handle released
     *     - Page data accessible via handle.data()
     *     On failure:
     *     - Returns invalid PageHandle
     *
     * ALGORITHM:
     *     1. Check L1 cache --> HIT: return pinned handle
     *     2. Check L2 cache --> HIT: promote to L1, return handle
     *     3. Load from backend
     *     4. Insert into L1 (or L2 if L1 full and page cold)
     *     5. Return pinned handle
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    template <typename BackendT>
    PageHandle get(
        std::size_t page_idx,             // Page index to get
        BackendT* backend                  // Backend to load from
    );

    /* -------------------------------------------------------------------------
     * METHOD: prefetch
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Hint that pages will be needed soon.
     *
     * PARAMETERS:
     *     pages   [in] - Page indices to prefetch
     *     backend [in] - Storage backend to load from
     *
     * POSTCONDITIONS:
     *     - Pages queued for background loading
     *     - Does not block caller
     *
     * BEHAVIOR:
     *     - Prefetched pages go to L2 (configurable)
     *     - Already cached pages are not reloaded
     *     - Backend prefetch hints also issued
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    template <typename BackendT>
    void prefetch(
        std::span<const std::size_t> pages, // Pages to prefetch
        BackendT* backend                    // Backend to load from
    );

    /* -------------------------------------------------------------------------
     * METHOD: contains
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if page is in cache (L1 or L2).
     *
     * PARAMETERS:
     *     page_idx [in] - Page index to check
     *     file_id  [in] - Backend file ID
     *
     * RETURNS:
     *     True if page is cached.
     *
     * NOTE:
     *     Result may be stale immediately (page could be evicted).
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    bool contains(
        std::size_t page_idx,             // Page index
        std::size_t file_id               // Backend file ID
    ) const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: invalidate
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Remove a page from cache.
     *
     * PARAMETERS:
     *     page_idx [in] - Page index to invalidate
     *     file_id  [in] - Backend file ID
     *
     * POSTCONDITIONS:
     *     - Page removed from L1 and L2
     *     - Writeback performed if dirty
     *     - Pinned pages wait for unpin
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void invalidate(
        std::size_t page_idx,             // Page index
        std::size_t file_id               // Backend file ID
    );

    /* -------------------------------------------------------------------------
     * METHOD: invalidate_all
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Remove all pages for a backend from cache.
     *
     * PARAMETERS:
     *     file_id [in] - Backend file ID to invalidate
     *
     * POSTCONDITIONS:
     *     - All pages for file_id removed
     *     - Dirty pages written back
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void invalidate_all(
        std::size_t file_id               // Backend file ID
    );

    /* -------------------------------------------------------------------------
     * METHOD: flush
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Write all dirty pages to backend.
     *
     * POSTCONDITIONS:
     *     - All dirty pages written back
     *     - Pages remain in cache (not evicted)
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void flush();

    /* -------------------------------------------------------------------------
     * METHOD: clear
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Remove all pages from cache.
     *
     * POSTCONDITIONS:
     *     - L1 and L2 cleared
     *     - Dirty pages written back first
     *     - Statistics preserved
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void clear();

    /* -------------------------------------------------------------------------
     * METHOD: resize
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Change cache capacity at runtime.
     *
     * PARAMETERS:
     *     l1_capacity [in] - New L1 capacity in pages
     *     l2_capacity [in] - New L2 capacity in pages
     *
     * POSTCONDITIONS:
     *     - Capacity adjusted
     *     - Excess pages evicted (LRU order)
     *
     * THREAD SAFETY:
     *     Thread-safe (but blocks other operations).
     * ---------------------------------------------------------------------- */
    void resize(
        std::size_t l1_capacity,          // New L1 size
        std::size_t l2_capacity           // New L2 size
    );

    /* -------------------------------------------------------------------------
     * METHOD: config
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Reference to current configuration.
     * ---------------------------------------------------------------------- */
    const TieredCacheConfig& config() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: stats
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Current cache statistics.
     * ---------------------------------------------------------------------- */
    CacheStats stats() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: reset_stats
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Reset statistics to zero.
     * ---------------------------------------------------------------------- */
    void reset_stats() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: l1_size
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Current number of pages in L1.
     * ---------------------------------------------------------------------- */
    std::size_t l1_size() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: l2_size
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Current number of pages in L2.
     * ---------------------------------------------------------------------- */
    std::size_t l2_size() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: memory_usage
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Total memory used by cache in bytes.
     * ---------------------------------------------------------------------- */
    std::size_t memory_usage() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* =============================================================================
 * FUNCTION: cache_tier_name
 * =============================================================================
 * SUMMARY:
 *     Get human-readable name for cache tier.
 *
 * PARAMETERS:
 *     is_l1 [in] - True for L1, false for L2
 *
 * RETURNS:
 *     "L1" or "L2".
 * -------------------------------------------------------------------------- */
inline const char* cache_tier_name(bool is_l1) noexcept;

} // namespace scl::mmap::cache
