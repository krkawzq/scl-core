// =============================================================================
// FILE: scl/mmap/cache/eviction.h
// BRIEF: API reference for cache eviction policies
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <functional>
#include <span>

namespace scl::mmap::cache {

/* =============================================================================
 * ENUM: EvictionPolicy
 * =============================================================================
 * SUMMARY:
 *     Cache eviction algorithm selection.
 *
 * VALUES:
 *     LRU       - Least Recently Used (simple, low overhead)
 *     LFU       - Least Frequently Used (access count based)
 *     ARC       - Adaptive Replacement Cache (self-tuning)
 *     Clock     - Clock/Second-Chance (approximates LRU, lower overhead)
 *     CostBased - Considers backend latency and access patterns
 *
 * SELECTION GUIDE:
 *     LRU:       Best for recency-dominated workloads
 *     LFU:       Best for frequency-dominated workloads
 *     ARC:       Best for mixed/unknown workloads (self-adapts)
 *     Clock:     Best for very large caches (lowest overhead)
 *     CostBased: Best for heterogeneous backends (SSD vs network)
 * -------------------------------------------------------------------------- */
enum class EvictionPolicy : std::uint8_t {
    LRU,
    LFU,
    ARC,
    Clock,
    CostBased
};

/* =============================================================================
 * STRUCT: PageMetadata
 * =============================================================================
 * SUMMARY:
 *     Metadata tracked for each cached page.
 *
 * FIELDS:
 *     page_idx        - Global page index
 *     file_id         - Source backend identifier
 *     access_count    - Number of accesses since cache entry
 *     last_access     - Timestamp of most recent access
 *     load_latency    - Time taken to load this page
 *     is_dirty        - Page has been modified
 *     is_pinned       - Page cannot be evicted
 *     priority        - User-assigned priority (higher = keep longer)
 * -------------------------------------------------------------------------- */
struct PageMetadata {
    std::size_t page_idx;
    std::size_t file_id;
    std::uint32_t access_count;
    std::chrono::steady_clock::time_point last_access;
    std::chrono::nanoseconds load_latency;
    bool is_dirty;
    bool is_pinned;
    std::int8_t priority;
};

/* =============================================================================
 * STRUCT: EvictionCandidate
 * =============================================================================
 * SUMMARY:
 *     Represents a page considered for eviction.
 *
 * FIELDS:
 *     slot_idx  - Index in cache slot array
 *     metadata  - Pointer to page metadata
 *     score     - Eviction score (lower = more likely to evict)
 * -------------------------------------------------------------------------- */
struct EvictionCandidate {
    std::size_t slot_idx;
    const PageMetadata* metadata;
    double score;
};

/* =============================================================================
 * STRUCT: EvictionConfig
 * =============================================================================
 * SUMMARY:
 *     Configuration for eviction policies.
 *
 * FIELDS:
 *     policy           - Eviction algorithm to use
 *     arc_p_init       - ARC: initial p parameter (0.0-1.0)
 *     lfu_decay_period - LFU: accesses before frequency decay
 *     cost_weight      - CostBased: weight for load latency
 *     recency_weight   - CostBased: weight for recency
 *     frequency_weight - CostBased: weight for frequency
 *     priority_weight  - CostBased: weight for user priority
 * -------------------------------------------------------------------------- */
struct EvictionConfig {
    EvictionPolicy policy = EvictionPolicy::ARC;
    double arc_p_init = 0.5;
    std::size_t lfu_decay_period = 1000;
    double cost_weight = 1.0;
    double recency_weight = 1.0;
    double frequency_weight = 0.5;
    double priority_weight = 2.0;

    /* -------------------------------------------------------------------------
     * FACTORY: lru
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for LRU eviction.
     * ---------------------------------------------------------------------- */
    static constexpr EvictionConfig lru() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: lfu
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for LFU eviction.
     * ---------------------------------------------------------------------- */
    static constexpr EvictionConfig lfu() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: arc
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for ARC eviction (default).
     * ---------------------------------------------------------------------- */
    static constexpr EvictionConfig arc() noexcept;

    /* -------------------------------------------------------------------------
     * FACTORY: cost_based
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Configuration for cost-based eviction.
     * ---------------------------------------------------------------------- */
    static constexpr EvictionConfig cost_based() noexcept;
};

/* =============================================================================
 * STRUCT: EvictionStats
 * =============================================================================
 * SUMMARY:
 *     Statistics for eviction policy performance.
 *
 * FIELDS:
 *     total_evictions   - Total pages evicted
 *     dirty_evictions   - Pages requiring writeback before eviction
 *     ghost_hits        - Hits in ghost lists (ARC only)
 *     adaptive_changes  - Number of p parameter adjustments (ARC)
 * -------------------------------------------------------------------------- */
struct EvictionStats {
    std::size_t total_evictions;
    std::size_t dirty_evictions;
    std::size_t ghost_hits;
    std::size_t adaptive_changes;
};

/* =============================================================================
 * CLASS: EvictionTracker
 * =============================================================================
 * SUMMARY:
 *     Tracks page access patterns and selects eviction candidates.
 *
 * DESIGN PURPOSE:
 *     Decouples eviction logic from cache storage:
 *     - Cache manages memory and page storage
 *     - EvictionTracker manages access tracking and victim selection
 *
 * ARCHITECTURE:
 *     LRU:  Doubly-linked list ordered by access time
 *     LFU:  Min-heap by access count with periodic decay
 *     ARC:  Four lists (T1, T2, B1, B2) with adaptive p
 *     Clock: Circular buffer with reference bits
 *     CostBased: Priority queue by composite score
 *
 * THREAD SAFETY:
 *     NOT thread-safe. Caller must synchronize.
 *     Cache layer handles thread safety.
 * -------------------------------------------------------------------------- */
class EvictionTracker {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: EvictionTracker
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     capacity [in] - Maximum number of pages to track
     *     config   [in] - Eviction configuration
     *
     * POSTCONDITIONS:
     *     - Tracker initialized with specified policy
     *     - No pages tracked initially
     * ---------------------------------------------------------------------- */
    explicit EvictionTracker(
        std::size_t capacity,             // Max pages to track
        EvictionConfig config = {}        // Eviction configuration
    );

    ~EvictionTracker();

    EvictionTracker(const EvictionTracker&) = delete;
    EvictionTracker& operator=(const EvictionTracker&) = delete;
    EvictionTracker(EvictionTracker&&) noexcept;
    EvictionTracker& operator=(EvictionTracker&&) noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: on_access
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Record a page access (hit or miss).
     *
     * PARAMETERS:
     *     slot_idx [in] - Cache slot index
     *     metadata [in] - Page metadata
     *     is_hit   [in] - True if page was already cached
     *
     * POSTCONDITIONS:
     *     - Page moved to appropriate position in tracking structure
     *     - Access count and timestamp updated
     *     - For ARC: ghost list checked, p adjusted if needed
     *
     * COMPLEXITY:
     *     LRU/LFU: O(1) amortized
     *     ARC: O(1) amortized
     *     Clock: O(1)
     *     CostBased: O(log n)
     * ---------------------------------------------------------------------- */
    void on_access(
        std::size_t slot_idx,             // Cache slot index
        const PageMetadata& metadata,     // Page metadata
        bool is_hit                       // True if cache hit
    );

    /* -------------------------------------------------------------------------
     * METHOD: on_evict
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Notify tracker that a page was evicted.
     *
     * PARAMETERS:
     *     slot_idx [in] - Cache slot index being evicted
     *
     * POSTCONDITIONS:
     *     - Page removed from active tracking
     *     - For ARC: page added to ghost list
     * ---------------------------------------------------------------------- */
    void on_evict(
        std::size_t slot_idx              // Cache slot being evicted
    );

    /* -------------------------------------------------------------------------
     * METHOD: select_victim
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Select a page to evict.
     *
     * PARAMETERS:
     *     exclude_pinned [in] - If true, skip pinned pages
     *
     * PRECONDITIONS:
     *     - At least one unpinned page in cache (if exclude_pinned)
     *
     * RETURNS:
     *     Slot index of page to evict, or SIZE_MAX if none available.
     *
     * ALGORITHM:
     *     LRU: Return tail of LRU list
     *     LFU: Return min-frequency page
     *     ARC: Choose from T1 or T2 based on p
     *     Clock: Scan until reference bit is 0
     *     CostBased: Return lowest score page
     *
     * COMPLEXITY:
     *     LRU/LFU/ARC: O(1) amortized
     *     Clock: O(n) worst case, O(1) amortized
     *     CostBased: O(log n)
     * ---------------------------------------------------------------------- */
    std::size_t select_victim(
        bool exclude_pinned = true        // Skip pinned pages
    ) const;

    /* -------------------------------------------------------------------------
     * METHOD: select_victims
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Select multiple pages to evict.
     *
     * PARAMETERS:
     *     count          [in]  - Number of victims to select
     *     exclude_pinned [in]  - If true, skip pinned pages
     *     output         [out] - Output span for slot indices
     *
     * RETURNS:
     *     Number of victims selected (may be less than count).
     *
     * POSTCONDITIONS:
     *     - Output contains up to count victim slot indices
     *     - Victims ordered by eviction priority (first = evict first)
     * ---------------------------------------------------------------------- */
    std::size_t select_victims(
        std::size_t count,                // Number of victims needed
        bool exclude_pinned,              // Skip pinned pages
        std::span<std::size_t> output     // Output buffer
    ) const;

    /* -------------------------------------------------------------------------
     * METHOD: pin
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Mark a page as pinned (cannot be evicted).
     *
     * PARAMETERS:
     *     slot_idx [in] - Cache slot to pin
     *
     * POSTCONDITIONS:
     *     - Page will not be selected as victim until unpinned
     * ---------------------------------------------------------------------- */
    void pin(
        std::size_t slot_idx              // Cache slot to pin
    );

    /* -------------------------------------------------------------------------
     * METHOD: unpin
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Mark a page as unpinned (can be evicted).
     *
     * PARAMETERS:
     *     slot_idx [in] - Cache slot to unpin
     *
     * POSTCONDITIONS:
     *     - Page eligible for eviction again
     * ---------------------------------------------------------------------- */
    void unpin(
        std::size_t slot_idx              // Cache slot to unpin
    );

    /* -------------------------------------------------------------------------
     * METHOD: set_priority
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Update user priority for a cached page.
     *
     * PARAMETERS:
     *     slot_idx [in] - Cache slot index
     *     priority [in] - New priority (-128 to 127, higher = keep longer)
     *
     * NOTE:
     *     Only affects CostBased policy. Ignored by other policies.
     * ---------------------------------------------------------------------- */
    void set_priority(
        std::size_t slot_idx,             // Cache slot index
        std::int8_t priority              // Priority level
    );

    /* -------------------------------------------------------------------------
     * METHOD: capacity
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Maximum number of pages this tracker can manage.
     * ---------------------------------------------------------------------- */
    std::size_t capacity() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: size
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Current number of tracked pages.
     * ---------------------------------------------------------------------- */
    std::size_t size() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: config
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Reference to current eviction configuration.
     * ---------------------------------------------------------------------- */
    const EvictionConfig& config() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: stats
     * -------------------------------------------------------------------------
     * RETURNS:
     *     Current eviction statistics.
     * ---------------------------------------------------------------------- */
    EvictionStats stats() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: reset_stats
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Reset eviction statistics to zero.
     * ---------------------------------------------------------------------- */
    void reset_stats() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: clear
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Remove all tracked pages.
     *
     * POSTCONDITIONS:
     *     - All tracking state cleared
     *     - Statistics preserved
     * ---------------------------------------------------------------------- */
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* =============================================================================
 * FUNCTION: eviction_policy_name
 * =============================================================================
 * SUMMARY:
 *     Convert EvictionPolicy enum to human-readable string.
 *
 * PARAMETERS:
 *     policy [in] - Eviction policy enum value
 *
 * RETURNS:
 *     Null-terminated C string (never nullptr).
 * -------------------------------------------------------------------------- */
const char* eviction_policy_name(EvictionPolicy policy) noexcept;

/* =============================================================================
 * FUNCTION: compute_eviction_score
 * =============================================================================
 * SUMMARY:
 *     Compute eviction score for a page under CostBased policy.
 *
 * PARAMETERS:
 *     metadata [in] - Page metadata
 *     config   [in] - Eviction configuration (for weights)
 *     now      [in] - Current timestamp
 *
 * RETURNS:
 *     Eviction score (lower = more likely to evict).
 *
 * ALGORITHM:
 *     score = cost_weight * (1 / load_latency)
 *           + recency_weight * (1 / age)
 *           + frequency_weight * access_count
 *           + priority_weight * priority
 * -------------------------------------------------------------------------- */
double compute_eviction_score(
    const PageMetadata& metadata,
    const EvictionConfig& config,
    std::chrono::steady_clock::time_point now
) noexcept;

} // namespace scl::mmap::cache
