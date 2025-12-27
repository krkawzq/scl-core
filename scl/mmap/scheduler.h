// =============================================================================
// FILE: scl/mmap/scheduler.h
// BRIEF: API reference for event-driven prefetch scheduler with pluggable policies
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <memory>
#include <span>
#include <vector>

namespace scl::mmap {

// Forward declarations
class PageStore;
class Page;

/* =============================================================================
 * TYPE ALIASES: Time Utilities
 * =============================================================================
 * Clock:      std::chrono::steady_clock for time measurements
 * Duration:   std::chrono::nanoseconds for latency tracking
 * TimePoint:  Clock::time_point for timestamp tracking
 * -------------------------------------------------------------------------- */
using Clock = std::chrono::steady_clock;
using Duration = std::chrono::nanoseconds;
using TimePoint = Clock::time_point;

/* =============================================================================
 * STRUCT: SchedulerStats
 * =============================================================================
 * SUMMARY:
 *     Statistics tracking for prefetch scheduler performance metrics.
 *
 * FIELDS:
 *     total_fetches:         Total number of page fetches executed
 *     cache_hits:            Number of cache hits (page already loaded)
 *     cache_misses:          Number of cache misses (page needs loading)
 *     evictions:             Number of pages evicted from cache
 *     total_fetch_latency_ns: Cumulative fetch latency in nanoseconds
 *
 * THREAD SAFETY:
 *     All fields are atomic. Methods are thread-safe.
 *
 * COMPUTED PROPERTIES:
 *     hit_rate:              Ratio of hits to (hits + misses), returns 1.0 if no accesses
 *     avg_fetch_latency:     Average latency per fetch, returns Duration{0} if no fetches
 * -------------------------------------------------------------------------- */
struct SchedulerStats {
    std::atomic<std::size_t> total_fetches;
    std::atomic<std::size_t> cache_hits;
    std::atomic<std::size_t> cache_misses;
    std::atomic<std::size_t> evictions;
    std::atomic<std::uint64_t> total_fetch_latency_ns;
    
    /* -------------------------------------------------------------------------
     * METHOD: hit_rate()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Compute cache hit rate as ratio of hits to total accesses.
     *
     * RETURNS:
     *     Hit rate in [0.0, 1.0]. Returns 1.0 if no accesses yet.
     *
     * THREAD SAFETY:
     *     Thread-safe (atomic loads).
     * ---------------------------------------------------------------------- */
    double hit_rate() const noexcept;
    
    /* -------------------------------------------------------------------------
     * METHOD: avg_fetch_latency()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Compute average fetch latency.
     *
     * RETURNS:
     *     Average latency per fetch. Returns Duration{0} if no fetches.
     *
     * THREAD SAFETY:
     *     Thread-safe (atomic loads).
     * ---------------------------------------------------------------------- */
    Duration avg_fetch_latency() const noexcept;
    
    /* -------------------------------------------------------------------------
     * METHOD: reset()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Reset all statistics counters to zero.
     *
     * POSTCONDITIONS:
     *     All atomic counters set to 0.
     *
     * THREAD SAFETY:
     *     Thread-safe (atomic stores).
     * ---------------------------------------------------------------------- */
    void reset() noexcept;
};

/* =============================================================================
 * STRUCT: SchedulerState
 * =============================================================================
 * SUMMARY:
 *     Read-only view of scheduler state for policy decision-making.
 *
 * FIELDS:
 *     current_row:           Current row being processed
 *     total_rows:            Total number of rows in computation
 *     resident_pages:        Number of pages currently in cache
 *     max_resident:          Maximum number of resident pages allowed
 *     pending_fetches:       Number of pages currently being fetched
 *     hit_rate:              Current cache hit rate
 *     avg_fetch_latency:     Average fetch latency
 *     row_lengths:           Span of row lengths (optional hint)
 *
 * THREAD SAFETY:
 *     Immutable struct. Safe to read concurrently.
 * -------------------------------------------------------------------------- */
struct SchedulerState {
    std::size_t current_row;
    std::size_t total_rows;
    std::size_t resident_pages;
    std::size_t max_resident;
    std::size_t pending_fetches;
    double hit_rate;
    Duration avg_fetch_latency;
    std::span<const std::size_t> row_lengths;
    
    /* -------------------------------------------------------------------------
     * METHOD: has_capacity()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if scheduler has capacity for more pages.
     *
     * RETURNS:
     *     True if resident_pages + pending_fetches < max_resident.
     * ---------------------------------------------------------------------- */
    bool has_capacity() const noexcept;
    
    /* -------------------------------------------------------------------------
     * METHOD: available_slots()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Compute number of available cache slots.
     *
     * RETURNS:
     *     max_resident - (resident_pages + pending_fetches), clamped to >= 0.
     * ---------------------------------------------------------------------- */
    std::size_t available_slots() const noexcept;
};

/* =============================================================================
 * ENUM: BlockMode
 * =============================================================================
 * VALUES:
 *     SpinWait:       Spin-wait with CPU pause instructions (low latency)
 *     ConditionWait:  Condition variable wait (saves CPU, higher latency)
 *     Hybrid:         Spin for fixed iterations, then condition wait (balanced)
 *     Callback:       No blocking (callback-based synchronization)
 * -------------------------------------------------------------------------- */
enum class BlockMode : std::uint8_t {
    SpinWait,
    ConditionWait,
    Hybrid,
    Callback
};

/* =============================================================================
 * STRUCT: PrefetchTask
 * =============================================================================
 * SUMMARY:
 *     Task description for prefetch worker threads.
 *
 * FIELDS:
 *     page_idx:       Page index to fetch
 *     store:          PageStore to load from
 *     priority:       Task priority (higher = more urgent)
 *     submit_time:    When task was submitted
 *
 * OPERATOR:
 *     operator<:      Comparison by priority (for priority_queue)
 * -------------------------------------------------------------------------- */
struct PrefetchTask {
    std::size_t page_idx;
    PageStore* store;
    int priority;
    TimePoint submit_time;
    
    bool operator<(const PrefetchTask& other) const noexcept;
};

/* =============================================================================
 * STRUCT: PolicyDecision
 * =============================================================================
 * SUMMARY:
 *     Decision output from scheduling policy.
 *
 * FIELDS:
 *     pages_to_fetch: Vector of page indices to prefetch
 *     priority:       Priority level for these prefetch tasks
 *     should_wait:    If true, scheduler should wait before scheduling more
 * -------------------------------------------------------------------------- */
struct PolicyDecision {
    std::vector<std::size_t> pages_to_fetch;
    int priority;
    bool should_wait;
};

/* =============================================================================
 * CLASS: SchedulePolicy
 * =============================================================================
 * SUMMARY:
 *     Abstract base class for prefetch scheduling policies.
 *
 * DESIGN PURPOSE:
 *     Pluggable policy interface allows different prefetch strategies:
 *     - LookaheadPolicy: Sequential prefetch with adaptive depth
 *     - Custom policies: User-defined scheduling algorithms
 *
 * THREAD SAFETY:
 *     Policy methods may be called from multiple threads. Implementations
 *     must be thread-safe or use thread-local state.
 * -------------------------------------------------------------------------- */
class SchedulePolicy {
public:
    virtual ~SchedulePolicy() = default;
    
    /* -------------------------------------------------------------------------
     * METHOD: decide()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Make scheduling decision based on current state.
     *
     * PARAMETERS:
     *     state    [in]  Current scheduler state snapshot
     *
     * RETURNS:
     *     PolicyDecision with pages to prefetch and scheduling hints.
     *
     * THREAD SAFETY:
     *     Must be thread-safe (may be called from worker threads).
     * ---------------------------------------------------------------------- */
    virtual PolicyDecision decide(const SchedulerState& state) = 0;
    
    /* -------------------------------------------------------------------------
     * CALLBACKS: Event notifications
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Optional callbacks for policy state updates.
     *
     * on_fetch_complete: Called when page fetch completes (for adaptive tuning)
     * on_cache_hit:      Called on cache hit
     * on_cache_miss:     Called on cache miss
     * on_eviction:       Called when page is evicted
     * on_row_begin:      Called at start of row processing
     * on_row_end:        Called at end of row processing
     *
     * THREAD SAFETY:
     *     May be called from multiple threads concurrently.
     * ---------------------------------------------------------------------- */
    virtual void on_fetch_complete(std::size_t page, Duration latency);
    virtual void on_cache_hit(std::size_t page);
    virtual void on_cache_miss(std::size_t page);
    virtual void on_eviction(std::size_t page);
    virtual void on_row_begin(std::size_t row);
    virtual void on_row_end(std::size_t row);
    
    /* -------------------------------------------------------------------------
     * METHODS: Configuration
     * -------------------------------------------------------------------------
     * set_lookahead:  Set prefetch lookahead depth
     * set_batch_size: Set prefetch batch size
     *
     * THREAD SAFETY:
     *     Must be thread-safe.
     * ---------------------------------------------------------------------- */
    virtual void set_lookahead(std::size_t lookahead);
    virtual void set_batch_size(std::size_t batch);
    
    /* -------------------------------------------------------------------------
     * METHOD: name()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get policy name for logging/debugging.
     *
     * RETURNS:
     *     Null-terminated C string with policy name.
     * ---------------------------------------------------------------------- */
    virtual const char* name() const noexcept;
};

/* =============================================================================
 * CLASS: LookaheadPolicy
 * =============================================================================
 * SUMMARY:
 *     Sequential prefetch policy with adaptive lookahead depth.
 *
 * ALGORITHM:
 *     Prefetches pages sequentially ahead of current row with adaptive
 *     lookahead depth based on fetch latency. Uses atomic flags to track
 *     loaded state for thread-safe coordination.
 *
 * ADAPTIVE BEHAVIOR:
 *     - Increases lookahead if latency > 2 * target_latency
 *     - Decreases lookahead if latency < target_latency / 2
 *     - Clamped between min_lookahead (8) and max_lookahead (128)
 *
 * THREAD SAFETY:
 *     Fully thread-safe using atomic operations.
 * -------------------------------------------------------------------------- */
class LookaheadPolicy : public SchedulePolicy {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: LookaheadPolicy
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     total_pages      [in] - Total number of pages to manage
     *     lookahead        [in] - Initial lookahead depth (default: 32)
     *     batch_size       [in] - Batch size for prefetch (default: 8)
     *     target_latency   [in] - Target fetch latency for adaptation
     *
     * PRECONDITIONS:
     *     - total_pages > 0
     *     - lookahead in [min_lookahead, max_lookahead]
     *
     * POSTCONDITIONS:
     *     - Policy initialized with atomic flags for all pages
     *     - All flags set to false (not loaded)
     * ---------------------------------------------------------------------- */
    explicit LookaheadPolicy(
        std::size_t total_pages,
        std::size_t lookahead = 32,
        std::size_t batch_size = 8,
        Duration target_latency = std::chrono::microseconds(100)
    );
    
    PolicyDecision decide(const SchedulerState& state) override;
    void on_fetch_complete(std::size_t page, Duration latency) override;
    void on_eviction(std::size_t page) override;
    void on_row_begin(std::size_t row) override;
    void set_lookahead(std::size_t lookahead) override;
    void set_batch_size(std::size_t batch) override;
    const char* name() const noexcept override;
};

/* =============================================================================
 * CLASS: ComputeHooks
 * =============================================================================
 * SUMMARY:
 *     Interface for computation lifecycle hooks.
 *
 * DESIGN PURPOSE:
 *     Allows scheduler to integrate with computation loops, receiving
 *     notifications at key points (row begin/end, computation begin/end).
 *
 * THREAD SAFETY:
 *     Methods may be called from computation thread(s). Implementations
 *     must be thread-safe.
 * -------------------------------------------------------------------------- */
class ComputeHooks {
public:
    virtual ~ComputeHooks() = default;
    
    /* -------------------------------------------------------------------------
     * METHODS: Lifecycle hooks
     * -------------------------------------------------------------------------
     * on_computation_begin: Called at start of computation (reset state)
     * on_computation_end:   Called at end (sync all pending operations)
     * on_row_begin:         Called at start of row processing (trigger prefetch)
     * on_row_end:           Called at end of row (notify waiting threads)
     *
     * THREAD SAFETY:
     *     Called from computation thread(s).
     * ---------------------------------------------------------------------- */
    virtual void on_computation_begin(std::size_t total_rows);
    virtual void on_computation_end();
    virtual void on_row_begin(std::size_t row);
    virtual void on_row_end(std::size_t row);
    
    /* -------------------------------------------------------------------------
     * METHODS: Synchronization
     * -------------------------------------------------------------------------
     * sync_barrier:    Wait until all pending fetches complete
     * sync_fence:      Wait until pages up to row are loaded
     *
     * PARAMETERS (sync_fence):
     *     up_to_row    [in] - Wait for pages in range [0, up_to_row]
     *     timeout      [in] - Maximum wait time (default: 5 seconds)
     *
     * RETURNS (sync_fence):
     *     True if all pages loaded, false on timeout.
     *
     * THREAD SAFETY:
     *     Called from computation thread(s). Blocks until condition met.
     * ---------------------------------------------------------------------- */
    virtual void sync_barrier();
    virtual bool sync_fence(
        std::size_t up_to_row,
        std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}
    );
    
    /* -------------------------------------------------------------------------
     * METHODS: Hints
     * -------------------------------------------------------------------------
     * hint_access_pattern:  Prefetch pages for given rows (hint)
     * hint_priority_boost:  Boost priority for specific row (hint)
     *
     * THREAD SAFETY:
     *     Thread-safe (adds tasks to prefetch queue).
     * ---------------------------------------------------------------------- */
    virtual void hint_access_pattern(std::span<const std::size_t> rows);
    virtual void hint_priority_boost(std::size_t row);
};

/* =============================================================================
 * CLASS: PageHandle
 * =============================================================================
 * SUMMARY:
 *     RAII wrapper for cached page access with automatic pin/unpin.
 *
 * DESIGN PURPOSE:
 *     Ensures pages are pinned during use and unpinned on destruction,
 *     preventing eviction while in use. Provides type-safe access to page data.
 *
 * LIFETIME:
 *     PageHandle pins the page on construction and unpins on destruction.
 *     Moving transfers ownership of pin to new handle.
 *
 * THREAD SAFETY:
 *     Not thread-safe. Each thread should have its own PageHandle.
 * -------------------------------------------------------------------------- */
class PageHandle {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: PageHandle() (default)
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     - valid() returns false
     *     - No pin acquired
     * ---------------------------------------------------------------------- */
    PageHandle() noexcept;
    
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: PageHandle(scheduler, idx, page)
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     scheduler    [in] - PrefetchScheduler instance
     *     idx          [in] - Page index
     *     page         [in] - Page pointer
     *
     * POSTCONDITIONS:
     *     - Page is pinned
     *     - owns_pin_ = true
     * ---------------------------------------------------------------------- */
    PageHandle(PrefetchScheduler* scheduler, std::size_t idx, Page* page) noexcept;
    
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: PageHandle(scheduler, idx, page, already_pinned)
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     scheduler      [in] - PrefetchScheduler instance
     *     idx            [in] - Page index
     *     page           [in] - Page pointer
     *     already_pinned [in] - If true, page is already pinned
     *
     * POSTCONDITIONS:
     *     - If already_pinned: no additional pin, owns_pin_ = false
     *     - Otherwise: page is pinned, owns_pin_ = true
     * ---------------------------------------------------------------------- */
    PageHandle(PrefetchScheduler* scheduler, std::size_t idx, Page* page, bool already_pinned) noexcept;
    
    /* -------------------------------------------------------------------------
     * DESTRUCTOR: ~PageHandle()
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     - Page is unpinned if owns_pin_ was true
     * ---------------------------------------------------------------------- */
    ~PageHandle();
    
    PageHandle(PageHandle&& other) noexcept;
    PageHandle& operator=(PageHandle&& other) noexcept;
    PageHandle(const PageHandle&) = delete;
    PageHandle& operator=(const PageHandle&) = delete;
    
    /* -------------------------------------------------------------------------
     * METHODS: Data access
     * -------------------------------------------------------------------------
     * data():  Get raw byte pointer to page data
     * as<T>(): Reinterpret page data as typed pointer
     *
     * RETURNS:
     *     Pointer to data, or nullptr if handle invalid.
     *
     * PRECONDITIONS:
     *     valid() must be true for safe access.
     * ---------------------------------------------------------------------- */
    const std::byte* data() const noexcept;
    std::byte* data() noexcept;
    
    template <typename T>
    const T* as() const noexcept;
    
    template <typename T>
    T* as() noexcept;
    
    /* -------------------------------------------------------------------------
     * METHODS: State queries
     * -------------------------------------------------------------------------
     * valid():     Check if handle is valid (page != nullptr)
     * operator bool(): Same as valid()
     * index():     Get page index
     * ---------------------------------------------------------------------- */
    bool valid() const noexcept;
    explicit operator bool() const noexcept;
    std::size_t index() const noexcept;
    
    /* -------------------------------------------------------------------------
     * METHOD: mark_dirty()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Mark page as dirty (needs writeback on eviction).
     *
     * THREAD SAFETY:
     *     Thread-safe (atomic store).
     * ---------------------------------------------------------------------- */
    void mark_dirty() noexcept;
};

/* =============================================================================
 * CLASS: PrefetchScheduler
 * =============================================================================
 * SUMMARY:
 *     Event-driven prefetch scheduler with pluggable policies and worker threads.
 *
 * DESIGN PURPOSE:
 *     Manages page cache with automatic prefetching based on computation
 *     events. Uses worker threads for asynchronous I/O and implements
 *     Clock algorithm for eviction. Supports pluggable scheduling policies.
 *
 * ARCHITECTURE:
 *     - Cache entries: Array of CacheEntry (one per page index)
 *     - Worker threads: Configurable number (default: hw_concurrency / 4)
 *     - Task queue: Priority queue for prefetch tasks
 *     - Eviction: Clock algorithm with access bits
 *     - Policies: Pluggable SchedulePolicy implementations
 *
 * THREAD SAFETY:
 *     All public methods are thread-safe. Internal synchronization uses:
 *     - Atomic operations for counters and flags
 *     - Mutexes for task queue and policy access
 *     - Condition variables for worker thread coordination
 *
 * CONCURRENCY FIXES:
 *     - Pin-before-check: request() pins page before checking if loaded
 *     - Load state machine: Prevents duplicate loads with atomic state
 *     - CAS-based eviction: Ensures atomic page removal
 *     - Pin-before-finish: do_load() pins before notifying waiters
 * -------------------------------------------------------------------------- */
class PrefetchScheduler : public ComputeHooks {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: PrefetchScheduler
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     num_pages     [in] - Total number of pages to manage
     *     max_resident  [in] - Maximum resident pages in cache
     *     num_workers   [in] - Number of worker threads (0 = auto)
     *     block_mode    [in] - Blocking mode for synchronization
     *
     * PRECONDITIONS:
     *     - num_pages > 0
     *     - max_resident > 0
     *     - max_resident <= num_pages
     *
     * POSTCONDITIONS:
     *     - Cache entries allocated for all pages
     *     - Worker threads started
     *     - Default LookaheadPolicy installed
     *
     * THROWS:
     *     ValueError if preconditions violated.
     *
     * COMPLEXITY:
     *     Time: O(num_pages) - allocation and initialization
     *     Space: O(num_pages) - cache entry array
     * ---------------------------------------------------------------------- */
    explicit PrefetchScheduler(
        std::size_t num_pages,
        std::size_t max_resident,
        std::size_t num_workers = 0,
        BlockMode block_mode = BlockMode::Hybrid
    );
    
    /* -------------------------------------------------------------------------
     * DESTRUCTOR: ~PrefetchScheduler()
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     - All worker threads stopped
     *     - All dirty pages written back
     *     - All pages released to GlobalPagePool
     * ---------------------------------------------------------------------- */
    ~PrefetchScheduler();
    
    PrefetchScheduler(const PrefetchScheduler&) = delete;
    PrefetchScheduler& operator=(const PrefetchScheduler&) = delete;
    
    /* -------------------------------------------------------------------------
     * METHOD: register_store()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Register PageStore for page loading.
     *
     * PARAMETERS:
     *     store    [in] - PageStore to register
     *
     * POSTCONDITIONS:
     *     - Store added to internal registry
     *     - Store accessible by file_id for writeback
     *
     * THREAD SAFETY:
     *     Thread-safe (protected by stores_lock_).
     * ---------------------------------------------------------------------- */
    void register_store(PageStore* store);
    
    /* -------------------------------------------------------------------------
     * METHOD: request()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Request page access with automatic loading if needed.
     *
     * ALGORITHM (Pin-Before-Check):
     *     1. Pin entry to prevent eviction
     *     2. Check if page already loaded (fast path)
     *     3. If loaded: mark accessed, return PageHandle
     *     4. If not loaded: unpin, perform load, return PageHandle
     *
     * PARAMETERS:
     *     page_idx [in] - Page index to request
     *     store    [in] - PageStore to load from
     *
     * RETURNS:
     *     PageHandle with pinned page. Returns invalid handle if:
     *     - page_idx >= num_pages
     *     - store is nullptr
     *
     * PRECONDITIONS:
     *     - page_idx < num_pages
     *     - store != nullptr
     *
     * POSTCONDITIONS:
     *     - Page is pinned in returned PageHandle
     *     - Page is loaded if not already in cache
     *     - Statistics updated (hit or miss)
     *
     * THREAD SAFETY:
     *     Thread-safe. Pin-before-check prevents race with eviction.
     *
     * COMPLEXITY:
     *     Fast path (cache hit): O(1)
     *     Slow path (cache miss): O(1) + I/O latency
     * ---------------------------------------------------------------------- */
    PageHandle request(std::size_t page_idx, PageStore* store);
    
    /* -------------------------------------------------------------------------
     * METHOD: is_loaded()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if page is currently loaded in cache.
     *
     * PARAMETERS:
     *     page_idx [in] - Page index to check
     *
     * RETURNS:
     *     True if page is loaded, false otherwise.
     *
     * THREAD SAFETY:
     *     Thread-safe (atomic load).
     * ---------------------------------------------------------------------- */
    bool is_loaded(std::size_t page_idx) const noexcept;
    
    /* -------------------------------------------------------------------------
     * METHODS: ComputeHooks Implementation
     * -------------------------------------------------------------------------
     * See ComputeHooks interface for documentation.
     * ---------------------------------------------------------------------- */
    void on_computation_begin(std::size_t total_rows) override;
    void on_computation_end() override;
    void on_row_begin(std::size_t row) override;
    void on_row_end(std::size_t row) override;
    void sync_barrier() override;
    bool sync_fence(std::size_t up_to_row, std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}) override;
    void hint_access_pattern(std::span<const std::size_t> rows) override;
    void hint_priority_boost(std::size_t row) override;
    
    /* -------------------------------------------------------------------------
     * METHODS: Policy Management
     * -------------------------------------------------------------------------
     * set_policy: Set scheduling policy (template or unique_ptr overload)
     * policy_name: Get current policy name
     *
     * THREAD SAFETY:
     *     Thread-safe (protected by policy_lock_).
     * ---------------------------------------------------------------------- */
    template <typename Policy, typename... Args>
    void set_policy(Args&&... args);
    
    void set_policy(std::unique_ptr<SchedulePolicy> policy);
    const char* policy_name() const;
    
    /* -------------------------------------------------------------------------
     * METHODS: Configuration
     * -------------------------------------------------------------------------
     * set_block_mode:  Set synchronization blocking mode
     * set_lookahead:   Set prefetch lookahead depth (delegates to policy)
     * set_batch_size:  Set prefetch batch size (delegates to policy)
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void set_block_mode(BlockMode mode) noexcept;
    void set_lookahead(std::size_t lookahead);
    void set_batch_size(std::size_t batch);
    
    /* -------------------------------------------------------------------------
     * METHODS: Statistics
     * -------------------------------------------------------------------------
     * stats:           Get scheduler statistics
     * resident_count:  Get number of resident pages
     * pending_count:   Get number of pending fetches
     *
     * THREAD SAFETY:
     *     Thread-safe (atomic loads).
     * ---------------------------------------------------------------------- */
    const SchedulerStats& stats() const noexcept;
    std::size_t resident_count() const noexcept;
    std::size_t pending_count() const noexcept;
    
    /* -------------------------------------------------------------------------
     * METHODS: Pin/Unpin (called by PageHandle)
     * -------------------------------------------------------------------------
     * pin:        Increment pin count for page
     * unpin:      Decrement pin count for page
     * mark_dirty: Mark page as dirty (needs writeback)
     *
     * THREAD SAFETY:
     *     Thread-safe (atomic operations).
     * ---------------------------------------------------------------------- */
    void pin(std::size_t idx) noexcept;
    void unpin(std::size_t idx) noexcept;
    void mark_dirty(std::size_t idx) noexcept;
};

/* =============================================================================
 * FUNCTION: make_scheduler()
 * =============================================================================
 * SUMMARY:
 *     Factory function to create PrefetchScheduler with default configuration.
 *
 * PARAMETERS:
 *     num_pages     [in] - Total number of pages
 *     max_resident  [in] - Maximum resident pages (default: 64)
 *     num_workers   [in] - Worker threads (0 = auto, default: 0)
 *     block_mode    [in] - Blocking mode (default: Hybrid)
 *
 * RETURNS:
 *     shared_ptr to PrefetchScheduler instance.
 *
 * COMPLEXITY:
 *     Same as PrefetchScheduler constructor.
 * -------------------------------------------------------------------------- */
std::shared_ptr<PrefetchScheduler> make_scheduler(
    std::size_t num_pages,
    std::size_t max_resident = 64,
    std::size_t num_workers = 0,
    BlockMode block_mode = BlockMode::Hybrid
);

} // namespace scl::mmap

