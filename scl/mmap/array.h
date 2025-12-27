// =============================================================================
// FILE: scl/mmap/array.h
// BRIEF: API reference for memory-mapped virtual arrays with prefetch scheduling
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <functional>
#include <memory>

namespace scl::mmap {

// Forward declarations
class PrefetchScheduler;
class PageStore;
class SchedulerStats;

// Type aliases (from page.h)
using LoadCallback = std::function<void(std::size_t page_idx, std::byte* dest)>;
using WriteCallback = std::function<void(std::size_t page_idx, const std::byte* src)>;

/* =============================================================================
 * CLASS: MmapArray<T>
 * =============================================================================
 * SUMMARY:
 *     Virtual array view with transparent paging and event-driven prefetch.
 *
 * DESIGN PURPOSE:
 *     Provides array-like interface over large datasets that may not fit in RAM:
 *     - Transparent paging: Access any element via operator[]
 *     - Event-driven prefetch: Automatic prefetch based on computation events
 *     - Pluggable policies: Customizable prefetch strategies
 *     - Write-through: Optional write-back to storage
 *
 * ARCHITECTURE:
 *     MmapArray (lightweight view)
 *         ↓ uses
 *     PrefetchScheduler (event-driven prefetch, worker threads)
 *         ↓ requests pages from
 *     GlobalPagePool (system-wide deduplication)
 *         ↓ loads data via
 *     PageStore (I/O backend callbacks)
 *
 * MEMORY OVERHEAD:
 *     Per MmapArray: ~48 bytes (3 shared_ptrs + metadata)
 *     Per Page: kPageSize + 64 bytes (data + control)
 *     Total = (resident_pages * kPageSize) + overhead
 *
 * THREAD SAFETY:
 *     - Read operations: Thread-safe (concurrent access allowed)
 *     - Write operations: Thread-safe (atomic dirty tracking)
 *     - Prefetch: Thread-safe (multi-threaded prefetch workers)
 *
 * CONSTRAINTS:
 *     T must be trivially copyable (verified at compile time).
 * -------------------------------------------------------------------------- */
template <typename T>
class MmapArray {
public:
    using value_type = T;
    using ValueType = T;

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: MmapArray(scheduler, num_elements, loader, writer)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Construct with existing PrefetchScheduler.
     *
     * PARAMETERS:
     *     scheduler    [in] - Shared PrefetchScheduler instance
     *     num_elements [in] - Number of T elements in array
     *     loader      [in] - Page load callback
     *     writer      [in] - Optional page write callback
     *
     * PRECONDITIONS:
     *     - scheduler != nullptr
     *     - num_elements > 0
     *     - num_elements * sizeof(T) does not overflow
     *     - loader != nullptr
     *
     * POSTCONDITIONS:
     *     - Array ready for access
     *     - size() returns num_elements
     *     - PageStore registered with scheduler
     *
     * THROWS:
     *     ValueError if preconditions violated.
     *
     * COMPLEXITY:
     *     Time: O(1)
     *     Space: O(1)
     * ---------------------------------------------------------------------- */
    explicit MmapArray(
        std::shared_ptr<PrefetchScheduler> scheduler,
        std::size_t num_elements,
        LoadCallback loader,
        WriteCallback writer = nullptr
    );

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: MmapArray(num_elements, loader, writer, max_resident, num_workers)
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Construct with auto-created PrefetchScheduler.
     *
     * PARAMETERS:
     *     num_elements  [in] - Number of T elements in array
     *     loader        [in] - Page load callback
     *     writer        [in] - Optional page write callback
     *     max_resident  [in] - Maximum resident pages (default: 64)
     *     num_workers   [in] - Worker threads (0 = auto, default: 0)
     *
     * ALGORITHM:
     *     1. Compute total_pages = ceiling(num_elements * sizeof(T) / kPageSize)
     *     2. Create PrefetchScheduler with total_pages, max_resident, num_workers
     *     3. Generate unique file_id
     *     4. Create PageStore with loader/writer
     *     5. Register PageStore with scheduler
     *
     * PRECONDITIONS:
     *     - num_elements > 0
     *     - num_elements * sizeof(T) does not overflow
     *     - loader != nullptr
     *
     * POSTCONDITIONS:
     *     - Array ready for access
     *     - Scheduler created and worker threads started
     *
     * THROWS:
     *     ValueError if preconditions violated.
     *
     * COMPLEXITY:
     *     Time: O(total_pages) - scheduler initialization
     *     Space: O(total_pages) - cache entry array
     * ---------------------------------------------------------------------- */
    explicit MmapArray(
        std::size_t num_elements,
        LoadCallback loader,
        WriteCallback writer = nullptr,
        std::size_t max_resident = 64,
        std::size_t num_workers = 0
    );

    ~MmapArray() = default;
    
    MmapArray(const MmapArray&) = delete;
    MmapArray& operator=(const MmapArray&) = delete;
    MmapArray(MmapArray&&) noexcept = default;
    MmapArray& operator=(MmapArray&&) noexcept = default;

    /* -------------------------------------------------------------------------
     * METHODS: ArrayLike Interface
     * -------------------------------------------------------------------------
     * size:  Number of elements
     * empty: Check if array is empty
     *
     * COMPLEXITY:
     *     O(1)
     *
     * THREAD SAFETY:
     *     Thread-safe (const methods)
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    /* -------------------------------------------------------------------------
     * OPERATOR: operator[]
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Element access with transparent paging (may load page).
     *
     * ALGORITHM:
     *     1. Compute byte_offset = i * sizeof(T)
     *     2. Compute page_idx = byte_to_page_idx(byte_offset)
     *     3. Compute page_off = byte_to_page_offset(byte_offset)
     *     4. Request page from PrefetchScheduler (may load, may evict)
     *     5. Return element at page_off within page
     *
     * PRECONDITIONS:
     *     i < size()
     *
     * POSTCONDITIONS:
     *     Returns element value.
     *     On error: Returns T{} (zero-initialized).
     *
     * THREAD SAFETY:
     *     Safe - uses PageHandle RAII pinning
     *
     * PERFORMANCE:
     *     - Cache hit: ~10ns (atomic load + pointer arithmetic)
     *     - Cache miss: ~100us - 10ms (page load + possible eviction)
     *
     * ERROR HANDLING:
     *     Returns T{} on:
     *     - Index out of bounds
     *     - Integer overflow (i > SIZE_MAX / sizeof(T))
     *     - Page load failure
     *
     * WARNING:
     *     Repeated random access may thrash cache. Use prefetch hints.
     * ---------------------------------------------------------------------- */
    [[nodiscard]] T operator[](std::size_t i) const;

    /* -------------------------------------------------------------------------
     * METHOD: at
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Bounds-checked element access.
     *
     * THROWS:
     *     DimensionError if i >= size().
     *
     * OTHERWISE:
     *     Identical to operator[].
     * ---------------------------------------------------------------------- */
    [[nodiscard]] T at(std::size_t i) const;

    /* -------------------------------------------------------------------------
     * METHOD: read_range
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Batch read with SIMD optimization and automatic prefetch hints.
     *
     * ALGORITHM:
     *     1. Hint scheduler about access pattern (hint_range)
     *     2. For each page spanning [start, start+count):
     *        a. Request page from scheduler
     *        b. Compute elements in this page
     *        c. SIMD copy to output buffer
     *        d. Advance to next page
     *
     * PARAMETERS:
     *     start [in]  - Start element index
     *     count [in]  - Number of elements to read
     *     out   [out] - Output buffer (must have space for count elements)
     *
     * PRECONDITIONS:
     *     - start < size()
     *     - out != nullptr
     *
     * POSTCONDITIONS:
     *     out[0..count-1] contains array elements [start, start+count).
     *     If start + count > size(): count clamped to valid range.
     *
     * PERFORMANCE:
     *     Uses scl::memory::copy_fast():
     *     - 4-way unrolled SIMD (AVX2/NEON)
     *     - Non-temporal stores for large copies
     *     - Approximately 3-5x faster than naive loop
     *     - Prefetch hints improve cache hit rate
     *
     * THREAD SAFETY:
     *     Safe - uses PageHandle pinning
     *
     * COMPLEXITY:
     *     Time: O(count + num_pages_spanned)
     *     Space: O(1)
     *
     * ERROR HANDLING:
     *     - Clamps count to valid range
     *     - Zeros output on page load failure
     *     - Detects integer overflow (breaks loop)
     * ---------------------------------------------------------------------- */
    void read_range(std::size_t start, std::size_t count, T* out) const;

    /* -------------------------------------------------------------------------
     * METHOD: write_range
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Batch write with automatic dirty tracking.
     *
     * ALGORITHM:
     *     For each page spanning [start, start+count):
     *         1. Request page from scheduler
     *         2. SIMD copy from input buffer to page
     *         3. Mark page as dirty (requires writeback)
     *         4. Advance to next page
     *
     * PARAMETERS:
     *     start [in] - Start element index
     *     count [in] - Number of elements to write
     *     in    [in] - Input buffer (must contain count elements)
     *
     * PRECONDITIONS:
     *     - start < size()
     *     - in != nullptr
     *     - PageStore must have write_cb (otherwise writes discarded)
     *
     * POSTCONDITIONS:
     *     Array elements [start, start+count) updated.
     *     Affected pages marked dirty.
     *
     * MUTABILITY:
     *     INPLACE - modifies array contents
     *
     * PERFORMANCE:
     *     Same SIMD optimizations as read_range().
     *
     * THREAD SAFETY:
     *     Safe - uses PageHandle pinning + atomic dirty flag
     * ---------------------------------------------------------------------- */
    void write_range(std::size_t start, std::size_t count, const T* in);

    /* -------------------------------------------------------------------------
     * METHOD: fill_range
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Fill range with single value (SIMD optimized).
     *
     * ALGORITHM:
     *     For each page spanning [start, start+count):
     *         1. Request page from scheduler
     *         2. SIMD fill with value (Set + Store intrinsics)
     *         3. Mark page as dirty
     *
     * PARAMETERS:
     *     start [in] - Start element index
     *     count [in] - Number of elements to fill
     *     value [in] - Fill value
     *
     * PERFORMANCE:
     *     Uses scl::memory::fill():
     *     - Broadcast value to SIMD register
     *     - 4-way unrolled Store operations
     *     - Approximately 4x faster than std::fill
     *
     * THREAD SAFETY:
     *     Safe
     * ---------------------------------------------------------------------- */
    void fill_range(std::size_t start, std::size_t count, T value);

    /* -------------------------------------------------------------------------
     * METHOD: copy_to
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Copy range to another MmapArray (handles different schedulers).
     *
     * ALGORITHM:
     *     Uses heap-allocated buffer (4KB) as intermediate:
     *     While remaining > 0:
     *         chunk = min(remaining, 4096 / sizeof(T))
     *         this->read_range(src_pos, chunk, buffer)
     *         dest.write_range(dest_pos, chunk, buffer)
     *
     * PARAMETERS:
     *     src_start  [in]    - Start index in source (this)
     *     count      [in]    - Number of elements to copy
     *     dest       [inout] - Destination array
     *     dest_start [in]    - Start index in destination
     *
     * PRECONDITIONS:
     *     - src_start < this->size()
     *     - dest_start < dest.size()
     *
     * POSTCONDITIONS:
     *     dest[dest_start..dest_start+count) = this[src_start..src_start+count)
     *     Count clamped to fit both source and destination.
     *
     * PERFORMANCE:
     *     Buffering prevents page ping-pong between schedulers.
     *     SIMD copy in both read and write phases.
     *
     * THREAD SAFETY:
     *     Safe - independent schedulers
     * ---------------------------------------------------------------------- */
    void copy_to(
        std::size_t src_start,
        std::size_t count,
        MmapArray<T>& dest,
        std::size_t dest_start
    ) const;

    /* -------------------------------------------------------------------------
     * METHODS: Prefetch Hints
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Provide access pattern hints to scheduler for prefetch optimization.
     *
     * hint_range:
     *     Hint scheduler about sequential range access.
     *     Limits to kMaxHintPages (256) to prevent excessive allocation.
     *
     * hint_priority:
     *     Boost priority for specific element (urgent prefetch).
     *
     * prefetch_sequential:
     *     Sequential prefetch with window size (delegates to hint_range).
     *
     * prefetch_strided:
     *     Prefetch strided access pattern with deduplication.
     *     - Collects unique pages from strided elements
     *     - Sorts and deduplicates (handles non-adjacent duplicates)
     *     - Limits to kMaxHintPages
     *
     * THREAD SAFETY:
     *     Thread-safe (adds hints to scheduler queue).
     *
     * COMPLEXITY:
     *     hint_range: O(hint_pages)
     *     prefetch_strided: O(count + pages log pages) - sorting overhead
     * ---------------------------------------------------------------------- */
    void hint_range(std::size_t start, std::size_t count) const;
    void hint_priority(std::size_t elem_idx) const;
    void prefetch_sequential(std::size_t start, std::size_t window_size);
    void prefetch_strided(std::size_t start, std::size_t count, std::size_t stride);

    /* -------------------------------------------------------------------------
     * METHODS: Compute Hooks Integration
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Integrate with computation loops for event-driven prefetch.
     *
     * begin_computation:
     *     Signal start of computation (resets scheduler state).
     *
     * end_computation:
     *     Signal end of computation (syncs all pending operations).
     *
     * begin_row / end_row:
     *     Signal row processing boundaries (triggers prefetch for next rows).
     *
     * sync_barrier:
     *     Wait until all pending fetches complete.
     *
     * sync_fence:
     *     Wait until pages up to row are loaded (with timeout).
     *
     * flush:
     *     Sync all pending operations (same as sync_barrier).
     *
     * THREAD SAFETY:
     *     Called from computation thread(s). Methods are thread-safe.
     *
     * USAGE PATTERN:
     *     arr.begin_computation(total_rows);
     *     for (size_t row = 0; row < total_rows; ++row) {
     *         arr.begin_row(row);
     *         // Process row...
     *         arr.end_row(row);
     *     }
     *     arr.end_computation();
     * ---------------------------------------------------------------------- */
    void begin_computation(std::size_t total_rows);
    void end_computation();
    void begin_row(std::size_t row);
    void end_row(std::size_t row);
    void sync_barrier();
    bool sync_fence(std::size_t up_to_row, std::chrono::milliseconds timeout = std::chrono::milliseconds{5000});
    void flush();

    /* -------------------------------------------------------------------------
     * METHODS: Policy Control
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Configure prefetch scheduling policy.
     *
     * set_policy:
     *     Set custom scheduling policy (template or unique_ptr overload).
     *
     * set_lookahead:
     *     Set prefetch lookahead depth (delegates to policy).
     *
     * set_batch_size:
     *     Set prefetch batch size (delegates to policy).
     *
     * set_block_mode:
     *     Set synchronization blocking mode (SpinWait, ConditionWait, Hybrid, Callback).
     *
     * THREAD SAFETY:
     *     Thread-safe (protected by scheduler locks).
     * ---------------------------------------------------------------------- */
    template <typename Policy, typename... Args>
    void set_policy(Args&&... args);
    
    void set_lookahead(std::size_t lookahead);
    void set_batch_size(std::size_t batch);
    void set_block_mode(BlockMode mode);

    /* -------------------------------------------------------------------------
     * METHODS: Statistics
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Access scheduler performance metrics.
     *
     * resident_pages:  Pages currently in cache
     * pending_pages:    Pages currently being fetched
     * total_pages:      Total pages in file
     * stats:            Full scheduler statistics (hits, misses, latency, etc.)
     * hit_rate:         Cache hit rate (0.0 - 1.0)
     *
     * COMPLEXITY:
     *     All O(1) (atomic counters).
     *
     * THREAD SAFETY:
     *     Safe (const methods, atomic loads)
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::size_t resident_pages() const noexcept;
    [[nodiscard]] std::size_t pending_pages() const noexcept;
    [[nodiscard]] std::size_t total_pages() const noexcept;
    [[nodiscard]] const SchedulerStats& stats() const noexcept;
    [[nodiscard]] double hit_rate() const noexcept;

    /* -------------------------------------------------------------------------
     * METHODS: Accessors
     * -------------------------------------------------------------------------
     * scheduler: Get PrefetchScheduler pointer (for advanced control)
     * store:     Get PageStore pointer (for direct I/O access)
     *
     * THREAD SAFETY:
     *     Safe (returns shared_ptr internal pointer).
     * ---------------------------------------------------------------------- */
    [[nodiscard]] PrefetchScheduler* scheduler() noexcept;
    [[nodiscard]] const PrefetchScheduler* scheduler() const noexcept;
    [[nodiscard]] PageStore* store() noexcept;
    [[nodiscard]] const PageStore* store() const noexcept;
};

/* =============================================================================
 * CLASS: ScopedComputation<T>
 * =============================================================================
 * SUMMARY:
 *     RAII guard for computation lifecycle (begin/end).
 *
 * DESIGN PURPOSE:
 *     Ensures end_computation() is called even on exception.
 *
 * USAGE:
 *     {
 *         ScopedComputation guard(arr, total_rows);
 *         // Computation code...
 *     }  // guard destructor calls end_computation()
 *
 * THREAD SAFETY:
 *     Not thread-safe. One guard per computation.
 * -------------------------------------------------------------------------- */
template <typename T>
class ScopedComputation {
public:
    explicit ScopedComputation(MmapArray<T>& array, std::size_t total_rows);
    ~ScopedComputation();
    
    ScopedComputation(const ScopedComputation&) = delete;
    ScopedComputation& operator=(const ScopedComputation&) = delete;
    ScopedComputation(ScopedComputation&&) noexcept;
    ScopedComputation& operator=(ScopedComputation&&) noexcept;
};

/* =============================================================================
 * CLASS: ScopedRow<T>
 * =============================================================================
 * SUMMARY:
 *     RAII guard for row processing lifecycle (begin_row/end_row).
 *
 * DESIGN PURPOSE:
 *     Ensures end_row() is called even on exception.
 *
 * USAGE:
 *     for (size_t row = 0; row < total_rows; ++row) {
 *         ScopedRow guard(arr, row);
 *         // Process row...
 *     }  // guard destructor calls end_row()
 *
 * THREAD SAFETY:
 *     Not thread-safe. One guard per row.
 * -------------------------------------------------------------------------- */
template <typename T>
class ScopedRow {
public:
    explicit ScopedRow(MmapArray<T>& array, std::size_t row);
    ~ScopedRow();
    
    ScopedRow(const ScopedRow&) = delete;
    ScopedRow& operator=(const ScopedRow&) = delete;
    ScopedRow(ScopedRow&&) noexcept;
    ScopedRow& operator=(ScopedRow&&) noexcept;
    
    [[nodiscard]] std::size_t row() const noexcept;
};

/* =============================================================================
 * TYPE ALIASES
 * =============================================================================
 * SUMMARY:
 *     Common instantiations for standard numeric types.
 * -------------------------------------------------------------------------- */
using MmapArrayReal = MmapArray<Real>;
using MmapArrayF32 = MmapArray<float>;
using MmapArrayF64 = MmapArray<double>;
using MmapArrayIndex = MmapArray<Index>;
using MmapArrayI32 = MmapArray<std::int32_t>;
using MmapArrayI64 = MmapArray<std::int64_t>;

/* =============================================================================
 * FACTORY FUNCTIONS
 * =============================================================================
 * SUMMARY:
 *     Convenience functions for creating MmapArray instances.
 *
 * make_mmap_array (auto scheduler):
 *     Creates MmapArray with auto-created PrefetchScheduler.
 *
 * make_mmap_array (with scheduler):
 *     Creates MmapArray with existing PrefetchScheduler.
 *
 * RETURNS:
 *     shared_ptr to MmapArray instance.
 *
 * COMPLEXITY:
 *     Same as MmapArray constructors.
 * -------------------------------------------------------------------------- */
template <typename T>
[[nodiscard]] std::shared_ptr<MmapArray<T>> make_mmap_array(
    std::size_t num_elements,
    LoadCallback loader,
    WriteCallback writer = nullptr,
    std::size_t max_resident = 64,
    std::size_t num_workers = 0
);

template <typename T>
[[nodiscard]] std::shared_ptr<MmapArray<T>> make_mmap_array(
    std::shared_ptr<PrefetchScheduler> scheduler,
    std::size_t num_elements,
    LoadCallback loader,
    WriteCallback writer = nullptr
);

} // namespace scl::mmap
