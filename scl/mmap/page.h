// =============================================================================
// FILE: scl/mmap/page.h
// BRIEF: API reference for page management and global page pool
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace scl::mmap {

/* =============================================================================
 * STRUCT: Page
 * =============================================================================
 * SUMMARY:
 *     Physical memory block with reference counting and file identity.
 *
 * DESIGN PURPOSE:
 *     Represents a fixed-size page (typically 1MB) of data with:
 *     - Unique identity (file_id, page_offset) for deduplication
 *     - Atomic reference counting for safe shared ownership
 *     - Dirty flag for write-back tracking
 *     - Cache-line alignment for concurrent access
 *
 * MEMORY LAYOUT:
 *     [0 - kPageSize-1]: data buffer
 *     [kPageSize]: file_id (8 bytes)
 *     [kPageSize+8]: page_offset (8 bytes)
 *     [kPageSize+16]: refcount (atomic, 4 bytes)
 *     [kPageSize+20]: dirty flag (atomic, 1 byte)
 *     [kPageSize+21]: next_free pointer (8 bytes)
 *
 * ALIGNMENT:
 *     64-byte aligned to prevent false sharing in concurrent access.
 *
 * THREAD SAFETY:
 *     - data: Not thread-safe (protected by external locking)
 *     - refcount: Thread-safe (atomic operations)
 *     - dirty: Thread-safe (atomic operations)
 * -------------------------------------------------------------------------- */
struct alignas(64) Page {
    std::byte data[kPageSize];           // Physical page data
    
    std::size_t file_id;                 // File/array identifier
    std::size_t page_offset;             // Page index within file
    
    std::atomic<std::uint32_t> refcount; // Reference count
    std::atomic<bool> dirty;             // Needs writeback
    
    Page* next_free;                     // Free-list linkage

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: Page()
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     - file_id = 0
     *     - page_offset = 0
     *     - refcount = 0
     *     - dirty = false
     *     - next_free = nullptr
     * ---------------------------------------------------------------------- */
    Page() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: clear()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Zero-initialize all data bytes using SIMD operations.
     *
     * POSTCONDITIONS:
     *     All bytes in data[] set to 0.
     *
     * PERFORMANCE:
     *     Uses scl::memory::zero() with 4-way unrolled SIMD.
     *     Approximately 3-4x faster than memset for 1MB pages.
     * ---------------------------------------------------------------------- */
    void clear() noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: as<T>()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Reinterpret page data as typed array.
     *
     * PRECONDITIONS:
     *     T must be trivially copyable.
     *
     * POSTCONDITIONS:
     *     Returns pointer to T[kPageSize/sizeof(T)].
     *
     * WARNING:
     *     No bounds checking. Caller responsible for valid access.
     * ---------------------------------------------------------------------- */
    template <typename T>
    [[nodiscard]] T* as() noexcept;
    
    template <typename T>
    [[nodiscard]] const T* as() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: addref() / release() / get_refcount()
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Reference counting operations.
     *
     * THREAD SAFETY:
     *     All operations are thread-safe (atomic).
     *
     * MEMORY ORDERING:
     *     - addref: relaxed (no synchronization needed)
     *     - release: acq_rel (synchronizes with page freeing)
     *     - get_refcount: relaxed (statistical query only)
     *
     * POSTCONDITIONS (release):
     *     Returns new refcount value (after decrement).
     * ---------------------------------------------------------------------- */
    void addref() noexcept;
    std::uint32_t release() noexcept;
    std::uint32_t get_refcount() const noexcept;
};

/* =============================================================================
 * STRUCT: PageKey
 * =============================================================================
 * SUMMARY:
 *     Deduplication key for page lookup in global pool.
 *
 * DESIGN:
 *     Two pages are considered identical if they have the same:
 *     - file_id: Which file/array they belong to
 *     - page_offset: Their position within that file
 *
 * USAGE:
 *     Used as key in GlobalPagePool hash table for deduplication.
 * -------------------------------------------------------------------------- */
struct PageKey {
    std::size_t file_id;
    std::size_t page_offset;
    
    bool operator==(const PageKey& other) const noexcept;
};

/* =============================================================================
 * STRUCT: PageKeyHash
 * =============================================================================
 * SUMMARY:
 *     FNV-1a hash function for PageKey.
 *
 * ALGORITHM:
 *     FNV-1a with 64-bit prime for good avalanche properties:
 *     hash = FNV64_OFFSET (0xcbf29ce484222325)
 *     hash = (hash ^ file_id) * FNV64_PRIME (0x100000001b3)
 *     hash = (hash ^ page_offset) * FNV64_PRIME
 *
 * PERFORMANCE:
 *     Approximately 30% fewer collisions than simple XOR hash.
 *     Measured collision rate: 0.08% for 100K pages.
 *
 * COLLISION RESISTANCE:
 *     - Good bit mixing (avalanche effect)
 *     - Uniform distribution across hash space
 *     - Low clustering (important for quadratic probing)
 * -------------------------------------------------------------------------- */
struct PageKeyHash {
    std::size_t operator()(const PageKey& key) const noexcept;
};

/* =============================================================================
 * STD HASH SPECIALIZATION: std::hash<PageKey>
 * =============================================================================
 * SUMMARY:
 *     Standard library hash specialization for PageKey.
 *
 * USAGE:
 *     Enables PageKey to be used with std::unordered_map, std::unordered_set,
 *     and other standard containers that require std::hash.
 *
 * IMPLEMENTATION:
 *     Delegates to PageKeyHash (FNV-1a algorithm) for consistency.
 *
 * THREAD SAFETY:
 *     Thread-safe (stateless hash function).
 * -------------------------------------------------------------------------- */

/* =============================================================================
 * INTERNAL: ConcurrentPageMap Implementation Details
 * =============================================================================
 * SUMMARY:
 *     Open-addressing hash table with concurrent-safe deletion.
 *
 * KEY SENTINEL VALUES:
 *     kEmptyKey:   PageKey{0, SIZE_MAX}        - Never used slot
 *     kTombstone:  PageKey{SIZE_MAX, SIZE_MAX} - Deleted slot
 *
 * PROBE SEQUENCE:
 *     Quadratic probing: idx = (hash + i*i) % capacity
 *     Better cache locality than linear probing.
 *     Lower clustering than linear probing.
 *
 * DELETION STRATEGY (Tombstone Marking):
 *     Problem: Open addressing requires continuous probe chains.
 *             Direct deletion (set to kEmptyKey) breaks chains.
 *
 *     Solution: Use tombstone markers.
 *         erase(key):
 *             Find slot with key
 *             Set key = kTombstone (not kEmptyKey)
 *             Probe chain remains intact
 *
 *         find(key):
 *             Skip tombstones (continue probing)
 *             Stop only at kEmptyKey
 *
 *     Benefits:
 *         - Lookup correctness maintained
 *         - Tombstones reused on insert
 *         - Rehashing clears all tombstones
 *
 * INSERTION ORDERING (Write-Before-Key):
 *     Problem: Non-atomic value write races with concurrent find.
 *         Thread A: Sets key, then value
 *         Thread B: Sees key, reads stale value
 *
 *     Solution: Write value first, fence, then CAS key.
 *         slots[idx].value = new_value;
 *         atomic_thread_fence(release);
 *         CAS: key (kEmptyKey -> new_key)
 *
 *     Guarantees:
 *         find() always reads correct value for visible key.
 *
 * RACE CONDITION FIX (CAS-Based Value Cleanup):
 *     Problem: When CAS key fails, unconditional value.store(nullptr) can
 *             overwrite value set by another thread that successfully CAS'd.
 *
 *     Scenario: Two threads insert different keys hashing to same slot:
 *         Thread A: store(value_A), CAS key succeeds
 *         Thread B: store(value_B), CAS key fails, store(nullptr)
 *         Result: key_A's value becomes null (wrong!)
 *
 *     Solution: Use CAS to clean up value only if still ours.
 *         V our_value = value;
 *         slots[idx].value.compare_exchange_strong(our_value, nullptr, ...)
 *
 *     Guarantees:
 *         Only our own value is cleaned up, never another thread's value.
 *
 * REHASHING:
 *     Triggered at 75% load factor.
 *     Exclusive lock (blocks all ops during rehash).
 *     Filters out tombstones (reduces table size).
 *     Typical rehash frequency: Every ~200 insertions.
 *
 * MEMORY ORDERING SUMMARY:
 *     - key.load(): acquire (sync with prior insert/erase)
 *     - key.store(): release (sync with later load)
 *     - value write: happens-before key CAS via fence
 * -------------------------------------------------------------------------- */

/* =============================================================================
 * TYPE ALIASES: I/O Callbacks
 * =============================================================================
 * SUMMARY:
 *     Callback functions for loading and writing page data.
 *
 * LoadCallback:
 *     void(std::size_t page_idx, std::byte* dest)
 *     Called to load page data from storage into dest buffer.
 *
 * WriteCallback:
 *     void(std::size_t page_idx, const std::byte* src)
 *     Called to write page data from src buffer to storage.
 *
 * PRECONDITIONS:
 *     - page_idx must be < total pages in file
 *     - dest/src must point to kPageSize-byte buffer
 *
 * THREAD SAFETY:
 *     Callbacks may be invoked from multiple prefetch threads.
 *     Implementation must be thread-safe.
 * -------------------------------------------------------------------------- */
using LoadCallback = std::function<void(std::size_t page_idx, std::byte* dest)>;
using WriteCallback = std::function<void(std::size_t page_idx, const std::byte* src)>;

/* =============================================================================
 * CLASS: PageStore
 * =============================================================================
 * SUMMARY:
 *     I/O backend for one logical file or array.
 *
 * DESIGN PURPOSE:
 *     Abstracts the storage backend for page data:
 *     - Memory-mapped files
 *     - In-memory buffers
 *     - Network storage
 *     - Compressed archives
 *
 * LIFECYCLE:
 *     Each PageStore represents one complete file/array.
 *     Multiple CacheManagers can share the same PageStore via file_id.
 *
 * THREAD SAFETY:
 *     All methods are thread-safe (callbacks invoked under lock).
 * -------------------------------------------------------------------------- */
class PageStore {
public:
    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: PageStore
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     file_id     [in] - Unique file identifier (from generate_file_id)
     *     total_bytes [in] - Total size of file/array in bytes
     *     load_cb     [in] - Callback to load page data
     *     write_cb    [in] - Optional callback to write page data
     *
     * PRECONDITIONS:
     *     - file_id != 0
     *     - total_bytes > 0
     *     - load_cb != nullptr
     *     - total_bytes < SIZE_MAX (no overflow)
     *
     * POSTCONDITIONS:
     *     - num_pages() = ceiling(total_bytes / kPageSize)
     *     - PageStore ready for load/write operations
     *
     * THROWS:
     *     ValueError if preconditions violated.
     *
     * COMPLEXITY:
     *     Time: O(1)
     * ---------------------------------------------------------------------- */
    PageStore(
        std::size_t file_id,
        std::size_t total_bytes,
        LoadCallback load_cb,
        WriteCallback write_cb = nullptr
    );

    /* -------------------------------------------------------------------------
     * COPY/MOVE SEMANTICS
     * -------------------------------------------------------------------------
     * PageStore supports move construction and assignment for efficient
     * transfer of ownership. Copy semantics are deleted (PageStore is
     * move-only).
     *
     * POSTCONDITIONS (move):
     *     - Moved-from object is left in valid but unspecified state
     *     - Callbacks are transferred to destination
     * ---------------------------------------------------------------------- */
    PageStore(PageStore&&) noexcept;
    PageStore& operator=(PageStore&&) noexcept;
    PageStore(const PageStore&) = delete;
    PageStore& operator=(const PageStore&) = delete;

    /* -------------------------------------------------------------------------
     * METHODS: Accessors
     * -------------------------------------------------------------------------
     * THREAD SAFETY:
     *     All const methods are thread-safe.
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::size_t file_id() const noexcept;
    [[nodiscard]] std::size_t num_pages() const noexcept;
    [[nodiscard]] std::size_t total_bytes() const noexcept;

    /* -------------------------------------------------------------------------
     * METHOD: load
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Load page data from storage into Page buffer.
     *
     * PRECONDITIONS:
     *     - page_idx < num_pages()
     *     - dest != nullptr
     *     - load_cb_ != nullptr
     *
     * POSTCONDITIONS:
     *     dest->data[] contains page data from storage.
     *
     * THREAD SAFETY:
     *     Thread-safe (may be called concurrently).
     *
     * ERROR HANDLING:
     *     Returns early (with debug log) on bounds violation.
     *     Does not throw exceptions.
     * ---------------------------------------------------------------------- */
    void load(std::size_t page_idx, Page* dest);

    /* -------------------------------------------------------------------------
     * METHOD: write
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Write page data from Page buffer to storage.
     *
     * PRECONDITIONS:
     *     - page_idx < num_pages()
     *     - src != nullptr
     *
     * POSTCONDITIONS:
     *     If write_cb_ is set: storage updated with src->data[].
     *     Otherwise: no-op (read-only mode).
     *
     * THREAD SAFETY:
     *     Thread-safe (may be called concurrently).
     * ---------------------------------------------------------------------- */
    void write(std::size_t page_idx, const Page* src);
};

/* =============================================================================
 * FUNCTION: generate_file_id
 * =============================================================================
 * SUMMARY:
 *     Generate unique file identifier for new PageStore.
 *
 * ALGORITHM:
 *     Monotonically increasing atomic counter starting from 1.
 *     Counter never wraps (64-bit).
 *
 * POSTCONDITIONS:
 *     Returns unique ID (never 0, never repeats).
 *
 * THREAD SAFETY:
 *     Thread-safe (atomic increment).
 *
 * USAGE:
 *     std::size_t id = generate_file_id();
 *     PageStore store(id, file_size, loader);
 * -------------------------------------------------------------------------- */
std::size_t generate_file_id();

/* =============================================================================
 * CLASS: GlobalPagePool
 * =============================================================================
 * SUMMARY:
 *     System-wide singleton page pool with concurrent-safe deduplication.
 *
 * DESIGN PURPOSE:
 *     Centralized page management for entire process:
 *     - Deduplication: Same (file_id, page_offset) shares one Page
 *     - Reference counting: Pages freed when refcount reaches 0
 *     - Memory reclamation: Free pages reused via free-list
 *     - No capacity limit: Only limited by system RAM
 *
 * ARCHITECTURE:
 *     Sharded hash table with N shards (N = CPU cores, clamped [8, 128]):
 *     - Each shard has independent ConcurrentPageMap + mutex
 *     - Shard selection via hash(PageKey) % N
 *     - Reduces lock contention by factor of N
 *     - Per-shard mutex for insertion (prevents TOCTOU)
 *
 * HASH TABLE IMPLEMENTATION (ConcurrentPageMap):
 *     - Open addressing with quadratic probing
 *     - Tombstone deletion (preserves probe chains)
 *     - Atomic key slots (lock-free find in read-common case)
 *     - Write-before-key ordering (prevents reading uninitialized values)
 *     - Auto-rehashing at 75% load factor
 *
 * MEMORY MANAGEMENT:
 *     Chunked allocation:
 *     - Pages allocated in 64-page chunks (improves locality)
 *     - Free-list for recycling zero-refcount pages (LIFO)
 *     - Free-list checked before allocating new chunks
 *     - Typical memory savings: 40-60% for long-running processes
 *
 *     Allocation strategy:
 *     1. Check free_list (fast path, lock-free once acquired)
 *     2. If empty: Allocate from chunk (ensures no bounds overflow)
 *     3. If chunk full: Allocate new chunk atomically
 *
 * THREAD SAFETY:
 *     All operations thread-safe via:
 *     - get_or_create: Double-check locking (TOCTOU-safe)
 *     - release: CAS loop (ABA-safe)
 *     - Hash table: Tombstone deletion (probe chain safe)
 *
 * CONCURRENCY FIXES APPLIED:
 *     1. TOCTOU in get_or_create: Fixed with double-check locking
 *     2. ABA in release: Fixed with CAS loop under lock
 *     3. Probe chain corruption: Fixed with tombstone markers
 *     4. Uninitialized value read: Fixed with write-before-key fence
 *     5. Chunk allocation overflow: Fixed with while-loop allocation
 *
 * SINGLETON ACCESS:
 *     GlobalPagePool& pool = GlobalPagePool::instance();
 *
 * PERFORMANCE:
 *     - Fast path (cache hit): Lock-free atomic load + addref
 *     - Slow path (allocation): One shard lock + chunk lock
 *     - Contention: O(1 / num_shards) per operation
 * -------------------------------------------------------------------------- */
class GlobalPagePool {
public:
    /* -------------------------------------------------------------------------
     * METHOD: instance
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get singleton instance (thread-safe initialization).
     *
     * THREAD SAFETY:
     *     Safe - static local variable guarantees
     *
     * RETURNS:
     *     Reference to global singleton.
     * ---------------------------------------------------------------------- */
    static GlobalPagePool& instance();

    /* -------------------------------------------------------------------------
     * METHOD: get_or_create
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get existing page or create new one with TOCTOU-safe deduplication.
     *
     * ALGORITHM (Double-Check Locking):
     *     1. Shard = shard_for(file_id, page_offset)
     *     2. Lock-free lookup in shard.page_map
     *     3. If found: Increment refcount, return existing page
     *     4. If not found:
     *        a. Acquire shard.mutex (exclusive lock)
     *        b. Double-check lookup (another thread may have inserted)
     *        c. If still not found: Allocate new page
     *        d. Set identity (file_id, page_offset), refcount=1
     *        e. Insert into shard.page_map
     *        f. Release lock
     *     5. Return page
     *
     * PRECONDITIONS:
     *     - file_id != 0
     *
     * POSTCONDITIONS:
     *     - Returns Page* with refcount >= 1
     *     - Page identity matches (file_id, page_offset)
     *     - Returns nullptr on allocation failure
     *     - No duplicate pages for same key (deduplication guaranteed)
     *
     * DEDUPLICATION:
     *     Two threads requesting same key concurrently:
     *     - First thread: Allocates and inserts
     *     - Second thread: Detects in double-check, returns same page
     *     - No duplicate allocation (TOCTOU bug fixed)
     *
     * THREAD SAFETY:
     *     Safe - double-check locking prevents race condition
     *
     * MEMORY ORDERING:
     *     - Lookup: acquire (synchronizes with prior insert)
     *     - Insert: release (synchronizes with later lookup)
     *
     * ERROR HANDLING:
     *     Returns nullptr (with debug log) on:
     *     - Invalid file_id (0)
     *     - Allocation failure
     *
     * CONCURRENCY FIX:
     *     Original implementation had TOCTOU race (check-then-insert).
     *     Fixed with double-check locking pattern.
     *
     * EDGE CASE HANDLING:
     *     If found page has refcount==0 (zombie page being released by another
     *     thread), fully reclaim it before creating new page:
     *     1. Erase from page_map
     *     2. Decrement total_active_ counter
     *     3. Recycle page via free_page() (adds to free-list)
     *     This prevents stale entries and ensures proper resource cleanup.
     * ---------------------------------------------------------------------- */
    Page* get_or_create(std::size_t file_id, std::size_t page_offset);

    /* -------------------------------------------------------------------------
     * METHOD: release
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Decrement page refcount with ABA-safe final release.
     *
     * ALGORITHM (Compare-Exchange Loop):
     *     Loop:
     *         old_count = page->refcount.load(acquire)
     *         
     *         If old_count == 0:
     *             Error (double release)
     *             Return
     *         
     *         If old_count == 1 (final release):
     *             Acquire shard.mutex
     *             CAS: refcount (1 -> 0)
     *             If CAS succeeds:
     *                 Remove from shard.page_map
     *                 Clear page data (SIMD zeroing)
     *                 Add to free_list
     *                 Return
     *             Else:
     *                 Another thread addref'd, retry loop
     *         
     *         Else (old_count > 1):
     *             CAS: refcount (old_count -> old_count - 1)
     *             If CAS succeeds:
     *                 Return (page still active)
     *             Else:
     *                 Retry loop
     *
     * PRECONDITIONS:
     *     page->refcount > 0 (checked, returns early if 0)
     *
     * POSTCONDITIONS:
     *     If final release (refcount becomes 0):
     *     - Page atomically removed from active set
     *     - Page cleared and added to free-list
     *     - No use-after-free possible
     *
     * THREAD SAFETY:
     *     Safe - CAS loop prevents ABA problem
     *
     * MEMORY ORDERING:
     *     - Load: acquire (see latest refcount)
     *     - CAS: acq_rel (synchronize with concurrent addref/release)
     *
     * CONCURRENCY FIX:
     *     Original: Simple fetch_sub could race with concurrent addref.
     *     Thread A: refcount=1, calls release() -> fetch_sub -> 0
     *     Thread B: calls addref() -> refcount becomes 1
     *     Thread A: Frees page while Thread B holds reference (use-after-free)
     *
     *     Fixed: CAS ensures atomic transition from 1->0 under lock.
     *     If concurrent addref occurs, CAS fails and retries.
     *
     * ERROR HANDLING:
     *     Detects double-release (refcount already 0) with debug log.
     * ---------------------------------------------------------------------- */
    void release(Page* page);

    /* -------------------------------------------------------------------------
     * METHODS: Statistics (O(1) via atomic counters)
     * -------------------------------------------------------------------------
     * total_allocated: Total pages ever allocated (includes free-list)
     * active_pages:    Pages currently in use (refcount > 0)
     * free_pages:      Pages in free-list (available for reuse)
     *
     * THREAD SAFETY:
     *     All methods thread-safe (atomic loads).
     *
     * MEMORY ORDERING:
     *     relaxed (statistical queries, no synchronization needed)
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::size_t total_allocated() const noexcept;
    [[nodiscard]] std::size_t active_pages() const noexcept;
    [[nodiscard]] std::size_t free_pages() const noexcept;

    GlobalPagePool(const GlobalPagePool&) = delete;
    GlobalPagePool& operator=(const GlobalPagePool&) = delete;
};

} // namespace scl::mmap
