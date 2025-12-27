// =============================================================================
// FILE: scl/core/registry.h
// BRIEF: API reference for unified high-performance memory registry with reference counting
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace scl {

// =============================================================================
// TYPE ALIASES
// =============================================================================

using BufferID = std::uint64_t;

// =============================================================================
// FORWARD DECLARATIONS
// =============================================================================

class Registry;
Registry& get_registry();

namespace detail {
    std::size_t get_default_shard_count();
}

// =============================================================================
// ENUM: AllocType
// =============================================================================

/* -----------------------------------------------------------------------------
 * ENUM: AllocType
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Allocation type enumeration for memory tracking.
 *
 * VALUES:
 *     ArrayNew:     Allocated via new[] (requires delete[])
 *     ScalarNew:    Allocated via new (requires delete)
 *     AlignedAlloc: Allocated via aligned_alloc (requires aligned_free)
 *     Custom:       Custom allocation with user-provided deleter
 * -------------------------------------------------------------------------- */
enum class AllocType : std::uint8_t {
    ArrayNew = 0,
    ScalarNew = 1,
    AlignedAlloc = 2,
    Custom = 3
};

// =============================================================================
// CLASS: Registry
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: Registry
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Unified high-performance memory registry with reference counting.
 *
 * DESIGN PURPOSE:
 *     Provides thread-safe memory tracking and automatic cleanup for:
 *     - Simple pointers (refcount = 1, immediate cleanup)
 *     - Reference-counted buffers (multiple aliases, cleanup on last unregister)
 *
 * ARCHITECTURE:
 *     - Sharded design: Reduces lock contention via hash-based sharding
 *     - ConcurrentFlatMap: Open-addressing hash table with striped locks
 *     - Atomic counters: O(1) statistics without locking
 *     - RAII guards: Automatic cleanup via RegistryGuard
 *
 * THREAD SAFETY:
 *     All public methods are thread-safe. Internal synchronization uses:
 *     - Shared mutex for rehashing (readers can proceed concurrently)
 *     - Striped locks for slot-level access (reduces contention)
 *     - Atomic operations for counters and reference counts
 *
 * PERFORMANCE:
 *     - O(1) average case for insert/find/erase (hash table)
 *     - O(n) worst case (hash collision chain)
 *     - Sharding reduces lock contention by factor of num_shards
 *
 * MEMORY OVERHEAD:
 *     Per pointer: ~32 bytes (hash table slot + metadata)
 *     Per buffer: ~48 bytes (RefCountedBuffer + hash table slot)
 *     Per shard: ~64 bytes (cache line alignment)
 * -------------------------------------------------------------------------- */
class Registry {
public:
    using Deleter = void (*)(void*);

    /* -------------------------------------------------------------------------
     * STRUCT: PtrRecord
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Record for simple pointer tracking (refcount = 1).
     *
     * FIELDS:
     *     byte_size:      Size in bytes
     *     type:           Allocation type
     *     custom_deleter: Custom deleter function (if type == Custom)
     * ---------------------------------------------------------------------- */
    struct PtrRecord {
        std::uint64_t byte_size;
        AllocType type;
        Deleter custom_deleter;
    };

    /* -------------------------------------------------------------------------
     * STRUCT: BufferInfo
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Metadata for reference-counted buffer.
     *
     * FIELDS:
     *     real_ptr:       Actual allocated pointer
     *     byte_size:      Size in bytes
     *     type:           Allocation type
     *     custom_deleter: Custom deleter function (if type == Custom)
     * ---------------------------------------------------------------------- */
    struct BufferInfo {
        void* real_ptr;
        std::uint64_t byte_size;
        AllocType type;
        Deleter custom_deleter;
    };

    /* -------------------------------------------------------------------------
     * STRUCT: RefCountedBuffer
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Reference-counted buffer with atomic reference count.
     *
     * FIELDS:
     *     info:     Buffer metadata
     *     refcount: Atomic reference count
     *
     * THREAD SAFETY:
     *     refcount is atomic. Move operations are only safe under single
     *     ownership (e.g., via unique_ptr). Do NOT move instances that may
     *     be accessed concurrently.
     * ---------------------------------------------------------------------- */
    struct RefCountedBuffer {
        BufferInfo info;
        std::atomic<std::uint32_t> refcount{0};
        
        RefCountedBuffer() = default;
        RefCountedBuffer(BufferInfo info_, std::uint32_t initial_refcount);
        
        RefCountedBuffer(const RefCountedBuffer&) = delete;
        RefCountedBuffer& operator=(const RefCountedBuffer&) = delete;
        RefCountedBuffer(RefCountedBuffer&&) noexcept;
        RefCountedBuffer& operator=(RefCountedBuffer&&) noexcept;
    };

    /* -------------------------------------------------------------------------
     * CONSTRUCTOR: Registry
     * -------------------------------------------------------------------------
     * PARAMETERS:
     *     num_shards [in] - Number of shards (default: hardware_concurrency)
     *
     * PRECONDITIONS:
     *     num_shards > 0
     *
     * POSTCONDITIONS:
     *     Registry initialized with num_shards shards.
     *     All counters set to 0.
     *
     * COMPLEXITY:
     *     Time: O(num_shards) - shard initialization
     *     Space: O(num_shards) - shard storage
     * ---------------------------------------------------------------------- */
    explicit Registry(std::size_t num_shards = detail::get_default_shard_count());

    /* -------------------------------------------------------------------------
     * DESTRUCTOR: ~Registry()
     * -------------------------------------------------------------------------
     * POSTCONDITIONS:
     *     All registered pointers and buffers freed.
     *     In DEBUG mode: Warns about leaked memory.
     * ---------------------------------------------------------------------- */
    ~Registry();

    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;
    Registry(Registry&&) = delete;
    Registry& operator=(Registry&&) = delete;

    // =========================================================================
    // Simple Pointer Registration (refcount = 1)
    // =========================================================================

    /* -------------------------------------------------------------------------
     * METHOD: register_ptr
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Register simple pointer with automatic cleanup on unregister.
     *
     * PARAMETERS:
     *     ptr            [in] - Pointer to register
     *     byte_size      [in] - Size in bytes
     *     type           [in] - Allocation type
     *     custom_deleter [in] - Custom deleter (if type == Custom)
     *
     * PRECONDITIONS:
     *     - ptr != nullptr
     *     - byte_size > 0
     *
     * POSTCONDITIONS:
     *     - Pointer registered in appropriate shard
     *     - total_ptrs_ incremented
     *     - total_ptr_bytes_ incremented
     *
     * THREAD SAFETY:
     *     Thread-safe (protected by shard locks).
     *
     * COMPLEXITY:
     *     Time: O(1) average, O(n) worst case (hash collision)
     * ---------------------------------------------------------------------- */
    void register_ptr(void* ptr, std::size_t byte_size, AllocType type, 
                      Deleter custom_deleter = nullptr);

    /* -------------------------------------------------------------------------
     * METHOD: unregister_ptr
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Unregister pointer and free memory immediately.
     *
     * PARAMETERS:
     *     ptr [in] - Pointer to unregister
     *
     * RETURNS:
     *     True if pointer was registered and freed, false otherwise.
     *
     * POSTCONDITIONS:
     *     - Pointer removed from registry
     *     - Memory freed via appropriate deleter
     *     - total_ptrs_ decremented
     *     - total_ptr_bytes_ decremented
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    bool unregister_ptr(void* ptr);

    /* -------------------------------------------------------------------------
     * METHOD: register_batch
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Register multiple pointers in batch (more efficient than individual calls).
     *
     * PARAMETERS:
     *     entries [in] - Span of (ptr, byte_size, type, deleter) tuples
     *
     * ALGORITHM:
     *     1. Partition entries by shard (O(n))
     *     2. Process each shard independently (parallelizable)
     *     3. Update atomic counters once per shard
     *
     * PERFORMANCE:
     *     More efficient than N individual register_ptr calls due to:
     *     - Reduced lock acquisition overhead
     *     - Better cache locality
     *     - Batch counter updates
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void register_batch(std::span<const std::tuple<void*, std::size_t, AllocType, Deleter>> entries);

    /* -------------------------------------------------------------------------
     * METHOD: unregister_batch
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Unregister multiple pointers in batch.
     *
     * PARAMETERS:
     *     ptrs [in] - Span of pointers to unregister
     *
     * ALGORITHM:
     *     1. Partition pointers by shard
     *     2. Collect deletion tasks (defer actual deletion)
     *     3. Execute deletions after releasing all locks
     *
     * PERFORMANCE:
     *     More efficient than N individual unregister_ptr calls.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void unregister_batch(std::span<void* const> ptrs);

    // =========================================================================
    // Typed Allocation Helpers
    // =========================================================================

    /* -------------------------------------------------------------------------
     * METHOD: new_array
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Allocate array and register automatically.
     *
     * PARAMETERS:
     *     T     [template] - Element type
     *     count [in]       - Number of elements
     *
     * RETURNS:
     *     Pointer to allocated array, or nullptr on failure.
     *
     * POSTCONDITIONS:
     *     - Array allocated and zero-initialized
     *     - Automatically registered with ArrayNew type
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    template <typename T>
    [[nodiscard]] T* new_array(std::size_t count);

    /* -------------------------------------------------------------------------
     * METHOD: new_aligned
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Allocate aligned array and register automatically.
     *
     * PARAMETERS:
     *     T         [template] - Element type
     *     count     [in]       - Number of elements
     *     alignment [in]       - Alignment in bytes (default: 64)
     *
     * RETURNS:
     *     Aligned pointer, or nullptr on failure.
     *
     * POSTCONDITIONS:
     *     - Array allocated with specified alignment
     *     - Automatically registered with AlignedAlloc type
     * ---------------------------------------------------------------------- */
    template <typename T>
    [[nodiscard]] T* new_aligned(std::size_t count, std::size_t alignment = 64);

    /* -------------------------------------------------------------------------
     * METHOD: new_object
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Allocate single object and register automatically.
     *
     * PARAMETERS:
     *     T    [template] - Object type
     *     args [in]       - Constructor arguments
     *
     * RETURNS:
     *     Pointer to constructed object, or nullptr on failure.
     *
     * POSTCONDITIONS:
     *     - Object constructed with provided arguments
     *     - Automatically registered with ScalarNew type
     * ---------------------------------------------------------------------- */
    template <typename T, typename... Args>
    [[nodiscard]] T* new_object(Args&&... args);

    // =========================================================================
    // Reference-Counted Buffer Registration
    // =========================================================================

    /* -------------------------------------------------------------------------
     * METHOD: register_buffer
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Register reference-counted buffer with initial reference count.
     *
     * PARAMETERS:
     *     real_ptr         [in] - Actual allocated pointer
     *     byte_size        [in] - Size in bytes
     *     initial_refcount [in] - Initial reference count (must be > 0)
     *     type             [in] - Allocation type
     *     custom_deleter   [in] - Custom deleter (if type == Custom)
     *
     * RETURNS:
     *     BufferID (unique identifier), or 0 on failure.
     *
     * PRECONDITIONS:
     *     - real_ptr != nullptr
     *     - byte_size > 0
     *     - initial_refcount > 0
     *
     * POSTCONDITIONS:
     *     - Buffer registered with unique BufferID
     *     - Reference count set to initial_refcount
     *     - total_buffers_ incremented
     *     - total_buffer_bytes_ incremented
     *
     * NOTE:
     *     BufferID starts at 2 to avoid conflicts with sentinel values 0 and 1.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    [[nodiscard]] BufferID register_buffer(void* real_ptr, std::size_t byte_size,
                                           std::uint32_t initial_refcount,
                                           AllocType type, Deleter custom_deleter = nullptr);

    /* -------------------------------------------------------------------------
     * METHOD: register_alias
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Register alias pointer for existing buffer (increments refcount).
     *
     * PARAMETERS:
     *     alias_ptr [in] - Alias pointer to register
     *     buffer_id [in] - BufferID from register_buffer
     *
     * RETURNS:
     *     True if alias registered, false on failure.
     *
     * PRECONDITIONS:
     *     - alias_ptr != nullptr
     *     - buffer_id != 0 (valid BufferID)
     *
     * POSTCONDITIONS:
     *     - Alias registered and mapped to buffer_id
     *     - Buffer reference count unchanged (aliases don't increment refcount)
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    bool register_alias(void* alias_ptr, BufferID buffer_id);

    /* -------------------------------------------------------------------------
     * METHOD: register_buffer_with_aliases
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Register buffer with multiple aliases in one call.
     *
     * PARAMETERS:
     *     real_ptr   [in] - Actual allocated pointer
     *     byte_size  [in] - Size in bytes
     *     alias_ptrs [in] - Span of alias pointers
     *     type       [in] - Allocation type
     *     custom_deleter [in] - Custom deleter (if type == Custom)
     *
     * RETURNS:
     *     True if all aliases registered, false on failure.
     *
     * ALGORITHM:
     *     1. Count non-null aliases (initial_refcount)
     *     2. Register buffer with initial_refcount
     *     3. Register all aliases mapping to buffer_id
     *
     * POSTCONDITIONS:
     *     - Buffer registered with refcount = number of non-null aliases
     *     - All aliases registered and mapped to buffer_id
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    bool register_buffer_with_aliases(void* real_ptr, std::size_t byte_size,
                                       std::span<void* const> alias_ptrs,
                                       AllocType type, Deleter custom_deleter = nullptr);

    // =========================================================================
    // Alias Unregistration (Reference Counting)
    // =========================================================================

    /* -------------------------------------------------------------------------
     * METHOD: unregister_alias
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Unregister alias and decrement buffer reference count.
     *
     * PARAMETERS:
     *     alias_ptr [in] - Alias pointer to unregister
     *
     * RETURNS:
     *     True if alias was registered and unregistered, false otherwise.
     *
     * ALGORITHM:
     *     1. Remove alias from alias map
     *     2. Decrement buffer refcount atomically
     *     3. If refcount reaches 0: free buffer and remove from buffer map
     *
     * POSTCONDITIONS:
     *     - Alias removed from registry
     *     - Buffer refcount decremented
     *     - If refcount == 0: buffer freed and removed
     *
     * THREAD SAFETY:
     *     Thread-safe. Uses atomic fetch_sub with acquire-release ordering.
     * ---------------------------------------------------------------------- */
    bool unregister_alias(void* alias_ptr);

    /* -------------------------------------------------------------------------
     * METHOD: unregister_aliases
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Unregister multiple aliases in batch (optimized).
     *
     * PARAMETERS:
     *     alias_ptrs [in] - Span of alias pointers to unregister
     *
     * ALGORITHM:
     *     1. Remove all aliases from alias maps
     *     2. Aggregate refcount decrements per buffer (unordered_map)
     *     3. Apply decrements atomically per buffer
     *     4. Free buffers that reach refcount == 0
     *
     * PERFORMANCE:
     *     Uses unordered_map for O(1) average lookup instead of O(n) linear search.
     *     More efficient than N individual unregister_alias calls.
     *
     * THREAD SAFETY:
     *     Thread-safe.
     * ---------------------------------------------------------------------- */
    void unregister_aliases(std::span<void* const> alias_ptrs);

    // =========================================================================
    // Queries
    // =========================================================================

    /* -------------------------------------------------------------------------
     * METHOD: contains_ptr
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if pointer is registered as simple pointer.
     *
     * RETURNS:
     *     True if pointer is registered, false otherwise.
     *
     * THREAD SAFETY:
     *     Thread-safe (read-only operation).
     * ---------------------------------------------------------------------- */
    [[nodiscard]] bool contains_ptr(const void* ptr) const;

    /* -------------------------------------------------------------------------
     * METHOD: contains_alias
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if pointer is registered as alias.
     *
     * RETURNS:
     *     True if pointer is registered as alias, false otherwise.
     * ---------------------------------------------------------------------- */
    [[nodiscard]] bool contains_alias(const void* ptr) const;

    /* -------------------------------------------------------------------------
     * METHOD: contains
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Check if pointer is registered (either as simple pointer or alias).
     *
     * RETURNS:
     *     True if pointer is registered in any form, false otherwise.
     * ---------------------------------------------------------------------- */
    [[nodiscard]] bool contains(const void* ptr) const;

    /* -------------------------------------------------------------------------
     * METHOD: size_of
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get size in bytes for registered pointer.
     *
     * PARAMETERS:
     *     ptr [in] - Pointer to query
     *
     * RETURNS:
     *     Size in bytes, or 0 if pointer not registered.
     *
     * THREAD SAFETY:
     *     Thread-safe (read-only operation).
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::size_t size_of(const void* ptr) const;

    /* -------------------------------------------------------------------------
     * METHOD: get_buffer_id
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get BufferID for alias pointer.
     *
     * PARAMETERS:
     *     alias_ptr [in] - Alias pointer
     *
     * RETURNS:
     *     BufferID, or 0 if alias not registered.
     * ---------------------------------------------------------------------- */
    [[nodiscard]] BufferID get_buffer_id(const void* alias_ptr) const;

    /* -------------------------------------------------------------------------
     * METHOD: get_refcount
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Get current reference count for buffer.
     *
     * PARAMETERS:
     *     buffer_id [in] - BufferID to query
     *
     * RETURNS:
     *     Current reference count, or 0 if buffer_id invalid.
     *
     * THREAD SAFETY:
     *     Thread-safe (atomic load).
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::uint32_t get_refcount(BufferID buffer_id) const;

    // =========================================================================
    // Statistics
    // =========================================================================

    /* -------------------------------------------------------------------------
     * METHODS: Statistics
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     O(1) statistics via atomic counters.
     *
     * ptr_count:      Number of registered simple pointers
     * ptr_bytes:      Total bytes in simple pointers
     * buffer_count:   Number of registered buffers
     * buffer_bytes:   Total bytes in buffers
     * total_count:    Total registered entries (ptrs + buffers)
     * total_bytes:    Total bytes (ptr_bytes + buffer_bytes)
     *
     * THREAD SAFETY:
     *     Thread-safe (atomic loads).
     *
     * COMPLEXITY:
     *     O(1) - atomic counter read
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::size_t ptr_count() const noexcept;
    [[nodiscard]] std::size_t ptr_bytes() const noexcept;
    [[nodiscard]] std::size_t buffer_count() const noexcept;
    [[nodiscard]] std::size_t buffer_bytes() const noexcept;
    [[nodiscard]] std::size_t total_count() const noexcept;
    [[nodiscard]] std::size_t total_bytes() const noexcept;

    // =========================================================================
    // Dump and Clear
    // =========================================================================

    /* -------------------------------------------------------------------------
     * METHOD: dump_ptrs
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Dump all registered simple pointers (for debugging).
     *
     * RETURNS:
     *     Vector of (pointer, byte_size) pairs.
     *
     * THREAD SAFETY:
     *     Thread-safe (read-only operation).
     *
     * USE CASE:
     *     Debugging memory leaks, inspection of registered pointers.
     * ---------------------------------------------------------------------- */
    [[nodiscard]] std::vector<std::pair<void*, std::size_t>> dump_ptrs() const;

    /* -------------------------------------------------------------------------
     * METHOD: clear_all_and_free
     * -------------------------------------------------------------------------
     * SUMMARY:
     *     Clear all registrations and free all memory.
     *
     * POSTCONDITIONS:
     *     - All registered pointers freed
     *     - All registered buffers freed
     *     - All counters reset to 0
     *     - Registry empty
     *
     * THREAD SAFETY:
     *     Thread-safe (acquires all shard locks).
     *
     * WARNING:
     *     This operation is expensive and should only be used for cleanup.
     * ---------------------------------------------------------------------- */
    void clear_all_and_free();
};

// =============================================================================
// Global Registry Access
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: get_registry
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Get global Registry singleton instance.
 *
 * RETURNS:
 *     Reference to global Registry instance.
 *
 * THREAD SAFETY:
 *     Thread-safe (static initialization is thread-safe in C++11+).
 *
 * LIFETIME:
 *     Singleton lives for program lifetime.
 * -------------------------------------------------------------------------- */
Registry& get_registry();

// =============================================================================
// Global Convenience Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: register_ptr
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Register pointer in global registry.
 *
 * PARAMETERS:
 *     ptr       [in] - Pointer to register
 *     byte_size [in] - Size in bytes
 *     type      [in] - Allocation type
 *     deleter   [in] - Custom deleter (if type == Custom)
 * -------------------------------------------------------------------------- */
inline void register_ptr(void* ptr, std::size_t byte_size, AllocType type,
                         Registry::Deleter deleter = nullptr);

/* -----------------------------------------------------------------------------
 * FUNCTION: unregister_ptr
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Unregister pointer from global registry.
 *
 * RETURNS:
 *     True if pointer was registered and freed, false otherwise.
 * -------------------------------------------------------------------------- */
inline bool unregister_ptr(void* ptr);

/* -----------------------------------------------------------------------------
 * FUNCTION: new_array
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Allocate array via global registry.
 *
 * RETURNS:
 *     Pointer to allocated array, or nullptr on failure.
 * -------------------------------------------------------------------------- */
template <typename T>
[[nodiscard]] inline T* new_array(std::size_t count);

/* -----------------------------------------------------------------------------
 * FUNCTION: new_aligned
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Allocate aligned array via global registry.
 *
 * RETURNS:
 *     Aligned pointer, or nullptr on failure.
 * -------------------------------------------------------------------------- */
template <typename T>
[[nodiscard]] inline T* new_aligned(std::size_t count, std::size_t alignment = 64);

/* -----------------------------------------------------------------------------
 * FUNCTION: new_object
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Allocate object via global registry.
 *
 * RETURNS:
 *     Pointer to constructed object, or nullptr on failure.
 * -------------------------------------------------------------------------- */
template <typename T, typename... Args>
[[nodiscard]] inline T* new_object(Args&&... args);

/* -----------------------------------------------------------------------------
 * FUNCTION: register_shared_buffer
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Register shared buffer with aliases via global registry.
 *
 * RETURNS:
 *     True if all aliases registered, false on failure.
 * -------------------------------------------------------------------------- */
inline bool register_shared_buffer(void* real_ptr, std::size_t byte_size,
                                   std::span<void* const> alias_ptrs,
                                   AllocType type,
                                   Registry::Deleter custom_deleter = nullptr);

/* -----------------------------------------------------------------------------
 * FUNCTION: unregister_alias
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Unregister alias from global registry.
 *
 * RETURNS:
 *     True if alias was registered and unregistered, false otherwise.
 * -------------------------------------------------------------------------- */
inline bool unregister_alias(void* alias_ptr);

/* -----------------------------------------------------------------------------
 * FUNCTION: unregister_aliases
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Unregister multiple aliases from global registry.
 * -------------------------------------------------------------------------- */
inline void unregister_aliases(std::span<void* const> alias_ptrs);

// =============================================================================
// CLASS: RegistryGuard
// =============================================================================

/* -----------------------------------------------------------------------------
 * CLASS: RegistryGuard
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     RAII guard for automatic pointer/alias unregistration.
 *
 * DESIGN PURPOSE:
 *     Ensures unregistration even on exception. Similar to std::unique_ptr
 *     but for registry management.
 *
 * USAGE:
 *     {
 *         RegistryGuard guard(ptr, is_alias);
 *         // Use ptr...
 *     }  // guard destructor unregisters automatically
 *
 * METHODS:
 *     release: Cancel automatic unregistration (manual management)
 *
 * THREAD SAFETY:
 *     Not thread-safe. Each thread should have its own guard.
 * -------------------------------------------------------------------------- */
class RegistryGuard {
public:
    explicit RegistryGuard(void* ptr, bool is_alias = false) noexcept;
    ~RegistryGuard();
    
    void release() noexcept;
    
    RegistryGuard(const RegistryGuard&) = delete;
    RegistryGuard& operator=(const RegistryGuard&) = delete;
    RegistryGuard(RegistryGuard&&) noexcept;
    RegistryGuard& operator=(RegistryGuard&&) noexcept;
};

// =============================================================================
// Legacy Compatibility
// =============================================================================

/* -----------------------------------------------------------------------------
 * TYPE ALIAS: HandlerRegistry
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Legacy alias for Registry (for migration from handler.hpp).
 * -------------------------------------------------------------------------- */
using HandlerRegistry = Registry;

/* -----------------------------------------------------------------------------
 * FUNCTION: get_handler_registry
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Legacy function for backward compatibility.
 *
 * RETURNS:
 *     Reference to global Registry instance.
 * -------------------------------------------------------------------------- */
inline Registry& get_handler_registry();

/* -----------------------------------------------------------------------------
 * FUNCTION: get_refcount_registry
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Legacy function for backward compatibility.
 *
 * RETURNS:
 *     Reference to global Registry instance.
 * -------------------------------------------------------------------------- */
inline Registry& get_refcount_registry();

} // namespace scl

