# Memory Registry

Unified high-performance memory registry with three-layer reference counting and lock-free concurrent access.

**Location**: `scl/core/registry.hpp`

---

## Architecture

The registry implements a three-layer reference counting system:

1. **Layer 1: Buffer** - Real memory block with alias_count
2. **Layer 2: Alias** - Access pointer with ref_count (can be shared by multiple instances)
3. **Layer 3: Instance** - Matrix objects that hold aliases

**Key Features:**
- Sharded reference counting for lock-free concurrent access
- Multiple instances can share the same alias (no need for is_view_ flag)
- Zero-copy slicing via alias sharing
- Optimized with SCL_FORCE_INLINE and branch prediction hints
- Uses scl::algo custom operators instead of std:: for zero-overhead

---

## BufferID

**SUMMARY:**
Unique identifier for memory buffers.

**SIGNATURE:**
```cpp
using BufferID = std::uint64_t;
```

---

## AllocType

**SUMMARY:**
Enumeration of allocation types for proper deallocation.

**SIGNATURE:**
```cpp
enum class AllocType : std::uint8_t {
    ArrayNew = 0,      // new T[]
    ScalarNew = 1,     // new T
    AlignedAlloc = 2,  // scl::memory::aligned_alloc
    Custom = 3         // Custom deleter
};
```

---

## ShardedRefCount

**SUMMARY:**
Lock-free sharded reference counter for high-concurrency scenarios.

**SIGNATURE:**
```cpp
class ShardedRefCount {
public:
    static constexpr std::size_t MAX_SHARDS = 16;
    static constexpr std::int32_t BORROW_THRESHOLD = 8;

    explicit ShardedRefCount(std::size_t num_shards = 4, std::int32_t initial = 1) noexcept;
    
    SCL_FORCE_INLINE void incref() noexcept;
    SCL_FORCE_INLINE void incref(std::int32_t amount) noexcept;
    SCL_FORCE_INLINE bool decref() noexcept;
    SCL_FORCE_INLINE bool decref(std::int32_t amount) noexcept;
    
    SCL_NODISCARD std::int32_t get_count() const noexcept;
    SCL_NODISCARD SCL_FORCE_INLINE bool is_likely_unique() const noexcept;
    SCL_NODISCARD SCL_FORCE_INLINE bool is_unique() const noexcept;
    
    void consolidate() noexcept;
};
```

**ALGORITHM:**
- Each thread has a dedicated shard (determined by thread_local index)
- Fast path: incref/decref operates on thread-local shard (no contention)
- Slow path: borrow from base counter when shard is empty
- Consolidation: merge all shards into base for precise operations

**OPTIMIZATION:**
- SCL_FORCE_INLINE for hot-path functions
- SCL_LIKELY/UNLIKELY branch hints for common paths
- Cache-line aligned shards (64 bytes) prevent false sharing
- Relaxed memory ordering for increments
- Acquire-release ordering only when necessary

---

## AliasRecord

**SUMMARY:**
Layer 2 record: access pointer with reference count.

**SIGNATURE:**
```cpp
struct AliasRecord {
    BufferID buffer_id;                    // Parent buffer
    std::atomic<std::uint32_t> ref_count;  // Instances holding this alias
    
    AliasRecord() noexcept;
    AliasRecord(BufferID bid, std::uint32_t initial_ref = 1) noexcept;
};
```

---

## Registry

**SUMMARY:**
Unified memory management with three-layer reference counting.

### Core Methods

#### register_ptr

**SUMMARY:**
Register a simple pointer without sharing (Layer 1 alternative).

**SIGNATURE:**
```cpp
void register_ptr(
    void* ptr,
    std::size_t byte_size,
    AllocType type,
    Deleter custom_deleter = nullptr
);
```

**PARAMETERS:**
- ptr            [in] Pointer to register
- byte_size      [in] Size in bytes
- type           [in] Allocation type for proper deallocation
- custom_deleter [in] Custom deleter (if type == Custom)

**PRECONDITIONS:**
- ptr must not be null
- byte_size > 0

**POSTCONDITIONS:**
- Pointer is tracked by registry
- Will be freed on unregister or registry destruction

**THREAD SAFETY:**
Safe - sharded hash map with fine-grained locking

**OPTIMIZATION:**
- SCL_UNLIKELY hint for null/zero checks
- SCL_LIKELY hint for successful insertion

---

#### unregister_ptr

**SUMMARY:**
Unregister and free a simple pointer.

**SIGNATURE:**
```cpp
bool unregister_ptr(void* ptr);
```

**PARAMETERS:**
- ptr [in] Pointer to unregister

**PRECONDITIONS:**
None

**POSTCONDITIONS:**
- If found: pointer is freed and removed
- If not found: returns false

**RETURN VALUE:**
true if pointer was found and freed

**THREAD SAFETY:**
Safe

**OPTIMIZATION:**
- SCL_UNLIKELY for edge cases
- SCL_LIKELY for common deleter case

---

#### create_buffer

**SUMMARY:**
Create a buffer (Layer 1) without initial aliases.

**SIGNATURE:**
```cpp
SCL_NODISCARD BufferID create_buffer(
    void* real_ptr,
    std::size_t byte_size,
    AllocType type,
    Deleter custom_deleter = nullptr
);
```

**PARAMETERS:**
- real_ptr       [in] Memory block pointer
- byte_size      [in] Size in bytes
- type           [in] Allocation type
- custom_deleter [in] Custom deleter (if needed)

**PRECONDITIONS:**
- real_ptr must not be null
- byte_size > 0

**POSTCONDITIONS:**
- Returns unique BufferID
- Buffer registered with alias_count = 0
- Caller must create aliases or buffer will leak

**RETURN VALUE:**
Non-zero BufferID on success, 0 on failure

**THREAD SAFETY:**
Safe

---

#### create_alias

**SUMMARY:**
Create a new alias pointing to a buffer (Layer 2).

**SIGNATURE:**
```cpp
SCL_FORCE_INLINE bool create_alias(
    void* alias_ptr,
    BufferID buffer_id,
    std::uint32_t initial_ref = 1
);
```

**PARAMETERS:**
- alias_ptr   [in] Access pointer
- buffer_id   [in] Parent buffer
- initial_ref [in] Initial reference count (default 1)

**PRECONDITIONS:**
- alias_ptr must not be null
- buffer_id must be valid
- initial_ref > 0

**POSTCONDITIONS:**
- Alias registered with ref_count = initial_ref
- Buffer's alias_count incremented
- Returns true on success

**THREAD SAFETY:**
Safe - lock-free alias insertion

**OPTIMIZATION:**
- SCL_FORCE_INLINE for hot path
- SCL_UNLIKELY/LIKELY hints for edge cases

---

#### alias_incref

**SUMMARY:**
Increment alias reference count (lock-free fast path).

**SIGNATURE:**
```cpp
SCL_FORCE_INLINE bool alias_incref(
    void* alias_ptr,
    std::uint32_t increment = 1
);
```

**PARAMETERS:**
- alias_ptr [in] Alias pointer
- increment [in] Amount to increment

**PRECONDITIONS:**
- alias_ptr must be registered

**POSTCONDITIONS:**
- Alias ref_count increased by increment
- Returns true if alias exists

**THREAD SAFETY:**
Safe - lock-free atomic increment

**OPTIMIZATION:**
- SCL_FORCE_INLINE
- SCL_UNLIKELY for error cases
- Relaxed memory ordering for fast increment

---

#### alias_decref

**SUMMARY:**
Decrement alias reference count, free if reaches zero.

**SIGNATURE:**
```cpp
SCL_FORCE_INLINE bool alias_decref(void* alias_ptr);
```

**PARAMETERS:**
- alias_ptr [in] Alias pointer

**PRECONDITIONS:**
None (safe for null or unregistered pointers)

**POSTCONDITIONS:**
- ref_count decremented
- If ref_count reaches 0: alias removed, buffer's alias_count decremented
- If buffer's alias_count reaches 0: buffer freed
- Returns true if alias was removed

**THREAD SAFETY:**
Safe - atomic decrement with lock-free erase

**OPTIMIZATION:**
- SCL_FORCE_INLINE
- SCL_UNLIKELY for removal case
- SCL_LIKELY for buffer cleanup
- Acquire-release ordering for correctness

---

#### alias_incref_batch

**SUMMARY:**
Batch increment alias reference counts for better cache locality.

**SIGNATURE:**
```cpp
void alias_incref_batch(
    std::span<void* const> alias_ptrs,
    std::uint32_t increment = 1
);
```

**PARAMETERS:**
- alias_ptrs [in] Array of alias pointers
- increment  [in] Amount to increment each

**PRECONDITIONS:**
None (skips null pointers)

**POSTCONDITIONS:**
- All valid aliases have ref_count increased

**ALGORITHM:**
1. Group aliases by shard for cache locality
2. Process each shard sequentially
3. Increment each alias in shard

**THREAD SAFETY:**
Safe

**OPTIMIZATION:**
- Shard-grouped processing for cache efficiency
- Skips null pointers automatically

---

#### alias_decref_batch

**SUMMARY:**
Batch decrement alias reference counts (more efficient than individual decrements).

**SIGNATURE:**
```cpp
void alias_decref_batch(std::span<void* const> alias_ptrs);
```

**PARAMETERS:**
- alias_ptrs [in] Array of alias pointers

**PRECONDITIONS:**
None (safe for null or unregistered pointers)

**POSTCONDITIONS:**
- All ref_counts decremented
- Removed aliases have buffer's alias_count decremented
- Buffers with alias_count=0 are freed

**ALGORITHM:**
1. Group aliases by shard
2. Decrement ref_counts and mark for removal
3. Batch erase removed aliases
4. Batch decrement buffer alias_counts

**THREAD SAFETY:**
Safe

**OPTIMIZATION:**
- Batched operations reduce lock overhead
- Collected buffer decrements applied together

---

### Allocation Helpers

#### new_array

**SUMMARY:**
Allocate and register array via new[].

**SIGNATURE:**
```cpp
template <typename T>
SCL_NODISCARD T* new_array(std::size_t count);
```

**PARAMETERS:**
- count [in] Number of elements

**PRECONDITIONS:**
- count > 0

**POSTCONDITIONS:**
- Returns allocated array or nullptr on failure
- Array registered with type = ArrayNew

**THREAD SAFETY:**
Safe

---

#### new_aligned

**SUMMARY:**
Allocate and register aligned array.

**SIGNATURE:**
```cpp
template <typename T>
SCL_NODISCARD T* new_aligned(
    std::size_t count,
    std::size_t alignment = 64
);
```

**PARAMETERS:**
- count     [in] Number of elements
- alignment [in] Alignment in bytes (default 64)

**PRECONDITIONS:**
- count > 0
- alignment is power of 2

**POSTCONDITIONS:**
- Returns aligned array or nullptr on failure
- Registered with type = AlignedAlloc

**THREAD SAFETY:**
Safe

---

### Query Methods

#### contains_ptr

**SUMMARY:**
Check if pointer is registered as simple pointer.

**SIGNATURE:**
```cpp
SCL_NODISCARD bool contains_ptr(const void* ptr) const;
```

**THREAD SAFETY:**
Safe - read-only

---

#### contains_alias

**SUMMARY:**
Check if pointer is registered as alias.

**SIGNATURE:**
```cpp
SCL_NODISCARD bool contains_alias(const void* ptr) const;
```

**THREAD SAFETY:**
Safe

---

#### alias_refcount

**SUMMARY:**
Get alias reference count.

**SIGNATURE:**
```cpp
SCL_NODISCARD std::uint32_t alias_refcount(void* alias_ptr) const;
```

**RETURN VALUE:**
Current ref_count or 0 if not found

**THREAD SAFETY:**
Safe

---

#### get_buffer_id

**SUMMARY:**
Get buffer ID for an alias.

**SIGNATURE:**
```cpp
SCL_NODISCARD BufferID get_buffer_id(const void* alias_ptr) const;
```

**RETURN VALUE:**
BufferID or 0 if not found

**THREAD SAFETY:**
Safe

---

### Statistics

#### ptr_count

**SUMMARY:**
Number of registered simple pointers.

**SIGNATURE:**
```cpp
SCL_NODISCARD std::size_t ptr_count() const noexcept;
```

---

#### buffer_count

**SUMMARY:**
Number of registered buffers.

**SIGNATURE:**
```cpp
SCL_NODISCARD std::size_t buffer_count() const noexcept;
```

---

#### alias_count

**SUMMARY:**
Number of registered aliases.

**SIGNATURE:**
```cpp
SCL_NODISCARD std::size_t alias_count() const noexcept;
```

---

#### total_bytes

**SUMMARY:**
Total memory under management.

**SIGNATURE:**
```cpp
SCL_NODISCARD std::size_t total_bytes() const noexcept;
```

---

## Global Functions

#### get_registry

**SUMMARY:**
Get singleton registry instance.

**SIGNATURE:**
```cpp
Registry& get_registry();
```

**THREAD SAFETY:**
Safe - thread-local initialization

---

## RegistryGuard

**SUMMARY:**
RAII guard for automatic cleanup.

**SIGNATURE:**
```cpp
class RegistryGuard {
public:
    explicit RegistryGuard(void* ptr, bool is_alias = false) noexcept;
    ~RegistryGuard();
    void release() noexcept;
};
```

**USAGE:**
Automatically unregisters pointer on scope exit unless released.

---

## Optimization Summary

**Lock-Free Operations:**
- alias_incref: lock-free atomic increment
- alias_decref: lock-free atomic decrement
- Thread-local shard selection eliminates contention

**Branch Prediction:**
- SCL_LIKELY for common paths (successful operations)
- SCL_UNLIKELY for error cases (null pointers, allocation failures)

**Memory Ordering:**
- Relaxed for increments (no synchronization needed)
- Acquire-release for decrements (synchronize cleanup)
- Full fence only for consolidation

**Inlining:**
- SCL_FORCE_INLINE for hot-path functions
- Reduces function call overhead to zero

**Custom Operators:**
- Uses scl::algo::zero instead of std::memset
- Eliminates std:: overhead in critical paths
