# Registry V2: Three-Layer Reference Counting with Sharded Concurrency

## Overview

This document describes the redesigned Registry module with:
1. Three-layer reference counting (Buffer → Alias → Instance)
2. Sharded reference counting for lock-free concurrent access
3. Smart detection strategies to minimize synchronization overhead

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 3: Instances                            │
│  Matrix A          Matrix B (slice)      Matrix C                │
│  holds [a1,a2]     holds [a1,a2]         holds [a3,a4]          │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 2: Aliases                              │
│  Alias a1          Alias a2              Alias a3    Alias a4    │
│  ref_count=2       ref_count=2           ref_count=1 ref_count=1 │
│  buffer=B1         buffer=B1             buffer=B2   buffer=B2   │
│  offset=0          offset=1024           offset=0    offset=512  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 1: Buffers                              │
│  Buffer B1 (4KB)                    Buffer B2 (2KB)              │
│  alias_count=2                      alias_count=2                │
│  [real_ptr, size, deleter]          [real_ptr, size, deleter]    │
└─────────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### 1. ShardedRefCount - Lock-Free Reference Counting

```cpp
// Cache-line aligned shard to prevent false sharing
struct alignas(64) RefCountShard {
    std::atomic<int32_t> count{0};  // Can be negative (borrowed)
    char padding[60];               // Ensure 64-byte alignment
};

class ShardedRefCount {
    static constexpr size_t MAX_SHARDS = 64;
    static constexpr int32_t BORROW_THRESHOLD = 16;  // When to borrow from base
    
    std::atomic<int32_t> base_{0};           // Global base count
    std::vector<RefCountShard> shards_;      // Per-thread shards
    size_t num_shards_;
    
public:
    explicit ShardedRefCount(size_t num_shards, int32_t initial = 1);
    
    // Fast path: increment current thread's shard (lock-free)
    void incref() noexcept;
    
    // Fast path: decrement current thread's shard
    // Returns true if count reached zero
    bool decref() noexcept;
    
    // Slow path: get exact total count (requires memory fence)
    int32_t get_count() const noexcept;
    
    // Fast check: is this the only reference?
    // May return false negative (safe for optimization decisions)
    bool is_likely_unique() const noexcept;
    
    // Precise check: exactly one reference?
    bool is_unique() const noexcept;
};
```

### 2. AliasRecord - Access Pointer with Sharded RefCount

```cpp
struct AliasRecord {
    BufferID buffer_id;              // Parent buffer
    std::ptrdiff_t offset;           // Offset from buffer.real_ptr
    ShardedRefCount ref_count;       // Sharded reference count
    
    AliasRecord(BufferID bid, std::ptrdiff_t off, size_t num_shards)
        : buffer_id(bid), offset(off), ref_count(num_shards, 1) {}
};
```

### 3. BufferRecord - Real Memory Block

```cpp
struct BufferRecord {
    void* real_ptr;
    uint64_t byte_size;
    AllocType type;
    Deleter custom_deleter;
    ShardedRefCount alias_count;     // Number of aliases pointing to this buffer
    
    BufferRecord(void* ptr, uint64_t size, AllocType t, Deleter d, size_t num_shards)
        : real_ptr(ptr), byte_size(size), type(t), custom_deleter(d)
        , alias_count(num_shards, 0) {}
};
```

## Operations

### incref (Fast Path - Lock-Free)

```cpp
void ShardedRefCount::incref() noexcept {
    // Get current thread's shard index (cached in TLS)
    size_t shard_idx = get_thread_shard_index() % num_shards_;
    
    // Increment local shard - no contention with other threads
    shards_[shard_idx].count.fetch_add(1, std::memory_order_relaxed);
}
```

### decref (Fast Path with Borrowing)

```cpp
bool ShardedRefCount::decref() noexcept {
    size_t shard_idx = get_thread_shard_index() % num_shards_;
    auto& shard = shards_[shard_idx];
    
    // Try to decrement local shard
    int32_t old_local = shard.count.load(std::memory_order_relaxed);
    
    if (old_local > 0) {
        // Fast path: local shard has count, just decrement
        shard.count.fetch_sub(1, std::memory_order_acq_rel);
        return false;  // Not zero
    }
    
    // Slow path: need to borrow from base or other shards
    return decref_slow_path(shard_idx);
}

bool ShardedRefCount::decref_slow_path(size_t shard_idx) noexcept {
    // Try to borrow from base
    int32_t old_base = base_.load(std::memory_order_acquire);
    
    while (old_base > 0) {
        int32_t borrow = std::min(old_base, BORROW_THRESHOLD);
        if (base_.compare_exchange_weak(old_base, old_base - borrow,
                                        std::memory_order_acq_rel)) {
            // Successfully borrowed, add to local shard
            shards_[shard_idx].count.fetch_add(borrow - 1, std::memory_order_relaxed);
            return false;
        }
    }
    
    // Base is zero, check if total count is zero
    return get_count() == 0;
}
```

### get_count (Slow Path - Aggregation)

```cpp
int32_t ShardedRefCount::get_count() const noexcept {
    // Memory fence to ensure visibility
    std::atomic_thread_fence(std::memory_order_acquire);
    
    int32_t total = base_.load(std::memory_order_relaxed);
    for (size_t i = 0; i < num_shards_; ++i) {
        total += shards_[i].count.load(std::memory_order_relaxed);
    }
    
    return total;
}
```

### is_likely_unique (Fast Heuristic)

```cpp
bool ShardedRefCount::is_likely_unique() const noexcept {
    // Quick check: if base > 1, definitely not unique
    int32_t base = base_.load(std::memory_order_relaxed);
    if (base > 1) return false;
    
    // Check current thread's shard only
    size_t shard_idx = get_thread_shard_index() % num_shards_;
    int32_t local = shards_[shard_idx].count.load(std::memory_order_relaxed);
    
    // Heuristic: if local == 1 and base == 0, likely unique
    // May have false negatives (other shards have count), but safe
    return (base + local == 1);
}
```

## Smart Detection Strategies

### 1. Thread Shard Assignment

```cpp
// Thread-local shard index with lazy initialization
inline size_t get_thread_shard_index() {
    static std::atomic<size_t> next_index{0};
    thread_local size_t cached_index = next_index.fetch_add(1, std::memory_order_relaxed);
    return cached_index;
}
```

### 2. Periodic Rebalancing

```cpp
// Called periodically (e.g., every 1000 operations) to consolidate shards
void ShardedRefCount::rebalance() noexcept {
    int32_t total = 0;
    
    for (size_t i = 0; i < num_shards_; ++i) {
        int32_t local = shards_[i].count.exchange(0, std::memory_order_acq_rel);
        total += local;
    }
    
    base_.fetch_add(total, std::memory_order_release);
}
```

### 3. Adaptive Shard Count

```cpp
// Determine optimal shard count based on hardware
inline size_t get_optimal_shard_count() {
    size_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 8;
    
    // Round up to power of 2 for efficient modulo
    size_t shards = 1;
    while (shards < hw_threads && shards < 64) {
        shards <<= 1;
    }
    
    return shards;
}
```

## Registry API Changes

### New API

```cpp
class Registry {
public:
    // =========================================================================
    // Buffer Management (Layer 1)
    // =========================================================================
    
    // Create a new buffer, returns BufferID
    BufferID create_buffer(void* ptr, size_t byte_size, AllocType type,
                           Deleter custom_deleter = nullptr);
    
    // Get buffer info (read-only)
    bool get_buffer_info(BufferID id, BufferInfo* out) const;
    
    // =========================================================================
    // Alias Management (Layer 2)
    // =========================================================================
    
    // Create a new alias pointing to buffer at offset
    // Initial ref_count = 1
    bool create_alias(void* alias_ptr, BufferID buffer_id, std::ptrdiff_t offset = 0);
    
    // Increment alias reference count (lock-free fast path)
    bool alias_incref(void* alias_ptr);
    
    // Decrement alias reference count
    // Returns true if alias was removed (ref_count reached 0)
    // If last alias of buffer, buffer is also freed
    bool alias_decref(void* alias_ptr);
    
    // Batch increment for multiple aliases (more efficient)
    void alias_incref_batch(std::span<void* const> aliases);
    
    // Batch decrement for multiple aliases
    void alias_decref_batch(std::span<void* const> aliases);
    
    // Get alias reference count
    uint32_t alias_refcount(void* alias_ptr) const;
    
    // Check if alias is unique (only one reference)
    bool alias_is_unique(void* alias_ptr) const;
    
    // =========================================================================
    // Convenience Functions
    // =========================================================================
    
    // Create buffer and register multiple aliases in one call
    BufferID create_buffer_with_aliases(
        void* real_ptr, size_t byte_size,
        std::span<void* const> alias_ptrs,
        std::span<const std::ptrdiff_t> offsets,
        AllocType type, Deleter custom_deleter = nullptr);
    
    // Clone aliases: create new instance reference to same data
    // Increments ref_count for each alias
    void clone_aliases(std::span<void* const> aliases);
};
```

### Backward Compatibility

```cpp
// Legacy API - implemented using new primitives
inline bool unregister_alias(void* alias_ptr) {
    return get_registry().alias_decref(alias_ptr);
}

inline void unregister_aliases(std::span<void* const> alias_ptrs) {
    get_registry().alias_decref_batch(alias_ptrs);
}

// Legacy: register_buffer_with_aliases still works
inline BufferID register_buffer_with_aliases(...) {
    std::vector<std::ptrdiff_t> offsets(alias_ptrs.size(), 0);
    return get_registry().create_buffer_with_aliases(..., offsets, ...);
}
```

## Sparse Matrix Changes

### Remove is_view_ Flag

```cpp
template <typename T, bool IsCSR>
struct Sparse {
    // REMOVED: bool is_view_;
    
    // All matrices now use unified lifecycle:
    // - owns_data_ = true: this matrix allocated the buffer
    // - owns_data_ = false: this matrix wrapped external data (no registry)
    
    // release_resources now uses alias_decref for all cases
    void release_resources() {
        if (!valid()) return;
        
        const Index pdim = primary_dim();
        auto& reg = get_registry();
        
        if (owns_data_) {
            // Collect aliases and decref them
            std::vector<void*> aliases;
            aliases.reserve(pdim * 2);
            
            for (Index i = 0; i < pdim; ++i) {
                if (data_ptrs[i]) aliases.push_back(data_ptrs[i]);
                if (indices_ptrs[i]) aliases.push_back(indices_ptrs[i]);
            }
            
            // Unified: just decref, no special view handling
            if (!aliases.empty()) {
                reg.alias_decref_batch(aliases);
            }
        }
        
        // Free metadata (always owned by this instance)
        reg.unregister_ptr(data_ptrs);
        reg.unregister_ptr(indices_ptrs);
        reg.unregister_ptr(lengths);
        
        // Reset state
        data_ptrs = indices_ptrs = nullptr;
        lengths = nullptr;
        rows_ = cols_ = nnz_ = 0;
        owns_data_ = true;
    }
};
```

### Zero-Copy Slice

```cpp
// New: Zero-copy slice that shares aliases
template <typename T, bool IsCSR>
Sparse<T, IsCSR> slice_view(
    const Sparse<T, IsCSR>& matrix,
    std::span<const Index> keep_indices)
{
    auto& reg = get_registry();
    const Size n_keep = keep_indices.size();
    
    // Allocate metadata only
    auto* dp = reg.new_array<Pointer>(n_keep);
    auto* ip = reg.new_array<Pointer>(n_keep);
    auto* len = reg.new_array<Index>(n_keep);
    
    // Share aliases with original matrix
    std::vector<void*> aliases;
    aliases.reserve(n_keep * 2);
    
    Index new_nnz = 0;
    for (Size i = 0; i < n_keep; ++i) {
        Index idx = keep_indices[i];
        dp[i] = matrix.data_ptrs[idx];
        ip[i] = matrix.indices_ptrs[idx];
        len[i] = matrix.lengths[idx];
        new_nnz += len[i];
        
        if (dp[i]) aliases.push_back(dp[i]);
        if (ip[i]) aliases.push_back(ip[i]);
    }
    
    // Increment reference counts for shared aliases
    reg.alias_incref_batch(aliases);
    
    Index new_rows = IsCSR ? static_cast<Index>(n_keep) : matrix.rows();
    Index new_cols = IsCSR ? matrix.cols() : static_cast<Index>(n_keep);
    
    // Create result with owns_data_ = true (it owns the alias references)
    return Sparse<T, IsCSR>(dp, ip, len, new_rows, new_cols, new_nnz, true);
}
```

## Performance Characteristics

| Operation | Complexity | Contention |
|-----------|------------|------------|
| incref | O(1) | None (thread-local) |
| decref (fast path) | O(1) | None (thread-local) |
| decref (slow path) | O(1) | Low (only base) |
| get_count | O(shards) | None (read-only) |
| is_likely_unique | O(1) | None |
| is_unique | O(shards) | None |

## Memory Overhead

Per Buffer/Alias:
- Base count: 4 bytes
- Per shard: 64 bytes (cache line aligned)
- Total: 4 + 64 * num_shards bytes

For 8 shards: 4 + 512 = 516 bytes per ref-counted object

Optimization: Use compact mode for low-contention objects:
- Single atomic counter for objects with < N references
- Switch to sharded mode when contention detected

## Operator Fix Plan

### 1. slice.hpp - Memory Leak Fix

**Current Problem**:
```cpp
// WRONG: allocates memory but wrap_traditional doesn't own it
T* data_ptr = scl::memory::aligned_alloc<T>(...);
return Sparse::wrap_traditional(..., data_ptr, ...);
```

**Fix Option A**: Use from_traditional (copies data, safe but slower)
```cpp
// Allocate temp buffers
T* data_ptr = scl::memory::aligned_alloc<T>(out_nnz);
Index* indices_ptr = scl::memory::aligned_alloc<Index>(out_nnz);
Index* indptr_ptr = scl::memory::aligned_alloc<Index>(n_keep + 1);

// Materialize slice
materialize_slice_primary(matrix, keep_indices, ...);

// Create owned copy via from_traditional
auto result = Sparse<T, IsCSR>::from_traditional(
    new_rows, new_cols,
    std::span<const T>(data_ptr, out_nnz),
    std::span<const Index>(indices_ptr, out_nnz),
    std::span<const Index>(indptr_ptr, n_keep + 1),
    BlockStrategy::contiguous()
);

// Free temp buffers
scl::memory::aligned_free(data_ptr);
scl::memory::aligned_free(indices_ptr);
scl::memory::aligned_free(indptr_ptr);

return result;
```

**Fix Option B**: Zero-copy slice using new alias system (preferred)
```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> slice_primary_view(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> keep_indices)
{
    auto& reg = get_registry();
    const Size n_keep = keep_indices.len;
    
    // Only allocate metadata
    auto* dp = reg.new_array<Pointer>(n_keep);
    auto* ip = reg.new_array<Pointer>(n_keep);
    auto* len = reg.new_array<Index>(n_keep);
    
    // Collect aliases to incref
    std::vector<void*> aliases;
    aliases.reserve(n_keep * 2);
    
    Index new_nnz = 0;
    for (Size i = 0; i < n_keep; ++i) {
        Index idx = keep_indices[i];
        dp[i] = matrix.data_ptrs[idx];
        ip[i] = matrix.indices_ptrs[idx];
        len[i] = matrix.lengths[idx];
        new_nnz += len[i];
        
        if (dp[i]) aliases.push_back(dp[i]);
        if (ip[i]) aliases.push_back(ip[i]);
    }
    
    // Increment refs for shared aliases
    reg.alias_incref_batch(aliases);
    
    Index new_rows = IsCSR ? static_cast<Index>(n_keep) : matrix.rows();
    Index new_cols = IsCSR ? matrix.cols() : static_cast<Index>(n_keep);
    
    return Sparse<T, IsCSR>(dp, ip, len, new_rows, new_cols, new_nnz, true);
}
```

### 2. sparse.hpp - from_contiguous_arrays Fix

**Current Problem**:
```cpp
if (take_ownership) {
    reg.register_ptr(data, ...);  // Registers to registry
}
return Sparse::wrap_traditional(...);  // But Sparse doesn't own it!
```

**Fix**:
```cpp
template <typename T, bool IsCSR>
Sparse<T, IsCSR> from_contiguous_arrays(
    T* data, Index* indices, Index* indptr,
    Index rows, Index cols, Index nnz,
    bool take_ownership = false)
{
    if (!data || !indices || !indptr) {
        return Sparse<T, IsCSR>{};
    }
    
    const Index primary_dim = IsCSR ? rows : cols;
    auto& reg = get_registry();
    
    if (take_ownership) {
        // Create buffer and register aliases properly
        // Data buffer
        BufferID data_buf = reg.create_buffer(
            data, nnz * sizeof(T), AllocType::ArrayNew);
        
        // Indices buffer  
        BufferID idx_buf = reg.create_buffer(
            indices, nnz * sizeof(Index), AllocType::ArrayNew);
        
        // Create Sparse using internal construction
        return Sparse<T, IsCSR>::from_registered_buffers(
            rows, cols, data_buf, idx_buf, indptr, primary_dim);
    } else {
        // Non-owning wrap (external lifetime management)
        return Sparse<T, IsCSR>::wrap_traditional(rows, cols, data, indices,
            std::span<const Index>(indptr, primary_dim + 1));
    }
}
```

### 3. Core Sparse Changes

**Remove is_view_ flag**:
```cpp
// Before
struct Sparse {
    bool owns_data_;
    bool is_view_;  // REMOVE THIS
    
    void release_resources() {
        if (owns_data_) {
            if (is_view_) {
                reg.decrement_buffer_refcounts(aliases);  // View logic
            } else {
                reg.unregister_aliases(aliases);  // Owner logic
            }
        }
    }
};

// After
struct Sparse {
    bool owns_data_;
    // NO is_view_
    
    void release_resources() {
        if (owns_data_) {
            // Unified: just decref the aliases
            // If ref_count reaches 0, alias and buffer are freed automatically
            reg.alias_decref_batch(aliases);
        }
    }
};
```

### 4. C API Additions

```c
// scl/binding/c_api/core/lifecycle.h

// Query ownership status
scl_error_t scl_sparse_is_owner(scl_sparse_t matrix, int* out);
scl_error_t scl_sparse_refcount(scl_sparse_t matrix, uint32_t* out);

// Create shared reference (zero-copy, increments refcount)
scl_error_t scl_sparse_share(scl_sparse_t src, scl_sparse_t* out);

// Create slice view (zero-copy, shares aliases)
scl_error_t scl_sparse_slice_rows(
    scl_sparse_t matrix,
    const scl_index_t* row_indices,
    scl_size_t n_rows,
    scl_sparse_t* out);

// Detach from registry (transfer ownership to caller)
scl_error_t scl_sparse_detach(
    scl_sparse_t* matrix,
    scl_real_t** out_data,
    scl_index_t** out_indices,
    scl_index_t** out_indptr);

// Attach external memory to registry
scl_error_t scl_sparse_attach(
    scl_real_t* data,
    scl_index_t* indices,
    scl_index_t* indptr,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    int is_csr,
    int transfer_ownership,
    scl_sparse_t* out);
```

## Migration Plan

### Phase 1: Core Infrastructure (Non-Breaking)

1. Add `ShardedRefCount` class to `registry.hpp`
2. Add `AliasRecord` with sharded ref_count
3. Add new API: `alias_incref`, `alias_decref`, `alias_incref_batch`, `alias_decref_batch`
4. Keep old API working (backward compatible)

### Phase 2: Update Sparse (Breaking)

1. Remove `is_view_` member from Sparse
2. Update `release_resources()` to use `alias_decref_batch`
3. Update all factory methods to use new alias registration
4. Update move constructor/assignment

### Phase 3: Fix Operators

1. Fix `slice.hpp`: `slice_primary`, `filter_secondary`
2. Fix `sparse.hpp`: `from_contiguous_arrays`
3. Add new zero-copy slice functions

### Phase 4: C API Updates

1. Add lifecycle management functions
2. Add share/slice functions
3. Update Python bindings

### Phase 5: Cleanup

1. Deprecate old registry API
2. Update documentation
3. Add migration guide

## Testing Strategy

1. **Unit Tests**: RefCount correctness under concurrent access
2. **Stress Tests**: High-contention incref/decref from multiple threads
3. **Memory Tests**: Valgrind/ASan for leak detection
4. **Benchmark**: Compare performance with old atomic-only approach

## Appendix: Thread-Local Shard Index

```cpp
namespace detail {

// Efficient thread-local shard index
class ThreadShardIndex {
    static constexpr size_t CACHE_LINE = 64;
    
    struct alignas(CACHE_LINE) Counter {
        std::atomic<size_t> value{0};
    };
    
    static Counter next_index_;
    
public:
    static size_t get() noexcept {
        thread_local size_t cached = next_index_.value.fetch_add(1, 
            std::memory_order_relaxed);
        return cached;
    }
};

inline size_t get_thread_shard_index() noexcept {
    return ThreadShardIndex::get();
}

} // namespace detail
```

