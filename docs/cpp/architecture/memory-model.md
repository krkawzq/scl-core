# Memory Model

This document describes SCL-Core's memory management strategy, focusing on the Registry system for lifetime tracking and Python integration.

## Overview

SCL-Core uses explicit memory management with centralized tracking via the `Registry` class. This design provides:

- **Deterministic cleanup** - No garbage collection delays
- **Python integration** - Safe memory transfer across language boundaries
- **Reference counting** - Shared buffers with automatic cleanup
- **Thread safety** - Concurrent access without data races

## Memory Ownership Patterns

### 1. Owning Pointers

Simple allocations with single ownership:

```cpp
auto& reg = scl::get_registry();

// Allocate and register
Real* data = reg.new_array<Real>(1000);

// Use data...

// Cleanup
reg.unregister_ptr(data);  // Calls delete[]
```

### 2. Reference-Counted Buffers

Shared buffers with multiple aliases:

```cpp
auto& reg = scl::get_registry();

// Allocate main buffer
Real* main_ptr = new Real[1000];

// Create aliases (e.g., column views)
std::vector<void*> aliases;
for (size_t i = 0; i < 10; ++i) {
    aliases.push_back(main_ptr + i * 100);
}

// Register with reference counting
BufferID id = reg.register_buffer_with_aliases(
    main_ptr,                    // Real pointer to free
    1000 * sizeof(Real),         // Byte size
    aliases,                     // Alias pointers
    AllocType::ArrayNew          // Allocation type
);

// Refcount = 11 (main + 10 aliases)

// Unregister aliases
for (auto* alias : aliases) {
    reg.unregister_ptr(alias);  // Decrements refcount
}

// Unregister main pointer
reg.unregister_ptr(main_ptr);  // Refcount = 0, frees memory
```

### 3. Non-Owning Views

Pointers to memory not managed by Registry:

```cpp
// External buffer (e.g., from Python)
Real* external_data = /* ... */;

// Create non-owning view
scl::Sparse<Real, true> matrix = 
    scl::Sparse<Real, true>::wrap_traditional(
        external_data,
        indices,
        indptr,
        rows, cols, nnz
    );

// matrix does not own data
// Caller responsible for lifetime
```

## Registry Architecture

### Sharded Design

The Registry uses sharding to reduce lock contention:

```
Registry
├── Shard 0 (hash % num_shards == 0)
│   ├── PtrMap: { ptr → PtrRecord }
│   └── BufferMap: { ptr → RefCountedBuffer }
├── Shard 1
│   ├── PtrMap
│   └── BufferMap
└── ...
```

**Benefits:**
- Parallel access to different shards
- Reduced lock contention
- Better cache locality per shard

### Allocation Types

```cpp
enum class AllocType {
    ArrayNew,      // new[] → delete[]
    ScalarNew,     // new → delete
    AlignedAlloc,  // aligned_alloc → aligned_free
    Custom         // Custom deleter function
};
```

### API Overview

```cpp
class Registry {
    // Simple pointer registration
    template <typename T>
    T* new_array(size_t count);
    
    void register_ptr(void* ptr, size_t bytes, AllocType type);
    void unregister_ptr(void* ptr);
    
    // Reference-counted buffers
    BufferID register_buffer_with_aliases(
        void* real_ptr,
        size_t byte_size,
        std::span<void*> alias_ptrs,
        AllocType type
    );
    
    // Query
    bool is_registered(void* ptr) const;
    size_t get_total_bytes() const;
    size_t get_num_pointers() const;
};
```

## Sparse Matrix Memory

### Discontiguous Storage

`Sparse<T, IsCSR>` uses pointer arrays for flexible memory management:

```cpp
template <typename T, bool IsCSR>
struct Sparse {
    using Pointer = T*;
    
    Pointer* data_ptrs_;      // Array of data pointers
    Pointer* indices_ptrs_;   // Array of index pointers
    Index* lengths_;          // Array of lengths
    
    Index rows_, cols_, nnz_;
};
```

**Memory Layout:**

```
Row 0: data_ptrs_[0] → [v0, v1, v2]
       indices_ptrs_[0] → [i0, i1, i2]
       lengths_[0] = 3

Row 1: data_ptrs_[1] → [v3, v4]
       indices_ptrs_[1] → [i3, i4]
       lengths_[1] = 2

...
```

### Block Allocation

For efficiency, rows/columns are allocated in blocks:

```cpp
struct BlockStrategy {
    Index min_block_elements = 4096;      // 16KB for float32
    Index max_block_elements = 262144;    // 1MB for float32
    
    Index compute_block_size(Index total_nnz, Index primary_dim) {
        // Balance:
        // - Memory reuse (larger blocks)
        // - Partial release (smaller blocks)
        // - Parallelism (multiple blocks)
        
        Index avg_nnz = total_nnz / primary_dim;
        Index block_size = std::max(min_block_elements, avg_nnz * 16);
        return std::min(block_size, max_block_elements);
    }
};
```

### Contiguous Conversion

For external libraries expecting contiguous CSR/CSC:

```cpp
// Convert to contiguous arrays
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// arrays.data, arrays.indices, arrays.indptr are registered
// Python can take ownership without copying

// Cleanup
auto& reg = scl::get_registry();
reg.unregister_ptr(arrays.data);
reg.unregister_ptr(arrays.indices);
reg.unregister_ptr(arrays.indptr);
```

## Python Integration

### Zero-Copy Transfer

Registry enables zero-copy memory transfer to Python:

```cpp
// C++ side: Create and register
auto& reg = scl::get_registry();
Real* data = reg.new_array<Real>(1000);

// ... fill data ...

// Python binding: Transfer ownership
py::capsule deleter(data, [](void* ptr) {
    scl::get_registry().unregister_ptr(ptr);
});

return py::array_t<Real>(
    {1000},           // Shape
    {sizeof(Real)},   // Strides
    data,             // Data pointer
    deleter           // Cleanup callback
);
```

**Flow:**
1. C++ allocates and registers memory
2. Python takes ownership via capsule
3. When Python array is deleted, capsule calls deleter
4. Deleter unregisters from Registry
5. Registry frees memory

### Reference Counting

For shared buffers (e.g., sparse matrix blocks):

```cpp
// C++ side: Register with aliases
BufferID id = reg.register_buffer_with_aliases(
    real_ptr, byte_size, aliases, AllocType::ArrayNew);

// Python side: Each alias gets a separate array
for (auto* alias : aliases) {
    py::capsule deleter(alias, [](void* ptr) {
        scl::get_registry().unregister_ptr(ptr);
    });
    
    // Create Python array for this alias
    // Deleter decrements refcount when array is deleted
}

// Memory freed when last Python array is deleted
```

## Aligned Allocation

For SIMD operations:

```cpp
namespace scl::memory {
    // Allocate aligned memory
    template <typename T>
    T* aligned_alloc(size_t count, size_t alignment = 64) {
        void* ptr = std::aligned_alloc(alignment, count * sizeof(T));
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    
    // Free aligned memory
    void aligned_free(void* ptr) {
        std::free(ptr);
    }
}
```

**Usage with Registry:**

```cpp
auto& reg = scl::get_registry();

// Allocate aligned
Real* data = scl::memory::aligned_alloc<Real>(1000, 64);

// Register with custom deleter
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::AlignedAlloc);

// Cleanup
reg.unregister_ptr(data);  // Calls aligned_free
```

## Workspace Pools

For temporary storage in parallel loops:

```cpp
template <typename T>
class WorkspacePool {
    std::vector<std::unique_ptr<T[]>> workspaces_;
    
public:
    WorkspacePool(size_t num_threads, size_t workspace_size) {
        for (size_t i = 0; i < num_threads; ++i) {
            workspaces_.push_back(std::make_unique<T[]>(workspace_size));
        }
    }
    
    T* get(size_t thread_rank) {
        return workspaces_[thread_rank].get();
    }
};
```

**Usage:**

```cpp
// Create pool
WorkspacePool<Real> pool(num_threads, 1024);

// Use in parallel loop
parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);
    // Use workspace for temporary storage
});

// Automatic cleanup when pool goes out of scope
```

## Memory Debugging

### Statistics

```cpp
auto& reg = scl::get_registry();

// Query current usage
size_t num_ptrs = reg.get_num_pointers();
size_t num_buffers = reg.get_num_buffers();
size_t total_bytes = reg.get_total_bytes();

std::cout << "Registered: " << num_ptrs << " pointers, "
          << num_buffers << " buffers, "
          << total_bytes << " bytes\n";
```

### Leak Detection

In debug builds, Registry warns about leaked memory:

```cpp
// At program exit
~Registry() {
    #ifdef SCL_DEBUG
    if (get_num_pointers() > 0 || get_num_buffers() > 0) {
        std::cerr << "WARNING: Memory leak detected!\n";
        std::cerr << "  Pointers: " << get_num_pointers() << "\n";
        std::cerr << "  Buffers: " << get_num_buffers() << "\n";
        std::cerr << "  Bytes: " << get_total_bytes() << "\n";
    }
    #endif
}
```

## Best Practices

### 1. Use RAII

Wrap Registry operations in RAII guards:

```cpp
class RegistryGuard {
    void* ptr_;
    
public:
    explicit RegistryGuard(void* ptr) : ptr_(ptr) {}
    
    ~RegistryGuard() {
        if (ptr_) {
            scl::get_registry().unregister_ptr(ptr_);
        }
    }
    
    void* release() {
        void* p = ptr_;
        ptr_ = nullptr;
        return p;
    }
};
```

### 2. Prefer Stack Allocation

For small, temporary buffers:

```cpp
// BAD: Heap allocation for small buffer
Real* temp = new Real[100];
// ...
delete[] temp;

// GOOD: Stack allocation
Real temp[100];
// Automatic cleanup
```

### 3. Pre-Allocate Workspaces

Avoid allocations in hot loops:

```cpp
// BAD: Allocate per iteration
for (size_t i = 0; i < n; ++i) {
    std::vector<Real> temp(1000);  // Allocation!
    // ...
}

// GOOD: Pre-allocate workspace
std::vector<Real> workspace(1000);
for (size_t i = 0; i < n; ++i) {
    // Reuse workspace
}
```

### 4. Document Ownership

Clearly document who owns memory:

```cpp
// Returns owning pointer - caller must free
Real* allocate_buffer(size_t n);

// Returns non-owning view - do not free
const Real* get_data_view() const;

// Takes ownership of ptr
void consume_buffer(Real* ptr);
```

## Performance Considerations

### Registry Overhead

- **Per-pointer:** ~32 bytes (hash table slot + metadata)
- **Per-buffer:** ~48 bytes (RefCountedBuffer + hash table slot)
- **Lookup:** O(1) average, O(n) worst case (hash collision)

### When to Use Registry

**Use Registry for:**
- Memory transferred to Python
- Shared buffers with multiple aliases
- Long-lived allocations

**Don't use Registry for:**
- Stack-allocated buffers
- Short-lived temporaries in hot loops
- Memory managed by external libraries

---

::: tip Memory Safety
Always pair allocations with cleanup. Use RAII guards to ensure cleanup even in the presence of exceptions.
:::

