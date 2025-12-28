# Memory Model

SCL-Core's memory management system is designed for three goals: deterministic cleanup, zero-copy Python integration, and high-performance concurrent access. This document explains the Registry system and memory ownership patterns.

## Design Principles

### 1. Explicit Over Implicit

Every memory allocation and deallocation is explicit and trackable. No hidden allocations in constructors, no implicit copies, no garbage collection pauses.

### 2. Deterministic Cleanup

Memory is freed immediately when ownership is released, not at some future garbage collection cycle. This predictable behavior is critical for large-dataset workloads.

### 3. Zero-Copy Integration

Memory can be transferred across language boundaries (C++ ↔ Python) without copying data. Python takes ownership of C++-allocated buffers directly through the capsule protocol.

### 4. Thread Safety

Multiple threads can allocate and free memory concurrently without data races. The Registry uses sharding to minimize lock contention.

## Memory Ownership Models

### Simple Ownership

Single owner, single pointer, deterministic lifetime:

```cpp
auto& reg = scl::get_registry();

// Allocate and register
Real* data = reg.new_array<Real>(1000);

// Use data...
process(data, 1000);

// Cleanup - memory freed immediately
reg.unregister_ptr(data);
```

**Characteristics:**
- One pointer registered
- Unregister immediately frees memory
- No reference counting overhead
- Ideal for temporary buffers

**When to Use:**
- Short-lived allocations
- Single-threaded ownership
- Simple data structures

### Reference-Counted Buffers

Multiple aliases, shared ownership, automatic cleanup when last reference is released:

```cpp
auto& reg = scl::get_registry();

// Allocate main buffer
Real* main_ptr = new Real[1000];

// Create aliases (e.g., matrix columns)
std::vector<void*> aliases;
for (size_t i = 0; i < 10; ++i) {
    aliases.push_back(main_ptr + i * 100);  // Column i
}

// Register with reference counting
BufferID id = reg.register_buffer_with_aliases(
    main_ptr,                    // Real pointer to free
    1000 * sizeof(Real),         // Byte size
    aliases,                     // Alias pointers
    AllocType::ArrayNew          // How to free: delete[]
);

// Initial refcount = 11 (main + 10 aliases)

// Unregister aliases as they go out of scope
for (auto* alias : aliases) {
    reg.unregister_ptr(alias);  // Decrements refcount
}
// Refcount = 1 (main only)

// Unregister main pointer
reg.unregister_ptr(main_ptr);  // Refcount = 0, memory freed
```

**Characteristics:**
- Multiple pointers refer to same underlying buffer
- Memory freed only when last reference is released
- Thread-safe reference counting (atomic operations)
- Handles aliasing correctly

**When to Use:**
- Sparse matrix blocks (multiple row pointers into same block)
- Shared buffers across multiple views
- Python integration (multiple NumPy arrays viewing same memory)
- Complex ownership graphs

### Non-Owning Views

Pointers to memory managed externally (not registered in Registry):

```cpp
// External buffer - e.g., from Python NumPy array
Real* external_data = get_numpy_buffer();

// Create non-owning view without registering
scl::Sparse<Real, true> matrix = 
    scl::Sparse<Real, true>::wrap_traditional(
        external_data,
        indices,
        indptr,
        rows, cols, nnz
    );

// Use matrix...
process(matrix);

// matrix destruction does NOT free external_data
// Caller (Python) remains responsible for lifetime
```

**Characteristics:**
- No registration with Registry
- No automatic cleanup
- Caller retains ownership
- Zero-overhead wrapper

**When to Use:**
- Python-owned buffers
- Memory-mapped files
- Stack-allocated buffers
- Any externally-managed memory

## Registry Architecture

### Sharded Design

The Registry uses multiple independent shards to reduce lock contention:

```
Registry (global singleton)
├── Shard 0 (hash(ptr) % num_shards == 0)
│   ├── Mutex
│   ├── PtrMap: unordered_map<void*, PtrRecord>
│   └── BufferMap: unordered_map<void*, RefCountedBuffer>
│
├── Shard 1 (hash(ptr) % num_shards == 1)
│   ├── Mutex
│   ├── PtrMap
│   └── BufferMap
│
├── ...
│
└── Shard N-1
```

**Benefits:**
- Parallel allocations go to different shards - no contention
- Lock-free reads via const operations on different shards
- Scales linearly with thread count for independent allocations
- Typical configuration: 16 shards

**Shard Selection:**

```cpp
size_t shard_index = hash_ptr(ptr) % num_shards;
```

Simple modulo hashing distributes pointers evenly across shards.

### Data Structures

#### PtrRecord

Simple ownership tracking:

```cpp
struct PtrRecord {
    void* ptr;         // Registered pointer
    size_t bytes;      // Allocation size
    AllocType type;    // How to free (delete[], free, etc.)
};
```

#### RefCountedBuffer

Shared ownership with aliases:

```cpp
struct RefCountedBuffer {
    void* real_ptr;                     // Actual allocation to free
    size_t byte_size;                   // Size in bytes
    AllocType type;                     // Deallocation method
    std::atomic<size_t> refcount;       // Thread-safe reference count
    std::unordered_set<void*> aliases;  // All aliases to this buffer
};
```

### Allocation Types

```cpp
enum class AllocType {
    ArrayNew,      // new[] → delete[]
    ScalarNew,     // new → delete
    AlignedAlloc,  // aligned_alloc → free
    Custom         // Custom deleter function
};
```

The Registry calls the appropriate deleter based on AllocType when refcount reaches zero.

### Core API

```cpp
class Registry {
public:
    // Simple allocation
    template <typename T>
    T* new_array(size_t count);
    
    // Registration
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
    size_t get_num_buffers() const;
    
    // Debug/profiling
    void print_statistics(std::ostream& os) const;
};
```

### Global Access

```cpp
Registry& get_registry();  // Thread-safe singleton
```

The global Registry is initialized on first use (Meyers singleton) and lives for program duration.

## Sparse Matrix Memory

### Discontiguous Storage

SCL-Core's `Sparse<T, IsCSR>` uses pointer arrays instead of traditional contiguous CSR:

```cpp
template <typename T, bool IsCSR>
struct Sparse {
    using Pointer = T*;
    
    Pointer* data_ptrs_;      // Array of data pointers [primary_dim]
    Pointer* indices_ptrs_;   // Array of index pointers [primary_dim]
    Index* lengths_;          // Array of lengths [primary_dim]
    
    Index rows_, cols_, nnz_;
};
```

**Memory Layout:**

```
Traditional CSR (contiguous):
    data:    [v0 v1 v2 | v3 v4 | v5 v6 v7 v8]  (single allocation)
    indices: [c0 c1 c2 | c3 c4 | c5 c6 c7 c8]  (single allocation)
    indptr:  [0, 3, 5, 9]

SCL-Core Discontiguous:
    Row 0: data_ptrs_[0] → [v0 v1 v2]      indices_ptrs_[0] → [c0 c1 c2]
    Row 1: data_ptrs_[1] → [v3 v4]         indices_ptrs_[1] → [c3 c4]
    Row 2: data_ptrs_[2] → [v5 v6 v7 v8]   indices_ptrs_[2] → [c5 c6 c7 c8]
```

**Advantages:**
- Block allocation: Multiple rows per block
- Flexible ownership: Each block can be managed independently
- Easy slicing: Row subset is just subset of pointer arrays
- Reference counting: Multiple views share same blocks
- Python integration: Each block can be separate NumPy array

**Trade-offs:**
- Slightly more complex indexing (one indirection)
- Pointer array overhead (8 bytes per row)
- Not compatible with libraries expecting contiguous CSR

### Block Allocation Strategy

Rows/columns are allocated in blocks for efficiency:

```cpp
struct BlockStrategy {
    // Configuration
    Index min_block_elements = 4096;      // Min: 16KB for float32
    Index max_block_elements = 262144;    // Max: 1MB for float32
    
    Index compute_block_size(Index total_nnz, Index primary_dim) const {
        // Target: 4-8 rows per block for good parallelism
        Index avg_nnz_per_row = total_nnz / primary_dim;
        Index target_rows_per_block = 8;
        
        // Block size = avg_nnz_per_row * target_rows_per_block
        Index block_size = avg_nnz_per_row * target_rows_per_block;
        
        // Clamp to [min, max] range
        block_size = std::max(block_size, min_block_elements);
        block_size = std::min(block_size, max_block_elements);
        
        return block_size;
    }
};
```

**Allocation Process:**

```cpp
// Allocate blocks for sparse matrix
auto& reg = get_registry();
BlockStrategy strategy;
Index block_size = strategy.compute_block_size(nnz, rows);

for (Index start = 0; start < rows; start += rows_per_block) {
    Index end = std::min(start + rows_per_block, rows);
    Index block_nnz = /* count non-zeros in [start, end) */;
    
    // Allocate block
    Real* data_block = reg.new_array<Real>(block_nnz);
    Index* idx_block = reg.new_array<Index>(block_nnz);
    
    // Assign to rows
    Index offset = 0;
    for (Index i = start; i < end; ++i) {
        data_ptrs_[i] = data_block + offset;
        indices_ptrs_[i] = idx_block + offset;
        lengths_[i] = row_nnz[i];
        offset += row_nnz[i];
    }
}
```

### Contiguous Conversion

For interoperability with libraries expecting traditional CSR:

```cpp
template <typename T, bool IsCSR>
ContiguousArraysT<T> to_contiguous_arrays(const Sparse<T, IsCSR>& matrix) {
    auto& reg = get_registry();
    
    // Allocate contiguous arrays
    T* data = reg.new_array<T>(matrix.nnz());
    Index* indices = reg.new_array<Index>(matrix.nnz());
    Index* indptr = reg.new_array<Index>(matrix.primary_dim() + 1);
    
    // Copy data from discontiguous to contiguous
    indptr[0] = 0;
    Index offset = 0;
    for (Index i = 0; i < matrix.primary_dim(); ++i) {
        Index len = matrix.primary_length(i);
        std::copy_n(matrix.primary_values(i).ptr, len, data + offset);
        std::copy_n(matrix.primary_indices(i).ptr, len, indices + offset);
        offset += len;
        indptr[i + 1] = offset;
    }
    
    return {data, indices, indptr, matrix.nnz(), matrix.primary_dim()};
}
```

All pointers in `ContiguousArraysT` are registered with the Registry. Python can take ownership without copying.

## Python Integration

### Zero-Copy Transfer

Transfer ownership from C++ to Python:

```cpp
// C++ side: Allocate and populate
auto& reg = scl::get_registry();
Real* data = reg.new_array<Real>(1000);
fill_data(data, 1000);

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

**Ownership Flow:**
1. C++ allocates memory and registers with Registry
2. Python array takes ownership via PyCapsule
3. Python reference count tracks lifetime
4. When Python array is deleted, capsule deleter is called
5. Deleter unregisters from Registry
6. Registry frees memory using appropriate deleter

**No Copy:** Data pointer is transferred directly, no memcpy.

### Reference Counting for Shared Buffers

Transfer multiple views of same buffer to Python:

```cpp
// C++ side: Allocate block with aliases
auto& reg = scl::get_registry();
Real* main_ptr = new Real[10000];

std::vector<void*> aliases;
for (size_t i = 0; i < 100; ++i) {
    aliases.push_back(main_ptr + i * 100);  // Column i
}

BufferID id = reg.register_buffer_with_aliases(
    main_ptr, 10000 * sizeof(Real), aliases, AllocType::ArrayNew);

// Python binding: Create array for each alias
py::list result;
for (size_t i = 0; i < 100; ++i) {
    Real* col_ptr = static_cast<Real*>(aliases[i]);
    
    py::capsule deleter(col_ptr, [](void* ptr) {
        scl::get_registry().unregister_ptr(ptr);
    });
    
    py::array_t<Real> col(
        {100},            // Shape
        {sizeof(Real)},   // Strides
        col_ptr,          // Data pointer
        deleter           // Cleanup callback
    );
    
    result.append(col);
}

return result;  // List of NumPy arrays, all views of same buffer
```

**Lifetime Management:**
- Initial refcount = 101 (main + 100 aliases)
- Each Python array holds one reference
- When any array is deleted, refcount decrements
- When last array is deleted, refcount → 0, memory freed
- Thread-safe: Python can delete arrays from any thread

## Aligned Allocation

SIMD operations require properly aligned memory:

```cpp
namespace scl::memory {
    template <typename T>
    T* aligned_alloc(size_t count, size_t alignment = 64) {
        // alignment must be power of 2
        assert((alignment & (alignment - 1)) == 0);
        
        void* ptr = std::aligned_alloc(alignment, count * sizeof(T));
        if (!ptr) throw std::bad_alloc();
        
        return static_cast<T*>(ptr);
    }
    
    void aligned_free(void* ptr) {
        std::free(ptr);  // C++17: free works for aligned_alloc
    }
}
```

**Usage with Registry:**

```cpp
auto& reg = scl::get_registry();

// Allocate aligned memory
Real* data = scl::memory::aligned_alloc<Real>(1000, 64);

// Register with aligned_alloc type
reg.register_ptr(data, 1000 * sizeof(Real), AllocType::AlignedAlloc);

// Use for SIMD operations
process_simd(data, 1000);

// Cleanup - Registry calls free()
reg.unregister_ptr(data);
```

**Alignment Requirements:**
- AVX2: 32-byte alignment (8 floats)
- AVX-512: 64-byte alignment (16 floats)
- Cache lines: 64-byte alignment
- Recommendation: Always use 64-byte alignment

## Workspace Pools

Pre-allocated thread-local workspaces avoid allocations in hot loops:

```cpp
template <typename T>
class WorkspacePool {
    std::vector<std::unique_ptr<T[]>> workspaces_;
    size_t workspace_size_;
    
public:
    WorkspacePool(size_t num_threads, size_t workspace_size)
        : workspace_size_(workspace_size) {
        workspaces_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workspaces_.push_back(std::make_unique<T[]>(workspace_size));
        }
    }
    
    T* get(size_t thread_rank) {
        return workspaces_[thread_rank].get();
    }
    
    size_t size() const {
        return workspace_size_;
    }
};
```

**Usage:**

```cpp
// Create pool before parallel loop
const size_t num_threads = get_num_threads();
const size_t workspace_size = compute_required_size();
WorkspacePool<Real> pool(num_threads, workspace_size);

// Use in parallel loop
parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);  // Lock-free access
    
    // Use workspace for temporary storage
    compute_with_workspace(workspace, workspace_size);
});

// Automatic cleanup when pool goes out of scope
```

**Benefits:**
- Zero allocations in hot loop
- No synchronization (thread-local)
- Cache-friendly (reused memory)
- Predictable performance

## Memory Debugging

### Statistics

Query current memory usage:

```cpp
auto& reg = scl::get_registry();

size_t num_ptrs = reg.get_num_pointers();
size_t num_buffers = reg.get_num_buffers();
size_t total_bytes = reg.get_total_bytes();

std::cout << "Registry Statistics:\n";
std::cout << "  Pointers: " << num_ptrs << "\n";
std::cout << "  Buffers: " << num_buffers << "\n";
std::cout << "  Memory: " << (total_bytes / 1024 / 1024) << " MB\n";
```

### Leak Detection

In debug builds, Registry warns about memory leaks:

```cpp
// At program exit
Registry::~Registry() {
    #ifdef SCL_DEBUG
    if (get_num_pointers() > 0 || get_num_buffers() > 0) {
        std::cerr << "WARNING: Memory leak detected!\n";
        std::cerr << "  Pointers: " << get_num_pointers() << "\n";
        std::cerr << "  Buffers: " << get_num_buffers() << "\n";
        std::cerr << "  Total: " << get_total_bytes() << " bytes\n";
        
        // Print details of leaked allocations
        print_statistics(std::cerr);
    }
    #endif
    
    // Force cleanup all remaining allocations
    cleanup_all();
}
```

### Allocation Tracking

Enable detailed tracking:

```cpp
#define SCL_TRACK_ALLOCATIONS

// Now each allocation records:
// - Allocation site (file:line)
// - Stack trace
// - Timestamp
// - Thread ID

// Query allocation details
auto info = reg.get_allocation_info(ptr);
std::cout << "Allocated at " << info.file << ":" << info.line << "\n";
std::cout << "Thread: " << info.thread_id << "\n";
std::cout << "Time: " << info.timestamp << "\n";
```

## Best Practices

### 1. RAII Wrappers

Wrap Registry operations in RAII guards:

```cpp
template <typename T>
class RegistryGuard {
    T* ptr_;
    
public:
    explicit RegistryGuard(size_t count) {
        ptr_ = scl::get_registry().new_array<T>(count);
    }
    
    ~RegistryGuard() {
        if (ptr_) {
            scl::get_registry().unregister_ptr(ptr_);
        }
    }
    
    // Move-only
    RegistryGuard(RegistryGuard&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    RegistryGuard& operator=(RegistryGuard&& other) noexcept {
        if (this != &other) {
            if (ptr_) scl::get_registry().unregister_ptr(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    // Non-copyable
    RegistryGuard(const RegistryGuard&) = delete;
    RegistryGuard& operator=(const RegistryGuard&) = delete;
    
    T* get() const { return ptr_; }
    T* release() {
        T* p = ptr_;
        ptr_ = nullptr;
        return p;
    }
};

// Usage
void process_data() {
    RegistryGuard<Real> buffer(1000);
    
    // Use buffer.get()
    compute(buffer.get(), 1000);
    
    // Automatic cleanup even if exception thrown
}
```

### 2. Prefer Stack Allocation

For small, temporary buffers:

```cpp
// AVOID: Heap allocation for small buffer
Real* temp = new Real[100];
process(temp, 100);
delete[] temp;

// PREFER: Stack allocation
Real temp[100];
process(temp, 100);
// Automatic cleanup
```

Stack allocation is faster and safer (no manual memory management).

### 3. Pre-Allocate Workspaces

Avoid repeated allocations in loops:

```cpp
// INEFFICIENT: Allocate per iteration
for (size_t i = 0; i < n; ++i) {
    std::vector<Real> temp(1000);  // Allocate + free each iteration!
    compute_with_temp(temp);
}

// EFFICIENT: Pre-allocate workspace
std::vector<Real> workspace(1000);
for (size_t i = 0; i < n; ++i) {
    compute_with_temp(workspace);  // Reuse allocation
}
```

### 4. Document Ownership

Clearly document who owns memory:

```cpp
// Returns owning pointer - caller must free via Registry
Real* allocate_buffer(size_t n);

// Returns non-owning view - do NOT free
const Real* get_data_view() const;

// Takes ownership of ptr - will free via Registry
void consume_buffer(Real* ptr);

// Borrows ptr - does NOT take ownership
void process_buffer(const Real* ptr, size_t n);
```

## Performance Considerations

### Registry Overhead

**Per-Pointer Overhead:**
- Hash table slot: ~24 bytes
- PtrRecord: 24 bytes
- Total: ~48 bytes per registered pointer

**Per-Buffer Overhead:**
- Hash table slot: ~24 bytes
- RefCountedBuffer: 56 bytes + 8 bytes per alias
- Atomic refcount: Minimal contention with sharding
- Total: ~80 bytes + aliases

**Lookup Performance:**
- Average: O(1) with low constant factor
- Worst case: O(n) for shard (rare with good hash function)
- Typical: < 50ns for register/unregister
- Contention: Minimal due to sharding

### When to Use Registry

**Use Registry For:**
- Memory transferred to Python (required)
- Shared buffers with aliases (reference counting)
- Long-lived allocations (lifecycle management)
- Large allocations (> 1MB) worth tracking

**Do NOT Use Registry For:**
- Stack-allocated buffers (no need)
- Short-lived temporaries in hot loops (overhead dominates)
- Small allocations (< 1KB) (overhead not worthwhile)
- Memory managed by external libraries (not our responsibility)

**Rule of Thumb:** If lifetime is obvious and local, skip Registry. If lifetime crosses function/module boundaries or involves Python, use Registry.

---

::: tip Memory Safety Through Design
SCL-Core's memory model prevents entire classes of bugs through design: use-after-free is impossible with RAII wrappers, memory leaks are detected automatically in debug builds, and zero-copy Python integration eliminates buffer copy bugs.
:::
