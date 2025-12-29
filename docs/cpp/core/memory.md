---
title: Memory Management
description: Aligned allocation, RAII wrappers, and memory utilities
---

# Memory Management

SCL-Core provides modern C++ memory management utilities with aligned allocation, RAII wrappers, and automatic lifetime management.

## Aligned Allocation

### Basic Allocation

```cpp
namespace scl::memory {
    // Allocate aligned memory (returns unique_ptr)
    template <typename T>
    auto aligned_alloc(Size count, std::size_t alignment = DEFAULT_ALIGNMENT)
        -> std::unique_ptr<T[], AlignedDeleter<T>>;
    
    // Legacy: Raw pointer allocation (deprecated)
    template <typename T>
    [[deprecated]] auto aligned_alloc_raw(Size count, std::size_t alignment = DEFAULT_ALIGNMENT)
        -> T*;
    
    // Free aligned memory
    template <typename T>
    void aligned_free(T* ptr, std::size_t alignment = DEFAULT_ALIGNMENT);
}
```

### Usage Examples

```cpp
// Modern: Use unique_ptr (recommended)
auto buffer = memory::aligned_alloc<Real>(1000, SCL_ALIGNMENT);
Array<Real> view = {buffer.get(), 1000};

// Access data
for (Size i = 0; i < 1000; ++i) {
    view[i] = static_cast<Real>(i);
}

// Automatic cleanup when buffer goes out of scope
```

### Alignment Constants

```cpp
namespace scl::memory {
    inline constexpr std::size_t DEFAULT_ALIGNMENT = 64;  // AVX-512 alignment
    inline constexpr std::size_t STREAM_ALIGNMENT = 64;    // Non-temporal stores
    inline constexpr std::size_t CACHE_LINE_SIZE = 64;    // Cache line size
}
```

## RAII Wrappers

### AlignedBuffer

```cpp
namespace scl::memory {
    template <typename T>
    struct AlignedBuffer {
        AlignedBuffer(Size count, std::size_t alignment = DEFAULT_ALIGNMENT);
        
        // Non-copyable, movable
        AlignedBuffer(const AlignedBuffer&) = delete;
        AlignedBuffer& operator=(const AlignedBuffer&) = delete;
        AlignedBuffer(AlignedBuffer&&) noexcept = default;
        AlignedBuffer& operator=(AlignedBuffer&&) noexcept = default;
        
        // Get array view
        Array<T> array() noexcept;
        Array<const T> array() const noexcept;
        
        // Direct access
        T* data() noexcept;
        const T* data() const noexcept;
        Size size() const noexcept;
    };
}
```

**Usage**:
```cpp
// Create buffer
AlignedBuffer<Real> buffer(1000);
Array<Real> view = buffer.array();

// Use view
for (Size i = 0; i < view.size(); ++i) {
    view[i] = static_cast<Real>(i);
}

// Automatic cleanup
```

## Memory Utilities

### Prefetching

```cpp
// Prefetch for read
SCL_PREFETCH_READ(ptr, locality);  // locality: 0-3 (0=no locality, 3=strong locality)

// Prefetch for write
SCL_PREFETCH_WRITE(ptr, locality);

// Example: Prefetch ahead in loop
for (Size i = 0; i < n; ++i) {
    if (i + PREFETCH_DISTANCE < n) {
        SCL_PREFETCH_READ(&data[i + PREFETCH_DISTANCE], 0);
    }
    process(data[i]);
}
```

### Memory Operations

```cpp
// Zero memory
void zero(Array<T> data);

// Copy memory
void copy(Array<const T> src, Array<T> dst);

// Fill memory
void fill(Array<T> data, T value);
```

## Registry System

The registry system manages memory lifetime for sparse matrices and shared data:

```cpp
namespace scl::registry {
    // Register pointer (increment reference count)
    void alias_incref(void* ptr, std::size_t size);
    
    // Unregister pointer (decrement reference count)
    void alias_decref(void* ptr);
    
    // Batch operations
    void alias_incref_batch(const void* const* ptrs, const std::size_t* sizes, Size count);
    void alias_decref_batch(const void* const* ptrs, Size count);
    
    // Check if pointer is registered
    bool is_registered(const void* ptr);
}
```

### Usage with Sparse Matrices

```cpp
// Factory methods automatically register data
auto matrix = CSR::create(rows, cols, nnz);
// Data is registered in registry

// Slicing creates aliases
auto submatrix = matrix.slice_rows(0, 100);
// submatrix shares data, ref count incremented

// When destroyed, ref count decremented
// Data freed when ref count reaches zero
```

### Manual Registration

For external data (e.g., from Python):

```cpp
// Register external arrays
Real* data_ptr = ...;
Index* indices_ptr = ...;
Size data_size = ...;
Size indices_size = ...;

registry::alias_incref(data_ptr, data_size * sizeof(Real));
registry::alias_incref(indices_ptr, indices_size * sizeof(Index));

// Use in matrix
auto matrix = CSR::wrap_traditional(rows, cols, nnz, ...);

// Unregister when done
registry::alias_decref(data_ptr);
registry::alias_decref(indices_ptr);
```

## Configuration

### Memory Alignment

```cpp
// Default alignment (64 bytes for AVX-512)
constexpr std::size_t DEFAULT_ALIGNMENT = 64;

// Cache line size
constexpr std::size_t CACHE_LINE_SIZE = 64;

// Prefetch configuration
constexpr std::size_t DEFAULT_PREFETCH_DISTANCE = 8;
constexpr std::size_t DEFAULT_MAX_PREFETCHES = 16;
```

### Stream Copy Threshold

For large arrays, non-temporal stores can improve performance:

```cpp
// Threshold for using non-temporal stores
constexpr std::size_t STREAM_THRESHOLD = 256 * 1024;  // 256KB

// Arrays larger than threshold use stream copy
if (size > STREAM_THRESHOLD) {
    // Use non-temporal stores
}
```

## Best Practices

### 1. Use RAII Wrappers

```cpp
// Good: Automatic cleanup
{
    AlignedBuffer<Real> buffer(1000);
    Array<Real> view = buffer.array();
    // Use view
}  // Automatic cleanup

// Avoid: Manual memory management
Real* ptr = memory::aligned_alloc_raw<Real>(1000);
// ... use ptr ...
memory::aligned_free(ptr, SCL_ALIGNMENT);  // Easy to forget
```

### 2. Prefer unique_ptr for Temporary Buffers

```cpp
// Good: Clear ownership
auto buffer = memory::aligned_alloc<Real>(n);
Array<Real> view = {buffer.get(), n};

// Avoid: Raw pointers
Real* buffer = memory::aligned_alloc_raw<Real>(n);  // Deprecated
```

### 3. Align for SIMD

```cpp
// Always align data used with SIMD
auto buffer = memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
Array<Real> view = {buffer.get(), n};

// Use with SIMD
namespace s = scl::simd;
const s::SimdTag d;
auto v = s::Load(d, view.data());  // Requires alignment
```

### 4. Use Prefetching Strategically

```cpp
// Prefetch ahead in hot loops
constexpr Size PREFETCH_DISTANCE = 64;
for (Size i = 0; i < n; ++i) {
    if (i + PREFETCH_DISTANCE < n) {
        SCL_PREFETCH_READ(&data[i + PREFETCH_DISTANCE], 0);
    }
    process(data[i]);
}
```

### 5. Register External Data

```cpp
// When receiving data from external sources (Python, etc.)
Real* external_data = ...;
Size size = ...;

// Register before use
registry::alias_incref(external_data, size * sizeof(Real));

// Use in SCL-Core
auto matrix = CSR::wrap_traditional(..., external_data, ...);

// Unregister when done
registry::alias_decref(external_data);
```

## Performance Considerations

### Alignment Impact

Aligned memory is crucial for SIMD performance:

```cpp
// Aligned: Fast SIMD operations
auto aligned = memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
// SIMD operations are fast

// Unaligned: Slower SIMD operations
auto unaligned = new Real[n];  // May not be aligned
// SIMD operations may be slower or require special handling
```

### Cache-Friendly Access

```cpp
// Sequential access: Optimal
for (Size i = 0; i < n; ++i) {
    process(data[i]);
}

// Strided access: May cause cache misses
for (Size i = 0; i < n; i += stride) {
    process(data[i]);
}
```

### Prefetching Benefits

Prefetching can hide memory latency:

```cpp
// Without prefetching
for (Size i = 0; i < n; ++i) {
    process(data[i]);  // May wait for memory
}

// With prefetching
for (Size i = 0; i < n; ++i) {
    if (i + 64 < n) {
        SCL_PREFETCH_READ(&data[i + 64], 0);
    }
    process(data[i]);  // Data may already be in cache
}
```

## Related Documentation

- [Core Types](./types.md) - Array views and types
- [Sparse Matrices](./sparse.md) - Registry usage with sparse matrices
- [SIMD](./simd.md) - Alignment requirements for SIMD
