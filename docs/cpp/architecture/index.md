# Architecture Overview

SCL-Core is designed from the ground up for maximum performance in biological data analysis. This section explains the architectural decisions and design patterns that make SCL-Core fast.

## Design Philosophy

### 1. Zero-Overhead Abstraction

All high-level APIs compile down to optimal machine code with no runtime cost:

- **No virtual functions** in hot paths
- **Templates** for compile-time polymorphism
- **Inline hints** for critical functions
- **Constexpr** for compile-time computation

```cpp
// High-level API
template <typename T, bool IsCSR>
void normalize_rows(const Sparse<T, IsCSR>& matrix, NormMode mode);

// Compiles to tight SIMD loops - zero overhead
```

### 2. Data-Oriented Design

Optimize for cache locality and memory bandwidth:

- **Contiguous memory layouts** over pointer chasing
- **Structure of Arrays (SoA)** when beneficial
- **Batch processing** to amortize setup costs
- **Prefetching** for predictable access patterns

### 3. Explicit Resource Management

No hidden allocations or implicit costs:

- **Pre-allocated workspace pools** for temporary storage
- **Manual memory management** with aligned allocation
- **Registry-based lifetime tracking** for Python integration
- **No std::vector in hot paths** - use pre-sized buffers

## Module Structure

```
scl/
├── core/           # Foundation: types, sparse, SIMD, memory
│   ├── type.hpp    # Real, Index, Size types
│   ├── sparse.hpp  # Discontiguous sparse matrix
│   ├── simd.hpp    # SIMD abstraction (Highway)
│   ├── registry.hpp # Memory lifetime management
│   └── ...
├── threading/      # Parallel processing
│   ├── parallel_for.hpp  # Work-stealing parallel loops
│   ├── scheduler.hpp     # Thread pool
│   └── workspace.hpp     # Per-thread workspace
├── kernel/         # Computational operators (400+ functions)
│   ├── normalize.hpp
│   ├── neighbors.hpp
│   ├── leiden.hpp
│   └── ...
├── math/           # Statistical functions
│   ├── stats.hpp
│   ├── regression.hpp
│   └── ...
└── mmap/           # Memory-mapped arrays
    ├── array.hpp
    ├── sparse.hpp
    └── ...
```

## Key Components

### Core Types

- **`Real`**: Configurable floating-point type (float32/float64/float16)
- **`Index`**: Signed integer for indexing (int16/int32/int64)
- **`Size`**: Unsigned integer for sizes and byte counts

### Sparse Matrix

Discontiguous storage with pointer arrays for flexible memory management:

```cpp
template <typename T, bool IsCSR>
struct Sparse {
    Pointer* data_ptrs;      // Per-row/col data pointers
    Pointer* indices_ptrs;   // Per-row/col index pointers
    Index* lengths;          // Per-row/col lengths
    Index rows_, cols_, nnz_;
};
```

### Registry

Centralized memory tracking for Python integration:

```cpp
class Registry {
    // Track all allocated buffers
    void register_ptr(void* ptr, size_t bytes, AllocType type);
    
    // Reference counting for shared buffers
    BufferID register_buffer_with_aliases(...);
    
    // Automatic cleanup
    void unregister_ptr(void* ptr);
};
```

### SIMD Abstraction

Portable SIMD via Google Highway:

```cpp
namespace scl::simd {
    using Tag = hn::ScalableTag<Real>;
    
    auto Load(Tag d, const T* ptr);
    auto Add(Vec a, Vec b);
    auto MulAdd(Vec a, Vec b, Vec c);  // FMA
    auto SumOfLanes(Tag d, Vec v);
}
```

## Performance Strategies

### 1. SIMD Optimization

- **Multi-accumulator pattern** to hide latency
- **4-way unrolling** for FMA pipelines
- **Fused operations** to reduce memory traffic
- **Masked operations** for conditional processing

See [Design Principles](/cpp/architecture/design-principles) for details.

### 2. Parallel Processing

- **Work-stealing scheduler** for load balancing
- **Automatic parallelization** based on problem size
- **Per-thread workspaces** to avoid synchronization
- **Batch processing** to reduce overhead

See [Threading](/cpp/threading/) for details.

### 3. Memory Management

- **Aligned allocations** for SIMD
- **Block allocation** for sparse matrices
- **Reference counting** for shared buffers
- **Registry tracking** for Python integration

See [Memory Model](/cpp/architecture/memory-model) for details.

## Documentation Standard

SCL-Core uses a dual-file documentation system:

### `.hpp` Files - Implementation

Minimal inline comments, clean and readable code:

```cpp
template <typename T, bool IsCSR>
void normalize_rows(const Sparse<T, IsCSR>& matrix, NormMode mode) {
    // Use Kahan summation for numerical stability
    parallel_for(0, matrix.rows(), [&](Index i) {
        // ... implementation
    });
}
```

### `.h` Files - API Documentation

Comprehensive documentation with structured sections:

```cpp
/* -----------------------------------------------------------------------------
 * FUNCTION: normalize_rows
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Normalize each row of a sparse matrix to unit norm.
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix to normalize
 *     mode   [in]  Norm type: L1, L2, Max, or Sum
 *
 * PRECONDITIONS:
 *     - matrix is valid sparse format
 *
 * POSTCONDITIONS:
 *     - Each row has unit norm under specified mode
 *     - Matrix structure unchanged
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 * -------------------------------------------------------------------------- */
```

See [Documentation Standard](/cpp/architecture/documentation-standard) for details.

## Dependency Management

### External Dependencies

- **Google Highway**: SIMD abstraction (header-only)
- **C++17 Standard Library**: Minimal usage in hot paths

### Internal Dependencies

```
core/ (no dependencies)
  ↓
threading/ (depends on core/)
  ↓
kernel/ (depends on core/, threading/)
  ↓
math/ (depends on core/, threading/, kernel/)
```

## Build System

CMake-based build with:

- **Compiler detection** for optimal flags
- **SIMD target selection** (AVX2, AVX-512, NEON)
- **LTO/IPO** for cross-module optimization
- **Unity builds** for faster compilation

## Next Steps

- [Design Principles](/cpp/architecture/design-principles) - Deep dive into optimization strategies
- [Module Structure](/cpp/architecture/module-structure) - Detailed module dependencies
- [Memory Model](/cpp/architecture/memory-model) - Registry and lifetime management
- [Documentation Standard](/cpp/architecture/documentation-standard) - Writing documentation

---

::: tip Performance First
Every architectural decision in SCL-Core prioritizes performance. When in doubt, measure and optimize for the hot path.
:::

