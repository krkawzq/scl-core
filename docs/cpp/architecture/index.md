# Architecture Overview

SCL-Core is a high-performance biological data analysis library built on three foundational principles: zero-overhead abstraction, data-oriented design, and explicit resource management. This section explains the architectural decisions that enable SCL-Core to deliver optimal performance for sparse matrix computations.

## Core Mission

**Build a high-performance biological operator library with zero-overhead C++ kernels and a stable C-ABI surface for Python integration.**

Strategic Focus: **Sparse + Nonlinear**

SCL-Core targets the intersection of sparse matrix operations and nonlinear computations - a space underserved by traditional dense linear algebra libraries (BLAS/LAPACK/Eigen). We avoid reinventing mature dense linear algebra primitives, focusing instead on algorithms that leverage sparsity while performing complex nonlinear transformations common in biological data analysis.

## Design Philosophy

### 1. Zero-Overhead Abstraction

Every abstraction layer must compile down to optimal machine code with no runtime penalty.

**Principles:**
- No virtual functions in performance-critical paths
- Template-based compile-time polymorphism
- Aggressive inlining of hot functions
- Constexpr for compile-time computation
- Type erasure only at API boundaries

**Example:**

```cpp
// High-level API
template <CSRLike MatrixT>
void normalize_rows(MatrixT& matrix, NormMode mode, Real eps);

// Compiles to tight SIMD loops with no abstraction overhead
// Same performance as hand-written intrinsics
```

The template system resolves all polymorphism at compile time, allowing the optimizer to generate specialized code paths for each combination of types and parameters.

### 2. Data-Oriented Design

Memory access patterns determine performance on modern hardware. SCL-Core organizes data to maximize cache locality and memory bandwidth utilization.

**Principles:**
- Contiguous memory layouts to enable prefetching
- Structure of Arrays (SoA) when vectorization benefits outweigh locality
- Batch processing to amortize operation overhead
- Explicit prefetching for predictable access patterns
- Minimize pointer indirection in hot loops

**Memory Layout Strategy:**

Instead of traditional CSR with contiguous storage, SCL-Core uses a discontiguous pointer-based structure that enables:
- Block allocation for better memory management
- Flexible row/column ownership models
- Zero-copy integration with Python
- Efficient partial matrix operations

### 3. Explicit Resource Management

No hidden allocations. No implicit costs. Every memory operation is explicit and trackable.

**Principles:**
- Pre-allocated workspace pools for temporary storage
- Manual memory management with aligned allocation
- Registry-based lifetime tracking for Python integration
- No std::vector in hot paths - use fixed-size buffers
- Reference counting for shared buffer management

**Registry Pattern:**

The global Registry tracks all allocations, enabling:
- Deterministic cleanup without garbage collection
- Safe memory transfer across language boundaries (C++ ↔ Python)
- Reference counting for shared buffers
- Thread-safe concurrent access
- Memory usage profiling and leak detection

## Module Architecture

SCL-Core follows a strict layered architecture with explicit dependencies:

```
┌─────────────────────────────────────────┐
│         Python Bindings (scl-py)        │
│      (NumPy/SciPy/AnnData interface)    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│              math/                      │
│    (stats, regression, approximation)   │
│       Statistical computations          │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│            kernel/                      │
│  Computational Operators (400+ fns)     │
│  normalize, neighbors, leiden, etc.     │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│          threading/                     │
│  Work-stealing scheduler, parallel_for  │
│     Thread pools and workspaces         │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│             core/                       │
│  Types, Sparse, SIMD, Registry, Memory  │
│         Foundation Layer                │
└─────────────────────────────────────────┘
```

**Dependency Rules:**
- Lower layers never depend on upper layers
- Each layer can only use APIs from layers below
- core/ has zero internal dependencies (except Highway for SIMD)

## Key Components

### Core Types

```cpp
namespace scl {
    // Configurable fundamental types
    using Real = /* float32 | float64 | float16 */;
    using Index = /* int16 | int32 | int64 */;
    using Size = size_t;
}
```

Type configuration is compile-time, enabling whole-program optimization for specific precision requirements.

### Sparse Matrix Infrastructure

Discontiguous storage with pointer arrays for flexible memory management:

```cpp
template <typename T, bool IsCSR>
struct Sparse {
    using Pointer = T*;
    
    Pointer* data_ptrs_;      // Per-row/col data pointers
    Pointer* indices_ptrs_;   // Per-row/col index pointers
    Index* lengths_;          // Per-row/col lengths
    
    Index rows_, cols_, nnz_;
};
```

**Memory Layout:**

```
Row 0: data_ptrs_[0] → [v0, v1, v2]    indices_ptrs_[0] → [c0, c1, c2]
Row 1: data_ptrs_[1] → [v3, v4]        indices_ptrs_[1] → [c3, c4]
...
```

This design allows:
- Block allocation with configurable block sizes
- Reference-counted buffer sharing
- Zero-copy views and slices
- Efficient Python integration via Registry

### Registry System

Centralized memory tracking for lifetime management:

```cpp
class Registry {
    // Simple allocation tracking
    template <typename T>
    T* new_array(size_t count);
    void register_ptr(void* ptr, size_t bytes, AllocType type);
    void unregister_ptr(void* ptr);
    
    // Reference-counted buffers for shared memory
    BufferID register_buffer_with_aliases(
        void* real_ptr, size_t byte_size,
        std::span<void*> alias_ptrs, AllocType type);
    
    // Introspection
    bool is_registered(void* ptr) const;
    size_t get_total_bytes() const;
};
```

The Registry uses a sharded design to minimize lock contention in parallel workloads.

### SIMD Abstraction

Portable SIMD via Google Highway:

```cpp
namespace scl::simd {
    using Tag = hn::ScalableTag<Real>;
    
    // Operations compile to native intrinsics
    auto Load(Tag d, const T* ptr);
    auto Add(Vec a, Vec b);
    auto MulAdd(Vec a, Vec b, Vec c);  // FMA
    auto SumOfLanes(Tag d, Vec v);
}
```

Highway provides compile-time dispatch to optimal SIMD instructions (AVX2, AVX-512, NEON, etc.) based on target architecture.

## Performance Strategies

### 1. SIMD Optimization

**Multi-Accumulator Pattern:**

Modern CPUs have multiple FMA units with 4-5 cycle latency. Using 4 independent accumulators hides this latency and achieves near-peak throughput:

```cpp
auto v_sum0 = s::Zero(d), v_sum1 = s::Zero(d);
auto v_sum2 = s::Zero(d), v_sum3 = s::Zero(d);

for (; i + 4*lanes <= n; i += 4*lanes) {
    v_sum0 = s::Add(v_sum0, s::Load(d, data + i + 0*lanes));
    v_sum1 = s::Add(v_sum1, s::Load(d, data + i + 1*lanes));
    v_sum2 = s::Add(v_sum2, s::Load(d, data + i + 2*lanes));
    v_sum3 = s::Add(v_sum3, s::Load(d, data + i + 3*lanes));
}

auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
```

**Fused Operations:**

Combine related computations to minimize memory traffic:

```cpp
// Compute mean and variance in single pass
auto v_sum = s::Zero(d), v_sumsq = s::Zero(d);
for (size_t i = 0; i < n; i += lanes) {
    auto v = s::Load(d, data + i);
    v_sum = s::Add(v_sum, v);
    v_sumsq = s::MulAdd(v, v, v_sumsq);  // FMA
}
```

See [Design Principles](design-principles.md) for comprehensive SIMD patterns.

### 2. Parallel Processing

**Work-Stealing Scheduler:**

SCL-Core uses a custom work-stealing thread pool that:
- Automatically parallelizes based on problem size
- Balances load across threads dynamically
- Minimizes synchronization overhead
- Supports nested parallelism

**Per-Thread Workspaces:**

Avoid synchronization by giving each thread its own workspace:

```cpp
WorkspacePool<Real> pool(num_threads, workspace_size);

parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    auto* workspace = pool.get(thread_rank);  // No locking
    // Use workspace for temporary storage
});
```

See [Threading Documentation](/cpp/threading/) for details.

### 3. Memory Management

**Aligned Allocation for SIMD:**

All arrays used in SIMD code are 64-byte aligned for optimal performance:

```cpp
Real* data = scl::memory::aligned_alloc<Real>(count, 64);
```

**Block Allocation Strategy:**

Sparse matrix rows/columns are allocated in blocks (4KB-1MB) to balance:
- Memory reuse (larger blocks reduce overhead)
- Partial release (smaller blocks allow granular freeing)
- Parallelism (multiple blocks enable concurrent operations)

See [Memory Model](memory-model.md) for comprehensive memory management patterns.

## Documentation System

SCL-Core uses a dual-file documentation approach that separates implementation from specification:

### Implementation Files (.hpp)

Contain actual code with minimal inline comments:

```cpp
// scl/kernel/normalize.hpp
template <CSRLike MatrixT>
void normalize_rows_inplace(MatrixT& matrix, NormMode mode, Real eps) {
    // Use Kahan summation for numerical stability
    parallel_for(Size(0), matrix.rows(), [&](Index i) {
        // Implementation...
    });
}
```

### API Documentation Files (.h)

Comprehensive documentation with structured sections:

```cpp
// scl/kernel/normalize.h
/* -----------------------------------------------------------------------------
 * FUNCTION: normalize_rows_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Normalize each row of a CSR matrix to unit norm in-place.
 *
 * PARAMETERS:
 *     matrix   [in,out] Mutable CSR matrix, modified in-place
 *     mode     [in]     Norm type for normalization
 *     epsilon  [in]     Small constant to prevent division by zero
 *
 * PRECONDITIONS:
 *     - matrix must be valid CSR format
 *     - matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - Each row has unit norm (if original norm > epsilon)
 *     - Matrix structure unchanged
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 * -------------------------------------------------------------------------- */
template <CSRLike MatrixT>
void normalize_rows_inplace(
    MatrixT& matrix,               // CSR matrix, modified in-place
    NormMode mode,                 // Normalization type
    Real epsilon = 1e-12           // Zero-norm threshold
);
```

**Key Documentation Sections:**
- SUMMARY: One-line purpose
- PARAMETERS: Each parameter with [in], [out], or [in,out] tag
- PRECONDITIONS: Requirements before calling
- POSTCONDITIONS: Guarantees after execution
- MUTABILITY: INPLACE, CONST, or ALLOCATES
- ALGORITHM: Step-by-step description
- COMPLEXITY: Time and space analysis
- THREAD SAFETY: Safe, Unsafe, or conditional
- NUMERICAL NOTES: Precision and stability considerations

See [Documentation Standard](documentation-standard.md) for complete guidelines.

## External Dependencies

**Minimal by Design:**

- **Google Highway**: SIMD abstraction (header-only, mandatory)
- **C++17 Standard Library**: Minimal usage in hot paths

No other dependencies. This keeps compilation fast and eliminates version conflicts.

## Build System

CMake-based build with:
- **Compiler detection**: Automatic selection of optimal flags for GCC/Clang/MSVC
- **SIMD target selection**: AVX2, AVX-512, NEON based on architecture
- **LTO/IPO**: Link-time optimization for cross-module inlining
- **Unity builds**: Optional for faster compilation

Configuration:

```cmake
# Configure precision
set(SCL_REAL_TYPE "float32")  # or float64, float16
set(SCL_INDEX_TYPE "int32")   # or int16, int64

# Enable features
set(SCL_ENABLE_SIMD ON)
set(SCL_ENABLE_OPENMP ON)
```

## Performance Characteristics

**Typical Performance Metrics:**

- **Normalization**: 5-10 GB/s memory bandwidth (near hardware peak)
- **KNN**: 2-3x faster than Scanpy/sklearn for sparse matrices
- **Statistical tests**: 10-100x faster than scipy.stats for gene-wise operations
- **Leiden clustering**: Competitive with original igraph implementation

Performance is architecture-dependent. SCL-Core is optimized for:
- x86_64 with AVX2 or AVX-512
- ARM with NEON (Apple Silicon, AWS Graviton)

## Development Principles

### 1. Performance First

Every architectural decision prioritizes performance. When in doubt, measure and optimize for the hot path.

### 2. Zero Hidden Costs

If an operation allocates memory, throws exceptions, or performs I/O, it must be explicit in the API.

### 3. Correctness Through Types

Use the type system to enforce correctness:
- CSR/CSC distinction at type level
- Const-correctness for read-only operations
- Concepts to constrain template parameters

### 4. Testable Design

All kernels are pure functions that are easy to test in isolation.

## Next Steps

- [Design Principles](design-principles.md) - Deep dive into optimization strategies
- [Documentation Standard](documentation-standard.md) - Writing API documentation
- [Memory Model](memory-model.md) - Registry and lifetime management
- [Module Structure](module-structure.md) - Detailed module dependencies

---

::: tip Performance Philosophy
The best optimization is the one you do not need. SCL-Core's architecture eliminates overhead before it happens, not after. Every abstraction is designed to compile away completely.
:::
