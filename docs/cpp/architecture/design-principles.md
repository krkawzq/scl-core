# Design Principles

This document details the optimization strategies and design patterns that enable SCL-Core's high performance. Every pattern is battle-tested and backed by performance measurements.

## 1. Zero-Overhead Abstraction

### Principle

Abstractions must compile down to optimal machine code indistinguishable from hand-written low-level code. If an abstraction introduces any runtime overhead, it violates this principle.

### Implementation Strategies

#### No Virtual Functions in Hot Paths

Virtual function calls introduce indirection and prevent inlining. Use compile-time polymorphism instead.

```cpp
// AVOID: Virtual function overhead
class Normalizer {
    virtual void normalize(Real* data, size_t n) = 0;
};

class L2Normalizer : public Normalizer {
    void normalize(Real* data, size_t n) override {
        // Implementation
    }
};

// PREFER: Template-based compile-time polymorphism
template <NormMode Mode>
SCL_FORCE_INLINE void normalize(Real* data, size_t n) {
    if constexpr (Mode == NormMode::L2) {
        // L2-specific implementation
        // Compiler generates specialized code
    } else if constexpr (Mode == NormMode::L1) {
        // L1-specific implementation
    }
}
```

**Why This Works:**
- No vtable lookup - direct function call
- Inlining opportunity - optimizer can see through call
- Dead code elimination - unused branches removed
- Constant propagation - mode-specific optimizations

#### Force Inlining for Critical Functions

Guide the compiler to inline performance-critical functions:

```cpp
#define SCL_FORCE_INLINE [[gnu::always_inline]] inline

SCL_FORCE_INLINE Real dot_product(const Real* a, const Real* b, size_t n) {
    // Hot path function - always inlined
    Real sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

**When to Use:**
- Functions called in tight loops
- Small functions (< 20 lines)
- Functions with constant parameters that enable optimization

#### Constexpr for Compile-Time Computation

Move computation from runtime to compile time whenever possible:

```cpp
constexpr size_t compute_block_size(size_t total, size_t threads) {
    return (total + threads - 1) / threads;
}

// If arguments are compile-time constants, result is too
constexpr size_t BLOCK_SIZE = compute_block_size(1000000, 8);

// Can be used as array size
Real workspace[BLOCK_SIZE];
```

**Benefits:**
- Zero runtime cost
- Can be used in template parameters and array sizes
- Enables further compile-time optimization

#### Type Erasure Only at Boundaries

Keep templates internal, erase types only at API boundaries:

```cpp
// Internal: Template for zero-overhead
template <typename T, bool IsCSR>
void normalize_impl(Sparse<T, IsCSR>& matrix, NormMode mode);

// API boundary: Type-erased C interface for Python
extern "C" {
    void scl_normalize_f32_csr(SparseMatrixHandle handle, int mode);
    void scl_normalize_f64_csr(SparseMatrixHandle handle, int mode);
}

void scl_normalize_f32_csr(SparseMatrixHandle handle, int mode) {
    auto* matrix = reinterpret_cast<Sparse<float, true>*>(handle);
    normalize_impl(*matrix, static_cast<NormMode>(mode));
}
```

## 2. SIMD Optimization

Modern CPUs can process 4-16 elements per cycle using SIMD instructions. Properly vectorized code achieves 5-10x speedup over scalar code.

### Multi-Accumulator Pattern

The most important SIMD optimization. Modern FMA units have 4-5 cycle latency but accept new operations every cycle. Using multiple independent accumulators hides this latency.

```cpp
// POOR: Single accumulator creates dependency chain
// Each Add must wait for previous Add to complete
auto v_sum = s::Zero(d);
for (size_t i = 0; i < n; i += lanes) {
    auto v = s::Load(d, data + i);
    v_sum = s::Add(v_sum, v);  // Dependency on previous iteration!
}

// OPTIMAL: 4-way accumulator hides latency
// Multiple FMA units can execute in parallel
auto v_sum0 = s::Zero(d), v_sum1 = s::Zero(d);
auto v_sum2 = s::Zero(d), v_sum3 = s::Zero(d);

size_t i = 0;
// Main loop: 4-way unrolled
for (; i + 4*lanes <= n; i += 4*lanes) {
    v_sum0 = s::Add(v_sum0, s::Load(d, data + i + 0*lanes));
    v_sum1 = s::Add(v_sum1, s::Load(d, data + i + 1*lanes));
    v_sum2 = s::Add(v_sum2, s::Load(d, data + i + 2*lanes));
    v_sum3 = s::Add(v_sum3, s::Load(d, data + i + 3*lanes));
}

// Cleanup loop: process remaining elements
for (; i + lanes <= n; i += lanes) {
    v_sum0 = s::Add(v_sum0, s::Load(d, data + i));
}

// Combine accumulators
auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
```

**Performance Impact:**
- Single accumulator: ~1 operation per 4-5 cycles (limited by latency)
- 4-way accumulator: ~4 operations per cycle (limited by throughput)
- **Speedup: 4-5x** for memory-bound operations

**Why 4 Accumulators?**
- Most CPUs have 2 FMA units
- FMA latency is 4-5 cycles
- 4 accumulators fully utilize both units

### Fused Operations

Combine multiple operations into single pass to reduce memory traffic:

```cpp
// INEFFICIENT: Multiple passes over data
Real sum = 0, sumsq = 0;
for (size_t i = 0; i < n; ++i) {
    sum += data[i];
}
for (size_t i = 0; i < n; ++i) {
    sumsq += data[i] * data[i];
}

// EFFICIENT: Single pass with SIMD fusion
auto v_sum = s::Zero(d), v_sumsq = s::Zero(d);
for (size_t i = 0; i < n; i += lanes) {
    auto v = s::Load(d, data + i);
    v_sum = s::Add(v_sum, v);
    v_sumsq = s::MulAdd(v, v, v_sumsq);  // FMA: v*v + v_sumsq
}
```

**Benefits:**
- 2x reduction in memory bandwidth
- Better cache utilization (data loaded once)
- Single loop overhead
- Enables multi-accumulator pattern for both operations

### Horizontal Reductions

Efficiently reduce SIMD vectors to scalar values:

```cpp
// Sum all lanes in vector
Real horizontal_sum(s::Tag d, s::Vec v) {
    // Highway provides efficient implementation
    return s::GetLane(s::SumOfLanes(d, v));
}

// Combine with multi-accumulator
auto v_sum0 = /* ... */;
auto v_sum1 = /* ... */;
auto v_sum2 = /* ... */;
auto v_sum3 = /* ... */;

// Combine vectors before reduction (fewer reductions)
auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
Real result = horizontal_sum(d, v_sum);
```

### Prefetching

For predictable access patterns, prefetch future data while processing current data:

```cpp
constexpr size_t PREFETCH_DISTANCE = 16;  // Tuned experimentally

for (size_t i = 0; i < n; i += lanes) {
    // Prefetch data that will be used in future iterations
    if (i + PREFETCH_DISTANCE * lanes < n) {
        SCL_PREFETCH_READ(data + i + PREFETCH_DISTANCE * lanes, 0);
    }
    
    // Process current data
    auto v = s::Load(d, data + i);
    // ... computation ...
}
```

**When to Prefetch:**
- Sequential access patterns
- Memory-bound operations
- Access latency > computation time

**When NOT to Prefetch:**
- Random access patterns (wastes cache lines)
- Computation-bound operations
- Small datasets that fit in cache

### Masked Operations

Process partial vectors at array boundaries without branching:

```cpp
// Process full vectors
size_t i = 0;
for (; i + lanes <= n; i += lanes) {
    auto v = s::Load(d, data + i);
    process(v);
}

// Process remaining elements with mask
if (i < n) {
    auto mask = s::FirstN(d, n - i);  // Mask for remaining elements
    auto v = s::MaskedLoad(mask, d, data + i);
    process(v);
    s::MaskedStore(mask, result, d, output + i);
}
```

## 3. Parallel Processing

Effectively utilize multi-core CPUs while minimizing synchronization overhead.

### Automatic Parallelization

Let the library decide when parallelization is beneficial:

```cpp
// Automatically parallelizes if n is large enough
// Uses serial execution for small n to avoid overhead
parallel_for(Size(0), n, [&](size_t i) {
    process(data[i]);
});

// Threshold is dynamically computed based on:
// - Operation cost per element
// - Number of available threads
// - Thread pool overhead
```

**Typical Threshold:**
- Lightweight operations: 10,000+ elements
- Medium operations: 1,000+ elements
- Heavy operations: 100+ elements

### Batched Processing

For lightweight operations, batch elements to reduce per-task overhead:

```cpp
constexpr Size BATCH_SIZE = 64;  // Tuned to balance overhead and parallelism
const Size num_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;

parallel_for(Size(0), num_batches, [&](size_t batch_idx) {
    const Size start = batch_idx * BATCH_SIZE;
    const Size end = std::min(start + BATCH_SIZE, n);
    
    // Process batch sequentially within thread
    for (Size i = start; i < end; ++i) {
        process(data[i]);  // Lightweight operation
    }
});
```

**Trade-offs:**
- Larger batches: Lower overhead, worse load balancing
- Smaller batches: Higher overhead, better load balancing
- Optimal size depends on operation cost and array size

### Per-Thread Workspaces

Avoid synchronization by giving each thread its own workspace:

```cpp
// INEFFICIENT: Shared workspace requires locking
std::vector<Real> workspace;
std::mutex mtx;

parallel_for(Size(0), n, [&](size_t i) {
    std::lock_guard lock(mtx);  // Contention bottleneck!
    workspace.resize(needed_size);
    use_workspace(workspace);
});

// EFFICIENT: Per-thread workspace, no locking
WorkspacePool<Real> pool(num_threads, workspace_size);

parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    auto* workspace = pool.get(thread_rank);  // Lock-free access
    use_workspace(workspace, workspace_size);
});
```

**Benefits:**
- Zero synchronization overhead
- Better cache locality (thread-local data)
- Predictable performance (no lock contention)

### False Sharing Prevention

Ensure thread-local data is cache-line aligned:

```cpp
// PROBLEMATIC: Adjacent counters share cache line
struct Counters {
    size_t thread_counts[8];  // 8 threads, 8 bytes each = 64 bytes
};
// All threads write to same cache line - false sharing!

// CORRECT: Cache-line aligned counters
struct alignas(64) AlignedCounter {
    size_t count;
    char padding[56];  // Pad to 64 bytes
};

struct Counters {
    AlignedCounter thread_counts[8];
};
// Each thread writes to separate cache line
```

## 4. Memory Management

Efficient memory management is critical for both performance and correctness.

### Aligned Allocation

SIMD operations require properly aligned memory:

```cpp
// Allocate aligned memory for SIMD
constexpr size_t ALIGNMENT = 64;  // Cache line size
Real* data = scl::memory::aligned_alloc<Real>(count, ALIGNMENT);

// SIMD loads are more efficient with aligned data
auto v = s::LoadU(d, data);      // Unaligned load: slower
auto v = s::Load(d, data);       // Aligned load: faster
```

**Alignment Requirements:**
- AVX2: 32-byte alignment recommended
- AVX-512: 64-byte alignment recommended
- Always align to cache line (64 bytes) when possible

### Block Allocation Strategy

For sparse matrices, allocate rows/columns in blocks:

```cpp
struct BlockStrategy {
    Index min_block_elements = 4096;      // 16KB for float32
    Index max_block_elements = 262144;    // 1MB for float32
    
    Index compute_block_size(Index total_nnz, Index primary_dim) const {
        // Target: 4-8 rows per block for good parallelism
        Index avg_nnz = total_nnz / primary_dim;
        Index target_rows = 8;
        Index block_size = avg_nnz * target_rows;
        
        // Clamp to min/max
        block_size = std::max(block_size, min_block_elements);
        block_size = std::min(block_size, max_block_elements);
        
        return block_size;
    }
};
```

**Trade-offs:**
- Larger blocks: Better memory reuse, harder to free partially
- Smaller blocks: Easier to free, more allocation overhead
- Optimal size: 256KB - 1MB per block

### Registry Pattern

Track all allocations for Python integration:

```cpp
auto& reg = get_registry();

// Register allocation with metadata
Real* data = reg.new_array<Real>(count);

// Reference counting for shared buffers
BufferID id = reg.register_buffer_with_aliases(
    real_ptr, byte_size, alias_ptrs, AllocType::ArrayNew);

// Automatic cleanup with proper deleter
reg.unregister_ptr(data);
```

### Memory Pooling

Reuse allocations to avoid repeated malloc/free:

```cpp
template <typename T>
class MemoryPool {
    std::vector<std::unique_ptr<T[]>> available_;
    std::vector<std::unique_ptr<T[]>> in_use_;
    size_t buffer_size_;
    
public:
    T* acquire() {
        if (available_.empty()) {
            return new T[buffer_size_];
        }
        auto buffer = std::move(available_.back());
        available_.pop_back();
        T* ptr = buffer.get();
        in_use_.push_back(std::move(buffer));
        return ptr;
    }
    
    void release(T* ptr) {
        // Move from in_use to available
    }
};
```

## 5. Loop Optimization

Small details in loop structure can have large performance impacts.

### Manual Unrolling

Unroll loops to expose instruction-level parallelism:

```cpp
// 4-way unroll for better ILP
size_t i = 0;
for (; i + 4 <= n; i += 4) {
    result[i+0] = process(data[i+0]);
    result[i+1] = process(data[i+1]);
    result[i+2] = process(data[i+2]);
    result[i+3] = process(data[i+3]);
}

// Cleanup loop for remainder
for (; i < n; ++i) {
    result[i] = process(data[i]);
}
```

**When to Unroll:**
- Simple loop bodies (few operations)
- Independent iterations
- Unroll factor 4-8 for best results

### Loop Fusion

Combine multiple loops over same data:

```cpp
// INEFFICIENT: Multiple passes
for (size_t i = 0; i < n; ++i) compute_A(i);
for (size_t i = 0; i < n; ++i) compute_B(i);
for (size_t i = 0; i < n; ++i) compute_C(i);

// EFFICIENT: Single pass, better cache utilization
for (size_t i = 0; i < n; ++i) {
    compute_A(i);
    compute_B(i);
    compute_C(i);
}
```

### Strength Reduction

Replace expensive operations with cheaper equivalents:

```cpp
// EXPENSIVE: Division in loop
for (size_t i = 0; i < n; ++i) {
    result[i] = data[i] / divisor;
}

// CHEAPER: Multiply by reciprocal
const Real inv_divisor = Real(1) / divisor;
for (size_t i = 0; i < n; ++i) {
    result[i] = data[i] * inv_divisor;
}
```

**Common Replacements:**
- Division → Multiplication by reciprocal
- Modulo by power-of-2 → Bitwise AND
- Multiplication by constant → Shift + Add

### Loop Invariant Code Motion

Move constant computations out of loops:

```cpp
// INEFFICIENT: Recomputes constant each iteration
for (size_t i = 0; i < n; ++i) {
    result[i] = data[i] * compute_scale_factor(params);
}

// EFFICIENT: Compute once before loop
const Real scale = compute_scale_factor(params);
for (size_t i = 0; i < n; ++i) {
    result[i] = data[i] * scale;
}
```

## 6. Algorithmic Optimization

Choose algorithms that match hardware characteristics.

### Early Exit

Check termination conditions before expensive operations:

```cpp
void normalize_rows(Sparse<Real, true>& matrix, Real eps) {
    parallel_for(Size(0), matrix.rows(), [&](Index i) {
        const Index len = matrix.primary_length(i);
        
        // Early exit for empty rows
        if (len == 0) return;
        
        auto* vals = matrix.primary_values(i).ptr;
        Real norm = compute_norm(vals, len);
        
        // Early exit for zero norm
        if (norm <= eps) return;
        
        // Now do expensive normalization
        const Real inv_norm = Real(1) / norm;
        for (Index j = 0; j < len; ++j) {
            vals[j] *= inv_norm;
        }
    });
}
```

### Numerical Stability

Use compensated summation for better precision:

```cpp
// Kahan summation: reduces floating-point error accumulation
Real kahan_sum(const Real* data, size_t n) {
    Real sum = 0, compensation = 0;
    
    for (size_t i = 0; i < n; ++i) {
        Real y = data[i] - compensation;
        Real t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}
```

**When to Use:**
- Large arrays (n > 10,000)
- High precision requirements
- Summation of mixed-magnitude values

### Cache-Friendly Access

Access memory in order to maximize cache hits:

```cpp
// CACHE-UNFRIENDLY: Column-major access of row-major data
for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
        process(data[i * cols + j]);  // Strided access - cache misses!
    }
}

// CACHE-FRIENDLY: Row-major access of row-major data
for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
        process(data[i * cols + j]);  // Sequential access - cache hits!
    }
}
```

**Rule of Thumb:**
- Access multi-dimensional arrays in storage order
- Prefer unit-stride access patterns
- Process data in cache-line-sized chunks (64 bytes)

## 7. Compile-Time Optimization

Leverage the compiler to generate optimal code.

### Template Specialization

Optimize for common cases:

```cpp
// General template
template <typename T>
void process(T* data, size_t n) {
    // Generic implementation
}

// Specialized for float32 - use SIMD
template <>
void process<float>(float* data, size_t n) {
    using Tag = hn::ScalableTag<float>;
    const Tag d;
    // SIMD-optimized implementation
}
```

### Constexpr If

Branch at compile time to eliminate dead code:

```cpp
template <NormMode Mode>
void normalize(Real* data, size_t n) {
    if constexpr (Mode == NormMode::L1) {
        // L1-specific code - other branches removed
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::abs(data[i]);
        }
    } else if constexpr (Mode == NormMode::L2) {
        // L2-specific code
        for (size_t i = 0; i < n; ++i) {
            data[i] = data[i] * data[i];
        }
    }
    // No runtime branching!
}
```

### Attribute Hints

Guide the compiler with attributes:

```cpp
// Function is pure - no side effects, result depends only on args
[[gnu::pure]] Real compute_norm(const Real* data, size_t n);

// Result must be used - warn if discarded
[[nodiscard]] Real* allocate_buffer(size_t n);

// Likely/unlikely for branch prediction
if (SCL_LIKELY(n > threshold)) {
    // Common path
} else {
    // Rare path
}
```

## Performance Checklist

Before declaring a function "optimized", verify:

- [ ] SIMD: Multi-accumulator pattern (4-way)
- [ ] SIMD: Fused operations (single pass)
- [ ] Memory: Aligned allocation (64-byte)
- [ ] Memory: Sequential access (cache-friendly)
- [ ] Parallel: Automatic parallelization
- [ ] Parallel: Per-thread workspaces (no locks)
- [ ] Loop: Manual unrolling (4-way)
- [ ] Loop: Strength reduction (div → mul)
- [ ] Algorithm: Early exit checks
- [ ] Algorithm: Numerical stability (Kahan sum)
- [ ] Compile: Constexpr for constants
- [ ] Compile: Template specialization
- [ ] Compile: Inline hints for hot paths

## Profiling and Benchmarking

Always measure before and after optimization:

```cpp
#include <chrono>

template <typename Func>
double benchmark(Func&& func, size_t iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / static_cast<double>(iterations);
}

// Usage
double avg_time = benchmark([&]() {
    normalize_rows(matrix, NormMode::L2);
});

std::cout << "Average time: " << avg_time << " μs\n";
```

**Metrics to Track:**
- **Throughput**: Elements processed per second
- **Bandwidth**: Memory bandwidth utilization (GB/s)
- **Efficiency**: Actual bandwidth / theoretical peak
- **Scalability**: Speedup vs number of threads

---

::: tip Optimization Philosophy
Optimization is not about clever tricks - it is about understanding hardware and writing code that matches how CPUs actually work. Every pattern in this document exists because modern hardware rewards it with measurable performance gains.
:::
