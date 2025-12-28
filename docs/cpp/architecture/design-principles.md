# Design Principles

This document details the optimization strategies and design patterns used throughout SCL-Core.

## 1. Zero-Overhead Abstraction

### Principle

All abstractions must compile down to optimal machine code with no runtime cost.

### Implementation

**No Virtual Functions in Hot Paths**

```cpp
// BAD: Virtual function call overhead
class Normalizer {
    virtual void normalize(Real* data, size_t n) = 0;
};

// GOOD: Template-based compile-time polymorphism
template <NormMode Mode>
SCL_FORCE_INLINE void normalize(Real* data, size_t n) {
    if constexpr (Mode == NormMode::L2) {
        // Specialized implementation
    }
}
```

**Inline Hints for Critical Functions**

```cpp
#define SCL_FORCE_INLINE [[gnu::always_inline]] inline

SCL_FORCE_INLINE Real compute_norm(const Real* data, size_t n) {
    // Hot path function - always inlined
}
```

**Constexpr for Compile-Time Computation**

```cpp
constexpr size_t compute_block_size(size_t total, size_t threads) {
    return (total + threads - 1) / threads;
}

// Computed at compile time
constexpr size_t BLOCK_SIZE = compute_block_size(1000, 8);
```

## 2. SIMD Optimization

### Multi-Accumulator Pattern

Use multiple accumulators to hide FMA latency:

```cpp
// BAD: Single accumulator creates dependency chain
auto v_sum = s::Zero(d);
for (size_t i = 0; i < n; i += lanes) {
    v_sum = s::Add(v_sum, s::Load(d, data + i));  // Dependency!
}

// GOOD: 4-way accumulator hides latency
auto v_sum0 = s::Zero(d), v_sum1 = s::Zero(d);
auto v_sum2 = s::Zero(d), v_sum3 = s::Zero(d);

for (; i + 4 * lanes <= n; i += 4 * lanes) {
    v_sum0 = s::Add(v_sum0, s::Load(d, data + i + 0 * lanes));
    v_sum1 = s::Add(v_sum1, s::Load(d, data + i + 1 * lanes));
    v_sum2 = s::Add(v_sum2, s::Load(d, data + i + 2 * lanes));
    v_sum3 = s::Add(v_sum3, s::Load(d, data + i + 3 * lanes));
}

// Combine accumulators
auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
```

**Why This Works:**
- Modern CPUs have multiple FMA units
- 4-way unrolling allows parallel execution
- Hides 4-5 cycle FMA latency

### Fused Operations

Combine related computations to reduce memory traffic:

```cpp
// BAD: Two passes over data
Real sum = 0, sumsq = 0;
for (size_t i = 0; i < n; ++i) {
    sum += data[i];
}
for (size_t i = 0; i < n; ++i) {
    sumsq += data[i] * data[i];
}

// GOOD: Single pass with fused SIMD
auto v_sum = s::Zero(d), v_sumsq = s::Zero(d);
for (size_t i = 0; i < n; i += lanes) {
    auto v = s::Load(d, data + i);
    v_sum = s::Add(v_sum, v);
    v_sumsq = s::MulAdd(v, v, v_sumsq);  // FMA: v*v + v_sumsq
}
```

**Benefits:**
- 2x reduction in memory bandwidth
- Better cache utilization
- Single loop overhead

### Prefetching

For predictable access patterns:

```cpp
constexpr size_t PREFETCH_DISTANCE = 16;

for (size_t i = 0; i < n; i += lanes) {
    // Prefetch future data
    if (i + PREFETCH_DISTANCE * lanes < n) {
        SCL_PREFETCH_READ(data + i + PREFETCH_DISTANCE * lanes, 0);
    }
    
    // Process current data
    auto v = s::Load(d, data + i);
    // ...
}
```

## 3. Parallel Processing

### Work Distribution

**Automatic Parallelization**

```cpp
// Automatically parallelizes if n >= threshold
parallel_for(Size(0), n, [&](size_t i) {
    process(data[i]);
});
```

**Batched Processing**

For lightweight operations, batch to reduce overhead:

```cpp
constexpr Size BATCH_SIZE = 64;
const Size num_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;

parallel_for(Size(0), num_batches, [&](size_t batch_idx) {
    const Size start = batch_idx * BATCH_SIZE;
    const Size end = std::min(start + BATCH_SIZE, n);
    
    for (Size i = start; i < end; ++i) {
        process(data[i]);  // Batch of 64 items per task
    }
});
```

### Per-Thread Workspaces

Avoid synchronization with thread-local storage:

```cpp
// BAD: Shared workspace requires locking
std::vector<Real> workspace;
std::mutex mtx;

parallel_for(Size(0), n, [&](size_t i) {
    std::lock_guard lock(mtx);  // Contention!
    workspace.resize(needed_size);
    // ...
});

// GOOD: Per-thread workspace
WorkspacePool<Real> pool(num_threads, workspace_size);

parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    auto* workspace = pool.get(thread_rank);  // No locking
    // ...
});
```

## 4. Memory Management

### Aligned Allocation

For SIMD operations:

```cpp
// Allocate aligned memory
constexpr size_t ALIGNMENT = 64;  // Cache line size
T* data = scl::memory::aligned_alloc<T>(count, ALIGNMENT);

// SIMD loads require alignment
auto v = s::Load(d, data);  // Efficient aligned load
```

### Block Allocation

For sparse matrices:

```cpp
struct BlockStrategy {
    Index min_block_elements = 4096;
    Index max_block_elements = 262144;  // 256K = ~1MB
    
    Index compute_block_size(Index total_nnz, Index primary_dim) {
        // Balance between:
        // - Memory reuse (larger blocks)
        // - Partial release (smaller blocks)
        // - Parallelism (multiple blocks)
    }
};
```

### Registry Pattern

Track all allocations for Python integration:

```cpp
auto& reg = get_registry();

// Register allocation
T* data = reg.new_array<T>(count);

// Reference counting for shared buffers
BufferID id = reg.register_buffer_with_aliases(
    real_ptr, byte_size, alias_ptrs, AllocType::ArrayNew);

// Automatic cleanup
reg.unregister_ptr(data);
```

## 5. Loop Optimization

### Unrolling

Manual unrolling for predictable loops:

```cpp
// 4-way unroll
for (; i + 4 <= n; i += 4) {
    result[i+0] = process(data[i+0]);
    result[i+1] = process(data[i+1]);
    result[i+2] = process(data[i+2]);
    result[i+3] = process(data[i+3]);
}

// Cleanup loop
for (; i < n; ++i) {
    result[i] = process(data[i]);
}
```

### Loop Fusion

Combine related loops:

```cpp
// BAD: Multiple passes
for (size_t i = 0; i < n; ++i) compute_A(i);
for (size_t i = 0; i < n; ++i) compute_B(i);
for (size_t i = 0; i < n; ++i) compute_C(i);

// GOOD: Single pass
for (size_t i = 0; i < n; ++i) {
    compute_A(i);
    compute_B(i);
    compute_C(i);
}
```

### Strength Reduction

Replace expensive operations:

```cpp
// BAD: Division in loop
for (size_t i = 0; i < n; ++i) {
    result[i] = data[i] / divisor;
}

// GOOD: Multiply by reciprocal
const Real inv_divisor = Real(1) / divisor;
for (size_t i = 0; i < n; ++i) {
    result[i] = data[i] * inv_divisor;
}
```

## 6. Algorithmic Optimization

### Early Exit

Check conditions before expensive operations:

```cpp
// Check for zero length before processing
if (len == 0) return;

// Check for trivial cases
if (n == 1) {
    result[0] = data[0];
    return;
}

// Now do expensive computation
```

### Numerical Stability

Use compensated summation for accuracy:

```cpp
// Kahan summation for better precision
Real sum = 0, compensation = 0;
for (size_t i = 0; i < n; ++i) {
    Real y = data[i] - compensation;
    Real t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
}
```

### Cache-Friendly Access

Access memory in order:

```cpp
// BAD: Column-major access of row-major data
for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
        process(data[i * cols + j]);  // Cache misses!
    }
}

// GOOD: Row-major access
for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
        process(data[i * cols + j]);  // Sequential access
    }
}
```

## 7. Compile-Time Optimization

### Template Specialization

Optimize for common cases:

```cpp
// General template
template <typename T>
void process(T* data, size_t n);

// Specialized for float
template <>
void process<float>(float* data, size_t n) {
    // Use float-specific SIMD intrinsics
}
```

### Constexpr If

Branch at compile time:

```cpp
template <NormMode Mode>
void normalize(Real* data, size_t n) {
    if constexpr (Mode == NormMode::L1) {
        // L1-specific code
    } else if constexpr (Mode == NormMode::L2) {
        // L2-specific code
    }
    // No runtime branching!
}
```

## Performance Checklist

When writing performance-critical code:

- [ ] Use SIMD with multi-accumulator pattern
- [ ] Minimize memory traffic (fused operations)
- [ ] Parallelize over large dimensions
- [ ] Use per-thread workspaces
- [ ] Align memory for SIMD
- [ ] Unroll loops manually (4-way)
- [ ] Prefetch predictable accesses
- [ ] Early exit for trivial cases
- [ ] Access memory sequentially
- [ ] Use constexpr for compile-time computation

## Benchmarking

Always measure performance:

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// Code to benchmark
process_data(data, n);

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

std::cout << "Time: " << duration.count() << " μs\n";
```

---

::: tip Remember
Premature optimization is the root of all evil, but SCL-Core is a performance library. Optimize the hot paths, measure everything, and document your assumptions.
:::

