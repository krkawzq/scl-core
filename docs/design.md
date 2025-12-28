# SCL-Core Algorithm Design Guide

This document summarizes the optimization strategies and design patterns used in scl-core for high-performance sparse matrix computations.

## Table of Contents

1. [Core Design Principles](#1-core-design-principles)
2. [SIMD Optimization](#2-simd-optimization)
3. [Parallel Processing](#3-parallel-processing)
4. [Memory Management](#4-memory-management)
5. [Loop Optimization](#5-loop-optimization)
6. [Algorithmic Optimization](#6-algorithmic-optimization)
7. [Dependency Reference](#7-dependency-reference)

---

## 1. Core Design Principles

### 1.1 Zero-Overhead Abstraction

All abstractions must compile down to optimal machine code:
- No virtual function calls in hot paths
- Templates for compile-time polymorphism
- Inline hints for critical functions

### 1.2 Data-Oriented Design

Optimize for cache locality and memory access patterns:
- Contiguous memory layouts over pointer chasing
- Structure of Arrays (SoA) over Array of Structures (AoS) when beneficial
- Batch processing to amortize setup costs

### 1.3 Explicit Resource Management

Avoid hidden allocations and implicit costs:
- Pre-allocated workspace pools
- Manual memory management with aligned allocation
- No std::vector in hot paths

---

## 2. SIMD Optimization

### 2.1 SIMD Abstraction Layer

Location: `scl/core/simd.hpp`

```cpp
namespace scl::simd {
    // Type tags for different widths
    using Tag = hn::ScalableTag<Real>;
    using IndexTag = hn::ScalableTag<Index>;

    // Core operations
    auto Load(Tag d, const T* ptr);
    void Store(Vec v, Tag d, T* ptr);
    auto Add(Vec a, Vec b);
    auto Mul(Vec a, Vec b);
    auto MulAdd(Vec a, Vec b, Vec c);  // FMA: a * b + c
    auto SumOfLanes(Tag d, Vec v);     // Horizontal sum
    auto Exp(Tag d, Vec v);            // Vectorized exp
}
```

### 2.2 Multi-Accumulator Pattern

Use multiple accumulators to hide latency and maximize throughput:

```cpp
// BAD: Single accumulator creates dependency chain
auto v_sum = s::Zero(d);
for (size_t i = 0; i < n; i += lanes) {
    v_sum = s::Add(v_sum, s::Load(d, data + i));
}

// GOOD: 4-way accumulator hides FMA latency
auto v_sum0 = s::Zero(d), v_sum1 = s::Zero(d);
auto v_sum2 = s::Zero(d), v_sum3 = s::Zero(d);

for (; i + 4 * lanes <= n; i += 4 * lanes) {
    v_sum0 = s::Add(v_sum0, s::Load(d, data + i + 0 * lanes));
    v_sum1 = s::Add(v_sum1, s::Load(d, data + i + 1 * lanes));
    v_sum2 = s::Add(v_sum2, s::Load(d, data + i + 2 * lanes));
    v_sum3 = s::Add(v_sum3, s::Load(d, data + i + 3 * lanes));
}

auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
```

### 2.3 Fused Operations

Combine related computations to reduce memory traffic:

```cpp
// Fused sum and sum-of-squares
for (; k + lanes <= n; k += lanes) {
    auto v = s::Load(d, vals + k);
    v_sum = s::Add(v_sum, v);
    v_sq = s::MulAdd(v, v, v_sq);  // FMA: v*v + v_sq
}
```

### 2.4 Masked Operations

Use SIMD masks for conditional processing:

```cpp
auto mask = s::Gt(v_mean, v_eps);
auto v_result = s::IfThenElse(mask, s::Div(v_var, v_mean), v_zero);
```

---

## 3. Parallel Processing

### 3.1 Thread Pool

Location: `scl/threading/scheduler.hpp`

```cpp
namespace scl::threading {
    class Scheduler {
        static size_t get_num_threads();
        static void set_num_threads(size_t n);
    };
}
```

### 3.2 Parallel For

Location: `scl/threading/parallel_for.hpp`

Two overloads for different use cases:

```cpp
// Simple iteration (no thread context needed)
parallel_for(Size(0), N, [&](size_t i) {
    // Process element i
});

// With thread rank (for per-thread workspace access)
parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
    auto* workspace = pool.get(thread_rank);
    // Process element i using workspace
});
```

### 3.3 Workspace Pools

Location: `scl/threading/workspace.hpp`

Pre-allocate per-thread buffers to avoid allocation in parallel loops:

```cpp
// Single buffer per thread
WorkspacePool<Real> pool;
pool.init(n_threads, buffer_size);

parallel_for(0, N, [&](size_t i, size_t thread_rank) {
    Real* buffer = pool.get(thread_rank);
    // Use buffer...
});

// Dual buffers per thread (e.g., for two arrays needed together)
DualWorkspacePool<Real> dual_pool;
dual_pool.init(n_threads, size1, size2);  // or init(n_threads, size) for equal sizes

parallel_for(0, N, [&](size_t i, size_t thread_rank) {
    Real* buf1 = dual_pool.get1(thread_rank);
    Real* buf2 = dual_pool.get2(thread_rank);
});
```

### 3.4 Parallel Reduction Pattern

For aggregating results across threads:

```cpp
Size* partial_sums = scl::memory::aligned_alloc<Size>(n_threads, SCL_ALIGNMENT);
scl::algo::zero(partial_sums, n_threads);

parallel_for(0, n_threads, [&](size_t tid) {
    Size start = tid * chunk_size;
    Size end = scl::algo::min2(start + chunk_size, n);

    Size local_sum = 0;
    for (Size i = start; i < end; ++i) {
        local_sum += compute(i);
    }
    partial_sums[tid] = local_sum;
});

Size total = 0;
for (Size t = 0; t < n_threads; ++t) {
    total += partial_sums[t];
}

scl::memory::aligned_free(partial_sums, SCL_ALIGNMENT);
```

---

## 4. Memory Management

### 4.1 Aligned Allocation

Location: `scl/core/memory.hpp`

```cpp
namespace scl::memory {
    // Allocate aligned memory (default: 64-byte alignment for cache lines)
    template <typename T>
    T* aligned_alloc(Size count, size_t alignment = SCL_ALIGNMENT);

    // Free aligned memory
    template <typename T>
    void aligned_free(T* ptr, size_t alignment = SCL_ALIGNMENT);

    // Zero-fill array
    void zero(Array<T> arr);

    // Fill with value
    void fill(Array<T> arr, T value);
}
```

### 4.2 Stack-Based Small Buffers

Use stack allocation for small, bounded sizes:

```cpp
// Stack buffer for small cases, heap for large
Size local_buffer[256];
Size* buffer = nullptr;
Size* heap_buffer = nullptr;

if (n <= 256) {
    buffer = local_buffer;
} else {
    heap_buffer = scl::memory::aligned_alloc<Size>(n, SCL_ALIGNMENT);
    buffer = heap_buffer;
}

// Use buffer...

if (heap_buffer) {
    scl::memory::aligned_free(heap_buffer, SCL_ALIGNMENT);
}
```

### 4.3 Contiguous Group Storage

Replace nested containers with flat arrays + offsets:

```cpp
// BAD: std::vector<std::vector<Index>> - many allocations, poor locality
std::vector<std::vector<Index>> groups(n_groups);

// GOOD: Single contiguous allocation with offset array
struct BatchGroups {
    Index* indices;      // All indices concatenated
    Size* offsets;       // offsets[b+1] - offsets[b] = size of group b
    Size n_batches;

    Size batch_size(Size b) const { return offsets[b + 1] - offsets[b]; }
    const Index* batch_data(Size b) const { return indices + offsets[b]; }
};
```

---

## 5. Loop Optimization

### 5.1 Loop Unrolling

Manual unrolling for predictable iteration counts:

```cpp
// 4-way unrolled loop
Size k = 0;
for (; k + 4 <= len; k += 4) {
    process(k + 0);
    process(k + 1);
    process(k + 2);
    process(k + 3);
}
// Scalar cleanup
for (; k < len; ++k) {
    process(k);
}
```

### 5.2 Software Prefetching

Location: `scl/core/macros.hpp`

```cpp
#define SCL_PREFETCH_READ(addr, locality)  __builtin_prefetch(addr, 0, locality)
#define SCL_PREFETCH_WRITE(addr, locality) __builtin_prefetch(addr, 1, locality)
// locality: 0 = no temporal locality, 3 = high temporal locality
```

Usage pattern:

```cpp
constexpr Size PREFETCH_DISTANCE = 16;

for (Size k = 0; k < len; ++k) {
    if (SCL_LIKELY(k + PREFETCH_DISTANCE < len)) {
        SCL_PREFETCH_READ(&data[k + PREFETCH_DISTANCE], 0);
    }
    process(data[k]);
}
```

### 5.3 Branch Hints

```cpp
#define SCL_LIKELY(x)   __builtin_expect(!!(x), 1)
#define SCL_UNLIKELY(x) __builtin_expect(!!(x), 0)

// Use for predictable branches
if (SCL_LIKELY(idx < n)) {
    // Common path
}

if (SCL_UNLIKELY(error_condition)) {
    // Rare error handling
}
```

### 5.4 Compiler Hints

```cpp
// Force inline for critical small functions
#define SCL_FORCE_INLINE inline __attribute__((always_inline))

// Mark functions as hot (frequently called)
#define SCL_HOT __attribute__((hot))

// Assume condition is true (enables optimizations)
#define SCL_ASSUME(cond) __builtin_assume(cond)

// Restrict pointer aliasing
#define SCL_RESTRICT __restrict__
```

---

## 6. Algorithmic Optimization

### 6.1 Early Exit and Pruning

Check preconditions to skip unnecessary work:

```cpp
// O(1) range disjointness check for sparse vectors
if (SCL_UNLIKELY(idx1[n1-1] < idx2[0] || idx2[n2-1] < idx1[0])) {
    return T(0);  // No overlap, dot product is zero
}

// Cauchy-Schwarz lower bound for distance pruning
T norm_diff = sqrt_norm_i - sqrt_norm_j;
T min_dist_sq = norm_diff * norm_diff;
if (min_dist_sq >= current_max_dist_sq) {
    continue;  // Skip this candidate
}
```

### 6.2 Adaptive Algorithm Selection

Choose algorithm based on input characteristics:

```cpp
namespace config {
    constexpr Size RATIO_THRESHOLD = 32;
    constexpr Size GALLOP_THRESHOLD = 256;
}

template <typename T>
T sparse_dot_adaptive(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
) {
    // Ensure n1 <= n2
    if (n1 > n2) {
        scl::algo::swap(idx1, idx2);
        scl::algo::swap(val1, val2);
        scl::algo::swap(n1, n2);
    }

    Size ratio = n2 / n1;

    if (ratio >= config::GALLOP_THRESHOLD) {
        return dot_gallop(idx1, val1, n1, idx2, val2, n2);
    } else if (ratio >= config::RATIO_THRESHOLD) {
        return dot_binary(idx1, val1, n1, idx2, val2, n2);
    } else {
        return dot_linear(idx1, val1, n1, idx2, val2, n2);
    }
}
```

### 6.3 Skip Optimization for Sparse Merge

Batch-skip non-overlapping ranges in merge operations:

```cpp
// 8-way skip for large gaps
while (i + 8 <= n1 && j + 8 <= n2) {
    if (idx1[i+7] < idx2[j]) { i += 8; continue; }
    if (idx2[j+7] < idx1[i]) { j += 8; continue; }
    break;
}

// 4-way skip for medium gaps
while (i + 4 <= n1 && j + 4 <= n2) {
    if (idx1[i+3] < idx2[j]) { i += 4; continue; }
    if (idx2[j+3] < idx1[i]) { j += 4; continue; }
    break;
}

// Scalar merge for remaining
while (i < n1 && j < n2) { /* ... */ }
```

### 6.4 Heap-Based Top-K Selection

Use max-heap for efficient top-k maintenance:

```cpp
template <typename T>
struct KHeap {
    struct Entry { T value; Index idx; };
    Entry* data;
    Size k, count;

    void try_insert(T value, Index idx) {
        if (count < k) {
            data[count] = {value, idx};
            sift_up(count++);
            return;
        }
        if (value < data[0].value) {  // Better than worst in heap
            data[0] = {value, idx};
            sift_down(0);
        }
    }

    T max_value() const {
        return (count > 0) ? data[0].value : std::numeric_limits<T>::max();
    }
};
```

### 6.5 Boundary Check Elimination

Location: `scl/core/algo.hpp`

All algo functions assume valid inputs - caller must verify preconditions:

```cpp
// Unchecked binary search
template <typename T, typename V>
const T* lower_bound(const T* first, const T* last, V target) noexcept {
    SCL_ASSUME(first <= last);  // Hint to compiler
    // No bounds checking inside
}

// Unchecked copy
template <typename T>
void copy(const T* SCL_RESTRICT src, T* SCL_RESTRICT dst, size_t n) noexcept {
    // Direct memcpy, no size validation
    std::memcpy(dst, src, n * sizeof(T));
}
```

---

## 7. Dependency Reference

### 7.1 Core Infrastructure

| File | Purpose | Key Components |
|------|---------|----------------|
| `scl/core/type.hpp` | Type definitions | `Real`, `Index`, `Size`, `Array<T>` |
| `scl/core/macros.hpp` | Compiler hints | `SCL_FORCE_INLINE`, `SCL_LIKELY`, `SCL_PREFETCH_*` |
| `scl/core/simd.hpp` | SIMD abstraction | `Tag`, `Load`, `Store`, `Add`, `Mul`, `MulAdd`, `Exp` |
| `scl/core/memory.hpp` | Memory management | `aligned_alloc`, `aligned_free`, `zero`, `fill` |
| `scl/core/algo.hpp` | Unchecked algorithms | `lower_bound`, `copy`, `fill`, `swap`, `min2`, `max2`, `partial_sort`, `sparse_dot` |
| `scl/core/sort.hpp` | High-performance sorting | `sort`, `sort_pairs`, VQSort wrapper |
| `scl/core/sparse.hpp` | Sparse matrix types | `Sparse<T, IsCSR>`, block storage |
| `scl/core/vectorize.hpp` | SIMD vector operations | `sum`, `sum_squared`, `dot` |
| `scl/core/error.hpp` | Error handling | `SCL_CHECK_DIM`, `SCL_CHECK_ARG`, `SCL_ASSERT` |

### 7.2 Threading Infrastructure

| File | Purpose | Key Components |
|------|---------|----------------|
| `scl/threading/scheduler.hpp` | Thread pool management | `Scheduler::get_num_threads()` |
| `scl/threading/parallel_for.hpp` | Parallel iteration | `parallel_for(start, end, func)` |
| `scl/threading/workspace.hpp` | Per-thread buffers | `WorkspacePool<T>`, `DualWorkspacePool<T>` |

### 7.3 Common Include Pattern

```cpp
#pragma once

// Core types and utilities
#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"

// For sparse matrix operations
#include "scl/core/sparse.hpp"
#include "scl/core/vectorize.hpp"

// For parallel processing
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/scheduler.hpp"
#include "scl/threading/workspace.hpp"

// Standard library (minimal)
#include <cmath>      // std::sqrt, std::exp, std::log
#include <limits>     // std::numeric_limits
#include <cstring>    // std::memcpy, std::memset
```

---

## 8. Algorithm Template

When implementing a new kernel, follow this template:

```cpp
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>

namespace scl::kernel::mykernel {

// Configuration constants
namespace config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size CHUNK_SIZE = 64;
    constexpr Size THRESHOLD = 256;
}

// Internal implementation details
namespace detail {

// SIMD helper for hot inner loop
template <typename T>
SCL_FORCE_INLINE T compute_simd(const T* data, Size len) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    // Multi-accumulator pattern
    auto v_acc0 = s::Zero(d);
    auto v_acc1 = s::Zero(d);

    Size k = 0;
    for (; k + 2 * lanes <= len; k += 2 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < len)) {
            SCL_PREFETCH_READ(data + k + config::PREFETCH_DISTANCE * lanes, 0);
        }
        v_acc0 = s::Add(v_acc0, s::Load(d, data + k));
        v_acc1 = s::Add(v_acc1, s::Load(d, data + k + lanes));
    }

    auto v_acc = s::Add(v_acc0, v_acc1);
    T result = s::GetLane(s::SumOfLanes(d, v_acc));

    // Scalar cleanup
    for (; k < len; ++k) {
        result += data[k];
    }

    return result;
}

} // namespace detail

// Main kernel function
template <typename T, bool IsCSR>
void my_operation(
    const Sparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    // Input validation
    SCL_CHECK_DIM(output.len >= N, "Output size mismatch");

    // Pre-allocate workspace for all threads
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    scl::threading::WorkspacePool<T> workspace;
    workspace.init(n_threads, config::CHUNK_SIZE);

    // Parallel processing with thread-local workspace
    scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
        T* local_buffer = workspace.get(thread_rank);

        const Index idx = static_cast<Index>(i);
        const Index len = matrix.primary_length(idx);

        // Early exit for empty rows
        if (SCL_UNLIKELY(len == 0)) {
            output[i] = T(0);
            return;
        }

        auto values = matrix.primary_values(idx);

        // Adaptive processing based on size
        T result;
        if (static_cast<Size>(len) < config::THRESHOLD) {
            // Simple scalar path for small inputs
            result = T(0);
            for (Index k = 0; k < len; ++k) {
                result += values[k];
            }
        } else {
            // SIMD path for larger inputs
            result = detail::compute_simd(values.ptr, static_cast<Size>(len));
        }

        output[i] = result;
    });
}

} // namespace scl::kernel::mykernel
```

---

## 9. Zero-Overhead Custom Operators (CRITICAL: No std:: in Hot Paths)

### 9.1 Mandatory Rule: Use Custom Operators Instead of std::

**CRITICAL**: All hot-path code MUST use SCL custom operators instead of standard library functions. This ensures:
- Zero-overhead abstractions
- SIMD optimization where applicable
- Cache-aligned memory access
- Parallel-friendly implementations

### 9.2 Memory Operations

**NEVER use**:
- `std::memcpy`, `std::memset`, `std::memmove`
- `std::vector`, `std::array`
- `new[]` / `delete[]` (use aligned allocation)

**ALWAYS use**:
```cpp
// Memory allocation
T* buffer = scl::memory::aligned_alloc<T>(count, SCL_ALIGNMENT);
scl::memory::aligned_free(buffer, SCL_ALIGNMENT);

// Memory operations
scl::memory::copy_fast(src, dst);        // Fast copy (no overlap)
scl::memory::copy(src, dst);             // Safe copy (handles overlap)
scl::memory::zero(Array<T>(ptr, len));   // SIMD-optimized zero
scl::memory::fill(Array<T>(ptr, len), value);  // SIMD-optimized fill

// Or use algo namespace for unchecked operations
scl::algo::copy(src_ptr, dst_ptr, n);    // Unchecked fast copy
scl::algo::fill(dst_ptr, n, value);      // Unchecked fill
scl::algo::zero(dst_ptr, n);             // Unchecked zero
```

### 9.3 Sorting Operations

**NEVER use**:
- `std::sort`, `std::stable_sort`, `std::partial_sort`
- `std::nth_element`
- `std::sort` with custom comparators

**ALWAYS use**:
```cpp
// Single array sorting (VQSort - SIMD-optimized, parallel)
scl::sort::sort(Array<Real>(data, n));                    // Ascending
scl::sort::sort_descending(Array<Real>(data, n));          // Descending

// Key-value pair sorting (for argsort-like operations)
scl::sort::sort_pairs(
    Array<Real>(keys, n),
    Array<Index>(values, n)
);  // Sort by keys, values follow

scl::sort::sort_pairs_descending(
    Array<Real>(keys, n),
    Array<Index>(values, n)
);  // Descending order

// Convenience functions
scl::sort::sort_real(Array<Real>(data, n));
scl::sort::sort_index(Array<Index>(data, n));
```

**Performance**: VQSort uses SIMD-optimized sorting with automatic parallelization, typically 2-5x faster than `std::sort`.

### 9.4 Mathematical Operations

**NEVER use**:
- `std::exp`, `std::log`, `std::sqrt` in loops (use SIMD versions)
- `std::min`, `std::max` in hot loops
- `std::abs` in vectorized code

**ALWAYS use**:
```cpp
// SIMD transcendental functions
namespace s = scl::simd;
const s::Tag d;
auto v_result = s::Exp(d, v_input);      // Vectorized exp
auto v_result = s::Log(d, v_input);      // Vectorized log
auto v_result = s::Sqrt(d, v_input);      // Vectorized sqrt

// For scalar operations in non-hot paths, use std:: versions
// But prefer SIMD when processing arrays

// Min/Max operations
scl::algo::min2(a, b);                   // Fast min of two values
scl::algo::max2(a, b);                   // Fast max of two values
scl::algo::min(data, n);                 // SIMD min reduction
scl::algo::max(data, n);                 // SIMD max reduction

// Vectorized operations
scl::vectorize::sum(Array<const Real>(data, n));         // SIMD sum
scl::vectorize::dot(a, b);                               // SIMD dot product
scl::vectorize::scale(Array<Real>(data, n), factor);     // SIMD scale
```

### 9.5 Reduction Operations

**NEVER use**:
- Manual loops for sum/max/min (unless SIMD-optimized)
- `std::accumulate`

**ALWAYS use**:
```cpp
// SIMD-optimized reductions
Real total = scl::vectorize::sum(Array<const Real>(data, n));
Real max_val = scl::algo::max(data, n);
Real min_val = scl::algo::min(data, n);

// Or use algo namespace for unchecked operations
Real total = scl::algo::sum(data, n);
```

### 9.6 Search Operations

**NEVER use**:
- `std::lower_bound`, `std::upper_bound` (unless in cold paths)

**ALWAYS use**:
```cpp
// Unchecked binary search (faster, caller must verify preconditions)
const Index* pos = scl::algo::lower_bound(first, last, target);
const Index* pos = scl::algo::upper_bound(first, last, target);
```

### 9.7 Partial Sorting / Selection

**NEVER use**:
- `std::partial_sort`, `std::nth_element`
- Insertion sort for large arrays

**ALWAYS use**:
```cpp
// Partial sort (top-k elements)
scl::algo::partial_sort(data, k, n, [](const T& a, const T& b) {
    return a < b;
});

// Nth element (quickselect)
scl::algo::nth_element(data, nth, last);
```

### 9.8 Container Operations

**NEVER use**:
- `std::vector` in hot paths
- `std::array` (use stack arrays or aligned allocation)
- `std::unordered_map`, `std::map` in hot paths

**ALWAYS use**:
```cpp
// Stack allocation for small buffers
SCL_STACK_ARRAY(Real, local_buffer, 256);

// Heap allocation for larger buffers
Real* buffer = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
// ... use buffer ...
scl::memory::aligned_free(buffer, SCL_ALIGNMENT);

// Or use AlignedBuffer for RAII
scl::memory::AlignedBuffer<Real> buffer(n);
Real* ptr = buffer.get();
```

### 9.9 Complete Replacement Table

| std:: Function | SCL Replacement | Location |
|---------------|----------------|----------|
| `std::sort` | `scl::sort::sort` | `scl/core/sort.hpp` |
| `std::partial_sort` | `scl::algo::partial_sort` | `scl/core/algo.hpp` |
| `std::nth_element` | `scl::algo::nth_element` | `scl/core/algo.hpp` |
| `std::lower_bound` | `scl::algo::lower_bound` | `scl/core/algo.hpp` |
| `std::upper_bound` | `scl::algo::upper_bound` | `scl/core/algo.hpp` |
| `std::min` / `std::max` | `scl::algo::min2` / `scl::algo::max2` | `scl/core/algo.hpp` |
| `std::memcpy` | `scl::memory::copy_fast` or `scl::algo::copy` | `scl/core/memory.hpp` / `algo.hpp` |
| `std::memset` | `scl::memory::zero` or `scl::algo::zero` | `scl/core/memory.hpp` / `algo.hpp` |
| `std::exp` (in loops) | `scl::simd::Exp` | `scl/core/simd.hpp` |
| `std::log` (in loops) | `scl::simd::Log` | `scl/core/simd.hpp` |
| `std::sqrt` (in loops) | `scl::simd::Sqrt` | `scl/core/simd.hpp` |
| `std::accumulate` | `scl::vectorize::sum` | `scl/core/vectorize.hpp` |
| `new[]` / `delete[]` | `scl::memory::aligned_alloc` / `aligned_free` | `scl/core/memory.hpp` |
| `std::vector` | `scl::memory::aligned_alloc` + manual management | `scl/core/memory.hpp` |

---

## 10. Performance Checklist

Before finalizing any kernel implementation, verify:

- [ ] **CRITICAL**: No `std::` functions in hot paths (use SCL custom operators)
- [ ] No `std::vector` in hot paths (use `scl::memory::aligned_alloc`)
- [ ] No `std::sort`, `std::min`, `std::max` (use `scl::algo::*` or `scl::sort::*`)
- [ ] No `std::exp`, `std::log`, `std::sqrt` in loops (use `scl::simd::*`)
- [ ] No `std::memcpy`, `std::memset` (use `scl::memory::*` or `scl::algo::*`)
- [ ] SIMD used for vectorizable loops with multi-accumulator pattern
- [ ] Prefetch hints for sequential memory access
- [ ] Early exit conditions for edge cases
- [ ] Adaptive algorithm selection based on input size/characteristics
- [ ] WorkspacePool for per-thread temporary storage
- [ ] Proper memory cleanup in all code paths
- [ ] Branch hints (`SCL_LIKELY`/`SCL_UNLIKELY`) for predictable branches
- [ ] Loop unrolling for short, predictable loops
- [ ] `SCL_RESTRICT` for non-aliased pointer parameters
- [ ] `SCL_FORCE_INLINE` for small, frequently-called helpers
