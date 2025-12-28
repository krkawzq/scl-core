# parallel_for.hpp

> scl/threading/parallel_for.hpp · Backend-agnostic parallel loop abstraction

## Overview

This file provides a unified interface for parallel iteration across multiple threading backends. The `parallel_for` function abstracts over OpenMP, TBB, BS::thread_pool, and Serial backends, providing consistent semantics regardless of the underlying implementation.

Key features:
- Backend-agnostic API
- Minimal abstraction overhead
- Consistent behavior across backends
- Compile-time backend selection
- Support for thread rank access

**Header**: `#include "scl/threading/parallel_for.hpp"`

---

## Main APIs

### parallel_for

::: source_code file="scl/threading/parallel_for.hpp" symbol="parallel_for" collapsed
:::

**Algorithm Description**

Execute loop iterations in parallel across worker threads:

1. **Backend Selection**: Based on compile-time defines from `scl/config.hpp`:
   - **OpenMP**: Uses `#pragma omp parallel for` with static scheduling
   - **TBB**: Uses `tbb::parallel_for` with work-stealing scheduler
   - **BS::thread_pool**: Manually chunks range and submits tasks with futures
   - **Serial**: Sequential `for` loop (no parallelism)

2. **Work Distribution**:
   - **OpenMP**: Static partition - range divided into N chunks, each thread processes its chunk
   - **TBB**: Dynamic work-stealing - scheduler distributes blocks, idle threads steal work
   - **BS**: Manual chunking based on thread count, each chunk submitted as task
   - **Serial**: Sequential execution

3. **Synchronization**: Implicit barrier at end - all iterations complete before return

**Edge Cases**

- **Empty range** (start >= end): Returns immediately, no iterations executed
- **Single iteration**: May or may not use parallel overhead (backend-dependent)
- **Nested parallelism**: OpenMP serializes nested calls, TBB/BS support nested parallelism
- **Exceptions**: Backend-dependent behavior:
  - Serial: Exception propagates normally
  - OpenMP: Implementation-defined (often terminates)
  - TBB: Exception propagates to caller
  - BS: Exception propagates via futures

**Data Guarantees (Preconditions)**

- `Func` must be callable as `func(size_t)` or `func(size_t, size_t)` (with thread_rank)
- `Func` must be thread-safe (no data races on shared mutable state)
- `start <= end` (undefined if start > end, but handled gracefully)

**Complexity Analysis**

- **Time**: O((end - start) * T(func) / N) expected
  - Where N = number of threads, T(func) = execution time of func
  - Overhead: ~100-1000 ns per task (backend-dependent)
- **Space**: O(N) thread stack space
  - O(N) auxiliary for BS backend (futures vector)
  - O(1) auxiliary for other backends

**Example**

```cpp
#include "scl/threading/parallel_for.hpp"

// Basic parallel loop
scl::threading::parallel_for(0, n, [&](size_t i) {
    output[i] = compute(input[i]);
});

// With thread rank for workspace access
scl::threading::WorkspacePool<Real> pool;
pool.init(num_threads, workspace_size);

scl::threading::parallel_for(0, n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);
    process(data[i], workspace);
});

// Matrix row processing
scl::threading::parallel_for(0, matrix.rows(), [&](size_t row) {
    process_row(matrix, row);
});
```

---

## Notes

**Backend Characteristics**

| Backend | Scheduling | Best For | Overhead |
|---------|-----------|----------|----------|
| OpenMP | Static | Uniform workloads | Minimal |
| TBB | Dynamic (work-stealing) | Irregular workloads | Low |
| BS | Task-based | Moderate parallelism | Moderate |
| Serial | Sequential | Debugging | Zero |

**Performance Guidelines**

- **Effective for**: (end - start) >> num_threads
- **Overhead becomes significant**: When range < 1000 elements
- **Best when**: func execution time >> scheduling overhead (> ~1 microsecond per call)
- **No benefit**: If func is too fast (< ~1 microsecond per call)

**Optimization Tips**

1. **Minimize false sharing**: Use thread-local accumulators, align data to cache lines
2. **Minimize synchronization**: Avoid locks in hot paths, use atomic operations sparingly
3. **Balance workload**: TBB handles imbalance automatically, OpenMP assumes uniform work
4. **Reduce overhead**: Don't parallelize tiny loops (< 1000 iterations)

**When to Use**

- Independent iterations (no loop-carried dependencies)
- Computationally expensive loop body
- Large iteration count (> 1000)
- Read-heavy or embarrassingly parallel workloads

**When NOT to Use**

- Small iteration counts (< 100)
- Fast loop body (< 1us per iteration)
- Heavy synchronization requirements
- Sequential dependencies between iterations

**Thread Safety**

The loop body must ensure thread safety for shared mutable state. Use:
- Thread-local accumulators for reductions
- Atomic operations for shared counters
- Disjoint writes (each thread writes to different locations)

## See Also

- [Scheduler](./scheduler) - Thread pool management
- [Workspace](./workspace) - Per-thread workspace pools

