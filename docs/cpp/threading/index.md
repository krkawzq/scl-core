# Threading

The `scl/threading/` module provides parallel processing infrastructure for SCL-Core.

## Overview

Threading module provides:

- **Parallel For** - Work-stealing parallel loops
- **Scheduler** - Thread pool with dynamic work distribution
- **Workspace** - Per-thread workspace pools

## Key Components

### parallel_for

Automatic parallelization for loops:

```cpp
#include "scl/threading/parallel_for.hpp"

// Parallel loop
scl::threading::parallel_for(Size(0), n, [&](size_t i) {
    // Process element i in parallel
    process(data[i]);
});

// With thread rank
scl::threading::parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    // Access per-thread workspace
    auto* workspace = pool.get(thread_rank);
    process(data[i], workspace);
});
```

### Scheduler

Thread pool for task execution:

```cpp
#include "scl/threading/scheduler.hpp"

auto& scheduler = scl::threading::get_scheduler();

// Get number of threads
size_t num_threads = scheduler.num_threads();

// Execute task
scheduler.execute([&]() {
    // Task code
});
```

### WorkspacePool

Per-thread temporary storage:

```cpp
#include "scl/threading/workspace.hpp"

// Create pool
scl::threading::WorkspacePool<Real> pool(num_threads, workspace_size);

// Use in parallel loop
parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);
    // Use workspace without synchronization
});
```

## Design Principles

### Work-Stealing

Automatic load balancing:
- Dynamic work distribution
- Idle threads steal work from busy threads
- Optimal for irregular workloads

### Automatic Parallelization

Based on problem size:
- Small problems run serially
- Large problems run in parallel
- Threshold automatically determined

### Per-Thread Workspaces

Avoid synchronization:
- Each thread has private workspace
- No locking required
- Better cache locality

## Best Practices

### 1. Use parallel_for for Loops

```cpp
// GOOD: Automatic parallelization
parallel_for(Size(0), matrix.rows(), [&](Index i) {
    process_row(matrix, i);
});

// BAD: Manual threading (error-prone)
std::vector<std::thread> threads;
for (size_t t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
        // Manual work distribution...
    });
}
for (auto& thread : threads) {
    thread.join();
}
```

### 2. Use Workspaces for Temporary Storage

```cpp
// GOOD: Per-thread workspace
WorkspacePool<Real> pool(num_threads, 1024);
parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    Real* temp = pool.get(thread_rank);
    // Use temp without synchronization
});

// BAD: Shared workspace with locking
std::vector<Real> shared_workspace(1024);
std::mutex mtx;
parallel_for(Size(0), n, [&](size_t i) {
    std::lock_guard lock(mtx);  // Contention!
    // Use shared_workspace
});
```

### 3. Batch Small Operations

```cpp
// GOOD: Batch to reduce overhead
constexpr Size BATCH_SIZE = 64;
const Size num_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;

parallel_for(Size(0), num_batches, [&](size_t batch_idx) {
    const Size start = batch_idx * BATCH_SIZE;
    const Size end = std::min(start + BATCH_SIZE, n);
    
    for (Size i = start; i < end; ++i) {
        lightweight_operation(data[i]);
    }
});

// BAD: Parallelize lightweight operations directly
parallel_for(Size(0), n, [&](size_t i) {
    lightweight_operation(data[i]);  // Too much overhead!
});
```

## Thread Safety

### Read Operations

Safe for concurrent access:

```cpp
// Safe: Multiple threads reading
parallel_for(Size(0), matrix.rows(), [&](Index i) {
    auto vals = matrix.primary_values(i);
    Real sum = compute_sum(vals.ptr, vals.size);
});
```

### Write Operations

Require synchronization or disjoint access:

```cpp
// Safe: Disjoint writes (each thread writes to different location)
parallel_for(Size(0), matrix.rows(), [&](Index i) {
    output[i] = compute_result(matrix, i);
});

// Unsafe: Concurrent writes to shared location
Real global_sum = 0;
parallel_for(Size(0), n, [&](size_t i) {
    global_sum += data[i];  // RACE CONDITION!
});

// Safe: Use atomic or reduction
std::atomic<Real> global_sum{0};
parallel_for(Size(0), n, [&](size_t i) {
    global_sum.fetch_add(data[i]);  // Thread-safe
});
```

## Performance Considerations

### Parallelization Overhead

- Thread creation: ~1-10 μs per thread
- Work distribution: ~100-1000 ns per task
- Synchronization: ~10-100 ns per barrier

**Rule of thumb:** Parallelize if work per element > 1 μs

### Cache Effects

- False sharing: Avoid writing to adjacent memory locations
- Cache line size: 64 bytes on most systems
- Padding: Add padding between thread-local data

### NUMA Awareness

For NUMA systems:
- First-touch policy: Data allocated on first access
- Parallel initialization: Initialize data in parallel

---

::: tip Performance
Always measure! Parallelization adds overhead. Profile to ensure parallel version is actually faster.
:::

