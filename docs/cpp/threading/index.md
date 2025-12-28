# Threading Module

> scl/threading/ · Parallel processing infrastructure for SCL-Core

## Overview

The threading module provides backend-agnostic parallel processing infrastructure for SCL-Core. It abstracts over multiple threading backends (OpenMP, TBB, BS::thread_pool, Serial) to provide a unified interface for parallel execution.

Key features:
- **Backend-agnostic**: Works with OpenMP, TBB, BS::thread_pool, or Serial
- **Automatic parallelization**: Simple API for parallel loops
- **Thread pool management**: Global scheduler for controlling parallelism
- **Per-thread workspaces**: Eliminate allocations in hot loops
- **Zero-overhead abstraction**: Minimal runtime overhead

## Files

| File | Description | Main APIs |
|------|-------------|-----------|
| [parallel_for.hpp](./parallel_for) | Parallel Loop Interface | `parallel_for` |
| [scheduler.hpp](./scheduler) | Thread Pool Management | `Scheduler::init`, `Scheduler::set_num_threads` |
| [workspace.hpp](./workspace) | Per-Thread Workspaces | `WorkspacePool`, `DualWorkspacePool` |

## Quick Start

### Parallel Loop

```cpp
#include "scl/threading/parallel_for.hpp"

// Simple parallel loop
scl::threading::parallel_for(0, n, [&](size_t i) {
    output[i] = compute(input[i]);
});

// With thread rank for workspace access
scl::threading::parallel_for(0, n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);
    process(data[i], workspace);
});
```

### Thread Pool Configuration

```cpp
#include "scl/threading/scheduler.hpp"

// Initialize with all cores
scl::threading::Scheduler::init();

// Or specify number of threads
scl::threading::Scheduler::init(4);

// Query thread count
size_t threads = scl::threading::Scheduler::get_num_threads();
```

### Per-Thread Workspaces

```cpp
#include "scl/threading/workspace.hpp"

// Create workspace pool
scl::threading::WorkspacePool<Real> pool;
pool.init(num_threads, workspace_size);

// Use in parallel loop
scl::threading::parallel_for(0, n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);
    // Use workspace without synchronization
});
```

## Backend Selection

Backend is selected at compile time via `scl/config.hpp`:

- **OpenMP**: `SCL_BACKEND_OPENMP` - HPC standard, widespread support
- **TBB**: `SCL_BACKEND_TBB` - Work-stealing, advanced load balancing
- **BS::thread_pool**: `SCL_BACKEND_BS` - Portable, header-only
- **Serial**: `SCL_BACKEND_SERIAL` - Debug/single-threaded fallback

## See Also

- [Core Types](/cpp/core/types) - Basic types used by threading
- [Memory Management](/cpp/core/memory) - Memory allocation utilities
