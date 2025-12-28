# scheduler.hpp

> scl/threading/scheduler.hpp · Global thread pool and concurrency configuration

## Overview

This file provides a unified interface for controlling parallelism across multiple threading backends. The `Scheduler` class is a static facade that abstracts over backend-specific thread pool management APIs.

Key features:
- Backend-agnostic thread pool control
- Hardware concurrency detection
- Thread count configuration
- Static class (no instances)

**Header**: `#include "scl/threading/scheduler.hpp"`

---

## Main APIs

### Scheduler::hardware_concurrency

::: source_code file="scl/threading/scheduler.hpp" symbol="hardware_concurrency" collapsed
:::

**Algorithm Description**

Get number of hardware threads available:

Returns `std::thread::hardware_concurrency()`, which typically equals the number of logical CPU cores. May return 0 if unable to detect hardware capabilities.

**Edge Cases**

- **Unable to detect**: Returns 0
- **Hyperthreading**: Returns number of logical cores (may be 2x physical cores)

**Data Guarantees (Preconditions)**

None - safe to call at any time.

**Complexity Analysis**

- **Time**: O(1)
- **Space**: O(1)

**Example**

```cpp
#include "scl/threading/scheduler.hpp"

size_t cores = scl::threading::Scheduler::hardware_concurrency();
// Typically equals number of logical CPU cores
```

---

### Scheduler::set_num_threads

::: source_code file="scl/threading/scheduler.hpp" symbol="set_num_threads" collapsed
:::

**Algorithm Description**

Set number of worker threads for parallel execution:

1. **Backend-specific behavior**:
   - **OpenMP**: Calls `omp_set_num_threads(n)` - affects all parallel regions
   - **TBB**: Creates `tbb::global_control` stored in static variable (lifetime managed automatically)
   - **BS**: Calls `pool.reset(n)` - recreates thread pool (expensive operation)
   - **Serial**: No-op (always uses 1 thread)

2. **Auto-detection**: If `n = 0`, uses `hardware_concurrency()`

**Edge Cases**

- **n = 0**: Auto-detects and uses all available cores
- **n > hardware_concurrency()**: May create more threads than cores (oversubscription)
- **BS backend**: Expensive operation - joins old threads and spawns new ones

**Data Guarantees (Preconditions)**

- `n >= 0` (0 means auto-detect)

**Complexity Analysis**

- **Time**: 
  - O(n) for BS backend (thread creation/destruction)
  - O(1) for OpenMP and TBB
- **Space**: 
  - O(n) thread stack space for BS
  - O(1) for OpenMP and TBB

**Example**

```cpp
// Set to 4 threads
scl::threading::Scheduler::set_num_threads(4);

// Auto-detect (use all cores)
scl::threading::Scheduler::set_num_threads(0);
```

---

### Scheduler::get_num_threads

::: source_code file="scl/threading/scheduler.hpp" symbol="get_num_threads" collapsed
:::

**Algorithm Description**

Get currently configured number of worker threads:

Returns the active thread count based on backend:
- **OpenMP**: `omp_get_max_threads()`
- **TBB**: `tbb::global_control::active_value()` (system-wide parallelism limit)
- **BS**: `pool.get_thread_count()`
- **Serial**: Always returns 1

**Edge Cases**

- **Before initialization**: Returns default (typically hardware_concurrency() or 1)
- **TBB**: Returns system-wide parallelism limit, not per-process limit

**Data Guarantees (Preconditions)**

None - safe to call at any time.

**Complexity Analysis**

- **Time**: O(1)
- **Space**: O(1)

**Example**

```cpp
size_t threads = scl::threading::Scheduler::get_num_threads();
// Returns currently configured thread count
```

---

### Scheduler::init

::: source_code file="scl/threading/scheduler.hpp" symbol="init" collapsed
:::

**Algorithm Description**

Initialize threading subsystem:

Equivalent to `set_num_threads(n)`, provided for clarity in initialization code. Should be called once during application startup, before any parallel operations.

**Edge Cases**

- **Multiple calls**: Safe but redundant (equivalent to calling `set_num_threads`)
- **n = 0**: Auto-detects and uses all available cores

**Data Guarantees (Preconditions)**

- `n >= 0` (0 means auto-detect)

**Complexity Analysis**

Same as `set_num_threads()`.

**Example**

```cpp
// Initialize with all cores (recommended)
scl::threading::Scheduler::init();

// Or specify number of threads
scl::threading::Scheduler::init(4);
```

---

## Notes

**Backend Behavior**

| Backend | set_num_threads | get_num_threads | Notes |
|---------|----------------|-----------------|-------|
| OpenMP | `omp_set_num_threads()` | `omp_get_max_threads()` | Affects all parallel regions |
| TBB | `tbb::global_control` | `tbb::global_control::active_value()` | System-wide parallelism limit |
| BS | `pool.reset(n)` | `pool.get_thread_count()` | Expensive (recreates pool) |
| Serial | No-op | Always 1 | Single-threaded |

**Performance Warnings**

- **BS backend**: `set_num_threads()` is expensive (joins and recreates threads)
  - Do NOT call in hot loops or performance-critical sections
  - Call once during initialization
- **OpenMP/TBB**: `set_num_threads()` is cheap (O(1))

**Best Practices**

1. **Call during initialization**: Set thread count once at startup
2. **Use auto-detection**: `init()` or `set_num_threads(0)` to use all cores
3. **Avoid runtime changes**: Don't change thread count in hot paths
4. **Query when needed**: Use `get_num_threads()` to size workspace pools

**Thread Safety**

- All methods are thread-safe
- `set_num_threads()` should not be called concurrently (not reentrant)
- Safe to query from multiple threads

## See Also

- [Parallel For](./parallel_for) - Parallel loop execution
- [Workspace](./workspace) - Per-thread workspace pools

