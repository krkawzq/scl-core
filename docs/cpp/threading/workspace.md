# workspace.hpp

> scl/threading/workspace.hpp · Per-thread workspace management to eliminate allocations

## Overview

This file provides thread-local workspace pools to eliminate per-iteration allocations in parallel loops. Each thread gets its own pre-allocated buffer, avoiding synchronization and improving cache locality.

Key features:
- Pre-allocated per-thread buffers
- Zero allocations in hot loops
- No synchronization required
- Cache-friendly memory layout
- Multiple pool types for different use cases

**Header**: `#include "scl/threading/workspace.hpp"`

---

## Main APIs

### WorkspacePool

::: source_code file="scl/threading/workspace.hpp" symbol="WorkspacePool" collapsed
:::

**Algorithm Description**

Pre-allocated buffer pool for per-thread temporary storage:

1. **Initialization**: Allocates single contiguous block for all threads
   - Total size: `n_threads * capacity`
   - Aligned allocation for SIMD operations
   - Each thread's buffer: `data + thread_rank * capacity`

2. **Access**: `get(thread_rank)` returns pointer to thread's buffer
   - O(1) pointer arithmetic
   - No locking or synchronization
   - Thread-safe (each thread accesses different memory)

3. **Memory Layout**: Contiguous allocation improves cache locality
   - Thread buffers are adjacent in memory
   - Better for sequential access patterns

**Edge Cases**

- **Uninitialized pool**: `get()` returns nullptr (undefined behavior if used)
- **Invalid thread_rank**: Out-of-bounds access (undefined behavior)
- **Zero capacity**: Valid but useless (no storage)

**Data Guarantees (Preconditions)**

- Pool must be initialized with `init(n_threads, capacity)` before use
- `thread_rank` must be in range [0, n_threads)
- `capacity > 0` for useful storage

**Complexity Analysis**

- **Time**: 
  - `init()`: O(n_threads * capacity) for allocation
  - `get()`: O(1) pointer arithmetic
- **Space**: O(n_threads * capacity) for all thread buffers

**Example**

```cpp
#include "scl/threading/workspace.hpp"

// Create and initialize pool
scl::threading::WorkspacePool<Real> pool;
pool.init(num_threads, workspace_size);  // Each thread gets workspace_size elements

// Use in parallel loop
scl::threading::parallel_for(0, n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);
    
    // Use workspace without synchronization
    workspace[0] = compute_intermediate(data[i]);
    Real result = process(workspace, workspace_size);
    output[i] = result;
});

// Zero-initialize thread's buffer
pool.zero(thread_rank);

// Get Array view
Array<Real> workspace = pool.span(thread_rank);
```

---

### DualWorkspacePool

::: source_code file="scl/threading/workspace.hpp" symbol="DualWorkspacePool" collapsed
:::

**Algorithm Description**

Dual buffer pool for algorithms needing two buffers per thread:

Wraps two `WorkspacePool` instances, providing `get1()` and `get2()` for accessing each buffer. Useful for algorithms that alternate between two buffers (e.g., iterative algorithms with ping-pong buffers).

**Edge Cases**

- Same as `WorkspacePool` - uninitialized or invalid thread_rank

**Data Guarantees (Preconditions)**

- Same as `WorkspacePool`
- Both buffers have same capacity

**Complexity Analysis**

- **Time**: Same as `WorkspacePool` (two pools)
- **Space**: O(2 * n_threads * capacity) for both buffers

**Example**

```cpp
scl::threading::DualWorkspacePool<Real> pool;
pool.init(num_threads, capacity);

scl::threading::parallel_for(0, n, [&](size_t i, size_t thread_rank) {
    Real* buf1 = pool.get1(thread_rank);
    Real* buf2 = pool.get2(thread_rank);
    
    // Alternate between buffers
    process(buf1, buf2);
});
```

---

### DynamicWorkspacePool

::: source_code file="scl/threading/workspace.hpp" symbol="DynamicWorkspacePool" collapsed
:::

**Algorithm Description**

Variable-size buffer pool for variable-length work items:

Each thread gets a `ThreadBuffer` with:
- Fixed capacity (max size)
- Variable size (current usage)
- `push_back()` for adding elements
- Iterator support

Useful when work items have variable sizes but bounded maximum.

**Edge Cases**

- **Buffer overflow**: `push_back()` when `size >= capacity` causes undefined behavior
- **Uninitialized**: Same as `WorkspacePool`

**Data Guarantees (Preconditions)**

- Pool must be initialized with `init(n_threads, max_capacity)`
- `push_back()` only when `size < capacity`
- `thread_rank` in range [0, n_threads)

**Complexity Analysis**

- **Time**: 
  - `init()`: O(n_threads * max_capacity)
  - `push_back()`: O(1)
  - `clear()`: O(1)
- **Space**: O(n_threads * max_capacity)

**Example**

```cpp
scl::threading::DynamicWorkspacePool<Index> pool;
pool.init(num_threads, max_capacity);

scl::threading::parallel_for(0, n, [&](size_t i, size_t thread_rank) {
    auto& buffer = pool.get(thread_rank);
    buffer.clear();  // Reset for new work item
    
    // Collect variable number of indices
    for (Index j = 0; j < variable_size; ++j) {
        buffer.push_back(indices[j]);
    }
    
    // Process collected indices
    process(buffer.begin(), buffer.end());
});
```

---

## Notes

**Memory Layout**

All pools use single contiguous allocation:
- Better cache locality
- Single allocation/deallocation
- Aligned for SIMD operations

**Thread Safety**

- Each thread accesses different memory (no races)
- No locking required
- Safe for concurrent access from different threads

**Use Cases**

- **Temporary arrays**: Store intermediate results per thread
- **Accumulators**: Per-thread reduction buffers
- **Scratch space**: Algorithm-specific temporary storage
- **Variable collections**: DynamicWorkspacePool for variable-length data

**Best Practices**

1. **Pre-allocate**: Initialize pool before parallel loop
2. **Size appropriately**: Allocate enough capacity for worst case
3. **Reuse**: Create pool once, reuse across multiple loops
4. **Zero when needed**: Use `zero()` to reset buffers between iterations

**Performance Benefits**

- **Eliminates allocations**: No `malloc`/`new` in hot loops
- **No synchronization**: Each thread has private buffer
- **Cache-friendly**: Contiguous memory layout
- **SIMD-aligned**: Aligned allocation for vectorization

## See Also

- [Parallel For](./parallel_for) - Parallel loop execution
- [Memory Management](/cpp/core/memory) - Memory allocation utilities

