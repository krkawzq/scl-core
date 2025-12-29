---
title: Threading and Parallelization
description: Parallel execution backends and parallel_for interface
---

# Threading and Parallelization

SCL-Core provides a unified parallelization interface that abstracts over multiple threading backends, enabling optimal performance across different platforms.

## Overview

SCL-Core supports multiple threading backends:

- **OpenMP**: Industry standard for HPC (default on Linux/Windows)
- **Intel TBB**: Task-based parallelism
- **BS::thread_pool**: Header-only thread pool (default on macOS)
- **Serial**: Single-threaded execution (for debugging)

The backend is selected at compile time based on platform and configuration.

## Parallel For Interface

### Basic Usage

```cpp
#include "scl/threading/parallel_for.hpp"

namespace scl::threading {
    template <typename Func>
    void parallel_for(size_t start, size_t end, Func&& func);
}
```

**Usage**:
```cpp
// Parallel loop
threading::parallel_for(Size(0), n, [&](size_t i) {
    // Process element i in parallel
    process(data[i]);
});
```

### With Thread Rank

The function can optionally accept a thread rank parameter:

```cpp
// Single argument (index only)
threading::parallel_for(0, n, [&](size_t i) {
    process(data[i]);
});

// Dual argument (index and thread rank)
threading::parallel_for(0, n, [&](size_t i, size_t thread_rank) {
    // Use thread-local storage
    thread_local_storage[thread_rank].process(data[i]);
});
```

The interface automatically detects which signature is used.

## Backend Selection

### Automatic Selection

SCL-Core automatically selects a backend based on platform:

```cpp
// macOS: BS::thread_pool (unless SCL_MAC_USE_OPENMP is defined)
// Linux/Windows: OpenMP
// Unknown: BS::thread_pool (fallback)
```

### Manual Selection

Configure at compile time:

```cpp
// In config.hpp or via compile flags
#define SCL_BACKEND_OPENMP   // Use OpenMP
#define SCL_BACKEND_TBB      // Use Intel TBB
#define SCL_BACKEND_BS       // Use BS::thread_pool
#define SCL_BACKEND_SERIAL   // Serial execution
```

**CMake Configuration**:
```cmake
# Set threading backend
set(SCL_THREADING_BACKEND "OPENMP" CACHE STRING "Threading backend")

# Or for macOS with OpenMP
set(SCL_MAC_USE_OPENMP ON)
```

## Backend Details

### OpenMP

**Pros**:
- Industry standard for HPC
- Excellent compiler support
- Mature and well-optimized

**Cons**:
- Requires libomp on macOS
- May have issues with nested parallelism

**Usage**:
```cpp
#define SCL_BACKEND_OPENMP
// OpenMP is used automatically
```

### Intel TBB

**Pros**:
- Task-based parallelism
- Good load balancing
- Supports nested parallelism

**Cons**:
- Requires TBB library
- Additional dependency

**Usage**:
```cpp
#define SCL_BACKEND_TBB
// TBB is used automatically
```

### BS::thread_pool

**Pros**:
- Header-only, zero dependencies
- Works everywhere
- Good default for macOS

**Cons**:
- Less optimized than OpenMP/TBB
- No nested parallelism support

**Usage**:
```cpp
#define SCL_BACKEND_BS
// BS::thread_pool is used automatically
```

### Serial

**Pros**:
- No threading overhead
- Easier debugging
- Deterministic execution

**Cons**:
- No parallelism
- Slower for large datasets

**Usage**:
```cpp
#define SCL_BACKEND_SERIAL
// Serial execution (for debugging)
```

## Thread Safety

### Safe Operations

These operations are thread-safe:

```cpp
// Reading from sparse matrix (different rows)
threading::parallel_for(0, matrix.rows(), [&](size_t i) {
    auto row = matrix.row_values(static_cast<Index>(i));
    // Safe: Each thread accesses different row
});

// Writing to different output locations
Array<Real> output = {output_ptr, n};
threading::parallel_for(0, n, [&](size_t i) {
    output[i] = compute(data[i]);
    // Safe: Each thread writes to different location
});
```

### Unsafe Operations

These operations are **not** thread-safe:

```cpp
// Writing to shared location
Real sum = 0;
threading::parallel_for(0, n, [&](size_t i) {
    sum += data[i];  // Race condition!
});

// Use reduction instead
Real sum = 0;
#pragma omp parallel for reduction(+:sum)
for (size_t i = 0; i < n; ++i) {
    sum += data[i];
}
```

### Thread-Local Storage

For thread-local data:

```cpp
// Option 1: Use thread rank
threading::parallel_for(0, n, [&](size_t i, size_t thread_rank) {
    thread_local_data[thread_rank].process(data[i]);
});

// Option 2: Use thread_local (C++11)
thread_local Real local_sum = 0;
threading::parallel_for(0, n, [&](size_t i) {
    local_sum += data[i];
});
```

## Parallel Patterns

### Row-Wise Processing

```cpp
// Process each row in parallel
threading::parallel_for(Size(0), static_cast<Size>(matrix.rows()), [&](size_t i) {
    Index row = static_cast<Index>(i);
    auto values = matrix.row_values(row);
    auto indices = matrix.row_indices(row);
    Index len = matrix.row_length(row);
    
    // Process row
    for (Index k = 0; k < len; ++k) {
        process_element(indices[k], values[k]);
    }
});
```

### Chunked Processing

```cpp
// Process in chunks
constexpr Size CHUNK_SIZE = 256;
Size n_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;

threading::parallel_for(Size(0), n_chunks, [&](size_t chunk_idx) {
    Size start = chunk_idx * CHUNK_SIZE;
    Size end = std::min(start + CHUNK_SIZE, n);
    
    for (Size i = start; i < end; ++i) {
        process(data[i]);
    }
});
```

### Reduction

```cpp
// Parallel reduction (using thread-local storage)
constexpr Size NUM_THREADS = 8;  // Or detect at runtime
std::array<Real, NUM_THREADS> partial_sums{};

threading::parallel_for(0, n, [&](size_t i, size_t thread_rank) {
    partial_sums[thread_rank] += data[i];
});

// Sequential reduction
Real total = 0;
for (Real sum : partial_sums) {
    total += sum;
}
```

## Performance Considerations

### Parallel Threshold

Most kernels automatically parallelize when data size exceeds a threshold:

```cpp
namespace config {
    constexpr Size PARALLEL_THRESHOLD = 256;  // Typical value
}

// Kernels check threshold before parallelizing
if (n >= config::PARALLEL_THRESHOLD) {
    threading::parallel_for(0, n, [&](size_t i) {
        process(data[i]);
    });
} else {
    // Sequential execution for small data
    for (size_t i = 0; i < n; ++i) {
        process(data[i]);
    }
}
```

### Work Distribution

Different backends have different work distribution strategies:

- **OpenMP**: Static scheduling by default (good for uniform work)
- **TBB**: Dynamic scheduling (good for variable work)
- **BS::thread_pool**: Static chunks (simple and effective)

### Nested Parallelism

```cpp
// OpenMP and TBB support nested parallelism
// BS::thread_pool does not

// Outer parallel loop
threading::parallel_for(0, n_outer, [&](size_t i) {
    // Inner parallel loop (only with OpenMP/TBB)
    threading::parallel_for(0, n_inner, [&](size_t j) {
        process(i, j);
    });
});
```

## Best Practices

### 1. Use Parallel For for Independent Work

```cpp
// Good: Independent operations
threading::parallel_for(0, n, [&](size_t i) {
    output[i] = compute(input[i]);  // Independent
});

// Avoid: Dependent operations
threading::parallel_for(0, n, [&](size_t i) {
    output[i] = compute(input[i], output[i-1]);  // Dependency!
});
```

### 2. Avoid False Sharing

```cpp
// Bad: Adjacent writes cause false sharing
struct Data {
    Real value;  // Adjacent in memory
};
std::vector<Data> data(n);

threading::parallel_for(0, n, [&](size_t i) {
    data[i].value = compute(i);  // May cause false sharing
});

// Good: Use padding or separate arrays
struct Data {
    Real value;
    char padding[64 - sizeof(Real)];  // Pad to cache line
};
```

### 3. Minimize Critical Sections

```cpp
// Bad: Frequent locking
std::mutex mtx;
threading::parallel_for(0, n, [&](size_t i) {
    std::lock_guard<std::mutex> lock(mtx);
    shared_resource.process(data[i]);  // Lock contention
});

// Good: Batch operations
thread_local std::vector<Real> local_results;
threading::parallel_for(0, n, [&](size_t i) {
    local_results.push_back(compute(data[i]));
});

// Sequential merge
for (auto& result : all_local_results) {
    merge(result);
}
```

### 4. Profile Before Optimizing

```cpp
// Measure performance
auto start = std::chrono::high_resolution_clock::now();
threading::parallel_for(0, n, [&](size_t i) {
    process(data[i]);
});
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
```

## Configuration

### Compile-Time Configuration

```cpp
// In config.hpp
#define SCL_BACKEND_OPENMP   // Select backend
#define SCL_MAC_USE_OPENMP   // Force OpenMP on macOS
```

### Runtime Configuration

Some backends support runtime configuration:

```cpp
// OpenMP: Set number of threads
omp_set_num_threads(4);

// TBB: Configure task arena
tbb::task_arena arena(4);
arena.execute([&]() {
    threading::parallel_for(0, n, [&](size_t i) {
        process(data[i]);
    });
});
```

## Related Documentation

- [Core Types](./core/types.md) - Array views and types
- [Kernels](./kernels/) - Parallel kernel implementations
- [Error Handling](./error-handling.md) - Thread-safe error handling

