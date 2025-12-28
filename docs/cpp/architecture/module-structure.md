# Module Structure

This document describes the organization and dependencies of SCL-Core modules.

## Directory Layout

```
scl/
├── config.hpp          # Build configuration
├── version.hpp         # Version information
├── core/               # Foundation layer
├── threading/          # Parallel processing
├── kernel/             # Computational operators
├── math/               # Statistical functions
├── mmap/               # Memory-mapped arrays
└── io/                 # I/O utilities
```

## Dependency Graph

```
┌─────────────────────────────────────────┐
│              Applications               │
│         (Python bindings, etc.)         │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│              math/                      │
│    (stats, regression, approximation)   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│            kernel/                      │
│  (normalize, neighbors, leiden, etc.)   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│          threading/                     │
│  (parallel_for, scheduler, workspace)   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│             core/                       │
│  (types, sparse, SIMD, registry, etc.)  │
└─────────────────────────────────────────┘
```

**Dependency Rules:**
- Lower layers never depend on upper layers
- Each layer can only depend on layers below it
- `core/` has no internal dependencies

## Core Layer (`scl/core/`)

Foundation types and utilities.

### Files

| File | Purpose | Dependencies |
|------|---------|--------------|
| `type.h/hpp` | Type system (Real, Index, Size) | C++ standard library |
| `macros.h/hpp` | Compiler macros and attributes | None |
| `error.h/hpp` | Error handling and assertions | `macros.hpp` |
| `memory.h/hpp` | Aligned memory allocation | `type.hpp`, `error.hpp` |
| `registry.h/hpp` | Memory lifetime tracking | `type.hpp`, `error.hpp` |
| `simd.h/hpp` | SIMD abstraction (Highway) | `type.hpp`, Highway |
| `vectorize.h/hpp` | Vectorized operations | `simd.hpp` |
| `sparse.h/hpp` | Sparse matrix infrastructure | `type.hpp`, `registry.hpp` |
| `sort.h/hpp` | Sorting algorithms | `type.hpp` |
| `argsort.h/hpp` | Argsort implementation | `type.hpp`, `sort.hpp` |
| `algo.hpp` | Generic algorithms | `type.hpp` |

### Key Types

```cpp
namespace scl {
    // Fundamental types
    using Real = /* float | double | _Float16 */;
    using Index = /* int16_t | int32_t | int64_t */;
    using Size = size_t;
    
    // Sparse matrix
    template <typename T, bool IsCSR>
    struct Sparse { /* ... */ };
    
    // Memory registry
    class Registry { /* ... */ };
}
```

### Design Notes

- **No external dependencies** except C++ standard library and Highway
- **Header-only** for most components
- **Minimal templates** to reduce compile time

## Threading Layer (`scl/threading/`)

Parallel processing infrastructure.

### Files

| File | Purpose | Dependencies |
|------|---------|--------------|
| `scheduler.h/hpp` | Thread pool and work queue | `core/` |
| `parallel_for.h/hpp` | Parallel loop abstraction | `scheduler.hpp` |
| `workspace.hpp` | Per-thread workspace pools | `core/memory.hpp` |

### Key APIs

```cpp
namespace scl::threading {
    // Parallel loop
    void parallel_for(Size begin, Size end, 
                      std::function<void(size_t, size_t)> func);
    
    // Thread pool
    class Scheduler {
        void execute(Task task);
        size_t num_threads() const;
    };
    
    // Workspace pool
    template <typename T>
    class WorkspacePool {
        T* get(size_t thread_rank);
    };
}
```

### Design Notes

- **Work-stealing scheduler** for load balancing
- **Automatic parallelization** based on problem size
- **Per-thread workspaces** to avoid synchronization

## Kernel Layer (`scl/kernel/`)

Computational operators (80+ files, 400+ functions).

### Organization by Function

| Category | Files | Functions |
|----------|-------|-----------|
| Sparse Tools | `sparse.hpp`, `sparse_opt.hpp` | Matrix conversion, validation, cleanup |
| Normalization | `normalize.hpp`, `scale.hpp`, `log1p.hpp` | Row/col normalization, scaling |
| Statistics | `ttest.hpp`, `mwu.hpp`, `metrics.hpp` | Statistical tests, metrics |
| Neighbors | `neighbors.hpp`, `bbknn.hpp` | KNN, batch-balanced KNN |
| Clustering | `leiden.hpp`, `louvain.hpp` | Community detection |
| Spatial | `spatial.hpp`, `hotspot.hpp` | Spatial analysis |
| Enrichment | `enrichment.hpp`, `scoring.hpp` | Gene set enrichment |
| Dimensionality | `pca.hpp`, `svd.hpp` | PCA, SVD |

### Key Patterns

**Functional API**

```cpp
namespace scl::kernel::normalize {
    // Pure function - no side effects
    void row_norms(const Sparse<Real, true>& matrix,
                   NormMode mode,
                   MutableSpan<Real> output);
    
    // In-place modification
    void normalize_rows_inplace(Sparse<Real, true>& matrix,
                                NormMode mode,
                                Real epsilon = 1e-12);
}
```

**Template-Based Polymorphism**

```cpp
// Works with any CSR-like type
template <CSRLike MatrixT>
void process_matrix(const MatrixT& matrix) {
    // ...
}
```

### Design Notes

- **Stateless functions** - no hidden state
- **Explicit parallelization** via `parallel_for`
- **SIMD-optimized** hot paths
- **Minimal allocations** - use workspaces

## Math Layer (`scl/math/`)

Statistical functions and regression.

### Files

| File | Purpose | Dependencies |
|------|---------|--------------|
| `stats.h/hpp` | Basic statistics | `core/`, `threading/` |
| `regression.h/hpp` | Linear regression | `core/`, `threading/`, `kernel/` |
| `mwu.h/hpp` | Mann-Whitney U test | `core/`, `threading/` |
| `approx/` | Fast approximations | `core/simd.hpp` |

### Key APIs

```cpp
namespace scl::math {
    // Statistics
    Real mean(const Real* data, size_t n);
    Real variance(const Real* data, size_t n, Real mean);
    
    // Regression
    void linear_regression(const Real* x, const Real* y, size_t n,
                          Real& slope, Real& intercept);
    
    // Statistical tests
    Real mann_whitney_u(const Real* x, size_t nx,
                        const Real* y, size_t ny);
}
```

## Memory-Mapped Layer (`scl/mmap/`)

Large dataset support via memory-mapped files.

### Files

| Directory | Purpose |
|-----------|---------|
| `backend/` | OS-specific mmap implementations |
| `cache/` | Page cache management |
| `memory/` | Memory pool for pages |
| `array.h/hpp` | Memory-mapped array |
| `sparse.hpp` | Memory-mapped sparse matrix |

### Key APIs

```cpp
namespace scl::mmap {
    // Memory-mapped array
    template <typename T>
    class Array {
        T& operator[](size_t i);
        void flush();
    };
    
    // Memory-mapped sparse matrix
    template <typename T, bool IsCSR>
    class MappedSparse {
        // Same API as scl::Sparse
    };
}
```

## I/O Layer (`scl/io/`)

File I/O utilities.

### Purpose

- Read/write sparse matrices
- Format conversion
- Serialization

## Inter-Module Communication

### Registry Pattern

Shared memory tracking across modules:

```cpp
// In kernel/
auto& reg = scl::get_registry();
Real* data = reg.new_array<Real>(count);

// In Python binding
// Registry allows Python to take ownership
```

### Workspace Pools

Shared temporary storage:

```cpp
// In threading/
WorkspacePool<Real> pool(num_threads, workspace_size);

// In kernel/
parallel_for(0, n, [&](size_t i, size_t thread_rank) {
    auto* workspace = pool.get(thread_rank);
    // Use workspace
});
```

## Build Configuration

### Compile-Time Options

```cpp
// In config.hpp
#define SCL_USE_FLOAT32    // or FLOAT64, FLOAT16
#define SCL_USE_INT32      // or INT16, INT64
#define SCL_ENABLE_SIMD    // Enable SIMD
#define SCL_ENABLE_OPENMP  // Enable OpenMP
```

### CMake Targets

```cmake
# Core library (header-only)
add_library(scl-core INTERFACE)

# Kernels (compiled)
add_library(scl-kernels STATIC
    kernel/normalize.cpp
    kernel/neighbors.cpp
    # ...
)

# Python bindings
add_library(scl-python SHARED
    binding/python.cpp
)
```

## Adding New Modules

### Guidelines

1. **Place in appropriate layer** based on dependencies
2. **Follow naming conventions** (`module.h` for docs, `module.hpp` for impl)
3. **Document all public APIs** in `.h` files
4. **Add to CMakeLists.txt**
5. **Update this document**

### Template

```cpp
// scl/kernel/mymodule.h
#pragma once

namespace scl::kernel::mymodule {

/* -----------------------------------------------------------------------------
 * FUNCTION: my_function
 * -----------------------------------------------------------------------------
 * SUMMARY: Brief description
 * 
 * PARAMETERS:
 *     input [in]  Description
 *     output [out] Description
 * 
 * PRECONDITIONS: ...
 * POSTCONDITIONS: ...
 * COMPLEXITY: Time O(...), Space O(...)
 * THREAD SAFETY: Safe / Unsafe
 * -------------------------------------------------------------------------- */
void my_function(const Real* input, Real* output, size_t n);

} // namespace scl::kernel::mymodule
```

```cpp
// scl/kernel/mymodule.hpp
#pragma once
#include "scl/kernel/mymodule.h"
#include "scl/core/type.hpp"
#include "scl/threading/parallel_for.hpp"

namespace scl::kernel::mymodule {

void my_function(const Real* input, Real* output, size_t n) {
    // Implementation
}

} // namespace scl::kernel::mymodule
```

---

::: tip Module Independence
Keep modules loosely coupled. Each module should be usable independently with minimal dependencies.
:::

