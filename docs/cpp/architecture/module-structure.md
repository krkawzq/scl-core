# Module Structure

This document describes SCL-Core's module organization, dependencies, and design patterns. The architecture follows strict layering to maintain clean separation of concerns and minimize compilation dependencies.

## Design Philosophy

### Strict Layering

Dependencies flow in one direction only: from upper layers to lower layers. Lower layers never depend on upper layers.

```
Upper Layers (Application-specific)
    ↓
    ↓ Depends on
    ↓
Lower Layers (Foundational)
```

This ensures:
- Lower layers can be used independently
- Changes to upper layers do not affect lower layers
- Compilation order is deterministic
- No circular dependencies

### Minimal Dependencies

Each module depends only on what it needs. The core layer has zero internal dependencies except the C++ standard library and Google Highway for SIMD.

### Header-Only When Possible

Most core components are header-only for:
- Zero runtime overhead (perfect inlining)
- Simplified build system
- Easy integration into other projects
- Template-heavy code works naturally

## Directory Layout

```
scl/
├── config.hpp          # Build configuration (types, features)
├── version.hpp         # Version information
│
├── core/               # Foundation layer (no dependencies)
│   ├── type.hpp        # Fundamental types (Real, Index, Size)
│   ├── macros.hpp      # Compiler macros and attributes
│   ├── error.hpp       # Error handling, assertions
│   ├── memory.hpp      # Aligned memory allocation
│   ├── registry.hpp    # Memory lifetime tracking
│   ├── simd.hpp        # SIMD abstraction (Highway)
│   ├── vectorize.hpp   # Vectorized operations
│   ├── sparse.hpp      # Sparse matrix infrastructure
│   ├── sort.hpp        # Sorting algorithms
│   ├── argsort.hpp     # Argsort implementation
│   └── algo.hpp        # Generic algorithms
│
├── threading/          # Parallel processing (depends: core/)
│   ├── scheduler.hpp   # Work-stealing thread pool
│   ├── parallel_for.hpp # Parallel loop abstraction
│   └── workspace.hpp   # Per-thread workspace pools
│
├── kernel/             # Computational operators (depends: core/, threading/)
│   ├── normalize.hpp   # Row/column normalization
│   ├── neighbors.hpp   # KNN computation
│   ├── leiden.hpp      # Leiden clustering
│   ├── ttest.hpp       # Statistical tests
│   ├── spatial.hpp     # Spatial analysis
│   └── ... (80+ files)
│
├── math/               # Statistical functions (depends: core/, threading/, kernel/)
│   ├── stats.hpp       # Basic statistics
│   ├── regression.hpp  # Linear regression
│   ├── mwu.hpp         # Mann-Whitney U test
│   └── approx/         # Fast approximations
│       ├── exp.hpp
│       ├── log.hpp
│       └── special.hpp
│
├── mmap/               # Memory-mapped arrays (depends: core/)
│   ├── backend/        # OS-specific implementations
│   │   ├── posix.hpp
│   │   └── windows.hpp
│   ├── cache/          # Page cache management
│   ├── memory/         # Memory pool for pages
│   ├── array.hpp       # Memory-mapped array
│   └── sparse.hpp      # Memory-mapped sparse matrix
│
└── io/                 # File I/O utilities (depends: core/)
    ├── matrix.hpp      # Matrix serialization
    └── format.hpp      # Format conversion
```

## Dependency Graph

```
┌──────────────────────────────────────┐
│        Python Bindings               │
│    (scl-py, external project)        │
└────────────────┬─────────────────────┘
                 │
                 │ uses
                 ↓
┌────────────────────────────────────┐
│              math/                 │
│  Statistical computations          │
│  (stats, regression, mwu)          │
└────────────────┬───────────────────┘
                 │
                 │ uses
                 ↓
┌────────────────────────────────────┐
│            kernel/                 │
│  Computational operators           │
│  (normalize, neighbors, leiden)    │
└────────────────┬───────────────────┘
                 │
                 │ uses
                 ↓
┌────────────────────────────────────┐
│          threading/                │
│  Parallel processing               │
│  (parallel_for, scheduler)         │
└────────────────┬───────────────────┘
                 │
                 │ uses
                 ↓
┌────────────────────────────────────┐
│             core/                  │
│  Foundation types and utilities    │
│  (types, sparse, SIMD, registry)   │
└────────────────────────────────────┘
         │
         │ depends on
         ↓
┌────────────────────────────────────┐
│  C++ Standard Library + Highway    │
└────────────────────────────────────┘
```

**Dependency Rules:**
- core/ → C++ stdlib, Highway only
- threading/ → core/
- kernel/ → core/, threading/
- math/ → core/, threading/, kernel/
- Python bindings → all of SCL-Core

## Layer Descriptions

### Core Layer (scl/core/)

**Purpose:** Foundation types, data structures, and utilities used by all other layers.

**Key Files:**

| File | Purpose | Dependencies |
|------|---------|--------------|
| type.hpp | Fundamental types (Real, Index, Size) | C++ stdlib |
| macros.hpp | Compiler macros (SCL_FORCE_INLINE, etc.) | None |
| error.hpp | Error handling (SCL_ASSERT, SCL_CHECK_*) | macros.hpp |
| memory.hpp | Aligned memory allocation | type.hpp, error.hpp |
| registry.hpp | Memory lifetime tracking | type.hpp, error.hpp |
| simd.hpp | SIMD abstraction (Highway wrapper) | type.hpp, Highway |
| vectorize.hpp | Vectorized operations | simd.hpp |
| sparse.hpp | Sparse matrix infrastructure | type.hpp, registry.hpp |
| sort.hpp | Sorting algorithms | type.hpp |
| argsort.hpp | Argsort implementation | type.hpp, sort.hpp |
| algo.hpp | Generic algorithms (min, max, etc.) | type.hpp |

**Key Types:**

```cpp
namespace scl {
    // Configurable fundamental types (compile-time)
    using Real = /* float | double | _Float16 */;
    using Index = /* int16_t | int32_t | int64_t */;
    using Size = size_t;
    
    // Sparse matrix (discontiguous storage)
    template <typename T, bool IsCSR>
    struct Sparse {
        using Pointer = T*;
        Pointer* data_ptrs_;      // Per-row/col data pointers
        Pointer* indices_ptrs_;   // Per-row/col index pointers
        Index* lengths_;          // Per-row/col lengths
        Index rows_, cols_, nnz_;
    };
    
    // Memory registry (singleton)
    class Registry {
        template <typename T> T* new_array(size_t count);
        void register_ptr(void* ptr, size_t bytes, AllocType type);
        void unregister_ptr(void* ptr);
        // ...
    };
}
```

**Design Notes:**
- Header-only except for Registry (singleton needs definition)
- Zero dependencies on other SCL modules
- Minimal templates to reduce compilation time
- All public APIs in scl namespace

### Threading Layer (scl/threading/)

**Purpose:** Parallel processing infrastructure for multi-core CPUs.

**Key Files:**

| File | Purpose | Dependencies |
|------|---------|--------------|
| scheduler.hpp | Work-stealing thread pool | core/ |
| parallel_for.hpp | Parallel loop abstraction | scheduler.hpp |
| workspace.hpp | Per-thread workspace pools | core/memory.hpp |

**Key APIs:**

```cpp
namespace scl::threading {
    // Parallel loop with automatic parallelization
    void parallel_for(Size begin, Size end, 
                      std::function<void(size_t, size_t)> func);
    
    // Thread pool
    class Scheduler {
        void execute(Task task);
        size_t num_threads() const;
        void set_num_threads(size_t n);
    };
    
    // Workspace pool for temporary storage
    template <typename T>
    class WorkspacePool {
        WorkspacePool(size_t num_threads, size_t workspace_size);
        T* get(size_t thread_rank);
    };
}
```

**Design Notes:**
- Work-stealing scheduler for load balancing
- Automatic parallelization based on problem size
- Per-thread workspaces avoid synchronization
- Nested parallelism supported

### Kernel Layer (scl/kernel/)

**Purpose:** Computational operators for biological data analysis (400+ functions across 80+ files).

**Organization by Function:**

| Category | Files | Key Functions |
|----------|-------|---------------|
| Sparse Tools | sparse.hpp, sparse_opt.hpp | to_csr, to_csc, to_contiguous_arrays |
| Normalization | normalize.hpp, scale.hpp, log1p.hpp | normalize_rows, scale_rows, log1p_inplace |
| Statistics | ttest.hpp, mwu.hpp, metrics.hpp | ttest_rows, mann_whitney_u, compute_metrics |
| Neighbors | neighbors.hpp, bbknn.hpp | compute_knn, batch_balanced_knn |
| Clustering | leiden.hpp, louvain.hpp | leiden_clustering, louvain_clustering |
| Spatial | spatial.hpp, hotspot.hpp | spatial_autocorr, hotspot_analysis |
| Enrichment | enrichment.hpp, scoring.hpp | gene_set_enrichment, compute_scores |
| Dimensionality | pca.hpp, svd.hpp | compute_pca, truncated_svd |

**Key Patterns:**

**Functional API - Stateless Functions:**

```cpp
namespace scl::kernel::normalize {
    // Pure function - no side effects
    void row_norms(
        const Sparse<Real, true>& matrix,
        NormMode mode,
        MutableSpan<Real> output
    );
    
    // In-place modification
    void normalize_rows_inplace(
        Sparse<Real, true>& matrix,
        NormMode mode,
        Real epsilon = 1e-12
    );
}
```

**Template-Based Polymorphism:**

```cpp
// Works with any CSR-like type
template <CSRLike MatrixT>
void process_matrix(const MatrixT& matrix) {
    // MatrixT can be Sparse<Real, true> or any compatible type
}

// Concept definition
template <typename T>
concept CSRLike = requires(T matrix) {
    { matrix.rows() } -> std::convertible_to<Index>;
    { matrix.cols() } -> std::convertible_to<Index>;
    { matrix.nnz() } -> std::convertible_to<Index>;
    { matrix.primary_length(0) } -> std::convertible_to<Index>;
    { matrix.primary_values(0) } -> std::convertible_to<Span<Real>>;
    { matrix.primary_indices(0) } -> std::convertible_to<Span<Index>>;
};
```

**Design Notes:**
- Stateless functions - no hidden state
- Explicit parallelization via parallel_for
- SIMD-optimized hot paths (multi-accumulator, fused ops)
- Minimal allocations - use workspace pools
- Dual-file documentation (.h docs, .hpp impl)

### Math Layer (scl/math/)

**Purpose:** Statistical functions and mathematical computations.

**Key Files:**

| File | Purpose | Dependencies |
|------|---------|--------------|
| stats.hpp | Basic statistics (mean, variance, etc.) | core/, threading/ |
| regression.hpp | Linear regression | core/, threading/, kernel/ |
| mwu.hpp | Mann-Whitney U test | core/, threading/ |
| approx/ | Fast approximations | core/simd.hpp |

**Key APIs:**

```cpp
namespace scl::math {
    // Basic statistics
    Real mean(const Real* data, size_t n);
    Real variance(const Real* data, size_t n, Real mean);
    Real std_dev(const Real* data, size_t n, Real mean);
    
    // Regression
    void linear_regression(
        const Real* x, const Real* y, size_t n,
        Real& slope, Real& intercept, Real& r_squared
    );
    
    // Statistical tests
    Real mann_whitney_u(
        const Real* x, size_t nx,
        const Real* y, size_t ny
    );
}

namespace scl::math::approx {
    // Fast approximations (SIMD-optimized)
    Vec exp_fast(Tag d, Vec x);     // ~2-3x faster than std::exp
    Vec log_fast(Tag d, Vec x);     // ~2-3x faster than std::log
    Vec erf_fast(Tag d, Vec x);     // ~5-10x faster than std::erf
}
```

**Design Notes:**
- Build on kernel/ for complex operations
- Use SIMD for numerical computation
- Provide both exact and fast approximations
- Thread-safe, can be called from parallel loops

### Memory-Mapped Layer (scl/mmap/)

**Purpose:** Support large datasets that do not fit in RAM via memory-mapped files.

**Key Files:**

| Directory | Purpose |
|-----------|---------|
| backend/ | OS-specific mmap implementations (POSIX, Windows) |
| cache/ | Page cache management (LRU eviction) |
| memory/ | Memory pool for pages |

| File | Purpose |
|------|---------|
| array.hpp | Memory-mapped dense array |
| sparse.hpp | Memory-mapped sparse matrix |

**Key APIs:**

```cpp
namespace scl::mmap {
    // Memory-mapped dense array
    template <typename T>
    class Array {
        Array(const std::string& path, size_t size);
        T& operator[](size_t i);        // Lazy loading
        void flush();                    // Write back to disk
    };
    
    // Memory-mapped sparse matrix
    template <typename T, bool IsCSR>
    class MappedSparse {
        // Same API as scl::Sparse
        Index rows() const;
        Index cols() const;
        Index nnz() const;
        Span<T> primary_values(Index i);
        // ...
    };
}
```

**Design Notes:**
- Lazy loading - only load pages when accessed
- LRU cache eviction for memory management
- OS-specific backends (posix mmap, Windows MapViewOfFile)
- Same API as in-memory structures for easy migration

### I/O Layer (scl/io/)

**Purpose:** File I/O utilities for serialization and format conversion.

**Key Files:**

| File | Purpose |
|------|---------|
| matrix.hpp | Matrix serialization (binary format) |
| format.hpp | Format conversion (CSR ↔ CSC, dense ↔ sparse) |

**Key APIs:**

```cpp
namespace scl::io {
    // Save/load sparse matrix
    void save_matrix(const std::string& path, const Sparse<Real, true>& matrix);
    Sparse<Real, true> load_matrix(const std::string& path);
    
    // Format conversion
    Sparse<Real, false> csr_to_csc(const Sparse<Real, true>& matrix);
    DenseMatrix sparse_to_dense(const Sparse<Real, true>& matrix);
}
```

## Inter-Module Communication

### Registry Pattern

Shared memory tracking across all modules:

```cpp
// In kernel/ - allocate memory
auto& reg = scl::get_registry();
Real* data = reg.new_array<Real>(count);

// In math/ - use memory
process(data, count);

// In Python binding - transfer ownership
py::array_t<Real> numpy_array = wrap_pointer(data, count);
// Python now owns memory, will call reg.unregister_ptr() on destruction
```

### Workspace Pools

Shared temporary storage pattern:

```cpp
// In threading/ - create pool
WorkspacePool<Real> pool(num_threads, workspace_size);

// In kernel/ - use pool in parallel loop
parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    auto* workspace = pool.get(thread_rank);
    compute_with_workspace(workspace, workspace_size);
});
```

### Type System

Shared types defined in core/:

```cpp
// All modules use same Real, Index, Size types
// Configured once at compile time

#include "scl/core/type.hpp"

void kernel_function(const Real* data, Index n);
void math_function(Real* output, Size count);
```

## Build Configuration

### Compile-Time Options

```cpp
// scl/config.hpp
#pragma once

// Type configuration
#ifdef SCL_USE_FLOAT32
    using Real = float;
#elif defined(SCL_USE_FLOAT64)
    using Real = double;
#elif defined(SCL_USE_FLOAT16)
    using Real = _Float16;
#else
    using Real = float;  // Default
#endif

#ifdef SCL_USE_INT16
    using Index = int16_t;
#elif defined(SCL_USE_INT32)
    using Index = int32_t;
#elif defined(SCL_USE_INT64)
    using Index = int64_t;
#else
    using Index = int32_t;  // Default
#endif

// Feature flags
#ifdef SCL_ENABLE_SIMD
    #define SCL_SIMD_ENABLED 1
#else
    #define SCL_SIMD_ENABLED 0
#endif

#ifdef SCL_ENABLE_OPENMP
    #define SCL_OPENMP_ENABLED 1
    #include <omp.h>
#else
    #define SCL_OPENMP_ENABLED 0
#endif
```

### CMake Configuration

```cmake
# Configure types
option(SCL_USE_FLOAT32 "Use 32-bit floats" ON)
option(SCL_USE_INT32 "Use 32-bit integers" ON)

# Configure features
option(SCL_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(SCL_ENABLE_OPENMP "Enable OpenMP" OFF)
option(SCL_BUILD_TESTS "Build tests" ON)

# Core library (header-only interface)
add_library(scl-core INTERFACE)
target_include_directories(scl-core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(scl-core INTERFACE cxx_std_17)

# Kernels (compiled library)
add_library(scl-kernels STATIC
    scl/kernel/normalize.cpp
    scl/kernel/neighbors.cpp
    scl/kernel/leiden.cpp
    # ... more kernels
)
target_link_libraries(scl-kernels PUBLIC scl-core)

# Find Highway for SIMD
if(SCL_ENABLE_SIMD)
    find_package(hwy REQUIRED)
    target_link_libraries(scl-core INTERFACE hwy::hwy)
    target_compile_definitions(scl-core INTERFACE SCL_ENABLE_SIMD)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(scl-kernels PRIVATE
        -march=native
        -mtune=native
        -ffast-math
        -fno-exceptions  # Optional: reduce binary size
    )
endif()
```

## Adding New Modules

### Module Placement Guidelines

1. **Core layer:** Only if needed by multiple upper layers and has no dependencies
2. **Threading layer:** Parallelization utilities only
3. **Kernel layer:** Most new operators go here
4. **Math layer:** If it builds on kernel/ operations
5. **New layer:** Only if it forms a coherent abstraction used by multiple modules

### Module Template

**Implementation file (scl/kernel/mymodule.hpp):**

```cpp
// =============================================================================
// FILE: scl/kernel/mymodule.hpp
// BRIEF: Implementation of my module
// =============================================================================
#pragma once

#include "scl/kernel/mymodule.h"
#include "scl/core/type.hpp"
#include "scl/threading/parallel_for.hpp"

namespace scl::kernel::mymodule {

template <typename T>
void my_function(const T* input, T* output, size_t n) {
    // Minimal inline comments
    parallel_for(Size(0), n, [&](size_t i) {
        output[i] = process(input[i]);
    });
}

} // namespace scl::kernel::mymodule
```

**Documentation file (scl/kernel/mymodule.h):**

```cpp
// =============================================================================
// FILE: scl/kernel/mymodule.h
// BRIEF: API reference for my module
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

namespace scl::kernel::mymodule {

/* -----------------------------------------------------------------------------
 * FUNCTION: my_function
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Brief one-line description.
 *
 * PARAMETERS:
 *     input  [in]  Input array, size n
 *     output [out] Output array, size n (pre-allocated)
 *     n      [in]  Array size
 *
 * PRECONDITIONS:
 *     - input and output are valid pointers
 *     - output has space for n elements
 *
 * POSTCONDITIONS:
 *     - output[i] contains processed value of input[i]
 *     - input is unchanged
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized internally
 * -------------------------------------------------------------------------- */
template <typename T>
void my_function(
    const T* input,    // Input array [n]
    T* output,         // Output array [n] (pre-allocated)
    size_t n           // Array size
);

} // namespace scl::kernel::mymodule
```

**CMakeLists.txt update:**

```cmake
# If module needs compilation
add_library(scl-kernels STATIC
    # ... existing files ...
    scl/kernel/mymodule.cpp
)
```

### Checklist for New Modules

- [ ] Module placed in appropriate layer
- [ ] Dependencies flow downward only
- [ ] .hpp implementation file with minimal comments
- [ ] .h documentation file with comprehensive docs
- [ ] Functions follow stateless functional pattern
- [ ] Templates used for zero-overhead abstraction
- [ ] SIMD optimization where appropriate
- [ ] Parallelization via parallel_for
- [ ] Memory management via Registry (if needed)
- [ ] Added to CMakeLists.txt
- [ ] Tests written
- [ ] Documentation updated

---

::: tip Module Independence
Good module design enables independent development and testing. Each module should be usable on its own with minimal dependencies. This accelerates development and simplifies debugging.
:::
