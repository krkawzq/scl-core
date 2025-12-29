---
title: Getting Started
description: Quick start guide for C++ developers
---

# Getting Started

This guide will help you get started with SCL-Core in your C++ project.

## Prerequisites

- **C++20** compatible compiler (GCC 11+, Clang 14+, MSVC 19.29+)
- **CMake** 3.15 or later
- **Git** for cloning the repository

### Optional Dependencies

- **OpenMP**: For parallel execution (default on Linux/Windows)
- **Intel TBB**: Alternative threading backend
- **HDF5**: For native h5ad file support

## Installation

### Building from Source

```bash
# Clone repository
git clone https://github.com/krkawzq/scl-core.git
cd scl-core

# Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Install (optional)
cmake --install build
```

### Using as CMake Subdirectory

```cmake
# In your CMakeLists.txt
add_subdirectory(scl-core)

# Link against SCL-Core
target_link_libraries(your_target PRIVATE scl_core_f32)
```

### Using FetchContent

```cmake
include(FetchContent)

FetchContent_Declare(
    scl_core
    GIT_REPOSITORY https://github.com/krkawzq/scl-core.git
    GIT_TAG main  # Or specific tag
)

FetchContent_MakeAvailable(scl_core)

target_link_libraries(your_target PRIVATE scl_core_f32)
```

## First Example

### Basic Usage

Create a file `example.cpp`:

```cpp
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/kernel/normalize.hpp"
#include "scl/threading/parallel_for.hpp"

using namespace scl;

int main() {
    // Create a sparse matrix
    constexpr Index rows = 1000;
    constexpr Index cols = 2000;
    constexpr Index nnz = 10000;
    
    auto matrix = CSR::create(rows, cols, nnz);
    
    // Fill with some data (example)
    // ... populate matrix ...
    
    // Normalize rows
    kernel::normalize::normalize_rows_inplace(matrix, 1e4);
    
    // Process in parallel
    threading::parallel_for(Size(0), static_cast<Size>(matrix.rows()), [&](size_t i) {
        auto row = matrix.row_values(static_cast<Index>(i));
        // Process row
    });
    
    return 0;
}
```

### Compile

```bash
# With CMake
cmake -B build
cmake --build build

# Or manually
g++ -std=c++20 -O3 -march=native example.cpp -lscl_core_f32 -o example
```

## Configuration

### Precision Selection

SCL-Core supports multiple precision modes:

```cpp
// In config.hpp or via compile flags
#define SCL_PRECISION 0  // float32 (default)
#define SCL_PRECISION 1  // float64
#define SCL_PRECISION 2  // float16
```

**CMake Configuration**:
```cmake
# Build both f32 and f64 versions (default)
# Or specify precision:
target_compile_definitions(scl_core_f32 PRIVATE SCL_PRECISION=0)
```

### Threading Backend

```cpp
// Auto-selected based on platform
// Or explicitly set:
#define SCL_BACKEND_OPENMP   // OpenMP (default on Linux/Windows)
#define SCL_BACKEND_TBB      // Intel TBB
#define SCL_BACKEND_BS       // BS::thread_pool (default on macOS)
#define SCL_BACKEND_SERIAL   // Serial execution
```

**CMake Configuration**:
```cmake
set(SCL_THREADING_BACKEND "OPENMP" CACHE STRING "Threading backend")
```

## Project Structure

### Include Headers

```cpp
// Core types
#include "scl/core/type.hpp"        // Real, Index, Array
#include "scl/core/sparse.hpp"      // Sparse matrices
#include "scl/core/memory.hpp"      // Memory management
#include "scl/core/simd.hpp"        // SIMD operations
#include "scl/core/error.hpp"       // Error handling

// Threading
#include "scl/threading/parallel_for.hpp"

// Kernels
#include "scl/kernel/normalize.hpp"
#include "scl/kernel/neighbors.hpp"
// ... other kernels ...
```

### Namespace Organization

```cpp
namespace scl {
    // Core types
    using Real = ...;
    using Index = ...;
    template <typename T> struct Array;
    template <typename T, bool IsCSR> struct Sparse;
    
    namespace kernel {
        namespace normalize { ... }
        namespace neighbors { ... }
        // ... other kernels ...
    }
    
    namespace threading {
        template <typename Func>
        void parallel_for(size_t start, size_t end, Func&& func);
    }
    
    namespace memory {
        template <typename T>
        auto aligned_alloc(Size count, std::size_t alignment = DEFAULT_ALIGNMENT)
            -> std::unique_ptr<T[], AlignedDeleter<T>>;
    }
}
```

## Common Patterns

### Working with Sparse Matrices

```cpp
#include "scl/core/sparse.hpp"

// Create matrix
auto matrix = CSR::create(rows, cols, nnz);

// Access rows
for (Index i = 0; i < matrix.rows(); ++i) {
    auto indices = matrix.row_indices(i);
    auto values = matrix.row_values(i);
    Index len = matrix.row_length(i);
    
    for (Index k = 0; k < len; ++k) {
        Index col = indices[k];
        Real val = values[k];
        // Process element
    }
}
```

### Using Array Views

```cpp
#include "scl/core/type.hpp"

// Create array view
Real* data = ...;
Size n = 1000;
Array<Real> view = {data, n};

// Access elements
for (Size i = 0; i < view.size(); ++i) {
    view[i] *= 2.0;
}

// Iterate
for (Real& val : view) {
    val += 1.0;
}
```

### Parallel Processing

```cpp
#include "scl/threading/parallel_for.hpp"

// Parallel loop
threading::parallel_for(Size(0), n, [&](size_t i) {
    process(data[i]);
});

// With thread rank
threading::parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    thread_local_data[thread_rank].process(data[i]);
});
```

### Error Handling

```cpp
#include "scl/core/error.hpp"

try {
    auto matrix = CSR::create(rows, cols, nnz);
    kernel::normalize::normalize_rows_inplace(matrix, target_sum);
} catch (const scl::ValueError& e) {
    std::cerr << "Value error: " << e.what() << std::endl;
} catch (const scl::Exception& e) {
    std::cerr << "SCL error: " << e.what() << std::endl;
}
```

## Next Steps

- **Learn Core Types**: Read [Core Types](./core/types.md)
- **Understand Sparse Matrices**: Read [Sparse Matrices](./core/sparse.md)
- **Explore Kernels**: Browse [Kernels Overview](./kernels/overview.md)
- **Threading Guide**: Read [Threading](./threading.md)

## Troubleshooting

### Compilation Errors

**Error**: `'Real' is not a type`
- **Solution**: Include `scl/core/type.hpp`

**Error**: `'CSR' is not a type`
- **Solution**: Include `scl/core/sparse.hpp`

**Error**: `'parallel_for' is not a member of 'scl::threading'`
- **Solution**: Include `scl/threading/parallel_for.hpp`

### Runtime Errors

**Error**: `Dimension mismatch`
- **Solution**: Check matrix dimensions before operations

**Error**: `Out of memory`
- **Solution**: Reduce matrix size or use memory-mapped arrays

### Performance Issues

**Slow performance**:
- Ensure you're using Release build (`-DCMAKE_BUILD_TYPE=Release`)
- Check that SIMD is enabled (not `SCL_ONLY_SCALAR`)
- Verify parallelization is working (check thread count)

## Related Documentation

- [Core Types](./core/types.md) - Type system
- [Sparse Matrices](./core/sparse.md) - Sparse matrix operations
- [Threading](./threading.md) - Parallelization
- [Kernels](./kernels/overview.md) - Computational kernels

