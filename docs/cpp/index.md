---
title: C++ Developer Guide
description: Complete guide for C++ developers using SCL-Core
---

# C++ Developer Guide

Welcome to the SCL-Core C++ Developer Guide. This documentation provides comprehensive information for developers who want to use, extend, or contribute to SCL-Core.

## Overview

SCL-Core is a high-performance biological operator library built with **C++20**. It provides zero-overhead abstractions, SIMD-accelerated kernels, and parallel-by-default operations for maximum performance in computational biology applications.

### Key Features

- **Zero-Overhead Abstractions**: All high-level APIs compile to optimal machine code
- **SIMD Acceleration**: Built-in support via Google Highway library
- **Parallel by Default**: Automatic parallelization with optimal work distribution
- **Memory Efficient**: Advanced sparse matrix infrastructure with block allocation
- **Modern C++20**: Uses concepts, `std::span`, `constexpr`, and other modern features
- **Stable C-ABI**: Clean C interface for Python and other language bindings

## Quick Start

### Basic Usage

```cpp
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/kernel/normalize.hpp"
#include "scl/threading/parallel_for.hpp"

using namespace scl;

// Create a sparse matrix
auto matrix = CSR::create(1000, 2000, 10000);

// Normalize rows in-place
kernel::normalize::normalize_rows_inplace(matrix, 1e4);

// Parallel processing
threading::parallel_for(Size(0), 1000, [&](size_t i) {
    // Process row i
});
```

### Core Types

```cpp
// Floating-point type (configurable: float32/float64/float16)
Real value = 3.14;

// Index type (configurable: int16/int32/int64)
Index idx = 42;

// Zero-overhead array view
Array<Real> data = {ptr, size};

// Sparse matrix (CSR or CSC)
CSR matrix = CSR::create(rows, cols, nnz);
```

## Documentation Structure

### [Getting Started](./getting-started/)
- Installation and build instructions
- Development environment setup
- First steps with the library

### [Core Types](./core/)
- **Type System**: `Real`, `Index`, `Array<T>`, `Size`
- **Sparse Matrices**: `Sparse<T, IsCSR>`, CSR/CSC formats
- **Memory Management**: Aligned allocation, RAII wrappers
- **SIMD**: Highway-based vectorization

### [Threading](./threading/)
- Parallel execution backends (OpenMP, TBB, BS::thread_pool)
- `parallel_for` interface
- Thread safety guidelines

### [Error Handling](./error-handling/)
- Exception hierarchy
- Error codes
- Validation macros

### [Kernels](./kernels/)
- Normalization operations
- Neighbor search
- Statistical tests
- Spatial analysis
- And 50+ more algorithm modules

### [Memory Management](./memory/)
- Aligned allocation
- Registry system
- Memory-mapped arrays (experimental)

## Design Principles

### 1. Zero-Overhead Abstraction

All abstractions compile down to optimal machine code with no runtime cost:

```cpp
// High-level API
Array<Real> view = {data, n};
Real sum = vectorize::sum(view);

// Compiles to tight SIMD loops
```

### 2. Data-Oriented Design

Optimize for cache locality and memory bandwidth:

```cpp
// Block-allocated sparse matrices
auto matrix = CSR::create(rows, cols, nnz, BlockStrategy::adaptive());

// Cache-friendly access patterns
for (Index i = 0; i < n_rows; ++i) {
    auto row = matrix.row_values(i);
    // Process contiguous row data
}
```

### 3. Explicit Resource Management

No hidden allocations or implicit costs:

```cpp
// Explicit aligned allocation
auto buffer = memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

// RAII wrapper
AlignedBuffer<Real> buf(n);
Array<Real> view = buf.array();
```

### 4. Compile-Time Polymorphism

Templates over virtual functions for maximum performance:

```cpp
template <typename T, bool IsCSR>
void process_matrix(const Sparse<T, IsCSR>& matrix) {
    // Compile-time dispatch, zero overhead
}
```

## Module Organization

```
scl/
├── config.hpp          # Configuration (precision, threading backend)
├── version.hpp         # Version information
├── core/               # Core infrastructure
│   ├── type.hpp        # Type system (Real, Index, Array, Sparse)
│   ├── sparse.hpp      # Sparse matrix implementation
│   ├── memory.hpp      # Memory management
│   ├── simd.hpp        # SIMD abstraction (Highway)
│   ├── error.hpp       # Exception system
│   ├── macros.hpp      # Compiler abstractions
│   └── ...
├── kernel/             # Computational kernels (70+ modules)
│   ├── normalize.hpp  # Normalization operations
│   ├── neighbors.hpp  # Neighbor search
│   ├── spatial.hpp    # Spatial analysis
│   └── ...
├── threading/          # Parallel processing
│   ├── parallel_for.hpp
│   └── scheduler.hpp
├── math/               # Mathematical functions
├── mmap/               # Memory-mapped arrays (experimental)
└── binding/            # C-ABI interface
```

## Configuration

SCL-Core supports compile-time configuration:

### Precision Control

```cpp
// In config.hpp or via compile flags
#define SCL_PRECISION 0  // 0=float32, 1=float64, 2=float16
#define SCL_INDEX_PRECISION 2  // 0=int16, 1=int32, 2=int64
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

## Performance Tips

### 1. Use Appropriate Data Types

```cpp
// For small datasets (< 32K elements)
#define SCL_INDEX_PRECISION 0  // int16 saves memory

// For large datasets
#define SCL_INDEX_PRECISION 2  // int64 (default, NumPy-compatible)
```

### 2. Choose Correct Sparse Format

```cpp
// Row-based operations → CSR
CSR matrix = CSR::create(rows, cols, nnz);

// Column-based operations → CSC
CSC matrix = CSC::create(rows, cols, nnz);
```

### 3. Enable SIMD

```cpp
// SIMD is enabled by default
// Disable only for debugging:
#define SCL_ONLY_SCALAR  // Force scalar execution
```

### 4. Parallel Thresholds

Most kernels automatically parallelize when data size exceeds thresholds:

```cpp
// Typically 256-500 elements
// Adjust in kernel config namespaces if needed
```

## Building from Source

```bash
# Clone repository
git clone https://github.com/krkawzq/scl-core.git
cd scl-core

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Install
cmake --install build
```

## Next Steps

- **New to SCL-Core?** Start with [Getting Started](./getting-started/)
- **Using the library?** Read [Core Types](./core/) and [Kernels](./kernels/)
- **Contributing?** See [Contributing Guide](./contributing/)
- **Need API reference?** Browse [API Documentation](./api/)

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Documentation**: Browse this guide for detailed information

---

::: tip Version
SCL-Core version 0.4.0 - Built with C++20
:::
