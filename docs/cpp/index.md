# C++ Developer Guide

Welcome to the SCL-Core C++ Developer Guide! This documentation is designed for developers who want to contribute to SCL-Core, extend its functionality, or integrate it into their own C++ projects.

## Overview

SCL-Core is a high-performance biological operator library built with modern C++17. It provides:

- **Zero-overhead abstractions** for maximum performance
- **SIMD-accelerated** compute kernels
- **Parallel-by-default** operations
- **Memory-efficient** sparse matrix infrastructure
- **Stable C-ABI** for language bindings

## Quick Navigation

### For Contributors

- [Getting Started](/cpp/getting-started/) - Set up your development environment
- [Building from Source](/cpp/getting-started/building) - Compile and test the library
- [Contributing Guide](/cpp/getting-started/contributing) - Code standards and workflow

### For Library Users

- [Architecture Overview](/cpp/architecture/) - Understand the design philosophy
- [Core Modules](/cpp/core/) - Types, sparse matrices, memory management
- [Threading](/cpp/threading/) - Parallel processing infrastructure
- [Kernels](/cpp/kernels/) - Computational operators
- [Memory-Mapped Arrays](/cpp/mmap/) - Out-of-core processing (Experimental)

### For Advanced Users

- [API Reference](/cpp/reference/) - Complete function reference
- [Design Principles](/cpp/architecture/design-principles) - Performance optimization strategies
- [Memory Model](/cpp/architecture/memory-model) - Registry and lifetime management

## Key Features

### Zero-Overhead Performance

All abstractions compile down to optimal machine code:

```cpp
// High-level API
scl::kernel::normalize::normalize_rows_inplace(matrix, NormMode::L2);

// Compiles to tight SIMD loops with no overhead
```

### SIMD Acceleration

Built-in SIMD support via Highway library:

```cpp
namespace s = scl::simd;
const s::Tag d;

auto v_sum = s::Zero(d);
for (size_t i = 0; i < n; i += s::lanes()) {
    auto v = s::Load(d, data + i);
    v_sum = s::Add(v_sum, v);
}
```

### Parallel by Default

Automatic parallelization with optimal work distribution:

```cpp
scl::threading::parallel_for(Size(0), n_rows, [&](size_t i) {
    // Process row i in parallel
    process_row(matrix, i);
});
```

### Memory Efficiency

Advanced sparse matrix infrastructure:

```cpp
// Discontiguous storage for flexible memory management
scl::Sparse<Real, true> matrix = 
    scl::Sparse<Real, true>::create(rows, cols, nnz_per_row);

// Registry-based lifetime management
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);
// Arrays are registered and tracked automatically
```

## Module Organization

```
scl/
├── core/           # Core types, sparse matrices, SIMD, memory
├── threading/      # Parallel processing infrastructure
├── kernel/         # Computational operators (80+ files, 400+ functions)
├── math/           # Statistical functions and regression
├── mmap/           # [Experimental] Memory-mapped arrays for out-of-core processing
└── io/             # I/O utilities
```

## Documentation Structure

This guide is organized into several sections:

### Getting Started
Learn how to set up your development environment, build the library, and contribute code.

### Architecture
Understand the design principles, module structure, and memory model that make SCL-Core fast.

### Core Modules
Deep dive into the fundamental building blocks: types, sparse matrices, SIMD, and memory management.

### Threading
Learn about the parallel processing infrastructure and how to write thread-safe code.

### Kernels
Explore the 400+ computational operators organized by functionality.

### Reference
Complete API reference extracted from source code documentation.

## Design Philosophy

SCL-Core follows these core principles:

1. **Zero-Overhead Abstraction** - No runtime cost for high-level APIs
2. **Data-Oriented Design** - Optimize for cache locality and memory bandwidth
3. **Explicit Resource Management** - No hidden allocations or implicit costs
4. **Compile-Time Polymorphism** - Templates over virtual functions
5. **Documentation as Code** - `.h` files contain comprehensive API docs

## Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/krkawzq/scl-core/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/krkawzq/scl-core/discussions)
- **Contributing**: See the [Contributing Guide](/cpp/getting-started/contributing)

## Next Steps

- **New to SCL-Core?** Start with [Getting Started](/cpp/getting-started/)
- **Want to contribute?** Read the [Contributing Guide](/cpp/getting-started/contributing)
- **Need API docs?** Browse the [Core Modules](/cpp/core/) or [Kernels](/cpp/kernels/)
- **Curious about design?** Explore the [Architecture](/cpp/architecture/)

---

::: tip Development Status
SCL-Core is actively developed and welcomes contributions! Check the [GitHub repository](https://github.com/krkawzq/scl-core) for the latest updates.
:::

