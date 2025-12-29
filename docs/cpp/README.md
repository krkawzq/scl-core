---
title: C++ Documentation Complete
description: Complete C++ developer documentation for SCL-Core
---

# C++ Developer Documentation

Complete documentation for C++ developers using SCL-Core.

## Documentation Structure

### Getting Started
- **[Getting Started](./getting-started.md)** - Installation, build, and first steps
- **[Index](./index.md)** - Main documentation index

### Core Modules
- **[Types](./core/types.md)** - Type system (Real, Index, Array, Sparse)
- **[Sparse Matrices](./core/sparse.md)** - Sparse matrix operations
- **[Memory Management](./core/memory.md)** - Aligned allocation and RAII
- **[SIMD](./core/simd.md)** - SIMD and vectorization

### System Modules
- **[Threading](./threading.md)** - Parallel execution backends
- **[Error Handling](./error-handling.md)** - Exception system

### Computational Kernels
- **[Kernels Overview](./kernels/overview.md)** - General kernel usage
- **[Kernels Index](./kernels/README.md)** - Complete kernel reference

#### Data Processing
- **[Normalization](./kernels/normalize.md)** - Normalization operations
- **[Scaling](./kernels/scale.md)** - Scaling and standardization
- **[Logarithmic Transforms](./kernels/log1p.md)** - Log1p, log2p1, expm1

#### Neighbor Search
- **[Neighbors](./kernels/neighbors.md)** - K-nearest neighbors
- **[Spatial Analysis](./kernels/spatial.md)** - Spatial statistics

#### Graph Algorithms
- **[Louvain](./kernels/louvain.md)** - Louvain clustering

#### Feature Selection
- **[HVG](./kernels/hvg.md)** - Highly variable genes

#### Dimensionality Reduction
- **[Projection](./kernels/projection.md)** - Random projection

#### Statistical Analysis
- **[Statistics](./kernels/statistics.md)** - Statistical tests

## Quick Start

### Basic Usage

```cpp
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/kernel/normalize.hpp"
#include "scl/threading/parallel_for.hpp"

using namespace scl;

// Create sparse matrix
auto matrix = CSR::create(1000, 2000, 10000);

// Normalize
kernel::normalize::normalize_rows_inplace(matrix, 1e4);
kernel::log1p::log1p_inplace(matrix);

// Parallel processing
threading::parallel_for(0, matrix.rows(), [&](size_t i) {
    // Process row i
});
```

## Documentation Status

### ✅ Completed

**Core Documentation**:
- ✅ Main index and overview
- ✅ Getting started guide
- ✅ Core types system
- ✅ Sparse matrices
- ✅ Memory management
- ✅ SIMD operations
- ✅ Threading and parallelization
- ✅ Error handling

**Kernel Documentation**:
- ✅ Kernels overview
- ✅ Normalization
- ✅ Logarithmic transforms
- ✅ Neighbor search
- ✅ Spatial analysis
- ✅ Scaling operations
- ✅ Louvain clustering
- ✅ Highly variable genes
- ✅ Random projection
- ✅ Statistical tests
- ✅ Kernels index

### 📝 Additional Kernels (Can be added as needed)

The following kernels are available but not yet documented in detail:
- Leiden algorithm
- Connected components
- Centrality measures
- Marker detection
- Diffusion maps
- Pseudotime inference
- And 50+ more modules

See [Kernels Index](./kernels/README.md) for complete list.

## Key Features Documented

### Zero-Overhead Abstractions
All high-level APIs compile to optimal machine code with no runtime cost.

### SIMD Acceleration
Built-in support via Google Highway library for vectorized operations.

### Parallel by Default
Automatic parallelization with optimal work distribution.

### Memory Efficient
Advanced sparse matrix infrastructure with block allocation.

### Modern C++20
Uses concepts, `std::span`, `constexpr`, and other modern features.

## Navigation

- **New to SCL-Core?** Start with [Getting Started](./getting-started.md)
- **Using the library?** Read [Core Types](./core/types.md) and [Kernels Overview](./kernels/overview.md)
- **Need specific functionality?** Browse [Kernels Index](./kernels/README.md)
- **Contributing?** See project contributing guidelines

## Related Resources

- **GitHub Repository**: [scl-core](https://github.com/krkawzq/scl-core)
- **Issue Tracker**: Report bugs and request features
- **Discussions**: Ask questions and share ideas

---

**Documentation Version**: 0.4.0  
**Last Updated**: 2024

