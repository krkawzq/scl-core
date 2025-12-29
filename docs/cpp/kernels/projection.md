---
title: Dimensionality Reduction
description: Random projection and dimensionality reduction methods
---

# Dimensionality Reduction

The `projection` kernel provides high-performance random projection methods for dimensionality reduction, optimized with SIMD and efficient sparse projections.

## Overview

Random projection is a fast dimensionality reduction technique that:
- Preserves pairwise distances (Johnson-Lindenstrauss lemma)
- Works with sparse matrices efficiently
- Supports multiple projection types
- Much faster than PCA for large datasets

## Projection Types

### Gaussian Projection

Dense Gaussian random projection with N(0, 1/k) distribution.

```cpp
ProjectionType::Gaussian
```

**Use Case**: High accuracy, dense output

### Achlioptas Projection

Sparse projection with values {+1, 0, -1} with probabilities {1/6, 2/3, 1/6}.

```cpp
ProjectionType::Achlioptas
```

**Use Case**: Faster computation, sparse output

### Sparse Projection

Very sparse projection with density 1/√d.

```cpp
ProjectionType::Sparse
```

**Use Case**: Maximum speed, very sparse output

### CountSketch

Sign flips with hash-based indexing.

```cpp
ProjectionType::CountSketch
```

**Use Case**: Deterministic, hash-based

### Feature Hash

Multiple hash functions for better accuracy.

```cpp
ProjectionType::FeatureHash
```

**Use Case**: Better accuracy than CountSketch

## Functions

### `random_projection`

Project sparse matrix to lower dimension.

```cpp
template <typename T, bool IsCSR>
void random_projection(
    const Sparse<T, IsCSR>& matrix,
    Array<Real> output,
    Index n_components,
    ProjectionType type = ProjectionType::Achlioptas,
    uint64_t seed = 0
);
```

**Parameters**:
- `matrix` [in]: Input sparse matrix (cells × genes)
- `output` [out]: Projected matrix (cells × n_components)
- `n_components` [in]: Target dimension
- `type` [in]: Projection type (default: Achlioptas)
- `seed` [in]: Random seed (default: 0)

**Example**:
```cpp
#include "scl/kernel/projection.hpp"

// Project to 50 dimensions
constexpr Index N_COMPONENTS = 50;
auto projected = memory::aligned_alloc<Real>(
    matrix.rows() * N_COMPONENTS
);
Array<Real> proj_view = {
    projected.get(),
    static_cast<Size>(matrix.rows() * N_COMPONENTS)
};

kernel::projection::random_projection(
    matrix, proj_view, N_COMPONENTS,
    ProjectionType::Achlioptas,
    seed=42
);
```

**Complexity**: O(nnz * n_components) time, O(n * n_components) space

## Performance Optimizations

### SIMD-Accelerated Random Generation

```cpp
// Generate 4 random numbers in parallel
Xoshiro256 rng(seed);
uint64_t randoms[4];
rng.next4(randoms);
```

### Block-Wise Processing

```cpp
// Process in blocks for cache efficiency
constexpr Size BLOCK_SIZE = 256;
for (Size block = 0; block < n_rows; block += BLOCK_SIZE) {
    process_block(block, std::min(BLOCK_SIZE, n_rows - block));
}
```

### Sparse Projection Structures

For sparse projections, precompute projection structure:

```cpp
// Precompute sparse projection indices
struct SparseProjection {
    Index* indices;
    Real* values;
    Size nnz_per_col;
};
```

### Multi-Accumulator FMA

```cpp
// Fused multiply-add with multiple accumulators
auto acc0 = s::Zero(d);
auto acc1 = s::Zero(d);
// ... accumulate in parallel
auto result = s::Add(acc0, acc1);
```

## Common Patterns

### Fast Dimensionality Reduction

```cpp
void fast_dimension_reduction(
    const CSR& matrix,
    Index n_components,
    Array<Real>& output
) {
    // Use sparse projection for speed
    kernel::projection::random_projection(
        matrix, output, n_components,
        ProjectionType::Sparse,  // Fastest
        seed=42
    );
}
```

### High-Accuracy Projection

```cpp
void accurate_projection(
    const CSR& matrix,
    Index n_components,
    Array<Real>& output
) {
    // Use Gaussian for accuracy
    kernel::projection::random_projection(
        matrix, output, n_components,
        ProjectionType::Gaussian,  // Most accurate
        seed=42
    );
}
```

### Deterministic Projection

```cpp
void deterministic_projection(
    const CSR& matrix,
    Index n_components,
    Array<Real>& output
) {
    // Use CountSketch for reproducibility
    kernel::projection::random_projection(
        matrix, output, n_components,
        ProjectionType::CountSketch,  // Deterministic
        seed=42
    );
}
```

## Configuration

```cpp
namespace scl::kernel::projection::config {
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 8;
    constexpr Size BLOCK_SIZE = 256;
    constexpr Size SMALL_OUTPUT_DIM = 64;
    constexpr Real DEFAULT_EPSILON = 0.1;
    constexpr Size MIN_PARALLEL_ROWS = 64;
    constexpr Size MAX_PRECOMPUTE_BYTES = 256 * 1024 * 1024;  // 256 MB
}
```

## Algorithm Selection

The implementation automatically selects optimal algorithm:

```cpp
// For small output dimensions: Use dense precomputation
if (n_components < SMALL_OUTPUT_DIM) {
    use_dense_projection();
} else {
    // For large dimensions: Use sparse/hash-based
    use_sparse_projection();
}
```

## Related Documentation

- [Kernels Overview](./overview.md) - General kernel usage
- [Sparse Matrices](../core/sparse.md) - Sparse matrix operations
- [SIMD](../core/simd.md) - SIMD operations
