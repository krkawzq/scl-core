# Projection

Sparse random projection kernels for dimensionality reduction with distance preservation guarantees.

## Overview

Projection provides:

- **Gaussian Projection** - Dense Gaussian random projection
- **Achlioptas Projection** - Sparse ternary projection (3x faster)
- **Sparse Projection** - Very sparse projection for high-dimensional data
- **Count-Sketch** - Hash-based projection (O(nnz) time)
- **On-the-Fly Methods** - Memory-efficient projection without storing matrix
- **Johnson-Lindenstrauss** - Dimension computation for distance preservation

## Projection Types

### ProjectionType Enum

```cpp
enum class ProjectionType {
    Gaussian,       // N(0, 1/k) entries - highest accuracy
    Achlioptas,     // {+1, 0, -1} with prob {1/6, 2/3, 1/6} - 3x faster
    Sparse,         // Density = 1/sqrt(d) - best for high-dim
    CountSketch     // Hash + sign - O(nnz) time
};
```

**Selection Guide:**
- **Gaussian**: Highest accuracy, small-medium datasets
- **Achlioptas**: Good balance, medium datasets
- **Sparse**: High-dimensional genomic/text data
- **CountSketch**: Streaming/online applications

## Pre-computed Matrix Projection

### create_gaussian_projection

Create a dense Gaussian random projection matrix:

```cpp
#include "scl/kernel/projection.hpp"

auto proj = scl::kernel::projection::create_gaussian_projection<Real>(
    input_dim,   // Original dimension d
    output_dim,  // Target dimension k
    42           // seed
);
```

**Parameters:**
- `input_dim`: Original dimension d (number of features)
- `output_dim`: Target dimension k (reduced features)
- `seed`: Random seed for reproducibility

**Returns:** `ProjectionMatrix<T>` with Gaussian entries scaled by 1/sqrt(k)

**Postconditions:**
- E[||Rx - Ry||^2] = ||x - y||^2 (unbiased)
- Deterministic given same seed

**Complexity:**
- Time: O(input_dim * output_dim) for generation
- Space: O(input_dim * output_dim)

**Use cases:**
- Highest quality projection
- When projection matrix can be stored
- Small to medium input dimensions

### create_achlioptas_projection

Create a sparse Achlioptas random projection matrix:

```cpp
auto proj = scl::kernel::projection::create_achlioptas_projection<Real>(
    input_dim,
    output_dim,
    42  // seed
);
```

**Advantages:**
- 3x faster to compute than Gaussian
- 2/3 of entries are zero
- Same theoretical guarantees as Gaussian

**Reference:**
Achlioptas, D. (2003). Database-friendly random projections.

**Use cases:**
- Good balance of speed and quality
- Medium-sized datasets
- When storage is a concern

### create_sparse_projection

Create a very sparse random projection matrix:

```cpp
auto proj = scl::kernel::projection::create_sparse_projection<Real>(
    input_dim,
    output_dim,
    Real(1.0 / sqrt(input_dim)),  // density
    42  // seed
);
```

**Advantages:**
- sqrt(d)x faster for high-dimensional data
- (1-density) fraction of computation skipped
- Same distance preservation guarantees

**Reference:**
Li, P., Hastie, T. J., & Church, K. W. (2006). Very sparse random projections.

**Use cases:**
- Very high-dimensional data (d > 10000)
- Genomic/text data
- When speed is critical

### project_with_matrix

Project sparse matrix using pre-computed projection matrix:

```cpp
Sparse<Real, true> matrix = /* ... */;  // n x d
auto proj = scl::kernel::projection::create_gaussian_projection<Real>(
    matrix.cols(),  // d
    output_dim      // k
);
Array<Real> output(matrix.rows() * output_dim);  // PRE-ALLOCATED

scl::kernel::projection::project_with_matrix(
    matrix,
    proj,
    output
);
```

**Parameters:**
- `matrix`: CSR sparse matrix X, shape (n_rows x n_cols)
- `proj`: Pre-computed projection matrix R, shape (n_cols x output_dim)
- `output`: Dense output buffer Y, size = n_rows * output_dim, PRE-ALLOCATED

**Postconditions:**
- `output[i*k ... (i+1)*k-1]` contains projected row i
- Y = X * R

**Algorithm:**
Parallel over rows:
1. For each row i:
   - Initialize output_row to zero
   - For each non-zero (j, v) in row i:
     - output_row += v * R[j, :]
   - Use SIMD FMA for large output_dim

**Complexity:**
- Time: O(nnz * output_dim)
- Space: O(1) auxiliary

**Performance Notes:**
- SIMD 4-way unrolled accumulation for output_dim >= 64
- Prefetching projection rows for long sparse rows
- Best when projection matrix fits in cache

## On-the-Fly Projection

### project_gaussian_otf

Project sparse matrix using on-the-fly Gaussian random generation:

```cpp
Sparse<Real, true> matrix = /* ... */;
Array<Real> output(matrix.rows() * output_dim);

scl::kernel::projection::project_gaussian_otf(
    matrix,
    output_dim,
    output,
    42  // seed
);
```

**Advantages:**
- O(1) auxiliary memory (no projection matrix storage)
- Deterministic given same seed
- Good for very high-dimensional data

**Disadvantages:**
- Slower than pre-computed for repeated projections
- More random number generation overhead

**Complexity:**
- Time: O(nnz * output_dim) with higher constant
- Space: O(1) auxiliary

**Use cases:**
- Very high-dimensional data where storing matrix is impractical
- One-time projections
- Memory-constrained environments

### project_achlioptas_otf

On-the-fly Achlioptas projection (ternary: +1, 0, -1):

```cpp
scl::kernel::projection::project_achlioptas_otf(
    matrix,
    output_dim,
    output,
    42  // seed
);
```

**Advantages:**
- Faster than Gaussian OTF (simpler random generation)
- 2/3 of random values are zero (skip computation)

**Use cases:**
- Medium-dimensional data
- When memory is limited
- Faster alternative to Gaussian OTF

### project_sparse_otf

On-the-fly very sparse projection with custom density:

```cpp
scl::kernel::projection::project_sparse_otf(
    matrix,
    output_dim,
    output,
    Real(1.0 / sqrt(matrix.cols())),  // density
    42  // seed
);
```

**Advantages:**
- Best for very high-dimensional data (d > 10000)
- Most computation skipped due to sparsity

**Use cases:**
- Extremely high-dimensional data
- When speed is critical
- Genomic/text applications

## Count-Sketch Projection

### project_countsketch

Count-Sketch projection using hash-based bucketing and sign flips:

```cpp
scl::kernel::projection::project_countsketch(
    matrix,
    output_dim,  // Number of buckets
    output,
    42  // seed
);
```

**Advantages:**
- O(nnz) time (not O(nnz * k))
- Unbiased: E[Y^T Y] = X^T X
- Good for streaming/online learning

**Disadvantages:**
- Higher variance than Gaussian for small k
- Collision effects reduce accuracy

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary

**Reference:**
Charikar, M., Chen, K., & Farach-Colton, M. (2004). Finding frequent items in data streams.

**Use cases:**
- Streaming data
- Online learning
- When O(nnz * k) is too expensive

## High-Level Interface

### project

Unified interface for sparse random projection:

```cpp
Sparse<Real, true> matrix = /* ... */;
Array<Real> output(matrix.rows() * output_dim);

scl::kernel::projection::project(
    matrix,
    output_dim,
    output,
    ProjectionType::Sparse,  // type
    42  // seed
);
```

**Parameters:**
- `matrix`: CSR sparse matrix X
- `output_dim`: Target dimension k
- `output`: Dense output buffer [n * k]
- `type`: Projection type (default: Sparse)
- `seed`: Random seed (default: 42)

**Postconditions:**
- Distance preservation: (1-eps)||x-y|| <= ||Rx-Ry|| <= (1+eps)||x-y||
- With high probability

**Selection:**
- Sparse type uses density = max(1/sqrt(cols), 0.01)

**Use cases:**
- Simple interface for common use cases
- Automatic method selection
- Quick prototyping

## Utility Functions

### compute_jl_dimension

Compute minimum target dimension for Johnson-Lindenstrauss guarantee:

```cpp
Size k = scl::kernel::projection::compute_jl_dimension(
    n_samples,  // Number of data points
    Real(0.1)   // epsilon (distance distortion tolerance)
);
```

**Parameters:**
- `n_samples`: Number of data points
- `epsilon`: Maximum relative distance distortion (default: 0.1)

**Returns:** Minimum target dimension k for (1 +/- epsilon) distance preservation

**Guarantee:**
With probability >= 1 - 1/n^2:
(1-epsilon)||x-y||^2 <= ||Rx-Ry||^2 <= (1+epsilon)||x-y||^2
for all pairs (x, y)

**Typical Values:**
- n=1000, eps=0.1: k ~= 300
- n=10000, eps=0.1: k ~= 400
- n=1000, eps=0.5: k ~= 20

**Reference:**
Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space.

**Use cases:**
- Determine required dimension for distance preservation
- Quality control for projection
- Theoretical guarantees

## Configuration

Default parameters in `scl::kernel::projection::config`:

```cpp
namespace config {
    constexpr Size SIMD_THRESHOLD = 64;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size SMALL_OUTPUT_DIM = 32;
    constexpr Real DEFAULT_EPSILON = 0.1;
}
```

## Performance Considerations

### Memory Efficiency

- **Pre-computed**: O(d * k) storage, faster projection
- **On-the-fly**: O(1) storage, slower projection
- Choose based on input dimension and projection reuse

### SIMD Optimization

- 4-way unrolled accumulation for output_dim >= 64
- Scalar path for small output_dim (< 32)
- Prefetching for cache efficiency

## Best Practices

### 1. Choose Appropriate Method

```cpp
// For highest accuracy
auto proj = scl::kernel::projection::create_gaussian_projection<Real>(
    input_dim, output_dim
);

// For speed with good quality
auto proj = scl::kernel::projection::create_achlioptas_projection<Real>(
    input_dim, output_dim
);

// For very high-dimensional data
scl::kernel::projection::project_sparse_otf(
    matrix, output_dim, output, 1.0/sqrt(input_dim)
);
```

### 2. Compute JL Dimension

```cpp
// Determine required dimension
Size k = scl::kernel::projection::compute_jl_dimension(
    n_samples,
    0.1  // 10% distance distortion
);

// Use computed dimension
scl::kernel::projection::project(matrix, k, output);
```

### 3. Use Pre-computed for Repeated Projections

```cpp
// Build once
auto proj = scl::kernel::projection::create_gaussian_projection<Real>(
    input_dim, output_dim
);

// Reuse for multiple matrices
for (auto& matrix : matrices) {
    scl::kernel::projection::project_with_matrix(matrix, proj, output);
}
```

## Examples

### Complete Projection Pipeline

```cpp
// 1. Compute required dimension
Size k = scl::kernel::projection::compute_jl_dimension(
    n_samples, 0.1
);

// 2. Create projection matrix
auto proj = scl::kernel::projection::create_achlioptas_projection<Real>(
    input_dim, k
);

// 3. Project data
Sparse<Real, true> data = /* ... */;
Array<Real> projected(data.rows() * k);
scl::kernel::projection::project_with_matrix(data, proj, projected);

// 4. Use projected data for downstream analysis
// (e.g., clustering, classification)
```

---

::: tip Method Selection
Use Gaussian for highest accuracy, Achlioptas for balance, Sparse for high-dimensional data, and CountSketch for streaming applications.
:::

::: warning Memory
Pre-computed projection matrices require O(d * k) storage. Use on-the-fly methods for very high-dimensional data (d > 100000).
:::

