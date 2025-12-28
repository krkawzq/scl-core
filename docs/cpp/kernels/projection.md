# projection.hpp

> scl/kernel/projection.hpp · Sparse random projection kernels for dimensionality reduction

## Overview

This file provides high-performance sparse random projection kernels for dimensionality reduction with Johnson-Lindenstrauss distance preservation guarantees.

This file provides:
- Gaussian random projection (dense, highest accuracy)
- Achlioptas projection (sparse ternary, 3x faster)
- Very sparse projection (best for high-dimensional data)
- Count-Sketch projection (O(nnz) time, hash-based)
- On-the-fly methods (memory-efficient, no matrix storage)
- Johnson-Lindenstrauss dimension computation

**Header**: `#include "scl/kernel/projection.hpp"`

---

## Main APIs

### project_with_matrix

::: source_code file="scl/kernel/projection.hpp" symbol="project_with_matrix" collapsed
:::

**Algorithm Description**

Project sparse matrix X using pre-computed projection matrix R: `Y = X * R`

1. **Parallel Processing**: Process each row of X in parallel:
   - Each thread handles independent rows
   - No synchronization needed

2. **Row Projection**: For each row i:
   - Initialize output row to zero
   - For each non-zero element (j, v) in row i:
     - Load projection row R[j, :]
     - Accumulate: `output_row += v * R[j, :]`
   - Use SIMD FMA (fused multiply-add) for large output_dim

3. **SIMD Optimization**: 
   - For output_dim >= 64: 4-way unrolled SIMD accumulation
   - Prefetch projection rows for long sparse rows
   - Cache-efficient access pattern

4. **Output**: Store projected data in dense buffer:
   - `output[i*k ... (i+1)*k-1]` contains projected row i

**Edge Cases**

- **Empty rows**: Rows with no non-zeros get zero output
- **Very sparse rows**: Handled efficiently with minimal overhead
- **Large output_dim**: SIMD path provides significant speedup
- **Small output_dim**: Scalar path avoids SIMD overhead

**Data Guarantees (Preconditions)**

- Matrix must be CSR format (IsCSR = true)
- `proj.input_dim == matrix.cols()`
- `output.len >= matrix.rows() * proj.output_dim`
- Output buffer must be pre-allocated
- Projection matrix must be valid

**Complexity Analysis**

- **Time**: O(nnz * output_dim) for matrix multiplication
  - Each non-zero contributes to output_dim elements
  - Parallelized over rows
  - SIMD reduces constant factor
- **Space**: O(1) auxiliary space per thread

**Example**

```cpp
#include "scl/kernel/projection.hpp"

scl::Sparse<Real, true> matrix = /* ... */;  // [n_rows x n_cols]

// Create Gaussian projection matrix
auto proj = scl::kernel::projection::create_gaussian_projection<Real>(
    n_cols,      // input_dim
    100,         // output_dim
    42           // seed
);

// Allocate output buffer
scl::Array<Real> output(n_rows * proj.output_dim);

// Project matrix
scl::kernel::projection::project_with_matrix(matrix, proj, output);

// output[i*100 ... (i+1)*100-1] contains projected row i
```

---

### project

::: source_code file="scl/kernel/projection.hpp" symbol="project" collapsed
:::

**Algorithm Description**

Unified interface for sparse random projection with automatic method selection:

1. **Method Selection**: Based on ProjectionType:
   - **Gaussian**: Uses on-the-fly Gaussian generation
   - **Achlioptas**: Uses on-the-fly ternary generation
   - **Sparse**: Uses on-the-fly sparse with density = max(1/sqrt(cols), 0.01)
   - **CountSketch**: Uses hash-based projection

2. **On-the-Fly Generation**: For each non-zero element:
   - Generate random projection values on-demand
   - No explicit projection matrix storage
   - Deterministic given same seed

3. **Projection**: Apply projection to sparse matrix:
   - Same algorithm as project_with_matrix
   - But generates projection values dynamically

4. **Output**: Store projected data in dense buffer

**Edge Cases**

- **Same as project_with_matrix**: Handles all edge cases similarly
- **Memory efficient**: No projection matrix storage needed
- **Deterministic**: Same seed produces same results

**Data Guarantees (Preconditions)**

- Matrix must be CSR format
- `output.len >= matrix.rows() * output_dim`
- Output buffer must be pre-allocated

**Complexity Analysis**

- **Time**: O(nnz * output_dim) with higher constant than pre-computed
  - Random generation overhead
  - Still parallelized and SIMD-optimized
- **Space**: O(1) auxiliary (no projection matrix storage)

**Example**

```cpp
scl::Sparse<Real, true> matrix = /* ... */;
scl::Array<Real> output(n_rows * 100);

// Use sparse projection (memory efficient)
scl::kernel::projection::project(
    matrix,
    100,                              // output_dim
    output,
    scl::kernel::projection::ProjectionType::Sparse,
    42                                // seed
);
```

---

### project_countsketch

::: source_code file="scl/kernel/projection.hpp" symbol="project_countsketch" collapsed
:::

**Algorithm Description**

Count-Sketch projection using hash-based bucketing and sign flips:

1. **Hash Functions**: For each feature j:
   - Hash to bucket: `h(j) = hash(j) % output_dim`
   - Generate sign: `s(j) = ±1` based on hash

2. **Projection**: For each non-zero element (i, j, v):
   - Bucket = h(j)
   - Sign = s(j)
   - Accumulate: `output[i, bucket] += sign * v`

3. **Output**: Store projected data:
   - Each feature contributes to exactly one bucket per row
   - O(nnz) time complexity (not O(nnz * k))

**Edge Cases**

- **Hash collisions**: Multiple features map to same bucket (expected)
- **Collision effects**: Reduce accuracy but maintain unbiased property
- **Small output_dim**: More collisions, lower accuracy
- **Large output_dim**: Fewer collisions, better accuracy

**Data Guarantees (Preconditions)**

- Matrix must be CSR format
- `output.len >= matrix.rows() * output_dim`
- Output buffer must be pre-allocated

**Complexity Analysis**

- **Time**: O(nnz) - linear in number of non-zeros
  - Each non-zero contributes to exactly one output element
  - Much faster than O(nnz * k) methods
- **Space**: O(1) auxiliary space

**Example**

```cpp
scl::Sparse<Real, true> matrix = /* ... */;
scl::Array<Real> output(n_rows * 100);

// Count-Sketch projection (O(nnz) time)
scl::kernel::projection::project_countsketch(
    matrix,
    100,    // output_dim (number of buckets)
    output,
    42      // seed
);
```

---

### create_gaussian_projection

::: source_code file="scl/kernel/projection.hpp" symbol="create_gaussian_projection" collapsed
:::

**Algorithm Description**

Create a dense Gaussian random projection matrix:

1. **Random Generation**: For each entry (i, j):
   - Generate Gaussian sample: `X ~ N(0, 1)`
   - Scale by: `R[i,j] = X / sqrt(output_dim)`
   - Uses Box-Muller transform for Gaussian samples

2. **Storage**: Store in row-major order:
   - `data[i * output_dim + j] = R[i,j]`
   - Cache-efficient for projection operation

3. **Output**: Return ProjectionMatrix struct:
   - Contains data pointer, dimensions, ownership flag

**Edge Cases**

- **Zero dimensions**: Returns invalid matrix if dimensions <= 0
- **Large dimensions**: Requires O(input_dim * output_dim) memory
- **Deterministic**: Same seed produces same matrix

**Data Guarantees (Preconditions)**

- `input_dim > 0`
- `output_dim > 0`
- Matrix will be allocated internally

**Complexity Analysis**

- **Time**: O(input_dim * output_dim) for generation
  - Each entry requires random number generation
- **Space**: O(input_dim * output_dim) for storage

**Example**

```cpp
// Create Gaussian projection matrix
auto proj = scl::kernel::projection::create_gaussian_projection<Real>(
    10000,   // input_dim
    100,     // output_dim
    42       // seed
);

// Use for projection
scl::kernel::projection::project_with_matrix(matrix, proj, output);
```

---

### create_achlioptas_projection

::: source_code file="scl/kernel/projection.hpp" symbol="create_achlioptas_projection" collapsed
:::

**Algorithm Description**

Create a sparse Achlioptas random projection matrix:

1. **Ternary Generation**: For each entry (i, j):
   - Generate random value: X ~ {+1, 0, -1}
   - Probabilities: P(+1) = 1/6, P(0) = 2/3, P(-1) = 1/6
   - Scale by: `R[i,j] = sqrt(3/output_dim) * X`

2. **Sparsity**: 2/3 of entries are zero:
   - Only 1/3 of entries stored (sparse representation)
   - 3x faster to compute than Gaussian

3. **Output**: Return ProjectionMatrix struct

**Edge Cases**

- **Same as Gaussian**: Handles dimension edge cases
- **Sparse storage**: More memory efficient than Gaussian
- **Same guarantees**: Theoretical guarantees same as Gaussian

**Data Guarantees (Preconditions)**

- `input_dim > 0`
- `output_dim > 0`

**Complexity Analysis**

- **Time**: O(input_dim * output_dim / 3) for generation (sparse)
- **Space**: O(input_dim * output_dim / 3) for storage (sparse)

**Example**

```cpp
// Create Achlioptas projection (3x faster than Gaussian)
auto proj = scl::kernel::projection::create_achlioptas_projection<Real>(
    10000,   // input_dim
    100,     // output_dim
    42       // seed
);
```

---

### create_sparse_projection

::: source_code file="scl/kernel/projection.hpp" symbol="create_sparse_projection" collapsed
:::

**Algorithm Description**

Create a very sparse random projection matrix with custom density:

1. **Sparse Generation**: For each entry (i, j):
   - Generate with probability = density
   - If selected: X ~ {+1, -1} with equal probability
   - Scale by: `R[i,j] = sqrt(1/(output_dim * density)) * X`

2. **Density Control**: 
   - Typical density = 1/sqrt(input_dim)
   - For high-dimensional data, most entries are zero
   - sqrt(input_dim)x faster than Gaussian

3. **Output**: Return ProjectionMatrix struct

**Edge Cases**

- **Very low density**: If density too small, may have insufficient non-zeros
- **High-dimensional**: Best for input_dim > 10000
- **Same guarantees**: Distance preservation guarantees maintained

**Data Guarantees (Preconditions)**

- `input_dim > 0`
- `output_dim > 0`
- `density` in (0, 1]

**Complexity Analysis**

- **Time**: O(input_dim * output_dim * density) for generation
- **Space**: O(input_dim * output_dim * density) for storage

**Example**

```cpp
// Create very sparse projection for high-dimensional data
Real density = 1.0 / std::sqrt(input_dim);
auto proj = scl::kernel::projection::create_sparse_projection<Real>(
    input_dim,
    output_dim,
    density,
    42
);
```

---

## Utility Functions

### compute_jl_dimension

Compute minimum target dimension for Johnson-Lindenstrauss guarantee.

::: source_code file="scl/kernel/projection.hpp" symbol="compute_jl_dimension" collapsed
:::

**Algorithm Description**

Computes minimum dimension k for (1 ± epsilon) distance preservation:

- Formula: `k >= 4 * ln(n) / (epsilon^2/2 - epsilon^3/3)`
- Guarantees: With probability >= 1 - 1/n^2, distances preserved within (1 ± epsilon)

**Complexity**

- Time: O(1)
- Space: O(1)

**Example**

```cpp
Size n_samples = 10000;
Real epsilon = 0.1;

Size min_dim = scl::kernel::projection::compute_jl_dimension(
    n_samples,
    epsilon
);

// min_dim is minimum output_dim for distance preservation
```

---

### project_gaussian_otf

On-the-fly Gaussian projection (memory efficient).

::: source_code file="scl/kernel/projection.hpp" symbol="project_gaussian_otf" collapsed
:::

**Complexity**

- Time: O(nnz * output_dim) with higher constant
- Space: O(1) auxiliary

---

### project_achlioptas_otf

On-the-fly Achlioptas projection.

::: source_code file="scl/kernel/projection.hpp" symbol="project_achlioptas_otf" collapsed
:::

**Complexity**

- Time: O(nnz * output_dim) with lower constant than Gaussian
- Space: O(1) auxiliary

---

### project_sparse_otf

On-the-fly very sparse projection.

::: source_code file="scl/kernel/projection.hpp" symbol="project_sparse_otf" collapsed
:::

**Complexity**

- Time: O(nnz * output_dim * density)
- Space: O(1) auxiliary

---

## Configuration

Default parameters in `scl::kernel::projection::config`:

- `SIMD_THRESHOLD = 64`: Minimum output_dim for SIMD accumulation
- `PREFETCH_DISTANCE = 16`: Cache line prefetch distance
- `SMALL_OUTPUT_DIM = 32`: Threshold for scalar path
- `DEFAULT_EPSILON = 0.1`: Default distance distortion for JL dimension

---

## Performance Notes

### Method Selection

- **Gaussian**: Highest accuracy, use for small-medium datasets
- **Achlioptas**: Good balance, 3x faster than Gaussian
- **Sparse**: Best for high-dimensional data (d > 10000)
- **CountSketch**: Best for streaming/online applications

### Memory vs Speed Trade-off

- **Pre-computed matrix**: Faster but requires O(d*k) memory
- **On-the-fly**: Slower but O(1) memory
- Choose based on available memory and projection reuse

---

## See Also

- [Sparse Matrices](../core/sparse)
- [SIMD Operations](../core/simd)
- [Dimensionality Reduction](../math)
