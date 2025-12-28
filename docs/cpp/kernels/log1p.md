# log1p.hpp

> scl/kernel/log1p.hpp · Logarithmic transform kernels with SIMD optimization

## Overview

This file provides efficient logarithmic transform operations for sparse matrices. All operations are SIMD-accelerated, parallelized over rows, and modify matrices in-place without memory allocation.

**Header**: `#include "scl/kernel/log1p.hpp"`

---

## Main APIs

### log1p_inplace

::: source_code file="scl/kernel/log1p.hpp" symbol="log1p_inplace" collapsed
:::

**Algorithm Description**

Apply log(1 + x) transform to all non-zero values in sparse matrix:

1. For each row in parallel:
   - Load values using 4-way SIMD unrolling with prefetch
   - Apply SIMD Log1p operation to value vectors
   - Store transformed values back
   - Handle tail elements with scalar operations

2. Uses numerical stable log1p implementation for accuracy near zero

3. Zero values remain zero (not stored in sparse format)

**Edge Cases**

- **Empty matrix**: Returns immediately without modification
- **Values < -1**: Results in NaN (log1p domain requirement)
- **Values = -1**: Results in -infinity
- **Very small values**: More accurate than log(1+x) for x near zero
- **Large values**: log1p(x) ≈ log(x) for large x

**Data Guarantees (Preconditions)**

- Matrix values must be >= -1 (log1p domain requirement)
- For expression data: values should be non-negative counts
- Matrix must be valid CSR or CSC format
- Matrix values must be mutable

**Complexity Analysis**

- **Time**: O(nnz) - process each non-zero element once
- **Space**: O(1) auxiliary - only SIMD registers and temporary variables

**Example**

```cpp
#include "scl/kernel/log1p.hpp"

// Load or create sparse matrix
Sparse<Real, true> expression = /* ... */;  // CSR format

// Apply log1p transform in-place
scl::kernel::log1p::log1p_inplace(expression);

// All non-zero values v are now log(1 + v)
// Matrix structure (indices, indptr) unchanged
// Ready for downstream analysis (PCA, clustering, etc.)

// Standard preprocessing pipeline
scl::kernel::normalize::normalize_total_inplace(expression, 1e4);
scl::kernel::log1p::log1p_inplace(expression);
// Now ready for PCA, clustering, etc.
```

---

### log2p1_inplace

::: source_code file="scl/kernel/log1p.hpp" symbol="log2p1_inplace" collapsed
:::

**Algorithm Description**

Apply log2(1 + x) transform to all non-zero values in sparse matrix:

1. For each row in parallel:
   - Load values using 4-way SIMD unrolling with prefetch
   - Apply SIMD Log1p operation
   - Multiply by INV_LN2 (1/ln(2)) to convert to base-2
   - Store transformed values
   - Handle tail elements with scalar operations

2. Computed as log(1+x) * (1/ln(2)) for efficiency

3. Base-2 logarithm is common in information theory applications

**Edge Cases**

- **Empty matrix**: Returns immediately
- **Values < -1**: Results in NaN
- **Values = -1**: Results in -infinity
- **Very small values**: More accurate than log2(1+x) for x near zero

**Data Guarantees (Preconditions)**

- Matrix values must be >= -1
- For expression data: values should be non-negative counts
- Matrix must be valid CSR or CSC format
- Matrix values must be mutable

**Complexity Analysis**

- **Time**: O(nnz) - process each non-zero element once
- **Space**: O(1) auxiliary - only SIMD registers

**Example**

```cpp
#include "scl/kernel/log1p.hpp"

Sparse<Real, true> matrix = /* ... */;

// Apply base-2 log transform
scl::kernel::log1p::log2p1_inplace(matrix);

// All non-zero values v are now log2(1 + v)
// Useful for information-theoretic measures (entropy, mutual information)
```

---

### expm1_inplace

::: source_code file="scl/kernel/log1p.hpp" symbol="expm1_inplace" collapsed
:::

**Algorithm Description**

Apply exp(x) - 1 transform to all non-zero values in sparse matrix:

1. For each row in parallel:
   - Load values using 4-way SIMD unrolling with prefetch
   - Apply SIMD Expm1 operation to value vectors
   - Store transformed values back
   - Handle tail elements with scalar operations

2. Uses numerically stable expm1 implementation for accuracy near zero

3. Inverse of log1p: expm1(log1p(x)) = x

**Edge Cases**

- **Empty matrix**: Returns immediately
- **Very large values**: May overflow to infinity
- **Very small values**: More accurate than exp(x)-1 for x near zero
- **Values = 0**: Results in 0 (exp(0) - 1 = 0)
- **Negative large values**: Results in -1 (exp(-inf) - 1 = -1)

**Data Guarantees (Preconditions)**

- Values should be in reasonable range to avoid overflow
- Typically used to reverse log1p transform
- Matrix must be valid CSR or CSC format
- Matrix values must be mutable

**Complexity Analysis**

- **Time**: O(nnz) - process each non-zero element once
- **Space**: O(1) auxiliary - only SIMD registers

**Example**

```cpp
#include "scl/kernel/log1p.hpp"

Sparse<Real, true> matrix = /* ... */;

// Apply log1p transform
scl::kernel::log1p::log1p_inplace(matrix);

// ... perform analysis on log-transformed data ...

// Reverse transform to original scale
scl::kernel::log1p::expm1_inplace(matrix);

// Matrix values are now back to original scale (approximately)
// expm1(log1p(x)) = x (exact for small x)
```

---

## Configuration

### Default Parameters

```cpp
namespace scl::kernel::log1p::config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr double INV_LN2 = 1.44269504088896340736;  // 1/ln(2)
    constexpr double LN2 = 0.6931471805599453;          // ln(2)
}
```

---

## Notes

**Numerical Stability**: log1p and expm1 are more accurate than log(1+x) and exp(x)-1 for small values near zero. This is critical for expression data where many values are small counts.

**Performance**: SIMD acceleration provides 4-8x speedup on modern CPUs. Parallelization scales linearly with CPU cores. All operations are in-place with zero allocations.

**Matrix Format**: For best performance, use CSR format (row-major) for row-wise operations. The implementation is optimized for CSR but works with CSC as well.

**Use Cases**:
- **log1p**: Standard preprocessing for single-cell RNA-seq data
- **log2p1**: Information-theoretic measures (entropy, mutual information)
- **expm1**: Reversing log transforms, converting back to count scale

---

## See Also

- [Normalize](/cpp/kernels/normalize) - Normalization before log transform
- [Softmax](/cpp/kernels/softmax) - Softmax normalization
