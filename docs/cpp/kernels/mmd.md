# mmd.hpp

> scl/kernel/mmd.hpp · Maximum Mean Discrepancy computation with RBF kernel

## Overview

This file provides efficient computation of Maximum Mean Discrepancy (MMD) between two distributions using the RBF (Radial Basis Function) kernel. MMD is a metric for comparing two probability distributions by embedding them in a reproducing kernel Hilbert space (RKHS).

Key features:
- Feature-wise MMD computation for sparse matrices
- Optimized RBF kernel computation with SIMD
- Block tiling for cache efficiency
- Thread-safe parallelization over features

**Header**: `#include "scl/kernel/mmd.hpp"`

---

## Main APIs

### mmd_rbf

::: source_code file="scl/kernel/mmd.hpp" symbol="mmd_rbf" collapsed
:::

**Algorithm Description**

Computes MMD^2 between two distributions X and Y using the RBF kernel:

```
MMD^2 = E[k(X,X)] + E[k(Y,Y)] - 2*E[k(X,Y)]
```

For each feature in parallel:
1. Extract non-zero values from both distributions for the feature
2. Compute unary exp sums: `sum(exp(-gamma * x^2))` for each distribution with caching
3. Compute self-kernel sums: `E[k(X,X)]` and `E[k(Y,Y)]` using symmetry
4. Compute cross-kernel sum: `E[k(X,Y)]` using block tiling (64x512 blocks)
5. Combine results: `(K_xx/n_x^2) + (K_yy/n_y^2) - 2*(K_xy/(n_x*n_y))`

The self-kernel computation exploits symmetry:
- Zero-zero pairs: `(N-nnz)^2` (kernel value = 1)
- Zero-nonzero pairs: `2 * (N-nnz) * sum_unary`
- Diagonal: `nnz` (kernel value = 1)
- Off-diagonal: `2 * sum_{i<j} exp(-gamma * (x_i - x_j)^2)` using 2-way SIMD unroll

The cross-kernel uses block tiling for cache efficiency, computing nonzero-nonzero pairs with 2-way SIMD unroll within each block.

**Edge Cases**

- **All-zero feature**: Returns MMD^2 = 0 (identical zero distributions)
- **Empty distributions**: Handled gracefully with zero kernel sums
- **Single sample per distribution**: Computes self-kernel correctly for n=1
- **Negative values from numerical error**: Clamped to 0 (MMD^2 must be non-negative)

**Data Guarantees (Preconditions)**

- `mat_x.primary_dim() == mat_y.primary_dim()` (same number of features)
- `output.len == mat_x.primary_dim()` (output size matches feature count)
- `gamma > 0` (kernel parameter must be positive)
- Matrices must be valid CSR/CSC format

**Complexity Analysis**

- **Time**: O(features * (nnz_x^2 + nnz_y^2 + nnz_x*nnz_y)) per feature
- **Space**: O(max(nnz_x, nnz_y)) per thread for caching exp terms

**Example**

```cpp
#include "scl/kernel/mmd.hpp"
#include "scl/core/sparse.hpp"

// Create sparse matrices for two distributions
// Each column is a sample, each row is a feature
Sparse<Real, true> mat_x(n_features, n_samples_x);
Sparse<Real, true> mat_y(n_features, n_samples_y);

// Fill matrices with data...

// Pre-allocate output
Array<Real> mmd_values(n_features);

// Compute MMD^2 with default gamma = 1.0
scl::kernel::mmd::mmd_rbf(mat_x, mat_y, mmd_values);

// Or with custom gamma
Real gamma = 0.5;  // Smaller gamma = wider kernel
scl::kernel::mmd::mmd_rbf(mat_x, mat_y, mmd_values, gamma);

// mmd_values[i] now contains MMD^2 between distributions
// for feature i. Higher values indicate greater distribution difference.
```

---

## Utility Functions

### unary_exp_sum

Compute sum of exp(-gamma * x^2) for non-zero values, caching individual exp terms for reuse in self-kernel and cross-kernel computations.

::: source_code file="scl/kernel/mmd.hpp" symbol="unary_exp_sum" collapsed
:::

**Complexity**

- Time: O(nnz) with 8-way SIMD unroll
- Space: O(nnz) for cache array

---

### self_kernel_sum

Compute sum of RBF kernel for all pairs within a single distribution, including implicit zeros. Exploits symmetry for efficiency.

::: source_code file="scl/kernel/mmd.hpp" symbol="self_kernel_sum" collapsed
:::

**Complexity**

- Time: O(nnz^2) with 2-way SIMD unroll
- Space: O(1) auxiliary

---

### cross_kernel_sum

Compute sum of RBF kernel between all pairs from two distributions. Uses block tiling (BLOCK_X=64, BLOCK_Y=512) for cache efficiency.

::: source_code file="scl/kernel/mmd.hpp" symbol="cross_kernel_sum" collapsed
:::

**Complexity**

- Time: O(nnz_x * nnz_y) with 2-way SIMD unroll
- Space: O(1) auxiliary

---

## Numerical Notes

- **RBF kernel**: k(x,y) = exp(-gamma * (x-y)^2)
- **MMD^2 properties**: Non-negative metric (may be slightly negative from numerical error, clamped to 0)
- **All-zero features**: MMD^2 = 0 (identical distributions)
- **Gamma parameter**: Controls kernel width; larger gamma = narrower kernel, more sensitive to small differences
- **Normalization**: Each term is normalized by the number of pairs (n^2 for self-kernel, n_x*n_y for cross-kernel)

## See Also

- [Neighbors](/cpp/kernels/neighbors) - KNN algorithms
- [Statistics](/cpp/kernels/statistics) - Statistical tests and metrics
