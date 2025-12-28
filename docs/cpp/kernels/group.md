# group.hpp

> scl/kernel/group.hpp · Group aggregation kernels for computing per-group statistics

## Overview

This file provides high-performance group aggregation operations for sparse matrices. It computes per-group mean and variance statistics for each feature, supporting multiple groups and flexible zero-handling options.

This file provides:
- Per-group statistics computation (mean and variance)
- Support for arbitrary number of groups
- Optional zero inclusion/exclusion in statistics
- SIMD-optimized accumulation with 4-way unrolling

**Header**: `#include "scl/kernel/group.hpp"`

---

## Main APIs

### group_stats

::: source_code file="scl/kernel/group.hpp" symbol="group_stats" collapsed
:::

**Algorithm Description**

Compute per-group mean and variance for each feature in a sparse matrix:

1. **Parallel feature processing**: For each feature in parallel:
   - Zero-initialize thread-local accumulators (sum and sum_sq per group)
   - Iterate over non-zero elements with 4-way unrolled loop
   - Use prefetch for indirect group access via group_ids[indices[k]]
   - Accumulate sum and sum_sq per group using indirect addressing

2. **Statistics finalization**: For each group:
   - Determine sample count N:
     - If include_zeros: N = group_sizes[g]
     - Otherwise: N = nnz_counts[g] (count of non-zeros in group)
   - Compute mean: mean[g] = sum[g] / N[g]
   - Compute variance: var[g] = (sum_sq[g] - N[g] * mean[g]^2) / (N[g] - ddof)
   - Handle edge cases: If N <= ddof, set mean=0 and var=0

3. **Output layout**: Results stored in row-major format by feature:
   - Index for feature f, group g: (f * n_groups + g)
   - Layout: [feat0_g0, feat0_g1, ..., feat0_gN, feat1_g0, ...]

**Edge Cases**

- **Empty groups**: Groups with N <= ddof have mean=0, var=0
- **Negative group IDs**: Samples with negative group_ids are ignored
- **Invalid group IDs**: Group IDs outside [0, n_groups) are ignored
- **Zero variance**: Variance clamped to >= 0 for numerical stability
- **All zeros in group**: If include_zeros=false and group has no non-zeros, N=0, mean=0, var=0

**Data Guarantees (Preconditions)**

- `group_ids[i]` must be in range [0, n_groups) or negative (ignored)
- `group_sizes.len >= n_groups` (must contain size for each group)
- `out_means.len >= n_features * n_groups` (pre-allocated output buffer)
- `out_vars.len >= n_features * n_groups` (pre-allocated output buffer)
- Matrix must be valid CSR format (sorted indices, no duplicates)
- If include_zeros=false, requires additional memory for nnz_counts per thread

**Complexity Analysis**

- **Time**: O(nnz + n_features * n_groups)
  - O(nnz) for iterating all non-zero elements
  - O(n_features * n_groups) for finalizing statistics
- **Space**: O(n_groups) per thread for accumulators
  - Additional O(n_groups) per thread for nnz_counts if include_zeros=false
  - Stack allocation for n_groups <= 256, heap allocation for larger counts

**Example**

```cpp
#include "scl/kernel/group.hpp"
#include "scl/core/sparse.hpp"

// Expression matrix: genes x cells
Sparse<Real, true> expression = /* ... */;
Index n_genes = expression.rows();
Index n_cells = expression.cols();

// Cell type labels per cell
Array<int32_t> cell_type_labels(n_cells);
// ... assign labels: 0, 1, 2, ... (n_types - 1) ...

Size n_types = 5;

// Compute group sizes
Array<Size> type_sizes(n_types, 0);
for (Index i = 0; i < n_cells; ++i) {
    if (cell_type_labels[i] >= 0 && cell_type_labels[i] < n_types) {
        type_sizes[cell_type_labels[i]]++;
    }
}

// Pre-allocate output buffers
Array<Real> means(n_genes * n_types);
Array<Real> vars(n_genes * n_types);

// Compute per-group statistics
scl::kernel::group::group_stats(
    expression,
    cell_type_labels,
    n_types,
    type_sizes,
    means,
    vars,
    1,      // ddof = 1 (sample variance)
    true    // include_zeros = true
);

// Access mean expression of gene g in cell type t
Real mean_expr = means[g * n_types + t];
Real var_expr = vars[g * n_types + t];

// Example: Compare expression between groups
Real mean_type0 = means[gene_idx * n_types + 0];
Real mean_type1 = means[gene_idx * n_types + 1];
Real fold_change = mean_type1 / (mean_type0 + 1e-10);
```

---

## Utility Functions

### detail::finalize_stats

Internal helper function that converts accumulated sums to mean and variance.

::: source_code file="scl/kernel/group.hpp" symbol="detail::finalize_stats" collapsed
:::

**Complexity**

- Time: O(n_groups)
- Space: O(1)

---

## Notes

**Memory Optimization**:
- Stack allocation for n_groups <= 256 (typical case)
- Heap allocation for larger group counts to avoid stack overflow

**Numerical Stability**:
- Uses Welford-style variance formula for numerical stability
- Variance clamped to >= 0 to handle floating-point rounding errors
- Handles division by zero gracefully (N <= ddof case)

**Zero Handling**:
- `include_zeros=true`: Count all samples in group, including zeros
  - Mean reflects overall expression including non-expressed samples
  - Useful for proportion/percentage calculations
- `include_zeros=false`: Count only non-zero samples
  - Mean reflects average expression among expressing samples
  - Useful for average expression level calculations

**Output Layout**:
Results are stored in row-major format by feature:
- For feature f, group g: index = f * n_groups + g
- This layout enables efficient access patterns for downstream analysis

## See Also

- [T-test](/cpp/kernels/ttest) - Two-group comparison with statistical testing
- [Statistics](/cpp/kernels/statistics) - Other statistical operations
