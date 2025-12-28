# auroc.hpp

> scl/kernel/stat/auroc.hpp · Area Under ROC Curve computation for binary classification

## Overview

This file provides efficient computation of AUROC (Area Under ROC Curve) for binary group comparisons:

- **AUROC Computation**: Compute AUROC for each feature comparing two groups
- **P-value Calculation**: Two-sided p-values using Mann-Whitney U test normal approximation
- **Fold Change**: Optional log2 fold change computation
- **High Performance**: SIMD-optimized, parallelized over features

AUROC measures the probability that a randomly selected value from group 1 exceeds a randomly selected value from group 0.

**Header**: `#include "scl/kernel/stat/auroc.hpp"`

---

## Main APIs

### auroc

::: source_code file="scl/kernel/stat/auroc.hpp" symbol="auroc" collapsed
:::

**Algorithm Description**

Compute AUROC (Area Under ROC Curve) for each feature using Mann-Whitney U test:

1. **Partition values by group**: For each feature in parallel
   - Extract non-zero values and their group assignments
   - Partition into group 0 and group 1 buffers
   - Count elements: n1 (group 0), n2 (group 1)

2. **Sort groups**: Use VQSort (Google Highway) for high-performance sorting
   - Sort group 0 values
   - Sort group 1 values

3. **Compute rank sum**: Merge sorted groups with tie handling
   - Merge sorted arrays while tracking ranks
   - For tied values: assign average rank
   - Compute rank sum R1 for group 1

4. **Compute U statistic**: U = R1 - n1 * (n1 + 1) / 2
   - U is the number of pairs where group 1 > group 0

5. **Compute AUROC**: AUROC = U / (n1 * n2)
   - Range: [0, 1]
   - 0.5 = no discrimination
   - > 0.5 = group 1 tends to have higher values
   - < 0.5 = group 0 tends to have higher values

6. **Compute p-value**: Normal approximation with tie correction
   - Mean: μ = n1 * n2 / 2
   - Variance: σ² = n1 * n2 * (n1 + n2 + 1) / 12 (with tie correction)
   - Z-score: z = (U - μ) / σ
   - Two-sided p-value: p = 2 * (1 - Φ(|z|))

**Edge Cases**

- **Empty group**: Throws ArgumentError if either group has zero members
- **All values equal**: AUROC = 0.5, p-value = 1.0
- **Perfect separation**: AUROC = 0.0 or 1.0, p-value ≈ 0
- **Sparse data**: Handled correctly by processing only non-zeros
- **Tied values**: Average rank assigned, variance corrected

**Data Guarantees (Preconditions)**

- Matrix is valid CSR/CSC sparse format
- `group_ids.len == matrix.secondary_dim()` (samples dimension)
- `out_auroc.len >= matrix.primary_dim()` (features dimension)
- `out_p_values.len >= matrix.primary_dim()`
- Both groups have at least one member (checked, throws ArgumentError if not)

**Complexity Analysis**

- **Time**: O(features * nnz_per_row * log(nnz_per_row)) parallelized over features
  - Partitioning: O(nnz_per_row)
  - Sorting: O(nnz_per_row * log(nnz_per_row))
  - Rank computation: O(nnz_per_row)
- **Space**: O(threads * max_row_length) for temporary buffers

**Example**

```cpp
#include "scl/kernel/stat/auroc.hpp"

// Expression matrix (features x samples, CSR format)
scl::Sparse<Real, true> matrix = /* ... */;

// Binary group assignment (0 or 1 for each sample)
scl::Array<int32_t> group_ids = /* ... */;  // [n_samples]

// Pre-allocate output buffers
scl::Array<Real> auroc_values(matrix.rows());      // [n_features]
scl::Array<Real> p_values(matrix.rows());          // [n_features]

// Compute AUROC and p-values
scl::kernel::stat::auroc::auroc(
    matrix, group_ids,
    auroc_values, p_values
);

// Filter significant features
for (scl::Index i = 0; i < matrix.rows(); ++i) {
    if (auroc_values[i] > 0.7 && p_values[i] < 0.05) {
        // Feature i is significantly higher in group 1
    } else if (auroc_values[i] < 0.3 && p_values[i] < 0.05) {
        // Feature i is significantly higher in group 0
    }
}
```

---

### auroc_with_fc

::: source_code file="scl/kernel/stat/auroc.hpp" symbol="auroc_with_fc" collapsed
:::

**Algorithm Description**

Compute AUROC with log2 fold change in a single pass:

1. **Partition with sum accumulation**: For each feature
   - Partition values by group while accumulating sums
   - Compute mean for each group: mean1, mean0
   - More efficient than separate AUROC and fold change computation

2. **Compute log2 fold change**: log2_fc = log2((mean1 + eps) / (mean0 + eps))
   - eps = 1e-9 for numerical stability
   - Handles zero means gracefully

3. **Compute AUROC and p-values**: Same as `auroc()` function

This function is more efficient than calling `auroc()` and computing fold change separately, as partitioning and sum accumulation are done once.

**Edge Cases**

- **Zero means**: log2_fc computed with epsilon, avoids division by zero
- **Both groups zero**: log2_fc = 0
- **Same as auroc()**: All edge cases from auroc() apply

**Data Guarantees (Preconditions)**

- Same as `auroc()` function
- `out_log2_fc.len >= matrix.primary_dim()`

**Complexity Analysis**

- **Time**: Same as `auroc()` (no additional overhead)
- **Space**: Same as `auroc()`

**Example**

```cpp
#include "scl/kernel/stat/auroc.hpp"

scl::Sparse<Real, true> matrix = /* ... */;
scl::Array<int32_t> group_ids = /* ... */;

scl::Array<Real> auroc_values(matrix.rows());
scl::Array<Real> p_values(matrix.rows());
scl::Array<Real> log2_fc(matrix.rows());

// Compute AUROC, p-values, and fold change in one pass
scl::kernel::stat::auroc::auroc_with_fc(
    matrix, group_ids,
    auroc_values, p_values, log2_fc
);

// Filter by AUROC, significance, and fold change
for (scl::Index i = 0; i < matrix.rows(); ++i) {
    if (std::abs(auroc_values[i] - 0.5) > 0.2 &&  // Significant separation
        p_values[i] < 0.05 &&                      // Significant p-value
        std::abs(log2_fc[i]) > 1.0) {              // >2x fold change
        // Feature i is differentially expressed
    }
}
```

---

## Utility Functions

### count_groups

Count elements in each of two groups using SIMD optimization.

::: source_code file="scl/kernel/stat/auroc.hpp" symbol="count_groups" collapsed
:::

**Complexity**

- Time: O(n) with SIMD optimization
- Space: O(1)

---

## Notes

- AUROC is equivalent to the Mann-Whitney U statistic normalized by (n1 * n2)
- AUROC = 0.5 indicates no discrimination between groups
- P-values use normal approximation, which is accurate for n1, n2 > 20
- Tie correction improves accuracy when many tied values exist
- All operations are parallelized over features and thread-safe
- Sparse data is handled efficiently by processing only non-zero values

## See Also

- [Effect Size Computation](./effect_size)
- [Statistical Tests](../ttest)
- [Multiple Testing Correction](../multiple_testing)

