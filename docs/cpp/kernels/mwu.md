# mwu.hpp

> scl/kernel/mwu.hpp · Mann-Whitney U test for comparing two groups

## Overview

This file provides the Mann-Whitney U test (also known as Wilcoxon rank-sum test), a non-parametric statistical test for comparing two independent groups. Unlike parametric tests like the t-test, it makes no assumptions about data distribution and is based on ranks.

Key features:
- Non-parametric test (no distribution assumptions)
- Rank-based comparison using sorted ranks
- Feature-wise parallel computation for sparse matrices
- Additional metrics: U statistics, p-values, log2 fold change, AUROC
- SIMD-optimized sorting and rank computation

**Header**: `#include "scl/kernel/mwu.hpp"`

---

## Main APIs

### mwu_test

Perform Mann-Whitney U test for each feature (row/column) in a sparse matrix, comparing two groups of samples.

::: source_code file="scl/kernel/mwu.hpp" symbol="mwu_test" collapsed
:::

**Algorithm Description**

The Mann-Whitney U test compares two groups by ranking all values and computing a U statistic:

1. For each feature in parallel:
   a. Partition non-zero values by group using pre-allocated buffers
   b. Sort each group using VQSort (SIMD-optimized)
   c. Merge sorted arrays to compute rank sum with tie correction
   d. Compute U statistic: U = R1 - n1*(n1+1)/2 where R1 is rank sum of group 1
   e. Apply continuity correction and normal approximation for p-value:
      - Mean: mu = n1*n2/2
      - Variance: sigma^2 = n1*n2*(n1+n2+1)/12 (with tie correction)
      - Z = (U - mu) / sigma (with continuity correction)
      - Two-sided p-value from standard normal
   f. Compute log2 fold change from group means
   g. Optionally compute AUROC = U / (n1 * n2)

Optimizations:
- Binary search for negative/positive boundary (O(log n))
- Prefetch in merge loop for cache efficiency
- Precomputed reciprocals to avoid division
- 4-way unrolled partition loop
- Thread-local workspace for sorting buffers

**Edge Cases**

- **All-zero feature**: Returns U=0, p=1, AUROC=0.5 (no difference between groups)
- **Empty groups**: Throws SCL_CHECK_ARG error (both groups must have at least one member)
- **Ties in values**: Tie correction applied to variance estimate
- **Very small groups (n1, n2 < 10)**: Normal approximation may be less accurate
- **Identical groups**: U ≈ n1*n2/2, p ≈ 1.0, AUROC ≈ 0.5

**Data Guarantees (Preconditions)**

- `matrix.secondary_dim() == group_ids.len` (one label per sample)
- `out_u_stats.len == out_p_values.len == out_log2_fc.len == matrix.primary_dim()`
- `group_ids` contains only values 0 or 1
- Both groups must have at least one member
- If `out_auroc` provided: `out_auroc.len == matrix.primary_dim()`

**Complexity Analysis**

- **Time**: O(features * nnz_per_feature * log(nnz_per_feature)) per feature
- **Space**: O(max_nnz) per thread for sorting buffers

**Example**

```cpp
#include "scl/kernel/mwu.hpp"
#include "scl/core/sparse.hpp"

// Create sparse matrix (features x samples)
Sparse<Real, true> matrix(n_features, n_samples);

// Create group labels (0 or 1 for each sample)
Array<int32_t> group_ids(n_samples);
// Fill: group_ids[i] = 0 or 1

// Pre-allocate output arrays
Array<Real> u_stats(n_features);
Array<Real> p_values(n_features);
Array<Real> log2_fc(n_features);
Array<Real> auroc(n_features);  // Optional

// Perform Mann-Whitney U test
scl::kernel::mwu::mwu_test(
    matrix,
    group_ids,
    u_stats,
    p_values,
    log2_fc,
    auroc  // Optional: omit for just U, p, log2FC
);

// Results:
// - u_stats[i]: U statistic for feature i
// - p_values[i]: Two-sided p-value (smaller = more significant)
// - log2_fc[i]: Log2(mean_group1 / mean_group0)
// - auroc[i]: Area under ROC curve = P(group1 > group0) + 0.5*P(ties)
```

---

## Utility Functions

### count_groups

Count the number of samples in each group (0 and 1). Uses SIMD-optimized counting.

::: source_code file="scl/kernel/mwu.hpp" symbol="count_groups" collapsed
:::

**Complexity**

- Time: O(n) with SIMD optimization
- Space: O(1)

---

## Numerical Notes

- **Normal approximation**: Valid for n1, n2 >= 10. For smaller samples, exact U distribution should be used (not implemented).
- **Continuity correction**: Applied when computing Z-score (U ± 0.5) to improve discrete-to-continuous approximation.
- **Tie correction**: Variance estimate adjusted when ties present: sigma^2 *= (1 - sum(t^3-t)/(n^3-n)) where t is tie group size.
- **Fold change computation**: EPS (1e-9) added to means to prevent division by zero: log2((mean1+EPS)/(mean0+EPS)).
- **AUROC interpretation**: 
  - AUROC = 0.5: No difference between groups
  - AUROC = 1.0: All group1 values > all group0 values
  - AUROC = 0.0: All group0 values > all group1 values
- **U statistic range**: U in [0, n1*n2]. Under null, E[U] = n1*n2/2.

## See Also

- [Multiple Testing](/cpp/kernels/multiple_testing) - FDR correction for p-values
- [T-test](/cpp/kernels/ttest) - Parametric alternative to MWU test
