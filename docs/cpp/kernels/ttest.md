# ttest.hpp

> scl/kernel/ttest.hpp · T-test computation kernels

## Overview

This file provides high-performance t-test computation for comparing two groups in sparse single-cell data. It supports both Welch's t-test (unequal variances) and Student's t-test (equal variances), with optimized sparse matrix operations.

**Header**: `#include "scl/kernel/ttest.hpp"`

Key features:
- Welch's and Student's t-test
- Sparse matrix optimization
- Log2 fold change computation
- Group statistics computation
- SIMD-optimized operations

---

## Main APIs

### ttest

::: source_code file="scl/kernel/ttest.hpp" symbol="ttest" collapsed
:::

**Algorithm Description**

Compute Welch's or Student's t-test for each feature comparing two groups:

1. For each feature f in parallel:
   - Partition non-zero values by group (4-way unrolled, prefetch)
   - Accumulate sum and sum_sq during partition
   - Compute mean including zeros: `mean = sum / n_total`
   - Compute variance with zero adjustment and Bessel correction:
     - `var = (sum_sq - sum^2/n) / (n-1)` for n > 1
   - Compute standard error:
     - Welch: `se = sqrt(var1/n1 + var2/n2)`
     - Pooled: `se = sqrt(pooled_var * (1/n1 + 1/n2))`
   - Compute t-statistic: `t = (mean2 - mean1) / se`
   - Compute p-value via normal approximation (fast erfc)
   - Compute log2 fold change: `log2FC = log2((mean2 + eps) / (mean1 + eps))`

**Edge Cases**

- **Empty group**: Throws ArgumentError
- **Constant feature**: t-statistic is 0, p-value is 1.0
- **Zero variance**: Standard error is clamped to SIGMA_MIN (1e-12)
- **Perfect separation**: Very large t-statistic, very small p-value

**Data Guarantees (Preconditions)**

- `matrix.secondary_dim() == group_ids.len`
- Output arrays have size >= `matrix.primary_dim()`
- `group_ids` contains only values 0 or 1
- Both groups must have at least one member

**Complexity Analysis**

- **Time**: O(features * nnz_per_feature) - parallel over features
- **Space**: O(threads * max_row_length) for workspace

**Example**

```cpp
#include "scl/kernel/ttest.hpp"

scl::Sparse<Real, true> matrix = /* feature matrix [n_features x n_samples] */;
scl::Array<int32_t> group_ids = /* binary group assignment */;
scl::Array<Real> t_stats(n_features);
scl::Array<Real> p_values(n_features);
scl::Array<Real> log2_fc(n_features);

scl::kernel::ttest::ttest(
    matrix, group_ids,
    t_stats, p_values, log2_fc,
    true  // use_welch
);

// t_stats[i] = t-statistic for feature i
// p_values[i] = two-tailed p-value
// log2_fc[i] = log2 fold change (group1/group0)
```

---

### compute_group_stats

::: source_code file="scl/kernel/ttest.hpp" symbol="compute_group_stats" collapsed
:::

**Algorithm Description**

Compute per-group mean, variance, and count for each feature:

1. For each feature f in parallel:
   - For each group g:
     - Extract non-zero values for samples in group g
     - Compute mean: `mean = sum(values) / n_nonzero`
     - Compute variance: `var = sum((values - mean)^2) / (n_nonzero - 1)`
     - Count: `count = n_nonzero`
2. Output layout is row-major: `[feat0_g0, feat0_g1, ..., feat1_g0, ...]`

**Edge Cases**

- **No samples in group**: Mean = 0, var = 0, count = 0
- **Single sample in group**: Variance is undefined (set to 0)
- **All zeros in group**: Mean = 0, var = 0, count = 0

**Data Guarantees (Preconditions)**

- `group_ids[i]` in range [0, n_groups) or negative (ignored)
- Output arrays sized >= `n_features * n_groups`
- Matrix must be valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(features * nnz_per_feature)
- **Space**: O(threads * max_row_length) workspace

**Example**

```cpp
Index n_groups = 3;
scl::Array<Real> means(n_features * n_groups);
scl::Array<Real> vars(n_features * n_groups);
scl::Array<Size> counts(n_features * n_groups);

scl::kernel::ttest::compute_group_stats(
    matrix, group_ids, n_groups,
    means, vars, counts
);

// means[f * n_groups + g] = mean of feature f in group g
// vars[f * n_groups + g] = variance of feature f in group g
// counts[f * n_groups + g] = count of non-zeros for feature f in group g
```

---

## Utility Functions

### count_groups

Count elements in each of two groups.

::: source_code file="scl/kernel/ttest.hpp" symbol="count_groups" collapsed
:::

**Complexity**

- Time: O(n) with SIMD optimization
- Space: O(1)

---

## Notes

- Welch's t-test is recommended for unequal variances (default)
- Student's t-test assumes equal variances (use_welch=false)
- Log2 fold change uses epsilon (1e-9) for numerical stability
- P-values are two-tailed by default
- Sparse matrix operations are optimized for single-cell data

## See Also

- [Multiple Testing Module](./multiple_testing) - For FDR correction
- [Statistics Module](../math/statistics) - For additional statistical tests
