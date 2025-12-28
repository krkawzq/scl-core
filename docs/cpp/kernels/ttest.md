# T-test

Welch's and Student's t-test for comparing two groups of samples.

## Overview

T-test provides:

- **Parametric test** - Assumes normal distribution (robust with large samples)
- **Two variants** - Welch's t-test (unequal variances) and Student's t-test (equal variances)
- **Multiple features** - Test all features in parallel
- **Additional metrics** - T-statistics, p-values, log2 fold change

## Basic Usage

### ttest

Compute Welch's or Student's t-test for each feature comparing two groups.

```cpp
#include "scl/kernel/ttest.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> matrix = /* ... */;  // features x samples
Array<int32_t> group_ids = /* ... */;    // 0 or 1 for each sample

Array<Real> t_stats(matrix.primary_dim());
Array<Real> p_values(matrix.primary_dim());
Array<Real> log2_fc(matrix.primary_dim());

scl::kernel::ttest::ttest(
    matrix,
    group_ids,
    t_stats,
    p_values,
    log2_fc,
    use_welch = true  // Use Welch's t-test (default)
);
```

**Parameters:**
- `matrix` [in] - Sparse matrix (features x samples)
- `group_ids` [in] - Binary group assignment (0 or 1) for each sample
- `out_t_stats` [out] - T-statistics for each feature
- `out_p_values` [out] - Two-tailed p-values for each feature
- `out_log2_fc` [out] - Log2 fold change (group1 / group0)
- `use_welch` [in] - Use Welch's t-test if true, Student's if false

**Preconditions:**
- `matrix.secondary_dim() == group_ids.len`
- All output arrays have size >= `matrix.primary_dim()`
- `group_ids` contains only values 0 or 1
- Both groups must have at least one member

**Postconditions:**
- `out_t_stats[i]` contains t-statistic for feature i
- `out_p_values[i]` contains two-tailed p-value
- `out_log2_fc[i]` = log2((mean_group1 + eps) / (mean_group0 + eps))

**Algorithm:**
For each feature in parallel:
1. Partition non-zero values by group (4-way unrolled, prefetch)
2. Accumulate sum and sum_sq during partition
3. Compute mean including zeros: mean = sum / n_total
4. Compute variance with zero adjustment and Bessel correction
5. Compute standard error:
   - Welch: se = sqrt(var1/n1 + var2/n2)
   - Pooled: se = sqrt(pooled_var * (1/n1 + 1/n2))
6. t_stat = (mean2 - mean1) / se
7. p_value via normal approximation (fast erfc)

**Complexity:**
- Time: O(features * nnz_per_feature)
- Space: O(threads * max_row_length) for workspace

**Thread Safety:**
Safe - parallelized over features with thread-local workspace

**Throws:**
`ArgumentError` - if either group is empty

**Numerical Notes:**
- EPS = 1e-9 added to means for log2FC stability
- SIGMA_MIN = 1e-12 threshold for valid standard error
- Variance clamped to >= 0 for numerical stability
- Uses fast erfc approximation (max error < 1.5e-7)

## Helper Functions

### count_groups

Count elements in each of two groups.

```cpp
Size n1, n2;
scl::kernel::ttest::count_groups(group_ids, n1, n2);
// n1 = count of group 0
// n2 = count of group 1
```

**Parameters:**
- `group_ids` [in] - Array of group assignments (0 or 1)
- `out_n1` [out] - Count of elements with group_id == 0
- `out_n2` [out] - Count of elements with group_id == 1

**Algorithm:**
Uses SIMD-optimized counting.

**Complexity:**
- Time: O(n) with SIMD optimization
- Space: O(1)

## Legacy Functions

### compute_group_stats

Compute per-group mean, variance, and count for each feature (retained for backward compatibility).

```cpp
Array<Real> out_means(n_features * n_groups);
Array<Real> out_vars(n_features * n_groups);
Array<Size> out_counts(n_features * n_groups);

scl::kernel::ttest::compute_group_stats(
    matrix,
    group_ids,
    n_groups,
    out_means,
    out_vars,
    out_counts
);
```

**Parameters:**
- `matrix` [in] - Sparse matrix, shape (n_features, n_samples)
- `group_ids` [in] - Group assignment per sample, size = n_samples
- `n_groups` [in] - Number of groups
- `out_means` [out] - Group means, size = n_features * n_groups
- `out_vars` [out] - Group variances, size = n_features * n_groups
- `out_counts` [out] - Group counts, size = n_features * n_groups

**Output Layout:**
Row-major: [feat0_g0, feat0_g1, ..., feat1_g0, feat1_g1, ...]

**Note:**
Consider using `ttest()` directly for two-group comparisons. This function is retained for k-group scenarios.

## Use Cases

### Differential Expression Analysis

Compare gene expression between two conditions:

```cpp
Sparse<Real, true> expression_matrix = /* ... */;  // genes x cells
Array<int32_t> condition_labels = /* ... */;      // 0=control, 1=treatment

Array<Real> t_stats(expression_matrix.primary_dim());
Array<Real> p_values(expression_matrix.primary_dim());
Array<Real> log2_fc(expression_matrix.primary_dim());

scl::kernel::ttest::ttest(
    expression_matrix,
    condition_labels,
    t_stats,
    p_values,
    log2_fc,
    true  // Use Welch's t-test for unequal variances
);

// Find significantly differentially expressed genes
for (Size i = 0; i < expression_matrix.primary_dim(); ++i) {
    if (p_values[i] < 0.05 && std::abs(log2_fc[i]) > 1.0) {
        // Gene i is significantly differentially expressed
    }
}
```

### Choosing Between Welch's and Student's t-test

- **Welch's t-test** (default): Use when group variances may differ
  - More robust and generally recommended
  - Slightly more conservative (less power)
  
- **Student's t-test**: Use when group variances are known to be equal
  - Slightly more powerful when assumption holds
  - Less robust if assumption is violated

## Performance

### Parallelization

- Parallelized over features
- Thread-local workspace for temporary buffers
- No synchronization overhead

### SIMD Optimization

- SIMD-optimized counting for group sizes
- 4-way unrolled partition loop with prefetch
- Fast erfc approximation for p-values

### Memory Efficiency

- Pre-allocated workspace pools
- Minimal allocations
- Efficient sparse matrix traversal

## Statistical Details

### T-statistic

Welch's t-test:
```
t = (mean2 - mean1) / sqrt(var1/n1 + var2/n2)
```

Student's t-test:
```
t = (mean2 - mean1) / sqrt(pooled_var * (1/n1 + 1/n2))
pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
```

### P-value Computation

Uses normal approximation for large samples:
- Fast erfc approximation (max error < 1.5e-7)
- Two-tailed p-value: 2 * (1 - Φ(|t|))

### Log2 Fold Change

```
log2_fc = log2((mean_group1 + eps) / (mean_group0 + eps))
```

- eps = 1e-9 added for numerical stability
- Positive values: higher in group1
- Negative values: higher in group0

## See Also

- [Mann-Whitney U Test](/cpp/kernels/mwu) - Non-parametric alternative
- [Statistics](/cpp/kernels/statistics) - Other statistical tests
