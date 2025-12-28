# Mann-Whitney U Test

Mann-Whitney U test (Wilcoxon rank-sum test) for comparing two groups of samples.

## Overview

Mann-Whitney U test provides:

- **Non-parametric test** - No distribution assumptions
- **Rank-based** - Uses ranks instead of raw values
- **Multiple features** - Test all features in parallel
- **Additional metrics** - U statistics, p-values, fold change, AUROC

## Basic Usage

### mwu_test

Perform Mann-Whitney U test for each feature in a sparse matrix.

```cpp
#include "scl/kernel/mwu.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> matrix = /* ... */;  // features x samples
Array<int32_t> group_ids = /* ... */;    // 0 or 1 for each sample

Array<Real> u_stats(matrix.primary_dim());
Array<Real> p_values(matrix.primary_dim());
Array<Real> log2_fc(matrix.primary_dim());
Array<Real> auroc(matrix.primary_dim());

scl::kernel::mwu::mwu_test(
    matrix,
    group_ids,
    u_stats,
    p_values,
    log2_fc,
    auroc  // Optional
);
```

**Parameters:**
- `matrix` [in] - Sparse matrix (features x samples)
- `group_ids` [in] - Group labels for each sample (0 or 1)
- `out_u_stats` [out] - U statistics for each feature
- `out_p_values` [out] - Two-sided p-values for each feature
- `out_log2_fc` [out] - Log2 fold change (group1 / group0)
- `out_auroc` [out] - Optional: AUROC values (U / (n1 * n2))

**Preconditions:**
- `matrix.secondary_dim() == group_ids.len`
- All output arrays have length `matrix.primary_dim()`
- `group_ids` contains only values 0 or 1
- Both groups must have at least one member

**Postconditions:**
- `out_u_stats[i]` contains U statistic for feature i
- `out_p_values[i]` contains two-sided p-value (normal approximation)
- `out_log2_fc[i]` contains log2(mean_group1 / mean_group0)
- `out_auroc[i]` (if provided) contains AUROC = U / (n1 * n2)
- For features with all zeros: U=0, p=1, AUROC=0.5

**Algorithm:**
For each feature in parallel:
1. Partition non-zero values by group using pre-allocated buffers
2. Sort each group using VQSort (SIMD-optimized)
3. Merge sorted arrays to compute rank sum with tie correction
4. Compute U statistic: U = R1 - n1*(n1+1)/2
5. Apply continuity correction and normal approximation for p-value
6. Compute log2 fold change from group means
7. Optionally compute AUROC = U / (n1 * n2)

**Optimizations:**
- Binary search for negative/positive boundary (O(log n))
- Prefetch in merge loop for cache efficiency
- Precomputed reciprocals to avoid division
- 4-way unrolled partition loop

**Complexity:**
- Time: O(features * nnz_per_feature * log(nnz_per_feature))
- Space: O(max_nnz) per thread for sorting buffers

**Thread Safety:**
Safe - parallelized over features with thread-local workspace

**Throws:**
`SCL_CHECK_ARG` - if either group is empty

**Numerical Notes:**
- Uses normal approximation for p-value (valid for n1, n2 >= 10)
- Continuity correction applied for discrete-to-continuous approximation
- Tie correction applied to variance estimate
- EPS (1e-9) added to means to prevent division by zero in fold change
- AUROC in [0, 1] represents P(group1 > group0) + 0.5 * P(ties)

## Helper Functions

### count_groups

Count the number of samples in each group.

```cpp
Size n1, n2;
scl::kernel::mwu::count_groups(group_ids, n1, n2);
// n1 = count of group 0
// n2 = count of group 1
```

**Parameters:**
- `group_ids` [in] - Array of group labels (0 or 1)
- `out_n1` [out] - Count of samples in group 0
- `out_n2` [out] - Count of samples in group 1

**Algorithm:**
Uses SIMD-optimized `scl::vectorize::count` for parallel counting.

**Complexity:**
- Time: O(n)
- Space: O(1)

## Use Cases

### Differential Expression Analysis

Compare gene expression between two conditions:

```cpp
Sparse<Real, true> expression_matrix = /* ... */;  // genes x cells
Array<int32_t> condition_labels = /* ... */;      // 0=control, 1=treatment

Array<Real> u_stats(expression_matrix.primary_dim());
Array<Real> p_values(expression_matrix.primary_dim());
Array<Real> log2_fc(expression_matrix.primary_dim());

scl::kernel::mwu::mwu_test(
    expression_matrix,
    condition_labels,
    u_stats,
    p_values,
    log2_fc
);

// Find significantly differentially expressed genes
for (Size i = 0; i < expression_matrix.primary_dim(); ++i) {
    if (p_values[i] < 0.05 && std::abs(log2_fc[i]) > 1.0) {
        // Gene i is significantly differentially expressed
    }
}
```

### Feature Selection

Rank features by effect size:

```cpp
// Compute AUROC for ranking
Array<Real> auroc(matrix.primary_dim());
scl::kernel::mwu::mwu_test(
    matrix, group_ids, u_stats, p_values, log2_fc, auroc
);

// Sort by AUROC (higher = better separation)
std::vector<std::pair<Real, Size>> ranked;
for (Size i = 0; i < matrix.primary_dim(); ++i) {
    ranked.push_back({auroc[i], i});
}
std::sort(ranked.rbegin(), ranked.rend());
```

### Quality Control

Check if groups are well-separated:

```cpp
Array<Real> auroc(matrix.primary_dim());
scl::kernel::mwu::mwu_test(
    matrix, group_ids, u_stats, p_values, log2_fc, auroc
);

Real mean_auroc = 0.0;
for (Size i = 0; i < matrix.primary_dim(); ++i) {
    mean_auroc += auroc[i];
}
mean_auroc /= matrix.primary_dim();

// mean_auroc close to 0.5: groups are similar
// mean_auroc close to 1.0: groups are well-separated
```

## Performance

### Parallelization

- Parallelized over features
- Thread-local workspace for sorting buffers
- No synchronization overhead

### SIMD Optimization

- SIMD-optimized counting for group sizes
- VQSort for fast sorting (SIMD-optimized)
- Prefetch in merge loops

### Memory Efficiency

- Pre-allocated workspace pools
- Reusable sorting buffers
- Minimal allocations

## Statistical Details

### U Statistic

U = R1 - n1*(n1+1)/2

Where:
- R1 = sum of ranks in group 1
- n1 = size of group 1

### P-value Computation

Uses normal approximation:
- Mean: n1*n2/2
- Variance: n1*n2*(n1+n2+1)/12 (with tie correction)
- Continuity correction: ±0.5

### AUROC Interpretation

AUROC = U / (n1 * n2)

- 0.5: No separation (random)
- 1.0: Perfect separation
- < 0.5: Group 0 tends to have higher values

## See Also

- [T-test](/cpp/kernels/ttest) - Parametric alternative
- [Statistics](/cpp/kernels/statistics) - Other statistical tests

