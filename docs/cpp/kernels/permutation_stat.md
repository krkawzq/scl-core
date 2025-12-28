# permutation_stat.hpp

> scl/kernel/stat/permutation_stat.hpp · Optimized permutation testing with sort-reuse optimization

## Overview

This file provides optimized permutation testing for comparing two groups. The key innovation is reusing sorted data structures across permutations, avoiding re-sorting for each permutation and achieving significant speedup.

**Header**: `#include "scl/kernel/stat/permutation_stat.hpp"`

---

## Main APIs

### batch_permutation_reuse_sort

::: source_code file="scl/kernel/stat/permutation_stat.hpp" symbol="batch_permutation_reuse_sort" collapsed
:::

**Algorithm Description**

Batch permutation test optimized by reusing sorted data structure:

1. For each feature in parallel:
   - Extract non-zero values with indices
   - Sort values ONCE using argsort (preserving index mapping to group_ids)
   - Compute observed statistic using original group_ids:
     - MWU: Mann-Whitney U statistic from sorted ranks
     - MeanDiff: Mean difference (group0 - group1)
     - KS: Kolmogorov-Smirnov D statistic (future)
   - For each permutation:
     - Shuffle group_ids using Fisher-Yates (O(n) time)
     - Recompute statistic using sorted data + shuffled groups:
       - Walk through sorted values
       - Assign groups based on shuffled group_ids
       - Compute statistic without re-sorting
     - Adaptive early stopping check every 100 permutations:
       - If p < 0.001 or p > 0.5, stop early
   - Compute two-sided p-value:
     - P-value = (count of |stat_perm| >= |stat_obs| + 1) / (n_perms + 1)

2. **Key Optimization**:
   - Standard approach: sort data for EACH permutation = O(P * n log n)
   - This approach: sort ONCE, permute groups = O(n log n + P * n)
   - Speedup factor: approximately log(n) for large n

3. Uses Xoshiro256++ PRNG with jump() for parallel independent streams

**Edge Cases**

- **Empty group 0 or group 1**: Throws ArgumentError
- **n_permutations < 100**: Uses minimum 100 permutations
- **n_permutations > 100000**: Clamped to maximum 100000
- **All values identical**: All permutation statistics = observed, p-value = 1.0
- **No overlap between groups**: p-value approaches 0
- **Early stopping**: May use fewer than n_permutations if p < 0.001 or p > 0.5

**Data Guarantees (Preconditions)**

- Both groups (0 and 1) must have at least one member
- Output array must have size >= matrix.primary_dim()
- `n_permutations` should be in range [100, 100000] (clamped if outside)
- `group_ids` must contain only 0 or 1 values
- Matrix must be valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(features * (nnz * log(nnz) + n_permutations * nnz))
  - Sort once: O(nnz * log(nnz))
  - Each permutation: O(nnz) (shuffle + statistic computation)
- **Space**: O(threads * (max_row_length + n_samples + n_permutations))
  - Sorted data, shuffled groups, permutation statistics

**Example**

```cpp
#include "scl/kernel/stat/permutation_stat.hpp"

// Prepare data
Sparse<Real, true> matrix = /* features x samples */;
Array<int32_t> group_ids = /* binary group assignment (0 or 1) */;
Size n_permutations = 10000;

// Pre-allocate output
Size n_features = matrix.rows();
Array<Real> p_values(n_features);

// Compute permutation test with MWU statistic
scl::kernel::stat::permutation_stat::batch_permutation_reuse_sort(
    matrix, group_ids, n_permutations,
    p_values,
    stat_type = scl::kernel::stat::permutation_stat::PermStatType::MWU,
    seed = 42
);

// Interpret results
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05) {
        std::cout << "Feature " << i 
                  << ": p = " << p_values[i]
                  << " (significant)\n";
    }
}

// Compare with mean difference statistic
Array<Real> p_values_mean(n_features);
scl::kernel::stat::permutation_stat::batch_permutation_reuse_sort(
    matrix, group_ids, n_permutations,
    p_values_mean,
    stat_type = scl::kernel::stat::permutation_stat::PermStatType::MeanDiff,
    seed = 42
);
```

---

### permutation_test_single

::: source_code file="scl/kernel/stat/permutation_stat.hpp" symbol="permutation_test_single" collapsed
:::

**Algorithm Description**

Single-feature permutation test with sort-reuse optimization:

1. Same algorithm as batch_permutation_reuse_sort but for single feature
2. Extract feature values
3. Sort once, then permute groups for each permutation
4. Compute empirical two-sided p-value

**Edge Cases**

- **Empty group 0 or group 1**: Throws ArgumentError
- **n_permutations < 100**: Uses minimum 100 permutations
- **All values identical**: p-value = 1.0

**Data Guarantees (Preconditions)**

- `values.len == group_ids.len`
- Both groups must have at least one member
- `n_permutations` in range [100, 100000]

**Complexity Analysis**

- **Time**: O(n * log(n) + n_permutations * n)
- **Space**: O(n + n_permutations)

**Example**

```cpp
#include "scl/kernel/stat/permutation_stat.hpp"

// Single feature test
Array<Real> values = /* feature values */;
Array<int32_t> group_ids = /* binary group assignment */;
Size n_permutations = 10000;

Real p_value = scl::kernel::stat::permutation_stat::permutation_test_single(
    values, group_ids, n_permutations,
    stat_type = scl::kernel::stat::permutation_stat::PermStatType::MWU,
    seed = 42
);

std::cout << "P-value: " << p_value << "\n";
```

---

## Statistic Types

### PermStatType

Enumeration of supported statistic types:

- **MWU**: Mann-Whitney U statistic (non-parametric rank test)
- **MeanDiff**: Mean difference (group0 - group1, t-test like)
- **KS**: Kolmogorov-Smirnov D statistic (future implementation)

---

## Notes

**When to Use**: Permutation tests are appropriate when:
- Exact p-values needed (no distributional assumptions)
- Small sample sizes
- Non-normal data
- Any statistic can be used (not limited to standard tests)

**Advantages**:
- **Exact**: No distributional assumptions
- **Flexible**: Works with any test statistic
- **Robust**: Valid for any sample size
- **Optimized**: Sort-reuse provides log(n) speedup

**Performance Optimization**:
- Standard approach: O(P * n log n) - sort for each permutation
- This approach: O(n log n + P * n) - sort once, permute groups
- Speedup: approximately log(n) for large n
- Example: n=1000, log(n)≈10, 10x speedup

**Adaptive Early Stopping**:
- Checks every 100 permutations
- Stops early if p < 0.001 (very significant) or p > 0.5 (not significant)
- Reduces computation for clear cases

**Random Number Generation**:
- Uses Xoshiro256++ PRNG (fast, high-quality)
- jump() method for parallel independent streams
- Lemire's nearly divisionless bounded random for shuffling

**Thread Safety**: Uses thread-local workspace and RNG for parallel processing over features, safe for concurrent execution.

**Comparison with Parametric Tests**:
- **Permutation test**: Exact, no assumptions, flexible statistic
- **t-test**: Assumes normality, faster, less flexible
- **Mann-Whitney U**: Non-parametric, faster, less flexible

---

## See Also

- [Mann-Whitney U](/cpp/kernels/mwu) - Non-parametric rank test
- [T-test](/cpp/kernels/ttest) - Parametric mean comparison
- [KS Test](/cpp/kernels/ks) - Distribution comparison

