# oneway_anova.hpp

> scl/kernel/stat/oneway_anova.hpp · One-way ANOVA F-test for parametric group comparison

## Overview

This file provides the one-way ANOVA (Analysis of Variance) F-test for comparing means across k groups. It is a parametric test that assumes normality and homogeneity of variance.

**Header**: `#include "scl/kernel/stat/oneway_anova.hpp"`

---

## Main APIs

### oneway_anova

::: source_code file="scl/kernel/stat/oneway_anova.hpp" symbol="oneway_anova" collapsed
:::

**Algorithm Description**

Compute One-way ANOVA F-test for k groups (parametric):

1. For each feature in parallel:
   - Compute group sums and grand mean (including sparse zeros):
     - Sum all values across all groups
     - Count total samples (including implicit zeros)
     - grand_mean = total_sum / total_count
   - Compute group means:
     - For each group g: mean_g = sum_g / n_g
     - Where sum_g is sum of values in group g, n_g is size of group g
   - Compute sum of squares:
     - SS_between = sum(n_g * (mean_g - grand_mean)^2)
     - SS_total = sum((x_i - grand_mean)^2) for all observations
     - SS_within = SS_total - SS_between
   - Compute F statistic:
     - df_between = k - 1
     - df_within = N - k
     - MS_between = SS_between / df_between
     - MS_within = SS_within / df_within
     - F = MS_between / MS_within
   - Compute p-value from F distribution:
     - Uses Wilson-Hilferty approximation for F distribution
     - P-value = P(F(df1, df2) >= F_observed)

2. The F statistic measures ratio of between-group variance to within-group variance

3. Large F with small p-value indicates group means differ significantly

**Edge Cases**

- **n_groups < 2**: Throws ArgumentError
- **Fewer than 2 groups have members**: Throws ArgumentError
- **N <= k (total samples <= number of groups)**: Throws ArgumentError (insufficient degrees of freedom)
- **All values identical**: F = 0, p-value = 1.0
- **Single value per group**: F computed but may have low power
- **Negative group_ids**: Ignored (treated as missing data)
- **Empty features**: F = NaN, p-value = NaN

**Data Guarantees (Preconditions)**

- `n_groups >= 2` (must have at least 2 groups)
- `group_ids[i]` in range [0, n_groups) or negative (ignored)
- At least 2 groups must have members
- `N > k` (total samples > number of groups, for degrees of freedom)
- Output arrays must have size >= matrix.primary_dim()
- Matrix must be valid CSR or CSC format
- Assumes normality and homogeneity of variance (not enforced, but required for validity)

**Complexity Analysis**

- **Time**: O(features * nnz_per_feature) - single pass through data per feature
- **Space**: O(threads * n_groups) - thread-local workspace for group statistics

**Example**

```cpp
#include "scl/kernel/stat/oneway_anova.hpp"

// Prepare data
Sparse<Real, true> matrix = /* features x samples */;
Array<int32_t> group_ids = /* group assignment [0, k-1] */;
Size n_groups = 3;  // k groups

// Pre-allocate output
Size n_features = matrix.rows();
Array<Real> F_stats(n_features);
Array<Real> p_values(n_features);

// Compute one-way ANOVA
scl::kernel::stat::oneway_anova::oneway_anova(
    matrix, group_ids, n_groups,
    F_stats, p_values
);

// Interpret results
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05) {
        std::cout << "Feature " << i 
                  << ": F = " << F_stats[i]
                  << ", p = " << p_values[i]
                  << " (group means differ)\n";
    }
}

// Filter significant features
std::vector<Size> significant_features;
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05) {
        significant_features.push_back(i);
    }
}
```

---

## Utility Functions

### count_k_groups

Count the number of samples in each of k groups.

::: source_code file="scl/kernel/stat/oneway_anova.hpp" symbol="count_k_groups" collapsed
:::

**Complexity**

- Time: O(n_samples)
- Space: O(1) auxiliary

---

## Notes

**When to Use**: One-way ANOVA is appropriate when:
- Data is normally distributed (or sample size is large)
- Homogeneity of variance across groups
- Comparing means across k groups (k >= 2)
- Parametric alternative to Kruskal-Wallis

**Assumptions**:
- **Normality**: Data in each group should be normally distributed
- **Homogeneity of variance**: Variances should be equal across groups
- **Independence**: Samples should be independent

**Interpretation**: 
- Large F statistic indicates group means differ significantly
- P-value < 0.05 suggests at least one group mean differs from others
- Post-hoc tests (e.g., Tukey's HSD) needed to identify which groups differ
- F > 1 suggests between-group variance exceeds within-group variance

**Sparse Data Handling**: The algorithm properly handles sparse zeros:
- Implicit zeros are included in mean and variance calculations
- Total count includes zeros for correct degrees of freedom
- Group means account for all samples (including zeros)

**Sensitivity**: ANOVA is sensitive to:
- Outliers (consider robust alternatives)
- Non-normality (consider Kruskal-Wallis)
- Unequal variances (consider Welch's ANOVA)

**Thread Safety**: Uses thread-local workspace for parallel processing over features, safe for concurrent execution.

**Comparison with Kruskal-Wallis**: 
- **ANOVA**: Parametric, assumes normality, more powerful when assumptions met
- **Kruskal-Wallis**: Non-parametric, robust to outliers, no normality assumption

---

## See Also

- [Kruskal-Wallis](/cpp/kernels/kruskal_wallis) - Non-parametric alternative
- [T-test](/cpp/kernels/ttest) - Two-group parametric test
- [Permutation Test](/cpp/kernels/permutation_stat) - Exact permutation testing

