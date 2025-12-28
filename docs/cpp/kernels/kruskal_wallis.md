# kruskal_wallis.hpp

> scl/kernel/stat/kruskal_wallis.hpp · Kruskal-Wallis H test for non-parametric ANOVA

## Overview

This file provides the Kruskal-Wallis H test, a non-parametric alternative to one-way ANOVA. It tests whether samples from different groups come from the same distribution without assuming normality.

**Header**: `#include "scl/kernel/stat/kruskal_wallis.hpp"`

---

## Main APIs

### kruskal_wallis

::: source_code file="scl/kernel/stat/kruskal_wallis.hpp" symbol="kruskal_wallis" collapsed
:::

**Algorithm Description**

Compute Kruskal-Wallis H test for k groups (non-parametric ANOVA):

1. For each feature in parallel:
   - Extract non-zero values with group tags
   - Sort values using argsort (preserving index mapping)
   - Compute rank sums per group with tie handling:
     - Assign average rank to tied values
     - Accumulate ranks for each group
   - Compute H statistic:
     - H = 12/(N(N+1)) * sum(R_i^2/n_i) - 3(N+1)
     - Where R_i is rank sum for group i, n_i is size of group i, N is total samples
   - Apply tie correction:
     - H_corrected = H / (1 - sum(t^3-t)/(N^3-N))
     - Where t is number of ties at each rank
   - Compute p-value from chi-squared distribution with df = k - 1

2. The H statistic measures whether groups differ significantly

3. P-value uses chi-squared approximation (accurate for large samples)

**Edge Cases**

- **n_groups < 2**: Throws ArgumentError
- **Fewer than 2 groups have members**: Throws ArgumentError
- **All values identical**: H = 0, p-value = 1.0
- **Single value per group**: H statistic computed but may have low power
- **Negative group_ids**: Ignored (treated as missing data)
- **Empty features**: H = NaN, p-value = NaN

**Data Guarantees (Preconditions)**

- `n_groups >= 2` (must have at least 2 groups)
- `group_ids[i]` in range [0, n_groups) or negative (ignored)
- At least 2 groups must have members
- Output arrays must have size >= matrix.primary_dim()
- Matrix must be valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(features * nnz_per_feature * log(nnz_per_feature)) - sorting dominates, then rank computation
- **Space**: O(threads * (max_row_length + n_groups)) - thread-local workspace for sorting and group statistics

**Example**

```cpp
#include "scl/kernel/stat/kruskal_wallis.hpp"

// Prepare data
Sparse<Real, true> matrix = /* features x samples */;
Array<int32_t> group_ids = /* group assignment [0, k-1] */;
Size n_groups = 3;  // k groups

// Pre-allocate output
Size n_features = matrix.rows();
Array<Real> H_stats(n_features);
Array<Real> p_values(n_features);

// Compute Kruskal-Wallis test
scl::kernel::stat::kruskal_wallis::kruskal_wallis(
    matrix, group_ids, n_groups,
    H_stats, p_values
);

// Interpret results
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05) {
        std::cout << "Feature " << i 
                  << ": H = " << H_stats[i]
                  << ", p = " << p_values[i]
                  << " (significant)\n";
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

## Notes

**When to Use**: Kruskal-Wallis is appropriate when:
- Data is not normally distributed
- Sample sizes are small
- Outliers are present
- Homogeneity of variance assumption is violated

**Interpretation**: 
- Large H statistic indicates groups differ significantly
- P-value < 0.05 suggests at least one group differs from others
- Post-hoc tests (e.g., Dunn's test) needed to identify which groups differ

**Tie Handling**: The algorithm properly handles tied values by assigning average ranks and applying tie correction to the H statistic.

**Thread Safety**: Uses thread-local workspace for parallel processing over features, safe for concurrent execution.

**Comparison with ANOVA**: 
- Kruskal-Wallis: Non-parametric, robust to outliers, no normality assumption
- One-way ANOVA: Parametric, assumes normality, more powerful when assumptions met

---

## See Also

- [One-way ANOVA](/cpp/kernels/oneway_anova) - Parametric alternative
- [Mann-Whitney U](/cpp/kernels/mwu) - Two-group non-parametric test
- [Permutation Test](/cpp/kernels/permutation_stat) - Exact permutation testing

