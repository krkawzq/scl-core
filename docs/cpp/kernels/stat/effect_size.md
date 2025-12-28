# effect_size.hpp

> scl/kernel/stat/effect_size.hpp · Effect size computation for group comparisons

## Overview

This file provides comprehensive effect size computation for comparing two groups:

- **Cohen's d**: Standardized mean difference using pooled standard deviation
- **Hedges' g**: Bias-corrected Cohen's d for small samples
- **Glass' delta**: Effect size using control group standard deviation
- **CLES**: Common Language Effect Size (from AUROC)

Effect sizes quantify the magnitude of difference between groups, complementing p-values.

**Header**: `#include "scl/kernel/stat/effect_size.hpp"`

---

## Main APIs

### effect_size

::: source_code file="scl/kernel/stat/effect_size.hpp" symbol="effect_size" collapsed
:::

**Algorithm Description**

Compute effect size for each feature in a sparse matrix:

1. **Partition values by group**: For each feature in parallel
   - Extract non-zero values and group assignments
   - Partition into group 0 and group 1
   - Accumulate moments (sum, sum-of-squares) during partitioning

2. **Compute group statistics**: For each group
   - Mean: mean = sum / n (including zeros for sparse data)
   - Variance: var = (sum_sq - sum * mean) / (n - 1)
   - Account for implicit zeros in sparse matrices

3. **Apply effect size formula** based on type:
   - **Cohen's d**: d = (mean2 - mean1) / sqrt(pooled_var)
   - **Hedges' g**: g = d * J (bias correction factor)
   - **Glass' delta**: delta = (mean2 - mean1) / sd1
   - **CLES**: Requires AUROC, returns 0.5 (not directly computable)

4. **Handle edge cases**: Zero variance, small samples, etc.

**Edge Cases**

- **Zero pooled variance**: Effect size = 0 (no variation)
- **Empty groups**: Throws error (checked in preconditions)
- **Small samples**: Hedges' g provides bias correction
- **Zero means**: Effect size = 0
- **Sparse data**: Implicit zeros included in mean/variance computation

**Data Guarantees (Preconditions)**

- Both groups have at least one member
- `out_effect_size.len >= matrix.primary_dim()`
- Matrix is valid sparse format
- `group_ids` has size equal to matrix secondary dimension

**Complexity Analysis**

- **Time**: O(features * nnz_per_feature) parallelized over features
- **Space**: O(threads * max_row_length) for temporary buffers

**Example**

```cpp
#include "scl/kernel/stat/effect_size.hpp"

// Expression matrix (features x samples)
scl::Sparse<Real, true> matrix = /* ... */;
scl::Array<int32_t> group_ids = /* ... */;  // Binary group assignment

// Pre-allocate output
scl::Array<Real> effect_sizes(matrix.rows());

// Compute Cohen's d effect size
scl::kernel::stat::effect_size::effect_size(
    matrix, group_ids, effect_sizes,
    scl::kernel::stat::effect_size::EffectSizeType::CohensD
);

// Interpret effect sizes (Cohen's conventions):
// |d| < 0.2: negligible
// 0.2 <= |d| < 0.5: small
// 0.5 <= |d| < 0.8: medium
// |d| >= 0.8: large
```

---

### compute_cohens_d

::: source_code file="scl/kernel/stat/effect_size.hpp" symbol="compute_cohens_d" collapsed
:::

**Algorithm Description**

Compute Cohen's d effect size from group statistics:

**Formula**: d = (mean2 - mean1) / sqrt(pooled_variance)

Where pooled_variance = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)

1. Compute pooled variance using Bessel's correction
2. Take square root to get pooled standard deviation
3. Compute d = (mean2 - mean1) / pooled_sd
4. Return 0 if pooled_sd < SIGMA_MIN (numerical stability)

**Edge Cases**

- **Zero pooled variance**: Returns 0
- **Very small pooled variance**: Returns 0 (below SIGMA_MIN threshold)
- **Equal means**: Returns 0
- **Small samples**: May be biased (use Hedges' g instead)

**Complexity**

- Time: O(1)
- Space: O(1)

---

### compute_hedges_g

::: source_code file="scl/kernel/stat/effect_size.hpp" symbol="compute_hedges_g" collapsed
:::

**Algorithm Description**

Compute Hedges' g (bias-corrected Cohen's d):

**Formula**: g = d * J

Where J = 1 - 3 / (4*df - 1), df = n1 + n2 - 2

1. Compute Cohen's d first
2. Compute degrees of freedom: df = n1 + n2 - 2
3. Compute bias correction factor: J = 1 - 3 / (4*df - 1)
4. Apply correction: g = d * J

Hedges' correction reduces small-sample bias in Cohen's d, making it more accurate for n < 20.

**Edge Cases**

- **Very small df**: J approaches 1, g ≈ d
- **Large df**: J ≈ 1, g ≈ d
- **df < 1**: Undefined, returns 0

**Complexity**

- Time: O(1)
- Space: O(1)

---

### compute_glass_delta

::: source_code file="scl/kernel/stat/effect_size.hpp" symbol="compute_glass_delta" collapsed
:::

**Algorithm Description**

Compute Glass' delta effect size:

**Formula**: delta = (mean2 - mean1) / sd1

Uses only the control group (group 1) standard deviation, not the pooled standard deviation. Useful when treatment affects variance.

**Edge Cases**

- **Zero control variance**: Returns 0
- **Equal means**: Returns 0

**Complexity**

- Time: O(1)
- Space: O(1)

---

### auroc_to_cles

::: source_code file="scl/kernel/stat/effect_size.hpp" symbol="auroc_to_cles" collapsed
:::

**Algorithm Description**

Convert AUROC to Common Language Effect Size (CLES):

**Interpretation**: CLES = probability that a randomly selected value from group 2 exceeds a randomly selected value from group 1.

AUROC and CLES are mathematically equivalent, so this function simply returns the AUROC value.

**Complexity**

- Time: O(1)
- Space: O(1)

---

### compute_effect_size

::: source_code file="scl/kernel/stat/effect_size.hpp" symbol="compute_effect_size" collapsed
:::

**Algorithm Description**

Generic effect size computation dispatcher:

1. Switch on effect size type
2. Call appropriate computation function
3. For CLES, returns 0.5 (requires AUROC, not directly computable from means/variances)

**Complexity**

- Time: O(1)
- Space: O(1)

---

### ttest_with_effect_size

::: source_code file="scl/kernel/stat/effect_size.hpp" symbol="ttest_with_effect_size" collapsed
:::

**Algorithm Description**

Combined t-test and effect size computation in single pass:

1. **Partition with moment accumulation**: Single pass through data
   - Partition values by group
   - Accumulate sums and sum-of-squares for both groups
   - More efficient than separate t-test and effect size calls

2. **Compute t-test statistics**: 
   - Means and variances from accumulated moments
   - T-statistic and p-value (Welch's or standard t-test)

3. **Compute log2 fold change**: log2((mean1 + eps) / (mean0 + eps))

4. **Compute effect size**: Using specified type (Cohen's d, Hedges' g, etc.)

This function is more efficient than calling `ttest()` and `effect_size()` separately, as partitioning and variance computation are done once.

**Edge Cases**

- **Same as ttest()**: All edge cases from t-test apply
- **Same as effect_size()**: All edge cases from effect size apply

**Data Guarantees (Preconditions)**

- Same as `ttest()` and `effect_size()` combined
- All output arrays have sufficient capacity

**Complexity Analysis**

- **Time**: O(features * nnz_per_feature) (same as separate calls, but single pass)
- **Space**: O(threads * max_row_length)

**Example**

```cpp
#include "scl/kernel/stat/effect_size.hpp"

scl::Sparse<Real, true> matrix = /* ... */;
scl::Array<int32_t> group_ids = /* ... */;

scl::Array<Real> t_stats(matrix.rows());
scl::Array<Real> p_values(matrix.rows());
scl::Array<Real> log2_fc(matrix.rows());
scl::Array<Real> effect_sizes(matrix.rows());

// Compute all statistics in one pass
scl::kernel::stat::effect_size::ttest_with_effect_size(
    matrix, group_ids,
    t_stats, p_values, log2_fc, effect_sizes,
    scl::kernel::stat::effect_size::EffectSizeType::HedgesG,
    true  // Use Welch's t-test
);
```

---

## Effect Size Types

The `EffectSizeType` enum provides different effect size measures:

- `CohensD`: Standardized mean difference (most common)
- `HedgesG`: Bias-corrected Cohen's d (recommended for small samples)
- `GlassDelta`: Uses control group SD (useful when variance differs)
- `CLES`: Common Language Effect Size (from AUROC)

---

## Notes

- Effect sizes complement p-values by quantifying magnitude of difference
- Cohen's conventions: |d| < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), ≥0.8 (large)
- Hedges' g is recommended for small samples (n < 20) due to bias correction
- Glass' delta is useful when treatment affects variance
- CLES provides intuitive interpretation as probability
- All operations are parallelized over features and thread-safe

## See Also

- [AUROC Computation](./auroc)
- [T-Test](../ttest)
- [Statistical Tests](../stat)

