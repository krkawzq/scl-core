# multiple_testing.hpp

> scl/kernel/multiple_testing.hpp · Multiple testing correction methods for controlling false discovery rate

## Overview

This file provides various methods for correcting p-values when performing multiple hypothesis tests. Multiple testing correction is essential to control false discovery rate (FDR) or family-wise error rate (FWER) when testing many hypotheses simultaneously.

Key features:
- Benjamini-Hochberg (BH) FDR control
- Bonferroni and Holm-Bonferroni FWER control
- Storey's q-value estimation
- Local FDR estimation using kernel density
- Empirical FDR from permutations
- Benjamini-Yekutieli correction for dependent tests

**Header**: `#include "scl/kernel/multiple_testing.hpp"`

---

## Main APIs

### benjamini_hochberg

Apply Benjamini-Hochberg FDR correction to p-values. This is the most commonly used FDR control method.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="benjamini_hochberg" collapsed
:::

**Algorithm Description**

The Benjamini-Hochberg procedure controls the false discovery rate:

1. Sort p-values in ascending order
2. For rank i, compute adjusted p = p_value[i] * n / i
3. Enforce monotonicity from right to left: if adjusted[i] > adjusted[i+1], set adjusted[i] = adjusted[i+1]
4. Map back to original order

This ensures that the expected proportion of false discoveries among rejected hypotheses is controlled at the specified FDR level.

**Edge Cases**

- **All p-values > FDR level**: All adjusted p-values remain above threshold, no discoveries
- **All p-values very small**: Adjusted values may still be small if tests are highly significant
- **Ties in p-values**: Handled correctly by sorting

**Data Guarantees (Preconditions)**

- `p_values.len == adjusted_p_values.len`
- All p-values in [0, 1]
- `fdr_level` typically 0.05

**Complexity Analysis**

- **Time**: O(n log n) for sorting
- **Space**: O(n) auxiliary for sorting indices

**Example**

```cpp
#include "scl/kernel/multiple_testing.hpp"

Array<Real> p_values(n_tests);  // Raw p-values from tests
Array<Real> adjusted(n_tests);  // Pre-allocated output

// Apply BH correction with FDR level 0.05
scl::kernel::multiple_testing::benjamini_hochberg(
    p_values, adjusted, 0.05
);

// Find significant tests (adjusted p < 0.05)
Index indices[n_tests];
Size count;
scl::kernel::multiple_testing::significant_indices(
    adjusted, 0.05, indices, count
);
```

---

### storey_qvalue

Estimate q-values using Storey's method with pi0 (proportion of true nulls) estimation.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="storey_qvalue" collapsed
:::

**Algorithm Description**

Storey's q-value is a more powerful alternative to BH-adjusted p-values:

1. Estimate pi0 (proportion of true nulls) using lambda tuning parameter:
   pi0 = (# p-values > lambda) / ((1 - lambda) * n)
2. Sort p-values in ascending order
3. Compute q-values from right to left with monotonicity:
   q[i] = min(pi0 * p[i] * n / i, q[i+1])
4. Map back to original order

The lambda parameter (default 0.5) controls the pi0 estimation. Storey's method is more powerful than BH when many tests are truly null.

**Edge Cases**

- **All p-values small**: pi0 estimation may be conservative
- **Very few nulls**: pi0 may be overestimated, making q-values more conservative
- **Lambda too high**: May overestimate pi0

**Data Guarantees (Preconditions)**

- `p_values.len == q_values.len`
- All p-values in [0, 1]
- `lambda` in (0, 1), typically 0.5

**Complexity Analysis**

- **Time**: O(n log n) for sorting
- **Space**: O(n) auxiliary

**Example**

```cpp
Array<Real> p_values(n_tests);
Array<Real> q_values(n_tests);

// Estimate q-values with default lambda = 0.5
scl::kernel::multiple_testing::storey_qvalue(p_values, q_values);

// Or with custom lambda
Real lambda = 0.75;
scl::kernel::multiple_testing::storey_qvalue(p_values, q_values, lambda);

// q_values[i] estimates the FDR if test i is called significant
```

---

### local_fdr

Estimate local false discovery rate using kernel density estimation on z-scores.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="local_fdr" collapsed
:::

**Algorithm Description**

Local FDR estimates the probability that a specific test is a false discovery:

1. Transform p-values to z-scores: z = -Phi^(-1)(p)
2. Estimate density f(z) using kernel density estimation (KDE)
3. Compute f0(z) (null density, standard normal N(0,1))
4. Estimate pi0 from the right tail of the z-score distribution
5. Compute lfdr = pi0 * f0(z) / f(z)

Local FDR provides test-specific FDR estimates rather than global correction.

**Edge Cases**

- **All p-values very large**: z-scores near 0, lfdr near pi0
- **Very small p-values**: z-scores very large, lfdr very small
- **KDE bandwidth**: Affects density estimation quality

**Data Guarantees (Preconditions)**

- `p_values.len == lfdr.len`
- All p-values in [0, 1]

**Complexity Analysis**

- **Time**: O(n^2) for KDE estimation
- **Space**: O(n) auxiliary

**Example**

```cpp
Array<Real> p_values(n_tests);
Array<Real> lfdr_values(n_tests);

scl::kernel::multiple_testing::local_fdr(p_values, lfdr_values);

// lfdr_values[i] is the estimated probability that test i is a false discovery
// Lower lfdr indicates higher confidence the test is truly significant
```

---

### bonferroni

Apply Bonferroni correction (multiply by number of tests). Most conservative FWER control.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="bonferroni" collapsed
:::

**Algorithm Description**

Bonferroni correction controls family-wise error rate (FWER):

1. Multiply each p-value by n (number of tests)
2. Clamp to [0, 1]

This is the most conservative correction method, controlling the probability of at least one false discovery.

**Edge Cases**

- **n * p > 1**: Clamped to 1.0
- **Very large n**: Most p-values become 1.0, very conservative

**Data Guarantees (Preconditions)**

- `p_values.len == adjusted_p_values.len`
- All p-values in [0, 1]

**Complexity Analysis**

- **Time**: O(n) with SIMD-optimized operations
- **Space**: O(1) auxiliary

**Example**

```cpp
Array<Real> p_values(n_tests);
Array<Real> adjusted(n_tests);

scl::kernel::multiple_testing::bonferroni(p_values, adjusted);

// adjusted[i] = min(p_values[i] * n, 1.0)
```

---

### holm_bonferroni

Apply Holm-Bonferroni step-down correction. More powerful than Bonferroni while still controlling FWER.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="holm_bonferroni" collapsed
:::

**Algorithm Description**

Holm-Bonferroni is a step-down procedure:

1. Sort p-values in ascending order
2. For rank i: adjusted = p_value[i] * (n - i + 1)
3. Enforce monotonicity from left to right

More powerful than Bonferroni but still controls FWER.

**Edge Cases**

- **Step-down nature**: Rejects hypotheses sequentially from smallest p-value
- **More powerful than Bonferroni**: Particularly for tests with small p-values

**Data Guarantees (Preconditions)**

- `p_values.len == adjusted_p_values.len`
- All p-values in [0, 1]

**Complexity Analysis**

- **Time**: O(n log n) for sorting
- **Space**: O(n) auxiliary

**Example**

```cpp
Array<Real> p_values(n_tests);
Array<Real> adjusted(n_tests);

scl::kernel::multiple_testing::holm_bonferroni(p_values, adjusted);

// More powerful than Bonferroni, still controls FWER
```

---

### hochberg

Apply Hochberg step-up correction. More powerful than Holm while controlling FWER.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="hochberg" collapsed
:::

**Algorithm Description**

Hochberg procedure is a step-up method:

1. Sort p-values in ascending order
2. For rank i: adjusted = p_value[i] * (n - i + 1)
3. Enforce monotonicity from right to left

More powerful than both Bonferroni and Holm, controls FWER under independence.

**Edge Cases**

- **Step-up nature**: More aggressive than step-down (Holm)
- **Independence assumption**: May not control FWER under dependency

**Data Guarantees (Preconditions)**

- `p_values.len == adjusted_p_values.len`
- All p-values in [0, 1]

**Complexity Analysis**

- **Time**: O(n log n) for sorting
- **Space**: O(n) auxiliary

**Example**

```cpp
Array<Real> p_values(n_tests);
Array<Real> adjusted(n_tests);

scl::kernel::multiple_testing::hochberg(p_values, adjusted);

// Most powerful FWER control under independence
```

---

### benjamini_yekutieli

Apply Benjamini-Yekutieli FDR correction. Works under arbitrary dependency.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="benjamini_yekutieli" collapsed
:::

**Algorithm Description**

Similar to BH but uses correction factor C(n) = sum(1/i) for i=1..n to handle dependency:

1. Compute correction factor C(n)
2. Sort p-values
3. Apply BH-like procedure with correction factor

More conservative than BH but works under arbitrary dependency structure.

**Edge Cases**

- **Independent tests**: Slightly more conservative than BH
- **Dependent tests**: Correctly controls FDR unlike standard BH

**Data Guarantees (Preconditions)**

- `p_values.len == adjusted_p_values.len`
- All p-values in [0, 1]

**Complexity Analysis**

- **Time**: O(n log n) for sorting
- **Space**: O(n) auxiliary

**Example**

```cpp
Array<Real> p_values(n_tests);
Array<Real> adjusted(n_tests);

scl::kernel::multiple_testing::benjamini_yekutieli(p_values, adjusted);

// Conservative FDR control that works under dependency
```

---

### empirical_fdr

Estimate FDR using permutation-based empirical null distribution.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="empirical_fdr" collapsed
:::

**Algorithm Description**

Computes FDR from permutation test results:

1. For each test i, count number of permutations where permuted_score >= observed_score[i]
2. Compute FDR[i] = (permutation_count + 1) / (n_permutations + 1)

This provides empirical FDR estimates without distributional assumptions.

**Edge Cases**

- **No permutations exceed observed**: FDR = 1/(n_perm+1)
- **All permutations exceed observed**: FDR = 1.0
- **Few permutations**: Less reliable estimates

**Data Guarantees (Preconditions)**

- `observed_scores.len == fdr.len`
- All permuted_scores arrays have same length as observed_scores

**Complexity Analysis**

- **Time**: O(n_tests * n_permutations), parallelized over tests
- **Space**: O(1) auxiliary

**Example**

```cpp
Array<Real> observed(n_tests);
std::vector<Array<Real>> permuted(n_permutations);
for (auto& p : permuted) {
    p = Array<Real>(n_tests);
}

// Fill with permutation results...

Array<Real> fdr(n_tests);
scl::kernel::multiple_testing::empirical_fdr(observed, permuted, fdr);

// fdr[i] is empirical FDR estimate for test i
```

---

## Utility Functions

### significant_indices

Get indices of tests with p-values below threshold.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="significant_indices" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1) auxiliary

---

### neglog10_pvalues

Compute negative log10 of p-values for visualization.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="neglog10_pvalues" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1) auxiliary

---

### fisher_combine

Combine p-values using Fisher's method. Returns chi-squared test statistic.

::: source_code file="scl/kernel/multiple_testing.hpp" symbol="fisher_combine" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1) auxiliary

---

## Configuration

Default configuration values in `scl::kernel::multiple_testing::config`:

- `DEFAULT_FDR_LEVEL = 0.05`
- `DEFAULT_LAMBDA = 0.5` (for Storey's method)
- `MIN_PVALUE = 1e-300`
- `MAX_PVALUE = 1.0`
- `SPLINE_KNOTS = 10`
- `MIN_TESTS_FOR_STOREY = 100`

## See Also

- [MWU Test](/cpp/kernels/mwu) - Mann-Whitney U test producing p-values
- [T-test](/cpp/kernels/ttest) - T-test producing p-values
