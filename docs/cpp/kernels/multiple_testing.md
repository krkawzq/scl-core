# Multiple Testing Correction

Multiple testing correction methods for controlling false discovery rate (FDR) and family-wise error rate (FWER).

## Overview

Multiple testing correction kernels provide:

- **FDR Control** - Benjamini-Hochberg, Storey's q-value, Benjamini-Yekutieli
- **FWER Control** - Bonferroni, Holm-Bonferroni, Hochberg
- **Local FDR** - Kernel density estimation-based local FDR
- **Empirical FDR** - Permutation-based FDR estimation
- **Utilities** - Significant indices, negative log10 transformation, Fisher's combination

## FDR Correction

### benjamini_hochberg

Apply Benjamini-Hochberg FDR correction:

```cpp
#include "scl/kernel/multiple_testing.hpp"

Array<Real> p_values = /* ... */;             // Input p-values [n_tests]
Array<Real> adjusted(n_tests);                 // Pre-allocated output

scl::kernel::multiple_testing::benjamini_hochberg(
    p_values, adjusted,
    fdr_level = 0.05                          // FDR control level
);

// adjusted[i] contains BH-adjusted p-value
```

**Parameters:**
- `p_values`: Input p-values, size = n_tests
- `adjusted_p_values`: Output adjusted p-values, must be pre-allocated, size = n_tests
- `fdr_level`: FDR control level (default: 0.05)

**Postconditions:**
- `adjusted_p_values[i]` contains BH-adjusted p-value
- Adjusted values are monotonic and in [0, 1]

**Algorithm:**
1. Sort p-values in ascending order
2. Compute adjusted p = p * n / rank
3. Enforce monotonicity from right to left
4. Map back to original order

**Complexity:**
- Time: O(n log n) for sorting
- Space: O(n) auxiliary

**Use cases:**
- Control FDR in differential expression analysis
- Genome-wide association studies
- High-throughput screening

### storey_qvalue

Estimate q-values using Storey's method:

```cpp
Array<Real> p_values = /* ... */;
Array<Real> q_values(n_tests);                // Pre-allocated output

scl::kernel::multiple_testing::storey_qvalue(
    p_values, q_values,
    lambda = 0.5                              // Pi0 estimation parameter
);

// q_values[i] contains estimated q-value
```

**Parameters:**
- `p_values`: Input p-values, size = n_tests
- `q_values`: Output q-values, must be pre-allocated, size = n_tests
- `lambda`: Tuning parameter for pi0 estimation (default: 0.5)

**Postconditions:**
- `q_values[i]` contains estimated q-value
- All q-values in [0, 1]

**Algorithm:**
1. Estimate pi0 (proportion of true nulls) using lambda
2. Sort p-values
3. Compute q-values from right to left with monotonicity

**Complexity:**
- Time: O(n log n) for sorting
- Space: O(n) auxiliary

## FWER Control

### bonferroni

Apply Bonferroni correction:

```cpp
Array<Real> p_values = /* ... */;
Array<Real> adjusted(n_tests);                 // Pre-allocated output

scl::kernel::multiple_testing::bonferroni(p_values, adjusted);

// adjusted[i] = min(p_values[i] * n, 1.0)
```

**Parameters:**
- `p_values`: Input p-values, size = n_tests
- `adjusted_p_values`: Output Bonferroni-adjusted p-values, must be pre-allocated, size = n_tests

**Postconditions:**
- `adjusted_p_values[i] = min(p_values[i] * n, 1.0)`
- All adjusted values in [0, 1]

**Complexity:**
- Time: O(n)
- Space: O(1) auxiliary

**Thread Safety:**
- Safe - uses SIMD-optimized operations

### holm_bonferroni

Apply Holm-Bonferroni step-down correction:

```cpp
Array<Real> adjusted(n_tests);
scl::kernel::multiple_testing::holm_bonferroni(p_values, adjusted);

// More powerful than Bonferroni, controls FWER
```

**Parameters:**
- `p_values`: Input p-values, size = n_tests
- `adjusted_p_values`: Output Holm-adjusted p-values, must be pre-allocated, size = n_tests

**Postconditions:**
- `adjusted_p_values[i]` contains Holm-adjusted p-value
- More powerful than Bonferroni, controls FWER

**Algorithm:**
1. Sort p-values
2. For rank i: adjusted = p * (n - i + 1)
3. Enforce monotonicity

**Complexity:**
- Time: O(n log n) for sorting
- Space: O(n) auxiliary

## Local FDR

### local_fdr

Estimate local false discovery rate using kernel density estimation:

```cpp
Array<Real> p_values = /* ... */;
Array<Real> lfdr(n_tests);                    // Pre-allocated output

scl::kernel::multiple_testing::local_fdr(p_values, lfdr);

// lfdr[i] contains local FDR estimate
```

**Parameters:**
- `p_values`: Input p-values, size = n_tests
- `lfdr`: Output local FDR estimates, must be pre-allocated, size = n_tests

**Postconditions:**
- `lfdr[i]` contains local FDR estimate for test i
- All lfdr values in [0, 1]

**Algorithm:**
1. Transform p-values to z-scores
2. Estimate density f(z) using KDE
3. Compute f0(z) (null density, standard normal)
4. Estimate pi0
5. Compute lfdr = pi0 * f0(z) / f(z)

**Complexity:**
- Time: O(n^2) for KDE estimation
- Space: O(n) auxiliary

## Utilities

### significant_indices

Get indices of tests with p-values below threshold:

```cpp
Array<Real> p_values = /* ... */;
Array<Index> indices(n_tests);                // Pre-allocated
Size count;

scl::kernel::multiple_testing::significant_indices(
    p_values, threshold = 0.05,
    indices.ptr, count
);

// indices[0..count-1] contains significant test indices
```

### neglog10_pvalues

Compute negative log10 of p-values for visualization:

```cpp
Array<Real> p_values = /* ... */;
Array<Real> neglog_p(n_tests);                // Pre-allocated

scl::kernel::multiple_testing::neglog10_pvalues(p_values, neglog_p);

// neglog_p[i] = -log10(p_values[i])
```

### fisher_combine

Combine p-values using Fisher's method:

```cpp
Array<Real> p_values = /* ... */;

Real chi2 = scl::kernel::multiple_testing::fisher_combine(p_values);

// Returns chi-squared test statistic
// Statistic follows chi2(2*n) under null
```

## Configuration

### Default Parameters

```cpp
namespace config {
    constexpr Real DEFAULT_FDR_LEVEL = Real(0.05);
    constexpr Real DEFAULT_LAMBDA = Real(0.5);
    constexpr Real MIN_PVALUE = Real(1e-300);
    constexpr Real MAX_PVALUE = Real(1.0);
    constexpr Size SPLINE_KNOTS = 10;
    constexpr Size MIN_TESTS_FOR_STOREY = 100;
}
```

---

::: tip FDR vs. FWER
FDR methods (BH, Storey) are less conservative and more powerful for exploratory analysis. FWER methods (Bonferroni, Holm) are stricter and appropriate for confirmatory analysis where any false positive is unacceptable.
:::

