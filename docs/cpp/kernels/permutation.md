# permutation.hpp

> scl/kernel/permutation.hpp · Permutation testing and multiple comparison correction

## Overview

This file provides kernels for permutation testing and multiple comparison correction, including generic permutation tests, correlation tests, FDR correction methods, and FWER correction methods.

This file provides:
- Generic permutation test with user-defined statistics
- Correlation permutation test
- FDR correction (Benjamini-Hochberg, Benjamini-Yekutieli)
- FWER correction (Bonferroni, Holm-Bonferroni)
- Batch permutation testing
- Utility functions for significance testing

**Header**: `#include "scl/kernel/permutation.hpp"`

---

## Main APIs

### permutation_test

::: source_code file="scl/kernel/permutation.hpp" symbol="permutation_test" collapsed
:::

**Algorithm Description**

Generic permutation test with user-defined test statistic:

1. **Label Preparation**: Copy labels to permutation buffer:
   - Preserve original labels (unchanged)
   - Create working copy for shuffling

2. **Permutation Loop**: For each permutation:
   - Shuffle labels using Fisher-Yates algorithm
   - Compute test statistic on shuffled labels
   - Store statistic in null distribution

3. **P-value Computation**: 
   - Count permutations with statistic >= observed (one-sided)
   - Or count with |statistic| >= |observed| (two-sided)
   - P-value = (count + 1) / (n_permutations + 1)
   - +1 avoids zero p-values

4. **Output**: Return p-value in range [1/(n_perm+1), 1]

**Edge Cases**

- **No permutations**: Returns p-value = 1.0 if n_permutations = 0
- **Perfect match**: If observed equals all permuted statistics, p-value = 1/(n_perm+1)
- **Extreme statistic**: Very extreme observed values get very small p-values
- **Tied statistics**: Handled correctly in counting

**Data Guarantees (Preconditions)**

- `labels.len > 0`
- `compute_statistic` must be callable with `Array<const Index>`
- `n_permutations` in [MIN_PERMUTATIONS, MAX_PERMUTATIONS]
- Labels array is unchanged after function returns

**Complexity Analysis**

- **Time**: O(n_permutations * (n + cost of compute_statistic))
  - Shuffling: O(n) per permutation
  - Statistic computation: Depends on user function
  - P-value computation: O(n_permutations)
- **Space**: O(n_permutations + n) for null distribution and permutation buffer

**Example**

```cpp
#include "scl/kernel/permutation.hpp"

// Define test statistic: mean difference between groups
auto compute_mean_diff = [](scl::Array<const Index> labels) -> Real {
    // Compute mean difference based on labels
    // Return test statistic
};

scl::Array<Index> group_labels = /* ... */;  // Group assignments
Real observed_statistic = /* ... */;         // Observed value

Real p_value = scl::kernel::permutation::permutation_test(
    compute_mean_diff,
    group_labels,
    observed_statistic,
    1000,    // n_permutations
    true,    // two_sided
    42       // seed
);

// p_value is significance of observed statistic
```

---

### permutation_correlation_test

::: source_code file="scl/kernel/permutation.hpp" symbol="permutation_correlation_test" collapsed
:::

**Algorithm Description**

Permutation test for Pearson correlation significance:

1. **Precomputation**: Compute statistics of x (constant across permutations):
   - Mean and standard deviation of x
   - Used for efficient correlation computation

2. **Permutation Loop**: For each permutation:
   - Shuffle indices (permute y while keeping x fixed)
   - Compute correlation with permuted y
   - Store correlation in null distribution

3. **P-value Computation**: 
   - Two-sided test: count |permuted_corr| >= |observed_corr|
   - P-value = (count + 1) / (n_permutations + 1)

4. **Output**: Return two-sided p-value

**Edge Cases**

- **Constant x or y**: If one variable is constant, correlation undefined (handled)
- **Perfect correlation**: If observed = 1.0 or -1.0, p-value very small
- **Zero correlation**: If observed = 0.0, p-value near 1.0
- **Small sample**: Need at least 3 points for correlation

**Data Guarantees (Preconditions)**

- `x.len == y.len`
- `x.len >= 3` (need at least 3 points for correlation)
- x and y are unchanged after function returns

**Complexity Analysis**

- **Time**: O(n_permutations * n) for shuffling and correlation
  - Precomputation: O(n)
  - Permutation loop: O(n_permutations * n)
  - Correlation computation: O(n) per permutation
- **Space**: O(n_permutations + n) for null distribution and indices

**Example**

```cpp
scl::Array<const Real> x = /* ... */;  // First variable
scl::Array<const Real> y = /* ... */; // Second variable
Real observed_corr = /* ... */;        // Observed correlation

Real p_value = scl::kernel::permutation::permutation_correlation_test(
    x,
    y,
    observed_corr,
    1000,  // n_permutations
    42     // seed
);

// p_value tests H0: rho = 0 vs H1: rho != 0
```

---

### batch_permutation_test

::: source_code file="scl/kernel/permutation.hpp" symbol="batch_permutation_test" collapsed
:::

**Algorithm Description**

Parallel permutation test for multiple features (rows of sparse matrix):

1. **Parallel Processing**: Process each feature (row) in parallel:
   - Each thread handles independent features
   - Uses WorkspacePool for thread-local buffers

2. **Per-feature Test**: For each feature:
   - Compute observed mean difference between groups
   - For each permutation:
     - Shuffle group labels (thread-local RNG)
     - Compute permuted mean difference
     - Store in null distribution
   - Compute two-sided p-value

3. **RNG Seeding**: Each thread uses independent RNG:
   - Seed = base_seed + row_index
   - Ensures reproducibility and independence

4. **Output**: Store p-values in output array:
   - `p_values[i]` = p-value for row i
   - Rows with no non-zeros get p-value = 1.0

**Edge Cases**

- **Empty rows**: Rows with no non-zeros get p-value = 1.0
- **Constant rows**: Rows with constant values get p-value = 1.0
- **Small groups**: If one group has < 2 samples, test may be unreliable
- **Many zeros**: Sparse rows handled efficiently

**Data Guarantees (Preconditions)**

- Matrix must be CSR format (IsCSR = true)
- `group_labels.len >= matrix.cols()`
- `p_values.len >= matrix.rows()`
- Group labels contain only 0 and 1

**Complexity Analysis**

- **Time**: O(n_features * n_permutations * avg_nnz_per_row)
  - Per-feature: O(n_permutations * nnz_per_row)
  - Parallelized over features
- **Space**: O(n_threads * (n_permutations + n_samples)) for thread-local buffers

**Example**

```cpp
scl::Sparse<Real, true> matrix = /* ... */;  // [n_features x n_samples]
scl::Array<const Index> group_labels = /* ... */;  // [n_samples], 0 or 1
scl::Array<Real> p_values(n_features);

scl::kernel::permutation::batch_permutation_test(
    matrix,
    group_labels,
    1000,    // n_permutations
    p_values,
    42       // seed
);

// p_values[i] contains p-value for feature i
// Apply FDR correction
scl::Array<Real> q_values(n_features);
scl::kernel::permutation::fdr_correction_bh(p_values, q_values);
```

---

### fdr_correction_bh

::: source_code file="scl/kernel/permutation.hpp" symbol="fdr_correction_bh" collapsed
:::

**Algorithm Description**

Benjamini-Hochberg FDR correction for multiple testing:

1. **Sorting**: Sort p-values in ascending order:
   - Store original indices for mapping back
   - Get rank of each p-value

2. **Adjustment**: From largest to smallest rank:
   - Adjusted = p_value * n / rank
   - Apply cumulative minimum: q = min(adjusted, previous_q)
   - Ensures monotonicity

3. **Mapping**: Map adjusted values back to original order:
   - `q_values[i]` = adjusted p-value for test i

4. **Output**: Store FDR-adjusted q-values:
   - `q_values[i] >= p_values[i]` (always more conservative)
   - Controls false discovery rate at level alpha

**Edge Cases**

- **All significant**: If all p-values very small, all q-values also small
- **All non-significant**: If all p-values large, q-values also large
- **Mixed**: Small p-values get small q-values, large p-values get larger q-values
- **Tied p-values**: Handled correctly in sorting

**Data Guarantees (Preconditions)**

- `q_values.len >= p_values.len`
- `p_values` in [0, 1]
- q_values must be pre-allocated

**Complexity Analysis**

- **Time**: O(n log n) for sorting
  - Sorting: O(n log n)
  - Adjustment: O(n)
  - Mapping: O(n)
- **Space**: O(n) for sorting indices

**Example**

```cpp
scl::Array<const Real> p_values = /* ... */;  // Raw p-values [n]
scl::Array<Real> q_values(n);                 // Pre-allocated

scl::kernel::permutation::fdr_correction_bh(p_values, q_values);

// Filter by FDR
for (Index i = 0; i < n; ++i) {
    if (q_values[i] < 0.05) {
        // Test i is significant at FDR = 0.05
    }
}
```

---

### fdr_correction_by

::: source_code file="scl/kernel/permutation.hpp" symbol="fdr_correction_by" collapsed
:::

**Algorithm Description**

Benjamini-Yekutieli FDR correction for dependent tests:

1. **Harmonic Sum**: Compute correction factor:
   - c_n = 1 + 1/2 + 1/3 + ... + 1/n
   - Used for arbitrary dependence

2. **BH-like Adjustment**: Same as BH, but multiplied by c_n:
   - Adjusted = p_value * c_n * n / rank
   - More conservative than BH

3. **Output**: Store adjusted q-values:
   - Controls FDR under arbitrary dependence
   - More conservative than BH correction

**Edge Cases**

- **Same as BH**: Handles all edge cases similarly
- **More conservative**: Always produces larger q-values than BH
- **Large n**: Harmonic sum grows as log(n), so correction factor increases

**Data Guarantees (Preconditions)**

- Same as fdr_correction_bh
- `q_values.len >= p_values.len`
- `p_values` in [0, 1]

**Complexity Analysis**

- **Time**: O(n log n) for sorting and adjustment
  - Harmonic sum: O(n)
  - Rest same as BH
- **Space**: O(n) for sorting indices

**Example**

```cpp
scl::Array<const Real> p_values = /* ... */;
scl::Array<Real> q_values(n);

// Use BY for dependent tests (e.g., correlated features)
scl::kernel::permutation::fdr_correction_by(p_values, q_values);
```

---

### bonferroni_correction

::: source_code file="scl/kernel/permutation.hpp" symbol="bonferroni_correction" collapsed
:::

**Algorithm Description**

Bonferroni correction for multiple testing (FWER control):

1. **Simple Adjustment**: For each p-value:
   - Adjusted = min(p_value * n, 1.0)
   - Very conservative correction

2. **Output**: Store adjusted p-values:
   - Controls family-wise error rate (FWER)
   - Reject when adjusted_p < alpha

**Edge Cases**

- **Very small p-values**: May become significant after correction
- **Large p-values**: Become even larger (often > 1, clamped to 1)
- **Many tests**: Very conservative when n is large

**Data Guarantees (Preconditions)**

- `adjusted_p_values.len >= p_values.len`
- Adjusted p-values must be pre-allocated

**Complexity Analysis**

- **Time**: O(n) for simple multiplication
- **Space**: O(1) auxiliary

**Example**

```cpp
scl::Array<const Real> p_values = /* ... */;
scl::Array<Real> adjusted_p_values(n);

scl::kernel::permutation::bonferroni_correction(p_values, adjusted_p_values);

// Filter by FWER
for (Index i = 0; i < n; ++i) {
    if (adjusted_p_values[i] < 0.05) {
        // Test i is significant at FWER = 0.05
    }
}
```

---

### holm_correction

::: source_code file="scl/kernel/permutation.hpp" symbol="holm_correction" collapsed
:::

**Algorithm Description**

Holm-Bonferroni step-down correction (less conservative than Bonferroni):

1. **Sorting**: Sort p-values in ascending order

2. **Step-down Adjustment**: For i = 1 to n:
   - Adjusted = p_value[i] * (n - i + 1)
   - Result[i] = max(adjusted, result[i-1])
   - Ensures monotonicity

3. **Mapping**: Map back to original order

4. **Output**: Store adjusted p-values:
   - Controls FWER
   - Uniformly more powerful than Bonferroni

**Edge Cases**

- **Same as Bonferroni for smallest p-value**: First adjustment same
- **Less conservative**: Subsequent p-values less adjusted than Bonferroni
- **Monotonic**: Adjusted values are non-decreasing

**Data Guarantees (Preconditions)**

- `adjusted_p_values.len >= p_values.len`
- Adjusted p-values must be pre-allocated

**Complexity Analysis**

- **Time**: O(n log n) for sorting
- **Space**: O(n) for sorting indices

**Example**

```cpp
scl::Array<const Real> p_values = /* ... */;
scl::Array<Real> adjusted_p_values(n);

scl::kernel::permutation::holm_correction(p_values, adjusted_p_values);

// Filter by FWER (less conservative than Bonferroni)
for (Index i = 0; i < n; ++i) {
    if (adjusted_p_values[i] < 0.05) {
        // Test i is significant at FWER = 0.05
    }
}
```

---

## Utility Functions

### count_significant

Count number of p-values below significance threshold.

::: source_code file="scl/kernel/permutation.hpp" symbol="count_significant" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1) auxiliary

---

### get_significant_indices

Get indices of significant tests.

::: source_code file="scl/kernel/permutation.hpp" symbol="get_significant_indices" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1) auxiliary

---

## Configuration

Default parameters in `scl::kernel::permutation::config`:

- `DEFAULT_N_PERMUTATIONS = 1000`: Default number of permutations
- `MIN_PERMUTATIONS = 100`: Minimum allowed permutations
- `MAX_PERMUTATIONS = 100000`: Maximum allowed permutations
- `PARALLEL_THRESHOLD = 500`: Minimum rows for parallel batch test

---

## Performance Notes

### Permutation Testing

- More permutations increase precision but also computation time
- For p-value resolution of 0.001, use at least 1000 permutations
- Batch testing uses WorkspacePool for efficient memory management

### Multiple Comparison Correction

- FDR methods (BH, BY) are less conservative than FWER methods (Bonferroni, Holm)
- Use FDR when many tests expected to be true
- Use FWER when need strict control of any false positives

---

## See Also

- [Multiple Testing](../multiple_testing)
- [Statistics](../stat)
- [Quality Control](../qc)
