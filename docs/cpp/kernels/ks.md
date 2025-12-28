# ks.hpp

> scl/kernel/stat/ks.hpp · Kolmogorov-Smirnov two-sample test

## Overview

This file provides the Kolmogorov-Smirnov (KS) two-sample test for comparing distributions between two groups. It tests whether two samples come from the same distribution by comparing their empirical cumulative distribution functions (ECDFs).

**Header**: `#include "scl/kernel/stat/ks.hpp"`

---

## Main APIs

### ks_test

::: source_code file="scl/kernel/stat/ks.hpp" symbol="ks_test" collapsed
:::

**Algorithm Description**

Compute two-sample Kolmogorov-Smirnov test for each feature:

1. For each feature in parallel:
   - Partition non-zero values by group (group 0 and group 1)
   - Sort each group using VQSort (high-performance sorting)
   - Merge sorted arrays while tracking ECDF difference:
     - Handle sparse zeros explicitly in ECDF computation
     - Values < 0: contribute to ECDF before zero point
     - Zeros (implicit): create jump at x=0 in ECDF
     - Values > 0: contribute to ECDF after zero point
   - Compute D statistic:
     - D = max |F1(x) - F2(x)| over all x
     - Where F1 and F2 are ECDFs of groups 1 and 2
   - Compute p-value via Kolmogorov distribution:
     - n_eff = n1 * n2 / (n1 + n2)
     - lambda = (sqrt(n_eff) + 0.12 + 0.11/sqrt(n_eff)) * D
     - P(D > d) = 2 * sum_{k=1}^inf (-1)^{k+1} * exp(-2*k^2*lambda^2)
     - Uses series expansion with 100-term limit

2. The D statistic measures maximum difference between ECDFs

3. P-value uses asymptotic Kolmogorov distribution (accurate for n1, n2 >= 25)

**Edge Cases**

- **Empty group 0 or group 1**: Throws ArgumentError
- **All values identical in both groups**: D = 0, p-value = 1.0
- **No overlap between groups**: D = 1.0, p-value approaches 0
- **Single value per group**: D computed but may have low power
- **Sparse zeros**: Explicitly handled in ECDF computation

**Data Guarantees (Preconditions)**

- `matrix.secondary_dim() == group_ids.len` (sample dimension must match)
- Output arrays must have size >= matrix.primary_dim()
- Both groups (0 and 1) must have at least one member
- `group_ids` must contain only 0 or 1 values
- Matrix must be valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(features * nnz_per_feature * log(nnz_per_feature)) - sorting dominates, then ECDF computation
- **Space**: O(threads * max_row_length) - thread-local workspace for sorting and ECDF tracking

**Example**

```cpp
#include "scl/kernel/stat/ks.hpp"

// Prepare data
Sparse<Real, true> matrix = /* features x samples */;
Array<int32_t> group_ids = /* binary group assignment (0 or 1) */;

// Pre-allocate output
Size n_features = matrix.rows();
Array<Real> D_stats(n_features);
Array<Real> p_values(n_features);

// Compute KS test
scl::kernel::stat::ks::ks_test(
    matrix, group_ids,
    D_stats, p_values
);

// Interpret results
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05) {
        std::cout << "Feature " << i 
                  << ": D = " << D_stats[i]
                  << ", p = " << p_values[i]
                  << " (distributions differ)\n";
    }
}

// Filter features with different distributions
std::vector<Size> different_features;
for (Size i = 0; i < n_features; ++i) {
    if (p_values[i] < 0.05 && D_stats[i] > 0.3) {
        // D > 0.3 indicates substantial difference
        different_features.push_back(i);
    }
}
```

---

## Notes

**When to Use**: Kolmogorov-Smirnov test is appropriate when:
- Comparing distributions between two groups
- No assumptions about distribution shape
- Detecting any difference in distributions (location, scale, shape)
- Non-parametric alternative to t-test

**Interpretation**: 
- D statistic ranges from 0 to 1
- D = 0: distributions are identical
- D = 1: distributions have no overlap
- Large D with small p-value: distributions differ significantly
- D > 0.3 typically indicates substantial difference

**Sparse Data Handling**: The algorithm explicitly handles sparse zeros in ECDF computation:
- Negative values contribute to ECDF before zero
- Implicit zeros create a jump at x=0
- Positive values contribute to ECDF after zero
- This ensures correct ECDF for sparse expression data

**Accuracy**: P-value computation uses asymptotic Kolmogorov distribution:
- Accurate for n1, n2 >= 25
- Uses series expansion with 100-term limit
- For smaller samples, consider permutation test

**Thread Safety**: Uses thread-local workspace for parallel processing over features, safe for concurrent execution.

**Comparison with Other Tests**:
- **KS test**: Detects any distribution difference, non-parametric
- **t-test**: Assumes normality, detects mean difference
- **Mann-Whitney U**: Non-parametric, detects location shift

---

## See Also

- [Mann-Whitney U](/cpp/kernels/mwu) - Non-parametric location test
- [Permutation Test](/cpp/kernels/permutation_stat) - Exact permutation testing
- [T-test](/cpp/kernels/ttest) - Parametric mean comparison

