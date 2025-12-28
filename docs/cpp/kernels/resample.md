# resample.hpp

> scl/kernel/resample.hpp · Resampling operations with fast RNG for count data

## Overview

This file provides resampling operations for count data, particularly useful for single-cell RNA sequencing data. Resampling can be used for downsampling to fixed counts, simulating technical noise, or normalizing across samples.

Key features:
- Downsampling to target counts (fixed or variable per row)
- Binomial and Poisson resampling
- In-place operations
- Deterministic with seed control
- Fast RNG implementation

**Header**: `#include "scl/kernel/resample.hpp"`

---

## Main APIs

### downsample

::: source_code file="scl/kernel/resample.hpp" symbol="downsample" collapsed
:::

**Algorithm Description**

Downsample each row to a target total count using binomial sampling:

1. **For each row in parallel**:
   - Compute current total count: `current_sum = sum(row)`
   - If `current_sum <= target_sum`: Skip row (already at or below target)
   - Otherwise, for each non-zero element in row:
     - Compute probability: `p = remaining_target / remaining_total`
     - Sample new count: `new_count = binomial(old_count, p)`
     - Update value and remaining counts
     - Continue until target reached or all elements processed

The algorithm uses sequential binomial sampling to ensure the total count approaches the target while preserving the relative proportions of counts.

**Edge Cases**

- **Row already at target**: Row is unchanged
- **Row below target**: Row is unchanged (no upsampling)
- **Zero counts**: Preserved as zero
- **Very small target**: May result in many zeros

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR format with mutable values
- `target_sum > 0`
- Matrix values must be non-negative integers (counts)

**Complexity Analysis**

- **Time**: O(nnz) - single pass through all non-zero elements
  - Parallelized over rows
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
#include "scl/kernel/resample.hpp"

Sparse<Real, true> matrix = /* ... */;  // Expression matrix (cells x genes)
Real target_sum = 10000.0;  // Target total count per cell

scl::kernel::resample::downsample(
    matrix,
    target_sum,
    42  // seed for reproducibility
);

// Each row now has total count approximately equal to target_sum
```

---

### downsample_variable

::: source_code file="scl/kernel/resample.hpp" symbol="downsample_variable" collapsed
:::

**Algorithm Description**

Downsample each row to a variable target count:

1. **For each row i in parallel**:
   - Get target count: `target = target_counts[i]`
   - Apply same algorithm as `downsample` with row-specific target

This allows different cells to be downsampled to different target counts, useful for normalizing to different sequencing depths.

**Edge Cases**

- **Variable targets**: Each row uses its own target from `target_counts`
- **Zero target**: Row becomes all zeros
- **Target larger than current**: Row unchanged (no upsampling)

**Data Guarantees (Preconditions)**

- `target_counts.len >= matrix.rows()`
- All `target_counts[i] > 0` (or handled as zero)
- Matrix values must be mutable

**Complexity Analysis**

- **Time**: O(nnz) - same as `downsample`
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
Array<Real> target_counts(n_cells);
// ... set target counts, e.g., based on sequencing depth ...

scl::kernel::resample::downsample_variable(
    matrix,
    target_counts,
    42  // seed
);

// Row i now has total count approximately equal to target_counts[i]
```

---

### binomial_resample

::: source_code file="scl/kernel/resample.hpp" symbol="binomial_resample" collapsed
:::

**Algorithm Description**

Resample each count value using binomial distribution:

1. **For each row in parallel**:
   - For each non-zero element:
     - Sample new count: `new_count = binomial(old_count, p)`
     - Replace value with sampled count

This simulates technical noise or reduces counts by a fixed probability factor.

**Edge Cases**

- **p = 0**: All counts become 0
- **p = 1**: All counts unchanged
- **Small p**: Many counts become 0 (sparsification effect)

**Data Guarantees (Preconditions)**

- `p` in range [0, 1]
- Matrix values must be mutable

**Complexity Analysis**

- **Time**: O(nnz) - single pass through all elements
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
Real p = 0.8;  // Keep 80% of counts on average

scl::kernel::resample::binomial_resample(
    matrix,
    p,
    42  // seed
);

// Each count is now binomial(original_count, 0.8)
```

---

### poisson_resample

::: source_code file="scl/kernel/resample.hpp" symbol="poisson_resample" collapsed
:::

**Algorithm Description**

Resample each count value using Poisson distribution:

1. **For each row in parallel**:
   - For each non-zero element:
     - Compute mean: `mean = old_count * lambda`
     - Sample new count: `new_count = Poisson(mean)`
     - Replace value with sampled count

This simulates Poisson noise or scales counts with variance equal to mean.

**Edge Cases**

- **lambda = 0**: All counts become 0
- **lambda = 1**: Mean preserved, variance = mean (Poisson property)
- **lambda > 1**: Counts increased on average
- **lambda < 1**: Counts decreased on average

**Data Guarantees (Preconditions)**

- `lambda > 0`
- Matrix values must be mutable

**Complexity Analysis**

- **Time**: O(nnz) - single pass through all elements
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
Real lambda = 0.9;  // Scale down by 10% on average

scl::kernel::resample::poisson_resample(
    matrix,
    lambda,
    42  // seed
);

// Each count is now Poisson(original_count * 0.9)
```

---

## Notes

**Random Number Generation**

All functions use fast RNG (Xoshiro128+ or similar) with:
- Deterministic behavior given seed
- Thread-local RNG states for parallelization
- High-quality randomness suitable for statistical sampling

**In-Place Operations**

All functions modify the matrix in-place:
- Matrix structure (indices, indptr) unchanged
- Only values are modified
- Memory efficient (no copies)

**Use Cases**

- **Downsampling**: Normalize sequencing depth across cells
- **Noise simulation**: Add technical noise to count data
- **Sparsification**: Reduce counts for faster computation
- **Data augmentation**: Generate synthetic variations

**Thread Safety**

All functions are thread-safe and parallelized over rows with independent RNG states.

## See Also

- [Sampling](/cpp/kernels/sampling) - Cell sampling strategies
- [Normalization](/cpp/kernels/normalize) - Normalization operations
