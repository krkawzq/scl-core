# Resampling

Resampling operations with fast RNG for downsampling and stochastic transformations.

## Overview

The `resample` module provides efficient resampling operations for single-cell data:

- **Downsampling**: Reduce total counts per cell to a target value
- **Binomial resampling**: Resample counts with fixed probability
- **Poisson resampling**: Resample counts with scaled Poisson distribution

All operations are:
- Fast RNG-based (xoshiro256++)
- Parallelized over rows with independent RNG states
- In-place modifications (no memory allocation)
- Deterministic given seed

## Functions

### downsample

Downsample each row to a target total count using binomial sampling.

```cpp
#include "scl/kernel/resample.hpp"

Sparse<Real, true> matrix = /* expression matrix */;

// Downsample each cell to 10,000 total counts
scl::kernel::resample::downsample(matrix, 10000.0, 42);
```

**Parameters:**
- `matrix` [in,out] - Expression matrix, modified in-place
- `target_sum` [in] - Target total count per row
- `seed` [in] - Random seed for reproducibility (default: 42)

**Preconditions:**
- `target_sum > 0`
- Matrix values must be mutable

**Postconditions:**
- Each row has total count approximately equal to target_sum
- Matrix structure (indices, indptr) unchanged
- Sampling is deterministic given seed

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary

**Thread Safety:** Safe - parallelized over rows with independent RNG states

**Algorithm:**
For each row in parallel:
1. Compute current total count
2. If current <= target, skip
3. For each non-zero element:
   a. Compute probability = remaining_target / remaining_total
   b. Sample binomial(count, probability)
   c. Update value and remaining counts

### downsample_variable

Downsample each row to a variable target count using binomial sampling.

```cpp
Array<Real> target_counts(n_cells);
// Set different target for each cell
for (Index i = 0; i < n_cells; ++i) {
    target_counts.ptr[i] = compute_target_for_cell(i);
}

scl::kernel::resample::downsample_variable(matrix, target_counts, 42);
```

**Parameters:**
- `matrix` [in,out] - Expression matrix, modified in-place
- `target_counts` [in] - Target count for each row [n_rows]
- `seed` [in] - Random seed for reproducibility (default: 42)

**Preconditions:**
- `target_counts.len >= matrix.rows()`
- All `target_counts[i] > 0`
- Matrix values must be mutable

**Postconditions:**
- Row i has total count approximately equal to `target_counts[i]`
- Matrix structure (indices, indptr) unchanged
- Sampling is deterministic given seed

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary

**Thread Safety:** Safe - parallelized over rows with independent RNG states

### binomial_resample

Resample each count value using binomial distribution with fixed probability.

```cpp
// Resample with 50% probability
scl::kernel::resample::binomial_resample(matrix, 0.5, 42);
```

**Parameters:**
- `matrix` [in,out] - Expression matrix, modified in-place
- `p` [in] - Success probability for binomial sampling
- `seed` [in] - Random seed for reproducibility (default: 42)

**Preconditions:**
- `p` in [0, 1]
- Matrix values must be mutable

**Postconditions:**
- Each value is replaced by binomial(value, p)
- Matrix structure (indices, indptr) unchanged
- Sampling is deterministic given seed

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary

**Thread Safety:** Safe - parallelized over rows with independent RNG states

**Algorithm:**
For each row in parallel:
- For each non-zero element:
  1. Sample binomial(count, p)
  2. Replace value with sampled count

### poisson_resample

Resample each count value using Poisson distribution with scaled mean.

```cpp
// Resample with lambda = 0.8 (reduce counts by 20%)
scl::kernel::resample::poisson_resample(matrix, 0.8, 42);
```

**Parameters:**
- `matrix` [in,out] - Expression matrix, modified in-place
- `lambda` [in] - Scaling factor for Poisson mean (mean = count * lambda)
- `seed` [in] - Random seed for reproducibility (default: 42)

**Preconditions:**
- `lambda > 0`
- Matrix values must be mutable

**Postconditions:**
- Each value is replaced by Poisson(value * lambda)
- Matrix structure (indices, indptr) unchanged
- Sampling is deterministic given seed

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary

**Thread Safety:** Safe - parallelized over rows with independent RNG states

**Algorithm:**
For each row in parallel:
- For each non-zero element:
  1. Compute mean = count * lambda
  2. Sample Poisson(mean)
  3. Replace value with sampled count

## Configuration

```cpp
namespace scl::kernel::resample::config {
    constexpr Size PREFETCH_DISTANCE = 16;
}
```

## Random Number Generator

The module uses a fast xoshiro256++ RNG:
- **Thread-safe**: Each parallel worker has independent RNG state
- **Deterministic**: Same seed produces same sequence
- **Fast**: Optimized for high-throughput sampling

## Use Cases

### Standard Downsampling

```cpp
// Downsample all cells to same depth
Sparse<Real, true> expression = /* ... */;
scl::kernel::resample::downsample(expression, 10000.0, 42);
```

### Variable Depth Downsampling

```cpp
// Downsample to different depths based on cell type
Array<Real> targets(n_cells);
for (Index i = 0; i < n_cells; ++i) {
    if (cell_types[i] == "T_cell") {
        targets.ptr[i] = 5000.0;
    } else {
        targets.ptr[i] = 10000.0;
    }
}
scl::kernel::resample::downsample_variable(expression, targets, 42);
```

### Stochastic Transformations

```cpp
// Apply binomial resampling for noise injection
scl::kernel::resample::binomial_resample(expression, 0.9, 42);

// Apply Poisson resampling for count scaling
scl::kernel::resample::poisson_resample(expression, 0.8, 42);
```

### Reproducible Analysis

```cpp
// Use same seed for reproducibility
uint64_t seed = 42;

// Downsample
scl::kernel::resample::downsample(matrix1, 10000.0, seed);

// Process...

// Downsample another dataset with same seed
scl::kernel::resample::downsample(matrix2, 10000.0, seed);
```

## Performance

- **Fast RNG**: xoshiro256++ is 2-3x faster than standard library RNGs
- **Parallelization**: Scales linearly with CPU cores
- **Zero allocations**: All operations are in-place
- **Deterministic**: Same seed produces reproducible results

---

::: tip Reproducibility
Always specify a seed for reproducible results. The same seed with the same input produces identical output.
:::

