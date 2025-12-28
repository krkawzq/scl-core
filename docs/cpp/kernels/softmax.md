# softmax.hpp

> scl/kernel/softmax.hpp · Softmax operations

## Overview

This file provides high-performance softmax and log-softmax operations for both dense arrays and sparse matrices. All operations are in-place for efficiency and use adaptive SIMD strategies based on input size. Supports temperature scaling for controlling distribution sharpness.

**Header**: `#include "scl/kernel/softmax.hpp"`

---

## Main APIs

### softmax_inplace (dense array)

::: source_code file="scl/kernel/softmax.hpp" symbol="softmax_inplace" collapsed
:::

**Algorithm Description**

Applies softmax normalization in-place to a dense array using 3-tier adaptive strategy:

1. **Short arrays (< 16)**: Scalar loop for minimal overhead
2. **Medium arrays (< 128)**: 4-way SIMD unroll with prefetch
3. **Long arrays (>= 128)**: 8-way SIMD unroll with 8 accumulators for instruction-level parallelism

Steps:
1. Find maximum value for numerical stability: `max_val = max(vals)`
2. Compute `exp(x - max)` and sum simultaneously using SIMD
3. Normalize: `vals[i] = exp(vals[i] - max) / sum`

**Edge Cases**

- **Empty array (len=0)**: No-op, returns immediately
- **All zeros**: Returns uniform distribution (1/len for each element)
- **All same value**: Returns uniform distribution
- **Very large values**: Max subtraction prevents overflow in exp()
- **Sum zero**: Returns uniform distribution to avoid division by zero

**Data Guarantees (Preconditions)**

- `vals` must be valid pointer if len > 0
- `len >= 0`
- Array memory is writable

**Complexity Analysis**

- **Time**: O(n) - single pass with SIMD acceleration
- **Space**: O(1) auxiliary - only accumulators needed

**Example**

```cpp
#include "scl/kernel/softmax.hpp"

Real* values = /* array of values */;
Size len = /* array length */;

scl::kernel::softmax::softmax_inplace(values, len);

// values[i] now in [0, 1] and sum(values) == 1.0
```

---

### softmax_inplace (dense array with temperature)

::: source_code file="scl/kernel/softmax.hpp" symbol="softmax_inplace" collapsed
:::

**Algorithm Description**

Applies softmax with temperature scaling in-place:

1. Scale all values: `vals[i] = vals[i] / temperature`
2. Apply standard softmax to scaled values
3. Temperature > 0: Produces softer distribution (higher temperature = more uniform)
4. Temperature <= 0: Produces one-hot at maximum value

**Edge Cases**

- **Temperature > 1**: Softer distribution, more uniform
- **Temperature < 1**: Sharper distribution, more peaked
- **Temperature = 1**: Standard softmax
- **Temperature <= 0**: One-hot encoding at maximum
- **Temperature = 0**: Division by zero avoided, treated as <= 0

**Data Guarantees (Preconditions)**

- `vals` must be valid pointer if len > 0
- `len >= 0`
- Array memory is writable

**Complexity Analysis**

- **Time**: O(n) - scaling plus softmax
- **Space**: O(1) auxiliary

**Example**

```cpp
Real* values = /* array of values */;
Size len = /* array length */;
Real temperature = 0.5;  // Sharper distribution

scl::kernel::softmax::softmax_inplace(values, len, temperature);

// values now represent temperature-scaled softmax distribution
```

---

### log_softmax_inplace (dense array)

::: source_code file="scl/kernel/softmax.hpp" symbol="log_softmax_inplace" collapsed
:::

**Algorithm Description**

Applies log-softmax in-place to a dense array:

1. Find maximum value: `max_val = max(vals)`
2. Compute sum of exponentials: `sum_exp = sum(exp(vals[i] - max))` using SIMD
3. Compute log-sum: `log_sum = log(sum_exp)`
4. Update values: `vals[i] = vals[i] - max - log_sum`

Formula: `log_softmax(x) = x - max - log(sum(exp(x - max)))`

**Edge Cases**

- **Empty array**: No-op
- **All zeros**: Returns uniform log probabilities (log(1/len))
- **All same value**: Returns uniform log probabilities
- **Very large values**: Max subtraction prevents overflow
- **Sum zero**: Returns uniform log probabilities

**Data Guarantees (Preconditions)**

- `vals` must be valid pointer if len > 0
- `len >= 0`
- Array memory is writable

**Complexity Analysis**

- **Time**: O(n) - single pass with SIMD
- **Space**: O(1) auxiliary

**Example**

```cpp
Real* values = /* array of values */;
Size len = /* array length */;

scl::kernel::softmax::log_softmax_inplace(values, len);

// values[i] <= 0 (log probabilities)
// exp(values) sums to 1.0
```

---

### softmax_inplace (sparse matrix)

::: source_code file="scl/kernel/softmax.hpp" symbol="softmax_inplace" collapsed
:::

**Algorithm Description**

Applies softmax row-wise in-place to a sparse matrix:

1. For each row in parallel:
   - Extract non-zero values in the row
   - Apply 3-tier adaptive softmax to non-zero values only
   - Update values in-place
2. Matrix structure (indices, pointers) unchanged
3. Empty rows remain unchanged (no non-zeros to normalize)

**Edge Cases**

- **Empty rows**: Unchanged (no non-zeros)
- **Single non-zero per row**: Becomes 1.0 after normalization
- **All zeros in row**: Remains unchanged
- **Very sparse rows**: Efficiently handles rows with few non-zeros

**Data Guarantees (Preconditions)**

- Matrix is valid sparse format (CSR or CSC)
- Matrix values must be mutable
- Matrix structure is valid

**Complexity Analysis**

- **Time**: O(nnz) - processes each non-zero once
- **Space**: O(1) auxiliary per thread - only accumulators

**Example**

```cpp
Sparse<Real, true> matrix = /* sparse matrix, CSR */;

scl::kernel::softmax::softmax_inplace(matrix);

// Each row now sums to 1.0 (considering only non-zeros)
// Matrix structure unchanged
```

---

### log_softmax_inplace (sparse matrix)

::: source_code file="scl/kernel/softmax.hpp" symbol="log_softmax_inplace" collapsed
:::

**Algorithm Description**

Applies log-softmax row-wise in-place to a sparse matrix:

1. For each row in parallel:
   - Extract non-zero values
   - Compute log-softmax: `log_softmax(x) = x - max - log(sum(exp(x - max)))`
   - Update values in-place
2. Matrix structure unchanged
3. All values become <= 0 (log probabilities)

**Edge Cases**

- **Empty rows**: Unchanged
- **Single non-zero**: Becomes 0.0 (log(1.0) = 0)
- **All zeros**: Remains unchanged
- **Sparse rows**: Efficiently handles few non-zeros

**Data Guarantees (Preconditions)**

- Matrix is valid sparse format
- Matrix values must be mutable

**Complexity Analysis**

- **Time**: O(nnz) - processes each non-zero once
- **Space**: O(1) auxiliary per thread

**Example**

```cpp
Sparse<Real, true> matrix = /* sparse matrix */;

scl::kernel::softmax::log_softmax_inplace(matrix);

// All values <= 0 (log probabilities)
// exp(values) per row sums to 1.0
```

---

## Numerical Notes

### Stability

- **Max subtraction**: Prevents overflow in exp() by subtracting maximum value
- **Log-softmax**: More numerically stable than log(softmax(x)) for large values
- **Uniform fallback**: Returns uniform distribution if sum is zero

### Temperature Scaling

- **Temperature > 1**: Softer distribution, reduces peakiness
- **Temperature < 1**: Sharper distribution, increases peakiness
- **Temperature = 1**: Standard softmax
- **Temperature <= 0**: One-hot encoding (hard maximum)

---

## See Also

- [Normalize Module](./normalize) - Other normalization operations
- [Sparse Matrix](../core/sparse) - Sparse matrix operations
