# Softmax

Softmax normalization operations with SIMD optimization and temperature scaling.

## Overview

Softmax operations provide:

- **In-place normalization** - Convert values to probability distributions
- **Temperature scaling** - Control distribution sharpness
- **Log-softmax** - Numerically stable log probabilities
- **Sparse matrix support** - Row-wise softmax for sparse matrices

## Dense Array Operations

### softmax_inplace

Apply softmax normalization in-place to a dense array.

```cpp
#include "scl/kernel/softmax.hpp"

Real values[100];
Size len = 100;

// Standard softmax
scl::kernel::softmax::softmax_inplace(values, len);

// With temperature scaling
scl::kernel::softmax::softmax_inplace(values, len, 0.5);
```

**Parameters:**
- `vals` [in,out] - Pointer to values array, modified in-place
- `len` [in] - Length of array
- `temperature` [in] - Optional temperature parameter (higher = more uniform)

**Postconditions:**
- All values in [0, 1] and sum to 1.0
- For temperature > 0: softmax(x / temperature)
- For temperature <= 0: one-hot at maximum value

**Algorithm:**
3-tier adaptive strategy based on array length:
1. Short (< 16): Scalar loop
2. Medium (< 128): 4-way SIMD unroll with prefetch
3. Long (>= 128): 8-way SIMD unroll with 8 accumulators for ILP

Steps:
1. Find max value for numerical stability
2. Compute exp(x - max) and sum simultaneously
3. Normalize by dividing each element by sum

**Complexity:**
- Time: O(n)
- Space: O(1) auxiliary

**Thread Safety:**
Safe for different arrays, unsafe for same array

**Numerical Notes:**
- Max subtraction prevents overflow in exp()
- Returns uniform distribution if sum is zero

### log_softmax_inplace

Apply log-softmax in-place to a dense array.

```cpp
Real values[100];
Size len = 100;

// Standard log-softmax
scl::kernel::softmax::log_softmax_inplace(values, len);

// With temperature scaling
scl::kernel::softmax::log_softmax_inplace(values, len, 0.5);
```

**Parameters:**
- `vals` [in,out] - Pointer to values array, modified in-place
- `len` [in] - Length of array
- `temperature` [in] - Optional temperature parameter

**Postconditions:**
- All values <= 0 (log probabilities)
- exp(vals) sums to 1.0
- For temperature > 0: log_softmax(x / temperature)
- For temperature <= 0: 0 at max, -inf elsewhere

**Algorithm:**
log_softmax(x) = x - max - log(sum(exp(x - max)))

3-tier adaptive strategy:
1. Find max value
2. Compute sum(exp(x - max)) with SIMD
3. Subtract (max + log(sum)) from each element

**Complexity:**
- Time: O(n)
- Space: O(1) auxiliary

**Numerical Notes:**
- More numerically stable than log(softmax(x))
- Avoids computing explicit probabilities

## Sparse Matrix Operations

### softmax_inplace (sparse)

Apply softmax row-wise in-place to a sparse matrix.

```cpp
#include "scl/core/sparse.hpp"
#include "scl/kernel/softmax.hpp"

Sparse<Real, true> matrix = /* ... */;

// Standard softmax
scl::kernel::softmax::softmax_inplace(matrix);

// With temperature scaling
scl::kernel::softmax::softmax_inplace(matrix, 0.5);
```

**Parameters:**
- `matrix` [in,out] - Sparse matrix (CSR or CSC), values modified in-place
- `temperature` [in] - Optional temperature parameter

**Postconditions:**
- Each row sums to 1.0 (considering only non-zero elements)
- Matrix structure (indices, pointers) unchanged
- Empty rows are unchanged

**Algorithm:**
For each row in parallel:
- Apply 3-tier adaptive softmax to non-zero values

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary per thread

**Thread Safety:**
Safe - parallelized over rows, no shared mutable state

### log_softmax_inplace (sparse)

Apply log-softmax row-wise in-place to a sparse matrix.

```cpp
Sparse<Real, true> matrix = /* ... */;

// Standard log-softmax
scl::kernel::softmax::log_softmax_inplace(matrix);

// With temperature scaling
scl::kernel::softmax::log_softmax_inplace(matrix, 0.5);
```

**Parameters:**
- `matrix` [in,out] - Sparse matrix, values modified in-place
- `temperature` [in] - Optional temperature parameter

**Postconditions:**
- All values <= 0 (log probabilities)
- Matrix structure unchanged

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary per thread

**Thread Safety:**
Safe - parallelized over rows

## Use Cases

### Probability Distributions

Convert raw scores to probability distributions:

```cpp
Real scores[10] = {3.0, 1.0, 4.0, 1.5, 2.0, 0.5, 2.5, 1.0, 3.5, 0.0};
scl::kernel::softmax::softmax_inplace(scores, 10);
// scores now sum to 1.0
```

### Temperature Scaling

Control distribution sharpness:

```cpp
// Sharp distribution (low temperature)
scl::kernel::softmax::softmax_inplace(values, len, 0.1);

// Uniform distribution (high temperature)
scl::kernel::softmax::softmax_inplace(values, len, 10.0);
```

### Log Probabilities

For numerical stability in log-space computations:

```cpp
scl::kernel::softmax::log_softmax_inplace(logits, len);
// Use in cross-entropy loss: -sum(y * log_softmax)
```

### Sparse Matrix Normalization

Normalize each row of a sparse matrix:

```cpp
Sparse<Real, true> expression_matrix = /* ... */;
scl::kernel::softmax::softmax_inplace(expression_matrix);
// Each row now represents a probability distribution
```

## Performance

### SIMD Optimization

All operations use SIMD-optimized exp and sum operations:
- 4-way unroll for medium arrays
- 8-way unroll with 8 accumulators for large arrays
- Prefetch for cache efficiency

### Parallelization

Sparse matrix operations are parallelized over rows:
- Automatic work distribution
- Thread-local accumulators
- No synchronization overhead

## See Also

- [Normalization](/cpp/kernels/normalization) - Other normalization operations
- [Sparse Tools](/cpp/kernels/sparse-tools) - Sparse matrix utilities

