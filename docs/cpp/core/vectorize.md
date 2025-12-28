# vectorize.hpp

> scl/core/vectorize.hpp · SIMD-optimized vectorized array operations

## Overview

This file provides high-performance SIMD-optimized operations on array views using Google Highway. All operations use aggressive unrolling (2-4 way) and handle scalar tails automatically.

Key features:
- Zero-overhead abstractions (compile away)
- Automatic SIMD vectorization (architecture-agnostic)
- Aggressive unrolling (2-4 way for maximum ILP)
- Automatic tail handling

**Header**: `` `#include "scl/core/vectorize.hpp"` ``

---

## Main APIs

### sum

Compute sum of all elements using SIMD-optimized reduction.

::: source_code file="scl/core/vectorize.hpp" symbol="sum" collapsed
:::

**Algorithm Description**

Computes sum using 4-way unrolled SIMD accumulation:
1. Process bulk with 4-way unrolled SIMD loop
2. Horizontal reduction using SumOfLanes
3. Scalar tail handling for remainder

**Edge Cases**

- **Empty array**: Returns T(0)
- **NaN/Inf**: Propagates through sum

**Data Guarantees (Preconditions)**

- span must be valid Array view

**Complexity Analysis**

- **Time**: O(N)
- **Space**: O(1)

**Example**

```cpp
#include "scl/core/vectorize.hpp"

Array<const Real> data = ...;
Real total = scl::vectorize::sum(data);
```

---

### dot

Compute dot product of two vectors.

::: source_code file="scl/core/vectorize.hpp" symbol="dot" collapsed
:::

**Algorithm Description**

Computes dot product: sum(a[i] * b[i]) using MulAdd (FMA) for optimal performance.

**Edge Cases**

- **Empty arrays**: Returns T(0)
- **Size mismatch**: Undefined behavior (caller must ensure a.len == b.len)

**Data Guarantees (Preconditions)**

- a.len == b.len
- Both arrays must be valid

**Complexity Analysis**

- **Time**: O(N)
- **Space**: O(1)

**Example**

```cpp
Array<const Real> a = ...;
Array<const Real> b = ...;
Real result = scl::vectorize::dot(a, b);
```

---

### norm

Compute L2 norm (Euclidean norm) of vector.

::: source_code file="scl/core/vectorize.hpp" symbol="norm" collapsed
:::

**Complexity**: O(N) time, O(1) space

---

### add / mul / sub / div

Element-wise arithmetic operations.

**Complexity**: O(N) time, O(1) space

---

### count

Count occurrences of value using SIMD.

**Complexity**: O(N) time, O(1) space

---

### find

Find first occurrence of value.

**Complexity**: O(N) time worst case, often better with early exit

## See Also

- [SIMD](./simd) - Underlying SIMD abstraction layer
- [Type System](./types) - Array<T> type used for views
