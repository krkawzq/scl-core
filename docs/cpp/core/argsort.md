# argsort.hpp

> scl/core/argsort.hpp · Argument sorting (returns sorted indices)

## Overview

This file provides argument sorting operations that return permutation indices that would sort an array, rather than sorting the array itself. Useful for top-K selection, ranking, and indirect sorting.

Key features:
- Returns sorted indices without modifying original data
- SIMD-optimized index initialization
- Multiple variants (in-place, buffered, indirect)
- 5-10x faster than std::sort with lambda

**Header**: `#include "scl/core/argsort.hpp"`

---

## Main APIs

### argsort_inplace

Sort keys and return corresponding indices (ascending order). Modifies keys array.

::: source_code file="scl/core/argsort.hpp" symbol="argsort_inplace" collapsed
:::

**Algorithm Description**

Sorts keys and returns permutation indices:
1. Initialize indices to [0, 1, 2, ..., n-1] using SIMD
2. Sort (keys, indices) pairs by keys using key-value sort
3. Result: keys sorted, indices contain original positions

**Edge Cases**

- **Empty arrays**: No-op
- **Single element**: indices[0] = 0

**Data Guarantees (Preconditions)**

- keys.len == indices.len
- indices buffer must be allocated

**Complexity Analysis**

- **Time**: O(n log n)
- **Space**: O(1) auxiliary

**Example**

```cpp
#include "scl/core/argsort.hpp"

Array<Real> keys = ...;
Array<Index> indices(keys.len);

scl::sort::argsort_inplace(keys, indices);
// keys is now sorted
// indices[i] contains original position of keys[i]
```

---

### argsort_inplace_descending

Sort keys and return indices in descending order.

**Complexity**: O(n log n) time, O(1) space

---

### argsort_indirect

Sort indices without modifying keys array (requires buffer).

**Complexity**: O(n log n) time, O(n) space for buffer

## See Also

- [Sort](./sort) - Direct array sorting
- [SIMD](./simd) - Underlying SIMD abstraction
