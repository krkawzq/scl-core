# sort.hpp

> scl/core/sort.hpp · High-performance sorting via Google Highway VQSort

## Overview

This file provides SIMD-accelerated sorting using Google Highway VQSort backend. Optimized for numerical computing with architecture-agnostic SIMD acceleration.

Key features:
- SIMD-accelerated sorting (2-5x faster than std::sort)
- Single-array sorting (ascending/descending)
- Key-value pair sorting
- Architecture-agnostic (AVX2/AVX-512/NEON)

**Header**: `#include "scl/core/sort.hpp"`

---

## Main APIs

### sort

Sort array in ascending order using SIMD-optimized VQSort.

::: source_code file="scl/core/sort.hpp" symbol="sort" collapsed
:::

**Algorithm Description**

Sorts array using Google Highway VQSort (vectorized quicksort variant):
- Uses SIMD partitioning and comparison
- Optimized for modern CPU cache hierarchies
- Best performance for arrays > 100 elements

**Edge Cases**

- **Empty array**: No-op
- **Single element**: No-op
- **Already sorted**: O(n log n) worst case, often faster

**Data Guarantees (Preconditions)**

- data.ptr must be valid or nullptr (if data.len == 0)
- T must be sortable (have < operator)

**Complexity Analysis**

- **Time**: O(n log n) average and worst-case
- **Space**: O(log n) stack for recursion

**Example**

```cpp
#include "scl/core/sort.hpp"

Array<Real> data = ...;
scl::sort::sort(data);  // Sorts in ascending order
```

---

### sort_descending

Sort array in descending order.

::: source_code file="scl/core/sort.hpp" symbol="sort_descending" collapsed
:::

**Complexity**: O(n log n) time, O(log n) space

---

### sort_key_value

Sort key-value pairs maintaining correspondence.

::: source_code file="scl/core/sort.hpp" symbol="sort_key_value" collapsed
:::

**Complexity**: O(n log n) time, O(log n) space

## See Also

- [Argsort](./argsort) - Argument sorting (returns sorted indices)
- [SIMD](./simd) - Underlying SIMD abstraction
