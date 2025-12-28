# High-Performance Sorting

SIMD-accelerated sorting using Google Highway VQSort backend.

## Overview

Sorting module provides:

- **SIMD-Accelerated** - Automatically selects best SIMD instructions
- **Single-Array Sorting** - Ascending/descending order
- **Key-Value Pair Sorting** - Maintains correspondence between keys and values
- **High Performance** - 2-5x faster than std::sort for numerical types

## Performance Characteristics

- Automatically selects best SIMD instructions (AVX2/AVX-512/NEON)
- Typical speedup: 2-5x vs std::sort for numerical types
- Near-optimal cache utilization
- Stable performance across data distributions

## Supported Types

- **Integers**: int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
- **Floats**: float, double
- **Custom types**: With < operator (may not use SIMD path)

**Thread Safety:**
- All operations are unsafe - concurrent access causes race conditions

## Basic Sorting

### sort

Sort array in ascending order:

```cpp
#include "scl/core/sort.hpp"

Array<Real> data = /* ... */;
scl::sort::sort(data);

// data is now sorted in ascending order
```

**Parameters:**
- `T` [template] - Element type (must be sortable)
- `data` - Array to sort

**PRECONDITIONS:**
- `data.ptr` must be valid or nullptr (if data.len == 0)

**POSTCONDITIONS:**
- `data` is sorted in ascending order
- `data[i] <= data[i+1]` for all valid i

**MUTABILITY:**
- INPLACE - modifies data array

**Algorithm:**
- Google Highway VQSort (vectorized quicksort variant)

**Complexity:**
- Time: O(n log n) average, O(n log n) worst-case
- Space: O(log n) stack for recursion

**Performance:**
- Uses SIMD partitioning and comparison
- Optimized for modern CPU cache hierarchies
- Best for arrays > 100 elements

### sort_descending

Sort array in descending order:

```cpp
scl::sort::sort_descending(data);

// data is now sorted in descending order
```

**POSTCONDITIONS:**
- `data[i] >= data[i+1]` for all valid i

## Key-Value Pair Sorting

### sort_key_value

Sort keys array and maintain correspondence with values:

```cpp
Array<Real> keys = /* ... */;
Array<Index> values = /* ... */;

scl::sort::sort_key_value(keys, values);

// keys is sorted, values maintains correspondence
// values[i] contains original index of keys[i]
```

**Parameters:**
- `T` [template] - Key type (must be sortable)
- `U` [template] - Value type
- `keys` - Keys to sort (modified)
- `values` - Values to reorder (modified)

**PRECONDITIONS:**
- `keys.len == values.len`

**POSTCONDITIONS:**
- `keys` is sorted in ascending order
- `values` is reordered to maintain correspondence
- After sorting: (keys[i], values[i]) pairs are sorted by keys[i]

**MUTABILITY:**
- INPLACE - modifies both arrays

**Algorithm:**
- Highway VQSort with key-value pairing
- Efficient SIMD-optimized swapping

**Complexity:**
- Time: O(n log n)
- Space: O(log n) stack

**Performance:**
- Faster than std::sort with custom comparator
- Optimized for numerical types

### sort_key_value_descending

Sort keys array in descending order while maintaining correspondence:

```cpp
scl::sort::sort_key_value_descending(keys, values);
```

## Usage Examples

### Sort Simple Array

```cpp
Array<Real> scores = /* ... */;

// Sort ascending
scl::sort::sort(scores);

// Find top 10
Index top_k = 10;
for (Index i = scores.len - top_k; i < scores.len; ++i) {
    std::cout << "Top score: " << scores[i] << "\n";
}
```

### Sort with Indices

```cpp
Array<Real> values = /* ... */;
Array<Index> indices(values.len);

// Initialize indices to [0, 1, 2, ..., n-1]
for (Index i = 0; i < indices.len; ++i) {
    indices[i] = i;
}

// Sort values and maintain index correspondence
scl::sort::sort_key_value(values, indices);

// Access original indices
for (Index i = 0; i < values.len; ++i) {
    std::cout << "Value " << values[i]
              << " was originally at index " << indices[i] << "\n";
}
```

### Custom Comparison (via argsort)

For custom comparison logic, use argsort module which supports lambda comparators:

```cpp
// See argsort.md for lambda-based sorting
```

---

::: tip When to Use
Use sort module for standard ascending/descending order. For custom comparisons or when you only need indices, use the argsort module instead.
:::

