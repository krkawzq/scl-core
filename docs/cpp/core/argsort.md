# Argument Sorting

Returns permutation indices that would sort an array, rather than sorting the array itself.

## Overview

Argsort provides:

- **Index Generation** - Returns indices that would sort an array
- **Preserve Original Data** - Original array remains unchanged (in buffered/indirect variants)
- **Multiple Variants** - In-place, buffered, and indirect (lambda-based)
- **High Performance** - 5-10x faster than std::sort with lambda

## Purpose

Argsort is fundamental for:

- Top-K selection (e.g., highly variable genes)
- Ranking and percentile calculations
- Indirect sorting of multiple related arrays
- Preserving original data while obtaining sorted order

## Variants

1. **In-Place**: Modifies keys array, fast for temporary arrays
2. **Buffered**: Preserves keys array, requires external buffer
3. **Indirect**: Never modifies keys, uses lambda comparisons (slowest)

## In-Place Argsort

### argsort_inplace

Sort keys and return corresponding indices (ascending order):

```cpp
#include "scl/core/argsort.hpp"

Array<Real> keys = /* ... */;
Array<Index> indices(keys.len);

scl::sort::argsort_inplace(keys, indices);

// keys is sorted
// indices[i] contains original position of keys[i]
```

**Parameters:**
- `T` [template] - Key type (must be sortable)
- `keys` - Array to sort (modified)
- `indices` - Output permutation indices

**PRECONDITIONS:**
- `keys.len == indices.len`
- `indices` buffer is allocated

**POSTCONDITIONS:**
- `keys` is sorted in ascending order
- `indices[i]` contains original position of `keys[i]`
- Applying indices to original data yields sorted order

**MUTABILITY:**
- INPLACE - modifies keys array

**Algorithm:**
1. Initialize indices to [0, 1, 2, ..., n-1] using SIMD
2. Sort (keys, indices) pairs by keys
3. Result: keys sorted, indices contain original positions

**Complexity:**
- Time: O(n log n)
- Space: O(1) auxiliary

**WHEN TO USE:**
- keys array is temporary and can be modified
- Need maximum performance
- keys won't be needed in original order

### argsort_inplace_descending

Sort keys in descending order and return corresponding indices:

```cpp
scl::sort::argsort_inplace_descending(keys, indices);
```

## Buffered Argsort

### argsort_buffered

Sort keys using external buffer, preserving original keys:

```cpp
Array<const Real> keys = /* ... */;
Array<Index> indices(keys.len);
Array<Real> buffer(keys.len);  // External buffer

scl::sort::argsort_buffered(keys, indices, buffer);

// keys is unchanged
// indices[i] contains position of i-th smallest element
```

**Parameters:**
- `T` [template] - Key type
- `keys` - Array to sort (not modified)
- `indices` - Output permutation indices
- `buffer` - External buffer (temporary workspace)

**PRECONDITIONS:**
- `keys.len == indices.len == buffer.len`

**POSTCONDITIONS:**
- `keys` is unchanged
- `indices` contains permutation that would sort keys
- `keys[indices[i]]` is sorted

**MUTABILITY:**
- CONST for keys, modifies buffer and indices

**WHEN TO USE:**
- Need to preserve original keys array
- Have buffer available
- Slightly slower than in-place but preserves data

### argsort_buffered_descending

Buffered argsort in descending order:

```cpp
scl::sort::argsort_buffered_descending(keys, indices, buffer);
```

## Indirect Argsort

### argsort_indirect

Sort using lambda comparator without modifying keys:

```cpp
Array<const Real> keys = /* ... */;
Array<Index> indices(keys.len);

scl::sort::argsort_indirect(keys, indices,
    [](Real a, Real b) { return a < b; }  // Comparator
);

// keys is unchanged
// indices contains sorting permutation
```

**Parameters:**
- `T` [template] - Key type
- `keys` - Array to sort (not modified)
- `indices` - Output permutation indices
- `comparator` - Lambda or function object: (T, T) -> bool

**PRECONDITIONS:**
- `keys.len == indices.len`
- comparator defines strict weak ordering

**POSTCONDITIONS:**
- `keys` is unchanged
- `indices` contains permutation that would sort keys according to comparator

**MUTABILITY:**
- CONST for keys, modifies indices only

**Complexity:**
- Time: O(n log n * comparator_cost)
- Space: O(n) for index array + O(log n) stack

**WHEN TO USE:**
- Need custom comparison logic
- Cannot modify keys
- Acceptable performance (slower than standard argsort)

**Performance:**
- Slower than in-place/buffered variants (lambda overhead)
- Still faster than std::sort with lambda (SIMD-optimized index initialization)

## Usage Examples

### Top-K Selection

```cpp
Array<Real> scores = /* ... */;
Array<Index> indices(scores.len);
Array<Real> buffer(scores.len);

// Get indices that would sort scores
scl::sort::argsort_buffered(scores, indices, buffer);

// Get top 10 indices
Index top_k = 10;
for (Index i = scores.len - top_k; i < scores.len; ++i) {
    Index original_idx = indices[i];
    std::cout << "Top score at index " << original_idx
              << ": " << scores[original_idx] << "\n";
}
```

### Rank Calculation

```cpp
Array<Real> values = /* ... */;
Array<Index> indices(values.len);
Array<Real> buffer(values.len);

scl::sort::argsort_buffered(values, indices, buffer);

// Create rank array
Array<Index> ranks(values.len);
for (Index i = 0; i < indices.len; ++i) {
    ranks[indices[i]] = i;  // Rank of original position i
}
```

### Custom Comparison

```cpp
// Sort by absolute value
Array<const Real> values = /* ... */;
Array<Index> indices(values.len);

scl::sort::argsort_indirect(values, indices,
    [](Real a, Real b) {
        return std::abs(a) < std::abs(b);
    }
);
```

---

::: tip Performance Trade-offs
In-place is fastest but modifies data. Buffered preserves data with minimal overhead. Indirect supports custom logic but is slower due to lambda overhead.
:::

