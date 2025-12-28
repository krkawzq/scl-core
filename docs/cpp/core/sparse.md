# sparse.hpp

> scl/core/sparse.hpp · Sparse matrix with discontiguous storage using pointer arrays

## Overview

This file provides the `` `Sparse<T, IsCSR>` `` struct, SCL-Core's primary sparse matrix data structure. Unlike traditional contiguous CSR/CSC formats, Sparse uses pointer arrays where each row/column can be stored in a separate allocation, enabling flexible integration with external data sources.

Key features:
- Discontiguous storage (each row/column can be in separate allocation)
- Non-owning view (wraps existing data without copying)
- Flexible integration (wrap NumPy arrays, memory-mapped files)
- Lazy loading support (load rows/columns on demand)
- Registry-managed metadata arrays

**Header**: `` `#include "scl/core/sparse.hpp"` ``

---

## Main APIs

### `` `Sparse<T, IsCSR>` ``

Sparse matrix structure with discontiguous storage.

::: source_code file="scl/core/sparse.hpp" symbol="Sparse" collapsed
:::

**Algorithm Description**

Sparse matrix structure using pointer arrays:

**Memory Layout** (CSR example):
```
data_ptrs    = [ptr_to_row0_vals, ptr_to_row1_vals, ptr_to_row2_vals]
indices_ptrs = [ptr_to_row0_cols, ptr_to_row1_cols, ptr_to_row2_cols]
lengths      = [len0, len1, len2]
```

Each row/column can be in a separate memory allocation, unlike traditional contiguous formats where all data is in single arrays.

**Design Philosophy**:
- **Non-owning view**: Does not manage lifetime of underlying data
- **Discontiguous**: Each row/column can be in separate allocation
- **Flexible**: Can wrap external data (NumPy, memory-mapped files)
- **Efficient**: O(1) row/column access via pointer indirection

**Edge Cases**

- **Empty matrix**: rows_ == 0 OR cols_ == 0 OR nnz_ == 0
- **Null pointers**: Default constructor creates invalid matrix (all pointers nullptr)
- **Out-of-bounds access**: Debug builds assert, release builds undefined behavior
- **Invalid matrix**: Call `valid()` or `operator bool()` to check validity

**Data Guarantees (Preconditions)**

- If constructed with pointer arrays, they must outlive the Sparse object
- Row/column indices must be in valid range [0, primary_dim())
- For CSR: row index i must be in [0, rows_)
- For CSC: column index j must be in [0, cols_)
- If using `new_registered()`, metadata arrays are registered with HandlerRegistry

**Complexity Analysis**

- **Construction**: O(1) - just stores pointers
- **Row/column access**: O(1) - pointer dereference
- **Memory**: (2*primary_dim + 1) pointers + nnz elements (vs 2*nnz + primary_dim+1 for contiguous)

**Example**

```cpp
#include "scl/core/sparse.hpp"

// Create CSR matrix from scratch
auto matrix = scl::Sparse<Real, true>::new_registered(
    1000,    // rows
    500,     // cols
    10000    // total_nnz
);

// Access row
Index row_idx = 0;
auto vals = matrix.primary_values(row_idx);
auto idxs = matrix.primary_indices(row_idx);
Index len = matrix.primary_length(row_idx);

// Iterate over non-zeros in row
for (Index j = 0; j < len; ++j) {
    Real value = vals[j];
    Index col = idxs[j];
    // Process value at (row_idx, col)
}

// Wrap existing data (non-owning)
Real* row0_data = ...;
Index* row0_indices = ...;
Pointer* data_ptrs = ...;
Pointer* indices_ptrs = ...;
Index* lengths = ...;

scl::Sparse<Real, true> wrapper(
    data_ptrs, indices_ptrs, lengths,
    1000, 500, 10000
);
// Wrapper points to existing data, does not own it
```

---

### primary_values / row_values / col_values

Get array view of values in a row (CSR) or column (CSC).

::: source_code file="scl/core/sparse.hpp" symbol="primary_values" collapsed
:::

**Algorithm Description**

Returns an `` `Array<T>` `` view of values in the specified row (CSR) or column (CSC):
- CSR: `primary_values(i)` = values in row i
- CSC: `primary_values(j)` = values in column j

The method performs pointer dereference: `data_ptrs[i]` cast to `` `T*` `` and wrapped in `` `Array<T>` `` with length from `lengths[i]`.

**Edge Cases**

- **Out-of-bounds index**: Debug builds assert, release builds undefined behavior
- **Null pointer in data_ptrs**: Debug builds assert, release builds undefined behavior
- **Zero-length row/column**: Returns empty `` `Array<T>` `` (ptr may be nullptr)

**Data Guarantees (Preconditions)**

- Index must be in [0, primary_dim())
- `data_ptrs` must not be nullptr
- `data_ptrs[i]` must point to valid array of at least `lengths[i]` elements

**Complexity Analysis**

- **Time**: O(1) - pointer dereference and `` `Array` `` construction
- **Space**: O(1) - returns `` `Array` `` view (non-owning)

**Example**

```cpp
Sparse<Real, true> matrix = ...;  // CSR

Index row = 5;
Array<Real> vals = matrix.primary_values(row);
// vals is a view of values in row 5

// Iterate
for (Index j = 0; j < vals.size(); ++j) {
    Real value = vals[j];
    // Process value
}
```

---

### primary_indices / row_indices / col_indices

Get array view of column indices (CSR) or row indices (CSC).

::: source_code file="scl/core/sparse.hpp" symbol="primary_indices" collapsed
:::

**Algorithm Description**

Returns an `` `Array<Index>` `` view of column indices (CSR) or row indices (CSC):
- CSR: `primary_indices(i)` = column indices in row i
- CSC: `primary_indices(j)` = row indices in column j

**Edge Cases**

- **Out-of-bounds index**: Debug builds assert, release builds undefined behavior
- **Null pointer in indices_ptrs**: Debug builds assert, release builds undefined behavior
- **Zero-length row/column**: Returns empty `` `Array<Index>` ``

**Data Guarantees (Preconditions)**

- Index must be in [0, primary_dim())
- `indices_ptrs` must not be nullptr
- `indices_ptrs[i]` must point to valid array of at least `lengths[i]` Index elements

**Complexity Analysis**

- **Time**: O(1) - pointer dereference and `` `Array` `` construction
- **Space**: O(1) - returns `` `Array` `` view

**Example**

```cpp
Sparse<Real, true> matrix = ...;  // CSR

Index row = 5;
Array<Real> vals = matrix.primary_values(row);
Array<Index> cols = matrix.primary_indices(row);

// Access value and column together
for (Index j = 0; j < vals.size(); ++j) {
    Real value = vals[j];
    Index col = cols[j];
    // Value at (row, col) is value
}
```

---

### primary_length / row_length / col_length

Get number of non-zeros in a row (CSR) or column (CSC).

::: source_code file="scl/core/sparse.hpp" symbol="primary_length" collapsed
:::

**Algorithm Description**

Returns the number of non-zero elements in the specified row (CSR) or column (CSC):
- CSR: `primary_length(i)` = number of non-zeros in row i
- CSC: `primary_length(j)` = number of non-zeros in column j

Simply returns `lengths[i]` where i is the primary dimension index.

**Edge Cases**

- **Out-of-bounds index**: Debug builds assert, release builds undefined behavior
- **Null pointer in lengths**: Debug builds assert, release builds undefined behavior

**Data Guarantees (Preconditions)**

- Index must be in [0, primary_dim())
- `lengths` must not be nullptr

**Complexity Analysis**

- **Time**: O(1) - array access
- **Space**: O(1)

**Example**

```cpp
Sparse<Real, true> matrix = ...;  // CSR

Index row = 5;
Index nnz_in_row = matrix.primary_length(row);
// nnz_in_row is the number of non-zeros in row 5
```

---

### new_registered

Factory method to create Sparse matrix with registry-managed metadata arrays.

::: source_code file="scl/core/sparse.hpp" symbol="new_registered" collapsed
:::

**Algorithm Description**

Allocates and registers metadata arrays (data_ptrs, indices_ptrs, lengths) with HandlerRegistry:
1. Allocates aligned memory for pointer arrays (primary_dim elements each)
2. Allocates aligned memory for lengths array (primary_dim Index elements)
3. Zero-initializes all arrays
4. Registers all arrays with HandlerRegistry
5. Returns Sparse view pointing to allocated arrays

**Edge Cases**

- **Allocation failure**: Returns empty Sparse (all pointers nullptr)
- **Zero dimensions**: Returns valid but empty matrix
- **Large dimensions**: May fail if allocation exceeds system limits

**Data Guarantees (Preconditions)**

- rows >= 0, cols >= 0, total_nnz >= 0
- HandlerRegistry must be available (via `scl::get_registry()`)

**Complexity Analysis**

- **Time**: O(primary_dim) for zero-initialization
- **Space**: O(primary_dim) for metadata arrays

**Example**

```cpp
#include "scl/core/sparse.hpp"

// Create CSR matrix with registered metadata
auto matrix = scl::Sparse<Real, true>::new_registered(
    1000,    // rows
    500,     // cols
    10000    // total_nnz (informational, not used for allocation)
);

// Matrix now has valid metadata arrays
// But actual row data arrays must be allocated separately
// and assigned to data_ptrs[i] and indices_ptrs[i]

// When done, unregister before transferring to Python
matrix.unregister_metadata();
```

---

### unregister_metadata

Unregister metadata arrays from HandlerRegistry.

::: source_code file="scl/core/sparse.hpp" symbol="unregister_metadata" collapsed
:::

**Algorithm Description**

Unregisters the three metadata arrays (data_ptrs, indices_ptrs, lengths) from HandlerRegistry and sets all pointers to nullptr, making the matrix invalid.

Used before transferring ownership of metadata arrays to Python or other external code.

**Edge Cases**

- **Already invalid**: Safe to call, does nothing
- **Not registered**: Safe to call, may log warning but does not crash

**Data Guarantees (Preconditions)**

- None - safe to call even if matrix is invalid

**Complexity Analysis**

- **Time**: O(1) - registry lookup and pointer reset
- **Space**: O(1)

**Example**

```cpp
auto matrix = scl::Sparse<Real, true>::new_registered(1000, 500, 10000);

// Use matrix...

// Before transferring ownership to Python
matrix.unregister_metadata();
// Matrix is now invalid (all pointers nullptr)
// Metadata arrays are still allocated but no longer tracked by registry
```

---

## Utility Methods

### rows / cols / nnz / primary_dim / secondary_dim

Dimension accessors, all O(1).

**Complexity**: O(1) for all methods

### empty / valid

State queries, all O(1).

**Complexity**: O(1) for all methods

---

## Type Aliases

```cpp
using CSR = Sparse<Real, true>;   // CSR matrix with Real values
using CSC = Sparse<Real, false>;  // CSC matrix with Real values
```

## Design Notes

### Non-Owning View

Sparse is a non-owning view - it does NOT:
- Allocate or free data arrays
- Manage lifetime of underlying memory
- Copy data

The caller is responsible for:
- Allocating data arrays
- Managing their lifetime
- Ensuring they outlive the Sparse object

### Discontiguous vs Contiguous

**Contiguous format** (traditional CSR):
- Single data array: `[v0, v1, v2, ...]`
- Single indices array: `[i0, i1, i2, ...]`
- Indptr array: `[0, 3, 5, 9, ...]`
- Cache-friendly for sequential access
- Less flexible for heterogeneous data

**Discontiguous format** (SCL Sparse):
- Pointer array: `[ptr0, ptr1, ptr2, ...]`
- Each pointer points to separate allocation
- Length array: `[len0, len1, len2, ...]`
- More flexible (wrap external data)
- May have cache misses (pointer indirection)

### Registry Integration

When using `new_registered()`:
- Metadata arrays are registered with HandlerRegistry
- Enables automatic tracking and cleanup
- Call `unregister_metadata()` before transferring to Python
- Actual data arrays can be registered separately if needed

## See Also

- [Type System](./types) - `` `Array<T>` `` type used for views
- [Registry](./registry) - HandlerRegistry for memory tracking
- [Memory Management](./memory) - Aligned allocation functions
