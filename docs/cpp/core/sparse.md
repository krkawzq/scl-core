# Sparse Matrix

The `Sparse<T, IsCSR>` struct is SCL-Core's primary sparse matrix data structure, using discontiguous storage for maximum flexibility.

## Overview

`Sparse` provides:

- **Discontiguous storage** - Each row/column can be in a separate allocation
- **Flexible integration** - Wrap external data without copying
- **Memory-mapped support** - Lazy loading of rows/columns
- **Python interop** - Zero-copy integration with NumPy/SciPy

## Template Parameters

```cpp
template <typename T, bool IsCSR>
struct Sparse {
    // ...
};
```

- **`T`** - Element type (typically `Real`)
- **`IsCSR`** - Format selector:
  - `true` = CSR (Compressed Sparse Row, row-major)
  - `false` = CSC (Compressed Sparse Column, column-major)

## Design Philosophy

### Discontiguous vs Contiguous

**Traditional CSR/CSC (Contiguous):**

```
data    = [v0, v1, v2, v3, v4, v5, ...]  // Single allocation
indices = [c0, c1, c2, c3, c4, c5, ...]  // Single allocation
indptr  = [0, 3, 5, ...]                 // Offsets into data/indices
```

**SCL-Core Sparse (Discontiguous):**

```
data_ptrs    = [ptr_to_row0, ptr_to_row1, ...]  // Pointer array
indices_ptrs = [ptr_to_row0, ptr_to_row1, ...]  // Pointer array
lengths      = [len0, len1, ...]                // Length array
```

### Benefits of Discontiguous Storage

1. **Wrap heterogeneous data** - Each row can come from different source
2. **Lazy loading** - Load rows on demand from disk/network
3. **Memory-mapped matrices** - Row-level granularity for mmap
4. **Python integration** - Wrap NumPy arrays without copying
5. **Flexible allocation** - Block-based allocation for efficiency

## Memory Layout

### CSR Example (3 rows)

```cpp
Sparse<Real, true> matrix;  // CSR

// Pointer arrays
matrix.data_ptrs[0] → [1.0, 2.0, 3.0]       // Row 0: 3 elements
matrix.data_ptrs[1] → [4.0, 5.0]            // Row 1: 2 elements
matrix.data_ptrs[2] → [6.0, 7.0, 8.0, 9.0]  // Row 2: 4 elements

matrix.indices_ptrs[0] → [0, 2, 4]          // Row 0 column indices
matrix.indices_ptrs[1] → [1, 3]             // Row 1 column indices
matrix.indices_ptrs[2] → [0, 1, 2, 3]       // Row 2 column indices

matrix.lengths = [3, 2, 4]                  // Lengths

// Dimensions
matrix.rows_ = 3
matrix.cols_ = 5
matrix.nnz_ = 9
```

### CSC Example (same data)

```cpp
Sparse<Real, false> matrix;  // CSC

// Pointer arrays (now per-column)
matrix.data_ptrs[0] → [1.0, 6.0]       // Column 0: 2 elements
matrix.data_ptrs[1] → [4.0, 7.0]       // Column 1: 2 elements
// ... etc
```

## Basic Usage

### Creating Matrices

**From scratch:**

```cpp
// Create with block allocation
auto matrix = Sparse<Real, true>::create(
    rows, cols, nnz_per_row);

// Empty matrix
auto empty = Sparse<Real, true>::zeros(rows, cols);
```

**From traditional CSR/CSC:**

```cpp
// Copy data
auto matrix = Sparse<Real, true>::from_traditional(
    data, indices, indptr, rows, cols, nnz);

// Wrap existing data (zero-copy)
auto view = Sparse<Real, true>::wrap_traditional(
    data, indices, indptr, rows, cols, nnz);
```

**From COO:**

```cpp
auto matrix = Sparse<Real, true>::from_coo(
    row_indices, col_indices, values, 
    rows, cols, nnz);
```

**From dense:**

```cpp
Real dense_data[9] = {
    1, 0, 2,
    0, 3, 0,
    4, 0, 5
};

auto matrix = Sparse<Real, true>::from_dense(
    dense_data, 3, 3, 0.0);  // threshold = 0.0
```

### Accessing Data

**Row/column access (CSR):**

```cpp
Sparse<Real, true> matrix;  // CSR

for (Index i = 0; i < matrix.rows(); ++i) {
    // Get row data
    auto vals = matrix.primary_values(i);
    auto idxs = matrix.primary_indices(i);
    Index len = matrix.primary_length(i);
    
    // Iterate over non-zeros in row i
    for (Index j = 0; j < len; ++j) {
        Real value = vals.ptr[j];
        Index col = idxs.ptr[j];
        
        std::cout << "(" << i << ", " << col << ") = " << value << "\n";
    }
}
```

**Element access:**

```cpp
// Check if element exists
if (matrix.has_element(i, j)) {
    Real value = matrix.get_element(i, j);
}

// Get with default
Real value = matrix.get_element_or(i, j, 0.0);
```

### Modifying Data

**In-place operations:**

```cpp
// Sort indices within each row/column
matrix.sort_indices();

// Scale all values
matrix.scale(2.0);

// Sum duplicates (if any)
matrix.sum_duplicates();
```

**Slicing:**

```cpp
// View slice (zero-copy)
auto sub = matrix.slice_view(row_start, row_end);

// Copy slice
auto sub_copy = matrix.slice_copy(row_start, row_end);
```

## Factory Methods

### create

Create with block allocation:

```cpp
static Sparse create(
    Index rows, 
    Index cols, 
    Index nnz_per_primary
);
```

**Example:**

```cpp
// Create 1000x500 CSR matrix with ~100 nnz per row
auto matrix = Sparse<Real, true>::create(1000, 500, 100);
```

### zeros

Create empty matrix:

```cpp
static Sparse zeros(Index rows, Index cols);
```

### from_traditional

Convert from contiguous CSR/CSC:

```cpp
static Sparse from_traditional(
    const T* data,
    const Index* indices,
    const Index* indptr,
    Index rows,
    Index cols,
    Index nnz
);
```

### wrap_traditional

Wrap contiguous CSR/CSC (zero-copy):

```cpp
static Sparse wrap_traditional(
    T* data,
    Index* indices,
    const Index* indptr,
    Index rows,
    Index cols,
    Index nnz
);
```

**Warning:** Caller must ensure data lifetime.

### from_coo

Convert from COO format:

```cpp
static Sparse from_coo(
    const Index* row_indices,
    const Index* col_indices,
    const T* values,
    Index rows,
    Index cols,
    Index nnz
);
```

### identity

Create identity matrix:

```cpp
static Sparse identity(Index n);
```

## Conversions

### Transpose

```cpp
// Create transposed copy
auto transposed = matrix.transpose();

// CSR → CSC or CSC → CSR
```

### To Contiguous

```cpp
// Convert to contiguous CSR/CSC
auto contiguous = matrix.to_contiguous();
```

### To Traditional Arrays

```cpp
#include "scl/kernel/sparse.hpp"

// Export to traditional CSR/CSC arrays
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// arrays.data, arrays.indices, arrays.indptr
// are registered with scl::Registry
```

### To COO

```cpp
#include "scl/kernel/sparse.hpp"

// Export to COO format
auto coo = scl::kernel::sparse::to_coo_arrays(matrix);

// coo.row_indices, coo.col_indices, coo.values
// are registered with scl::Registry
```

### To Dense

```cpp
// Export to dense array
std::vector<T> dense = matrix.to_dense();

// Or with pre-allocated buffer
T* dense_buffer = new T[rows * cols];
matrix.to_dense(dense_buffer);
```

## Ownership and Lifetime

### Non-Owning View

`Sparse` is a **non-owning view** by default:

```cpp
// External data
Real* data = /* ... */;
Index* indices = /* ... */;
Index* indptr = /* ... */;

// Wrap as Sparse (zero-copy)
auto matrix = Sparse<Real, true>::wrap_traditional(
    data, indices, indptr, rows, cols, nnz);

// matrix does not own data
// Caller must ensure data outlives matrix
```

### Registry-Managed

Use factory methods for automatic memory management:

```cpp
// create() registers memory with scl::Registry
auto matrix = Sparse<Real, true>::create(rows, cols, nnz_per_row);

// Memory automatically tracked
// Can transfer to Python without copying
```

### Manual Registration

```cpp
auto& reg = scl::get_registry();

// Allocate
Real* data = reg.new_array<Real>(nnz);
Index* indices = reg.new_array<Index>(nnz);
// ...

// Create Sparse view
auto matrix = Sparse<Real, true>::wrap_traditional(
    data, indices, indptr, rows, cols, nnz);

// Cleanup
reg.unregister_ptr(data);
reg.unregister_ptr(indices);
// ...
```

## Performance Considerations

### Cache Performance

**Discontiguous storage:**
- Less cache-friendly for sequential access
- More cache misses when iterating rows
- Better for random access patterns

**Contiguous storage:**
- Better cache locality
- Faster sequential iteration
- Less flexible

**Recommendation:**
- Use `Sparse` for flexibility and Python interop
- Convert to contiguous for performance-critical kernels
- Use `to_contiguous()` or `to_contiguous_arrays()`

### Block Allocation

`Sparse::create()` uses block allocation:

```cpp
struct BlockStrategy {
    Index min_block_elements = 4096;      // 16KB for float32
    Index max_block_elements = 262144;    // 1MB for float32
    
    // Balance:
    // - Memory reuse (larger blocks)
    // - Partial release (smaller blocks)
    // - Parallelism (multiple blocks)
};
```

**Benefits:**
- Fewer allocations
- Better memory locality within blocks
- Efficient for typical sparse matrices

### Memory Overhead

**Per matrix:**
- Pointer arrays: `2 * primary_dim * sizeof(void*)`
- Length array: `primary_dim * sizeof(Index)`
- Metadata: `3 * sizeof(Index)`

**Example (1000 rows, 64-bit pointers):**
- Pointer arrays: `2 * 1000 * 8 = 16KB`
- Length array: `1000 * 4 = 4KB`
- Total overhead: ~20KB

## Thread Safety

**Read operations:**
- Safe for concurrent access
- Multiple threads can read simultaneously

**Write operations:**
- Require external synchronization
- Use locks or ensure single-writer

**Example:**

```cpp
// Safe: Concurrent reads
parallel_for(Size(0), matrix.rows(), [&](Index i) {
    auto vals = matrix.primary_values(i);
    // Read-only operations
});

// Unsafe: Concurrent writes
// Need synchronization!
parallel_for(Size(0), matrix.rows(), [&](Index i) {
    auto vals = matrix.primary_values(i);
    vals.ptr[0] = 1.0;  // RACE CONDITION!
});
```

## Advanced Usage

### Custom Allocation

```cpp
// Allocate pointer arrays
Pointer* data_ptrs = new Pointer[rows];
Pointer* indices_ptrs = new Pointer[rows];
Index* lengths = new Index[rows];

// Allocate each row separately
for (Index i = 0; i < rows; ++i) {
    Index len = /* compute length */;
    data_ptrs[i] = new Real[len];
    indices_ptrs[i] = new Index[len];
    lengths[i] = len;
    
    // Fill data...
}

// Create Sparse view
Sparse<Real, true> matrix(
    data_ptrs, indices_ptrs, lengths,
    rows, cols, nnz);

// Use matrix...

// Cleanup
for (Index i = 0; i < rows; ++i) {
    delete[] data_ptrs[i];
    delete[] indices_ptrs[i];
}
delete[] data_ptrs;
delete[] indices_ptrs;
delete[] lengths;
```

### Wrapping Python Arrays

```cpp
// Python side: NumPy arrays
// data_list = [row0_data, row1_data, ...]
// indices_list = [row0_indices, row1_indices, ...]

// C++ side: Extract pointers
std::vector<Real*> data_ptrs(rows);
std::vector<Index*> indices_ptrs(rows);
std::vector<Index> lengths(rows);

for (Index i = 0; i < rows; ++i) {
    py::array_t<Real> row_data = data_list[i];
    py::array_t<Index> row_indices = indices_list[i];
    
    data_ptrs[i] = row_data.mutable_data();
    indices_ptrs[i] = row_indices.mutable_data();
    lengths[i] = row_data.size();
}

// Create Sparse view (zero-copy)
Sparse<Real, true> matrix(
    data_ptrs.data(),
    indices_ptrs.data(),
    lengths.data(),
    rows, cols, nnz);

// Use matrix without copying data!
```

## Best Practices

### 1. Use Factory Methods

```cpp
// GOOD: Automatic memory management
auto matrix = Sparse<Real, true>::create(rows, cols, nnz_per_row);

// BAD: Manual allocation (error-prone)
Pointer* data_ptrs = new Pointer[rows];
// ... complex allocation logic ...
```

### 2. Convert for Performance

```cpp
// For performance-critical kernels
auto contiguous = matrix.to_contiguous();
process_fast(contiguous);

// Or use to_contiguous_arrays() for traditional format
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);
external_library(arrays.data, arrays.indices, arrays.indptr);
```

### 3. Document Ownership

```cpp
// Returns owning matrix
Sparse<Real, true> create_matrix();

// Returns non-owning view
Sparse<Real, true> get_matrix_view();

// Takes ownership
void consume_matrix(Sparse<Real, true>&& matrix);
```

### 4. Check Validity

```cpp
if (matrix.valid()) {
    // Matrix has data
    process(matrix);
} else {
    // Empty or null matrix
}
```

---

::: tip Flexibility vs Performance
`Sparse` prioritizes flexibility. For maximum performance, convert to contiguous format using `to_contiguous()` or `to_contiguous_arrays()`.
:::

