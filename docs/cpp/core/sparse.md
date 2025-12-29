---
title: Sparse Matrices
description: Sparse matrix types, operations, and memory management
---

# Sparse Matrices

SCL-Core provides a high-performance sparse matrix implementation with block-allocated discontiguous storage, reference counting, and zero-copy slicing.

## Overview

The sparse matrix implementation (`Sparse<T, IsCSR>`) supports both CSR (Compressed Sparse Row) and CSC (Compressed Sparse Column) formats with:

- **Block Allocation**: Flexible memory layout for efficient memory reuse
- **Reference Counting**: Automatic lifetime management via registry
- **Zero-Copy Slicing**: Efficient submatrix operations
- **Sorted Invariant**: Indices within each row/column are strictly ascending

## Type Definitions

```cpp
namespace scl {
    // CSR matrix (row-major)
    template <typename T>
    using CSRMatrix = Sparse<T, true>;
    using CSR = CSRMatrix<Real>;
    
    // CSC matrix (column-major)
    template <typename T>
    using CSCMatrix = Sparse<T, false>;
    using CSC = CSCMatrix<Real>;
}
```

## Creating Sparse Matrices

### Factory Methods

```cpp
// Create empty matrix
auto matrix = CSR::create(rows, cols, nnz);

// Create with block allocation strategy
auto matrix = CSR::create(
    rows, cols, nnz,
    BlockStrategy::adaptive()  // or contiguous(), small_blocks(), large_blocks()
);

// Create from traditional CSR arrays (zero-copy)
auto matrix = CSR::from_traditional(
    rows, cols, nnz,
    row_ptr, col_indices, values
);

// Wrap existing arrays (caller manages lifetime)
auto matrix = CSR::wrap_traditional(
    rows, cols, nnz,
    row_ptr, col_indices, values
);
```

### Block Allocation Strategies

```cpp
namespace scl {
    struct BlockStrategy {
        Index min_block_elements = 4096;
        Index max_block_elements = 262144;
        Index target_block_count = 0;  // 0 = auto
        bool force_contiguous = false;
        
        // Predefined strategies
        static constexpr BlockStrategy contiguous();  // Traditional CSR
        static constexpr BlockStrategy small_blocks();
        static constexpr BlockStrategy large_blocks();
        static constexpr BlockStrategy adaptive();   // Default
    };
}
```

**Strategy Selection**:
- `contiguous()`: Single block, compatible with traditional CSR format
- `adaptive()`: Balanced for most use cases (default)
- `small_blocks()`: Many small blocks for fine-grained memory control
- `large_blocks()`: Few large blocks for reduced fragmentation

## Accessing Matrix Data

### Dimensions

```cpp
Index n_rows = matrix.rows();
Index n_cols = matrix.cols();
Index nnz = matrix.nnz();
```

### Row/Column Access (CSR)

```cpp
// Get row data
Index row = 0;
Array<Index> indices = matrix.row_indices(row);
Array<Real> values = matrix.row_values(row);
Index length = matrix.row_length(row);

// Unsafe access (no bounds checking)
Array<Index> indices = matrix.row_indices_unsafe(row);
Array<Real> values = matrix.row_values_unsafe(row);
Index length = matrix.row_length_unsafe(row);
```

### Column Access (CSC)

```cpp
// Get column data
Index col = 0;
Array<Index> indices = matrix.col_indices(col);
Array<Real> values = matrix.col_values(col);
Index length = matrix.col_length(col);
```

### Primary Dimension Access

For generic code that works with both CSR and CSC:

```cpp
// Primary dimension: rows for CSR, columns for CSC
Index primary_dim = matrix.primary_dim();
Index secondary_dim = matrix.secondary_dim();

// Access primary dimension data
Array<Index> indices = matrix.primary_indices(row_or_col);
Array<Real> values = matrix.primary_values(row_or_col);
Index length = matrix.primary_length(row_or_col);
```

## Matrix Operations

### Element Access

```cpp
// Get element at (row, col)
// Returns optional<Real> (empty if element doesn't exist)
auto value = matrix.get(row, col);
if (value.has_value()) {
    Real val = value.value();
}

// Set element (creates if doesn't exist)
matrix.set(row, col, value);

// Check if element exists
bool exists = matrix.has(row, col);
```

### Slicing

```cpp
// Slice rows [start, end)
auto submatrix = matrix.slice_rows(start_row, end_row);

// Slice columns [start, end)
auto submatrix = matrix.slice_cols(start_col, end_col);

// Slice both dimensions
auto submatrix = matrix.slice(
    start_row, end_row,
    start_col, end_col
);

// Slicing is zero-copy (shares data via reference counting)
```

### Transposition

```cpp
// Create transpose (zero-copy for CSR/CSC)
auto transposed = matrix.transpose();

// Transpose in-place (requires temporary storage)
matrix.transpose_inplace();
```

### Cloning

```cpp
// Deep copy
auto copy = matrix.clone();

// Clone with different format
CSC csc_copy = csr_matrix.clone_as_csc();
```

## Memory Management

### Registry System

SCL-Core uses a registry system for automatic memory management:

```cpp
// Data is automatically registered when created via factory methods
auto matrix = CSR::create(rows, cols, nnz);
// Data is registered in the registry

// Slicing creates aliases (zero-copy)
auto submatrix = matrix.slice_rows(0, 100);
// submatrix shares data with matrix via reference counting

// When matrix is destroyed, data is released if ref count reaches zero
```

### Manual Registration

For external data (e.g., from Python/NumPy):

```cpp
// Register external arrays
registry::alias_incref(data_ptr, size);
registry::alias_incref(indices_ptr, size);

// Use in matrix
auto matrix = CSR::wrap_traditional(rows, cols, nnz, ...);

// Unregister when done
registry::alias_decref(data_ptr);
registry::alias_decref(indices_ptr);
```

## Layout Information

```cpp
// Get memory layout information
SparseLayoutInfo info = matrix.layout_info();

// Access layout details
Index data_blocks = info.data_block_count;
Index index_blocks = info.index_block_count;
std::size_t data_bytes = info.data_bytes;
std::size_t index_bytes = info.index_bytes;
bool is_contiguous = info.is_contiguous;
bool is_traditional = info.is_traditional_format;
```

## Iteration Patterns

### Row Iteration (CSR)

```cpp
Index n_rows = matrix.rows();
for (Index i = 0; i < n_rows; ++i) {
    auto indices = matrix.row_indices(i);
    auto values = matrix.row_values(i);
    Index len = matrix.row_length(i);
    
    for (Index k = 0; k < len; ++k) {
        Index col = indices[k];
        Real val = values[k];
        // Process element
    }
}
```

### Column Iteration (CSC)

```cpp
Index n_cols = matrix.cols();
for (Index j = 0; j < n_cols; ++j) {
    auto indices = matrix.col_indices(j);
    auto values = matrix.col_values(j);
    Index len = matrix.col_length(j);
    
    for (Index k = 0; k < len; ++k) {
        Index row = indices[k];
        Real val = values[k];
        // Process element
    }
}
```

### Parallel Iteration

```cpp
// Parallel row processing
threading::parallel_for(Size(0), static_cast<Size>(matrix.rows()), [&](size_t i) {
    auto row = matrix.row_values(static_cast<Index>(i));
    // Process row in parallel
});
```

## Best Practices

### 1. Choose Correct Format

```cpp
// Row-based operations → CSR
CSR matrix = CSR::create(rows, cols, nnz);
// Fast row access, slow column access

// Column-based operations → CSC
CSC matrix = CSC::create(rows, cols, nnz);
// Fast column access, slow row access
```

### 2. Use Unsafe Access in Hot Loops

```cpp
// In performance-critical code
for (Index i = 0; i < n_rows; ++i) {
    // Use unsafe access (no bounds checking)
    auto values = matrix.row_values_unsafe(i);
    auto indices = matrix.row_indices_unsafe(i);
    Index len = matrix.row_length_unsafe(i);
    
    // Process row
}
```

### 3. Leverage Slicing for Submatrix Operations

```cpp
// Instead of creating new matrix
auto submatrix = matrix.slice_rows(0, 100);
// Zero-copy, efficient

// Process submatrix
process(submatrix);
```

### 4. Prefer Factory Methods

```cpp
// Good: Automatic memory management
auto matrix = CSR::create(rows, cols, nnz);

// Avoid: Manual memory management (unless necessary)
auto matrix = CSR::wrap_traditional(...);  // Caller manages lifetime
```

## Performance Considerations

### Memory Layout

- **Contiguous**: Single block, cache-friendly, traditional CSR format
- **Block-allocated**: Multiple blocks, flexible, better for large matrices

### Access Patterns

```cpp
// Good: Sequential row access (CSR)
for (Index i = 0; i < n_rows; ++i) {
    process_row(matrix, i);
}

// Slow: Random column access (CSR)
for (Index j = 0; j < n_cols; ++j) {
    process_col(matrix, j);  // Use CSC instead
}
```

### Sorted Invariant

All indices within each row/column are **strictly ascending**. This invariant is:
- Enforced by factory methods
- Maintained by all operations
- Assumed by all access methods

**Do not** manually modify indices to violate this invariant.

## Related Documentation

- [Core Types](./types.md) - Fundamental types
- [Memory Management](./memory.md) - Registry and allocation
- [Kernels](../kernels/) - Matrix operations and algorithms
