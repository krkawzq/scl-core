# Sparse Tools

Sparse matrix infrastructure tools for conversion, validation, and cleanup.

## Overview

Sparse tools provide:

- **Format Conversion** - Export to CSR/CSC/COO arrays
- **Validation** - Check structural integrity
- **Cleanup** - Remove zeros, prune small values
- **Memory Info** - Query memory usage
- **Layout Conversion** - Make contiguous, resize

## Format Conversion

### to_contiguous_arrays

Export to contiguous CSR/CSC arrays:

```cpp
#include "scl/kernel/sparse.hpp"

Sparse<Real, true> matrix = /* ... */;

// Export to contiguous arrays
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// arrays.data, arrays.indices, arrays.indptr are registered
// with scl::Registry for automatic memory management

// Use arrays...
external_library(arrays.data, arrays.indices, arrays.indptr,
                 arrays.nnz, arrays.primary_dim);

// Cleanup
auto& reg = scl::get_registry();
reg.unregister_ptr(arrays.data);
reg.unregister_ptr(arrays.indices);
reg.unregister_ptr(arrays.indptr);
```

**Returns:** `ContiguousArraysT<T>`

```cpp
template <typename T>
struct ContiguousArraysT {
    T* data;             // registry registered values array
    Index* indices;      // registry registered indices array
    Index* indptr;       // registry registered offset array
    Index nnz;
    Index primary_dim;
};
```

**Use cases:**
- Integration with external libraries (SciPy, cuSPARSE)
- Python bindings (zero-copy transfer)
- Performance-critical kernels (contiguous is faster)

### to_coo_arrays

Export to COO (Coordinate) format:

```cpp
auto coo = scl::kernel::sparse::to_coo_arrays(matrix);

// coo.row_indices, coo.col_indices, coo.values are registered

// Use COO format...
external_coo_library(coo.row_indices, coo.col_indices, coo.values,
                     coo.nnz, coo.rows, coo.cols);

// Cleanup
auto& reg = scl::get_registry();
reg.unregister_ptr(coo.row_indices);
reg.unregister_ptr(coo.col_indices);
reg.unregister_ptr(coo.values);
```

**Returns:** `COOArraysT<T>`

```cpp
template <typename T>
struct COOArraysT {
    Index* row_indices;  // registry registered row indices
    Index* col_indices;  // registry registered column indices
    T* values;           // registry registered values
    Index nnz;
    Index rows;
    Index cols;
};
```

**Use cases:**
- Format conversion
- Interfacing with COO-based libraries
- Sorting by different orders

### from_contiguous_arrays

Create Sparse matrix from contiguous arrays:

```cpp
// Existing CSR arrays
Real* data = /* ... */;
Index* indices = /* ... */;
Index* indptr = /* ... */;

// Wrap as Sparse (zero-copy)
auto matrix = scl::kernel::sparse::from_contiguous_arrays<Real, true>(
    data, indices, indptr, rows, cols, nnz,
    false  // take_ownership = false (just wrap)
);

// Or take ownership (register with Registry)
auto matrix = scl::kernel::sparse::from_contiguous_arrays<Real, true>(
    data, indices, indptr, rows, cols, nnz,
    true  // take_ownership = true (register)
);
```

**Parameters:**
- `take_ownership`: If true, registers arrays with Registry

**Use cases:**
- Loading from external sources
- Python bindings (wrap NumPy arrays)
- Zero-copy integration

## Validation

### validate

Check structural integrity:

```cpp
auto result = scl::kernel::sparse::validate(matrix);

if (result.is_valid) {
    std::cout << "Matrix is valid\n";
} else {
    std::cout << "Matrix is invalid:\n";
    for (const auto& error : result.errors) {
        std::cout << "  - " << error << "\n";
    }
}
```

**Returns:** `ValidationResult`

```cpp
struct ValidationResult {
    bool is_valid;
    std::vector<std::string> errors;
};
```

**Checks:**
- Index bounds (all indices within [0, secondary_dim))
- Sorted indices (within each row/column)
- NNZ consistency (sum of lengths == nnz)
- No duplicates (optional)

**Use cases:**
- Debugging matrix construction
- Validating external data
- Pre-condition checking

## Memory Information

### memory_info

Query memory usage:

```cpp
auto info = scl::kernel::sparse::memory_info(matrix);

std::cout << "Data bytes:    " << info.data_bytes << "\n";
std::cout << "Indices bytes: " << info.indices_bytes << "\n";
std::cout << "Metadata bytes:" << info.metadata_bytes << "\n";
std::cout << "Total bytes:   " << info.total_bytes << "\n";
std::cout << "Block count:   " << info.block_count << "\n";
std::cout << "Is contiguous: " << info.is_contiguous << "\n";
```

**Returns:** `MemoryInfo`

```cpp
struct MemoryInfo {
    Size data_bytes;       // Bytes for values
    Size indices_bytes;    // Bytes for indices
    Size metadata_bytes;   // Bytes for pointers/lengths
    Size total_bytes;      // Total memory usage
    Index block_count;     // Number of memory blocks
    bool is_contiguous;    // True if single-block layout
};
```

**Use cases:**
- Memory profiling
- Optimization decisions
- Debugging memory usage

## Cleanup Operations

### eliminate_zeros

Remove zero-valued elements:

```cpp
// Remove exact zeros
scl::kernel::sparse::eliminate_zeros(matrix);

// Remove near-zeros (within tolerance)
scl::kernel::sparse::eliminate_zeros(matrix, 1e-10);
```

**Parameters:**
- `tolerance`: Values with |x| <= tolerance are removed

**Effects:**
- Removes zero/near-zero elements
- Updates nnz and lengths
- Preserves matrix structure (rows/cols unchanged)

**Performance:**
- Parallel processing
- Two-pass algorithm (count, then copy)
- Efficient memory reuse

**Use cases:**
- Cleanup after arithmetic operations
- Reduce memory usage
- Improve performance (fewer elements to process)

### prune

Remove small-valued elements:

```cpp
// Remove elements with |value| < threshold
scl::kernel::sparse::prune(matrix, threshold);

// Or set to zero while preserving structure
scl::kernel::sparse::prune(matrix, threshold, true);
```

**Parameters:**
- `threshold`: Elements with |x| < threshold are removed/zeroed
- `keep_structure`: If true, set to zero; if false, remove

**Effects:**
- Removes or zeros small elements
- Updates nnz if removing
- Preserves structure if keep_structure=true

**Use cases:**
- Sparsification
- Noise reduction
- Memory optimization

## Layout Conversion

### make_contiguous

Convert to contiguous layout:

```cpp
// Check if already contiguous
auto info = scl::kernel::sparse::memory_info(matrix);
if (!info.is_contiguous) {
    // Convert to contiguous
    auto contiguous = scl::kernel::sparse::make_contiguous(matrix);
    
    // Use contiguous matrix for better performance
    process_fast(contiguous);
}
```

**Effects:**
- Creates new matrix with contiguous storage
- All rows/columns in single memory block
- Better cache locality

**Use cases:**
- Performance optimization
- Preparing for external libraries
- Sequential access patterns

### resize_secondary

Resize secondary dimension:

```cpp
// Resize columns (for CSR) or rows (for CSC)
scl::kernel::sparse::resize_secondary(matrix, new_secondary_dim);
```

**Parameters:**
- `new_secondary_dim`: New size for secondary dimension

**Effects:**
- Updates secondary dimension metadata
- Debug mode: Asserts no out-of-bounds indices if shrinking
- Does NOT modify data (metadata-only operation)

**Use cases:**
- Dimension adjustment
- Subset operations
- Matrix slicing

## Performance Considerations

### Parallelization

Most operations are parallelized:

```cpp
// to_coo_arrays: Parallel offset computation and copying
// eliminate_zeros: Parallel counting and copying
// validate: Parallel checking
```

### Memory Efficiency

- **Two-pass algorithms**: Count first, allocate once
- **In-place when possible**: eliminate_zeros, prune
- **Registry management**: Automatic cleanup

### Cache Optimization

- **Sequential access**: Optimized for cache lines
- **Prefetching**: For predictable patterns
- **Block processing**: Better locality

## Best Practices

### 1. Validate External Data

```cpp
// Always validate data from external sources
auto matrix = load_from_file(filename);

auto result = scl::kernel::sparse::validate(matrix);
if (!result.is_valid) {
    std::cerr << "Invalid matrix:\n";
    for (const auto& error : result.errors) {
        std::cerr << "  " << error << "\n";
    }
    return;
}
```

### 2. Convert for Performance

```cpp
// Convert to contiguous for performance-critical sections
auto info = scl::kernel::sparse::memory_info(matrix);
if (!info.is_contiguous) {
    auto contiguous = scl::kernel::sparse::make_contiguous(matrix);
    // Use contiguous for better performance
    process_intensive(contiguous);
}
```

### 3. Cleanup After Operations

```cpp
// After arithmetic that may produce zeros
matrix = matrix + other_matrix;
scl::kernel::sparse::eliminate_zeros(matrix);

// After thresholding
apply_threshold(matrix, threshold);
scl::kernel::sparse::prune(matrix, threshold);
```

### 4. Profile Memory Usage

```cpp
void print_matrix_info(const auto& matrix) {
    auto info = scl::kernel::sparse::memory_info(matrix);
    
    std::cout << "Matrix: " << matrix.rows() << " x " << matrix.cols() 
              << ", nnz = " << matrix.nnz() << "\n";
    std::cout << "Memory: " << info.total_bytes / 1024.0 / 1024.0 << " MB\n";
    std::cout << "Blocks: " << info.block_count 
              << (info.is_contiguous ? " (contiguous)" : " (discontiguous)") 
              << "\n";
}
```

## Examples

### Python Integration

```cpp
// C++ side: Export to contiguous arrays
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// Python binding: Zero-copy transfer
py::capsule data_deleter(arrays.data, [](void* ptr) {
    scl::get_registry().unregister_ptr(ptr);
});
py::capsule indices_deleter(arrays.indices, [](void* ptr) {
    scl::get_registry().unregister_ptr(ptr);
});
py::capsule indptr_deleter(arrays.indptr, [](void* ptr) {
    scl::get_registry().unregister_ptr(ptr);
});

return py::make_tuple(
    py::array_t<Real>({arrays.nnz}, {sizeof(Real)}, 
                      arrays.data, data_deleter),
    py::array_t<Index>({arrays.nnz}, {sizeof(Index)}, 
                       arrays.indices, indices_deleter),
    py::array_t<Index>({arrays.primary_dim + 1}, {sizeof(Index)}, 
                       arrays.indptr, indptr_deleter)
);
```

### External Library Integration

```cpp
// Convert to format expected by external library
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// Call external library (e.g., cuSPARSE, MKL)
external_sparse_mv(arrays.data, arrays.indices, arrays.indptr,
                   arrays.primary_dim, matrix.cols(), arrays.nnz,
                   x, y);

// Cleanup
auto& reg = scl::get_registry();
reg.unregister_ptr(arrays.data);
reg.unregister_ptr(arrays.indices);
reg.unregister_ptr(arrays.indptr);
```

---

::: tip Registry Management
All exported arrays are registered with `scl::Registry`. Remember to unregister them when done to avoid memory leaks.
:::

