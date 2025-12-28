# C API Overview

## Introduction

The SCL-Core C API provides a stable, language-agnostic interface to the high-performance sparse matrix operations implemented in C++. It is designed with two layers:

1. **Safe API**: Type-safe opaque handles with automatic memory management
2. **Unsafe API**: Direct struct access for zero-overhead operations (ABI unstable)

## Design Principles

### Opaque Handle Pattern

The C API uses forward-declared structs as type-safe opaque handles:

```c
typedef struct scl_sparse_matrix* scl_sparse_t;
typedef struct scl_dense_matrix* scl_dense_t;
```

This provides:
- Type safety at compile time
- ABI stability across library versions
- Clear ownership semantics
- Automatic resource management via registry

### Dynamic Dispatch

A single `scl_sparse_t` handle can represent both CSR and CSC matrices. The format is determined by the `is_csr` flag at creation time, and operations dispatch dynamically to the appropriate implementation.

### Factory Pattern

All matrix creation goes through factory functions:

```c
// Copy data (safe, registry-managed)
scl_sparse_create(&matrix, rows, cols, nnz, indptr, indices, data, is_csr);

// Wrap external data (zero-copy, caller manages lifetime)
scl_sparse_wrap(&matrix, rows, cols, nnz, indptr, indices, data, is_csr);
```

### Registry Integration

All allocations are managed by the internal registry system:
- Automatic reference counting for shared data
- Safe cleanup on destruction
- Memory leak detection in debug builds
- Thread-safe allocation and deallocation

## API Layers

### Core API (core.h)

Basic types and error handling:
- Opaque handle types
- Error codes and error message retrieval
- Thread-local error state

### Sparse Matrix API (sparse.h)

Safe operations on sparse matrices:
- Lifecycle: create, wrap, clone, destroy
- Properties: rows, cols, nnz, format
- Data export: get pointers, copy data
- Conversion: transpose, to contiguous

### Dense Matrix API (dense.h)

Safe operations on dense matrices:
- Lifecycle: create, wrap, clone, destroy
- Properties: rows, cols, stride
- Data access: get pointer, export
- Conversion: to sparse

### Unsafe API (unsafe.h)

Direct struct access (use with caution):
- Raw struct layouts matching C++ internals
- Conversion between safe handles and raw structs
- Zero-overhead access for advanced users
- ABI unstable - may change between versions

## Error Handling

All API functions return `scl_error_t`:

```c
scl_error_t err = scl_sparse_create(&matrix, ...);
if (err != SCL_OK) {
    const char* msg = scl_get_last_error();
    fprintf(stderr, "Error: %s\n", msg);
    return err;
}
```

Error codes:
- `SCL_OK` - Success
- `SCL_ERROR_NULL_POINTER` - Null pointer argument
- `SCL_ERROR_INVALID_ARGUMENT` - Invalid argument value
- `SCL_ERROR_DIMENSION_MISMATCH` - Incompatible dimensions
- `SCL_ERROR_OUT_OF_MEMORY` - Memory allocation failed
- `SCL_ERROR_INTERNAL` - Internal error

## Memory Management

### Safe API

Memory is automatically managed:

```c
scl_sparse_t matrix;
scl_sparse_create(&matrix, ...);  // Allocates and registers

// Use matrix...

scl_sparse_destroy(&matrix);       // Cleans up and sets to NULL
```

### Wrapped Data

For zero-copy wrapping, caller owns the data:

```c
float* data = malloc(...);
scl_sparse_t matrix;
scl_sparse_wrap(&matrix, ..., data, ...);  // No copy

// Use matrix...

scl_sparse_destroy(&matrix);  // Only frees wrapper, not data
free(data);                   // Caller frees data
```

### Unsafe API

Direct struct access bypasses registry:

```c
scl_sparse_raw_t raw;
scl_sparse_unsafe_get_raw(matrix, &raw);

// Direct access to raw.data_ptrs, etc.
// WARNING: Modifying these may corrupt registry state!
```

## Thread Safety

All API functions are thread-safe:
- Registry uses fine-grained locking
- Error state is thread-local
- Matrix objects are immutable after creation (except in-place operations)

## Type Configuration

Types adapt to compile-time configuration:

```c
// Configured via SCL_USE_FLOAT32/FLOAT64/FLOAT16
typedef float/double/_Float16 scl_real_t;

// Configured via SCL_USE_INT16/INT32/INT64  
typedef int16_t/int32_t/int64_t scl_index_t;
```

## Example Usage

### Creating and Using a Sparse Matrix

```c
#include "scl/binding/c_api/sparse.h"

// Traditional CSR format data
scl_index_t rows = 3, cols = 3, nnz = 5;
scl_index_t indptr[] = {0, 2, 3, 5};
scl_index_t indices[] = {0, 2, 1, 0, 2};
scl_real_t data[] = {1.0, 2.0, 3.0, 4.0, 5.0};

// Create CSR matrix
scl_sparse_t matrix;
scl_error_t err = scl_sparse_create(
    &matrix, rows, cols, nnz,
    indptr, indices, data,
    1  // is_csr = true
);

if (err != SCL_OK) {
    fprintf(stderr, "Error: %s\n", scl_get_last_error());
    return err;
}

// Query properties
scl_index_t mat_rows, mat_cols, mat_nnz;
scl_sparse_rows(matrix, &mat_rows);
scl_sparse_cols(matrix, &mat_cols);
scl_sparse_nnz(matrix, &mat_nnz);

printf("Matrix: %ld x %ld with %ld nonzeros\n", 
       mat_rows, mat_cols, mat_nnz);

// Transpose to CSC
scl_sparse_t transposed;
scl_sparse_transpose(matrix, &transposed);

// Clean up
scl_sparse_destroy(&transposed);
scl_sparse_destroy(&matrix);
```

### Converting Dense to Sparse

```c
#include "scl/binding/c_api/dense.h"
#include "scl/binding/c_api/sparse.h"

// Dense matrix data (row-major)
scl_real_t dense_data[] = {
    1.0, 0.0, 2.0,
    0.0, 3.0, 0.0,
    4.0, 0.0, 5.0
};

// Create dense matrix
scl_dense_t dense;
scl_dense_create(&dense, 3, 3, dense_data);

// Convert to sparse CSR with threshold
scl_sparse_t sparse;
scl_dense_to_sparse(dense, &sparse, 1, 1e-10);

// Clean up
scl_sparse_destroy(&sparse);
scl_dense_destroy(&dense);
```

## See Also

- [Sparse Matrix API Reference](sparse.md)
- [Dense Matrix API Reference](dense.md)
- [Unsafe API Reference](unsafe.md)
- [Error Handling Guide](errors.md)

