---
title: C API Reference
description: SCL Core C API documentation
---

# C API Reference

The C API provides a stable ABI surface for integration with Python, R, Julia, and other languages.

## Quick Start

```c
#include <scl/c_api.h>

// Create a sparse matrix
scl_sparse_t matrix;
scl_error_t err = scl_sparse_create(&matrix, rows, cols, nnz,
                                     indptr, indices, data, true);
if (scl_is_error(err)) {
    printf("Error: %s\n", scl_get_last_error());
    return 1;
}

// Use the matrix...
scl_algebra_spmv(matrix, x, x_size, y, y_size, 1.0, 0.0);

// Clean up
scl_sparse_destroy(&matrix);
```

## Error Handling

All functions return `scl_error_t`. Check with `scl_is_ok()` or `scl_is_error()`:

```c
scl_error_t err = scl_some_function(...);
if (scl_is_error(err)) {
    const char* msg = scl_get_last_error();
    // Handle error
}
```

## Modules

### Core

| Module | Description |
|--------|-------------|
| [core](./core) | Version info, error handling, basic types |
| [sparse](./sparse) | Sparse matrix creation and manipulation |
| [dense](./dense) | Dense matrix views (zero-copy) |

### Linear Algebra

| Module | Description |
|--------|-------------|
| [algebra](./algebra) | SpMV, SpMM, row operations |

### Preprocessing

| Module | Description |
|--------|-------------|
| [normalize](./normalize) | Data normalization |
| [scale](./scale) | Row/column scaling |
| [log1p](./log1p) | Log transformation |

### Quality Control

| Module | Description |
|--------|-------------|
| [qc](./qc) | Basic QC metrics |
| [hvg](./hvg) | Highly variable genes |

### Clustering

| Module | Description |
|--------|-------------|
| [leiden](./leiden) | Leiden clustering |
| [louvain](./louvain) | Louvain clustering |
| [components](./components) | Connected components |

### Statistics

| Module | Description |
|--------|-------------|
| [ttest](./ttest) | T-test |
| [mwu](./mwu) | Mann-Whitney U test |
| [ks](./ks) | Kolmogorov-Smirnov test |
| [auroc](./auroc) | AUROC computation |

## Type Reference

### Core Types

```c
typedef double scl_real_t;           // Numeric type
typedef int64_t scl_index_t;         // Index type
typedef size_t scl_size_t;           // Size type
typedef int scl_bool_t;              // Boolean type
typedef int32_t scl_error_t;         // Error code type

typedef struct scl_sparse_matrix* scl_sparse_t;  // Opaque sparse handle
typedef struct scl_dense_matrix* scl_dense_t;    // Opaque dense handle
```

### Error Codes

| Code | Value | Description |
|------|-------|-------------|
| `SCL_OK` | 0 | Success |
| `SCL_ERROR_UNKNOWN` | 1 | Unknown error |
| `SCL_ERROR_OUT_OF_MEMORY` | 2 | Memory allocation failed |
| `SCL_ERROR_NULL_POINTER` | 4 | Null pointer argument |
| `SCL_ERROR_DIMENSION_MISMATCH` | 11 | Array dimensions don't match |
| `SCL_ERROR_INVALID_ARGUMENT` | 12 | Invalid argument value |
| `SCL_ERROR_NOT_IMPLEMENTED` | 40 | Feature not implemented |

## Thread Safety

<Callout type="info" title="Thread Safety Guidelines">

- **Read-only operations** are generally thread-safe
- **In-place modifications** are NOT thread-safe on the same data
- Use separate matrix handles for concurrent writes
- Error messages are stored in thread-local storage

</Callout>

## Memory Management

<Callout type="warning" title="Memory Ownership">

- Matrices created with `scl_*_create()` must be destroyed with `scl_*_destroy()`
- View functions (`scl_*_view()`) share memory with the source - do not destroy source first
- Output arrays are typically caller-allocated unless documented otherwise

</Callout>
