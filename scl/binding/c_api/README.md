# C API Bindings

This directory contains C API bindings for the SCL-Core library.

## Structure

```
c_api/
├── core/           # Core data structures (sparse, dense matrices)
│   ├── core.h      # Basic types and error handling
│   ├── sparse.h    # Sparse matrix API
│   ├── dense.h     # Dense matrix API
│   ├── unsafe.h    # Unsafe raw struct access
│   └── internal.hpp # Internal C++ wrappers
└── (future)        # Kernel operations will be added here
```

## Design Philosophy

### Two-Layer API

1. **Safe API** (Recommended)
   - Opaque handles: `scl_sparse_t`, `scl_dense_t`
   - Automatic memory management via registry
   - Type-safe at compile time
   - ABI stable across versions

2. **Unsafe API** (Advanced)
   - Direct struct access: `scl_sparse_raw_t`, `scl_dense_raw_t`
   - Zero-overhead operations
   - Manual memory management
   - ABI unstable (may change)

### Key Features

- **Dynamic Dispatch**: Single handle for CSR/CSC formats
- **Registry Integration**: Automatic reference counting
- **Thread-Safe**: Fine-grained locking and thread-local errors
- **Factory Pattern**: Clean separation of create/wrap/clone

## Usage

### Quick Example

```c
#include "scl/binding/c_api/core/sparse.h"

// Create sparse matrix from CSR format
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

// Use matrix...

// Cleanup
scl_sparse_destroy(&matrix);
```

## Documentation

See `docs/cpp/c_api/` for comprehensive documentation:
- [Overview](../../docs/cpp/c_api/overview.md)
- [Quick Start](../../docs/cpp/c_api/README.md)
- [Implementation Summary](../../docs/cpp/c_api/summary.md)

## Migration Status

✅ **Phase 1 Complete**: Core data structures
- Sparse matrix API (CSR/CSC)
- Dense matrix API
- Unsafe raw struct access
- Error handling infrastructure

🚧 **Phase 2 In Progress**: Kernel operations
- Kernel bindings will be added incrementally
- Each kernel module will have its own subdirectory

## Building

The C API is built as part of the main SCL-Core library:

```bash
cmake -B build -DSCL_BUILD_C_API=ON
cmake --build build
```

## Language Bindings

The C API serves as the foundation for bindings to other languages:
- Python (via ctypes/cffi)
- Julia (via ccall)
- R (via .Call)
- C# (via P/Invoke)
- Java (via JNI)
