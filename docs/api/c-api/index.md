# C API Reference

The SCL-Core C API provides a stable, high-performance interface for direct system integration and language bindings.

## Design Principles

### Stable C-ABI
All public C functions follow a strict ABI stability guarantee:
- No breaking changes within major versions
- Explicit versioning for all interfaces
- Clear deprecation paths for legacy functions

### Zero-Overhead Performance
- Direct access to optimized kernels
- No Python interpreter overhead
- Minimal abstraction layers
- SIMD and cache-optimized implementations

### Memory Management
- Manual memory control with clear ownership semantics
- Allocate/deallocate pairs for all resources
- Error codes for all operations
- No hidden allocations

### Thread Safety
- All functions document thread-safety guarantees
- Explicit synchronization requirements
- Parallel execution where applicable

---

## API Organization

### Core
Fundamental types, error handling, and configuration:
- [Types](/api/c-api/core/types) - Basic data types and structures
- [Error Handling](/api/c-api/core/error) - Error codes and diagnostics
- [Sparse Matrices](/api/c-api/core/sparse) - CSR/CSC matrix types
- [Memory](/api/c-api/memory) - Allocation and lifetime management

### Kernels
High-performance computational kernels:
- [Normalize](/api/c-api/kernels/normalize) - Matrix normalization
- Neighbors (coming soon) - K-nearest neighbors
- Algebra (coming soon) - Matrix operations
- Statistics (coming soon) - Statistical tests

---

## Quick Start

### Basic Usage Pattern

```c
#include <scl/capi.h>

int main() {
    // 1. Initialize library
    scl_error_t err = scl_init();
    if (err != SCL_SUCCESS) {
        fprintf(stderr, "Init failed: %s\n", scl_error_message(err));
        return 1;
    }
    
    // 2. Create sparse matrix
    scl_sparse_matrix_t* matrix = NULL;
    err = scl_sparse_matrix_create(&matrix, n_rows, n_cols, nnz);
    if (err != SCL_SUCCESS) { /* handle error */ }
    
    // 3. Populate matrix data
    // ... set values, indices, indptr ...
    
    // 4. Call kernel
    err = scl_normalize_rows(matrix, SCL_NORM_L2, 1e-12);
    if (err != SCL_SUCCESS) { /* handle error */ }
    
    // 5. Clean up
    scl_sparse_matrix_destroy(matrix);
    scl_finalize();
    
    return 0;
}
```

### Error Handling

All C API functions return `scl_error_t`. Always check return values:

```c
scl_error_t err = scl_function(...);
if (err != SCL_SUCCESS) {
    const char* msg = scl_error_message(err);
    const char* detail = scl_error_detail();
    fprintf(stderr, "Error: %s - %s\n", msg, detail);
    // Clean up and return
}
```

### Memory Ownership

Functions follow clear naming conventions for ownership:
- `_create` - Caller must call `_destroy`
- `_borrow` - No ownership transfer, do not destroy
- `_transfer` - Ownership transferred to callee
- `_copy` - Creates new owned copy

---

## Compilation

### Header Files
```bash
# Single header for all functionality
#include <scl/capi.h>
```

### Linking
```bash
# Dynamic linking
gcc -o app app.c -lscl

# Static linking
gcc -o app app.c -lscl -static
```

### CMake Integration
```cmake
find_package(SCL REQUIRED)
target_link_libraries(your_target PRIVATE SCL::scl)
```

---

## Conventions

### Naming
- Types: `scl_typename_t`
- Functions: `scl_module_action()`
- Constants: `SCL_CONSTANT_NAME`
- Enums: `SCL_ENUM_VALUE`

### Return Values
- `scl_error_t` for status (0 = success)
- Output parameters via pointers
- NULL for optional parameters

### Threading
- Functions annotated with thread-safety level
- Most kernels are thread-safe
- Explicit locking for shared state

---

## Next Steps

1. Read [Core Types](/api/c-api/core/types) to understand fundamental structures
2. Review [Error Handling](/api/c-api/core/error) for robust error management
3. Explore [Kernels](/api/c-api/kernels/) for computational functionality
4. Check [Memory Management](/api/c-api/memory) for advanced usage patterns

