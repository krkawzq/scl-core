# C API Implementation Summary

## Files Created

### Public Headers (scl/binding/c_api/)

1. **core.h** - Core types and error handling
   - Opaque handle types: `scl_sparse_t`, `scl_dense_t`
   - Basic value types: `scl_real_t`, `scl_index_t`, `scl_size_t`
   - Error codes and error message API

2. **sparse.h** - Sparse matrix safe API
   - Lifecycle: create, wrap, clone, destroy
   - Properties: rows, cols, nnz, format queries
   - Data export: get pointers, export to buffers
   - Conversions: transpose, to contiguous

3. **dense.h** - Dense matrix safe API
   - Lifecycle: create, wrap, clone, destroy
   - Properties: rows, cols, stride queries
   - Data access: get pointer, export
   - Conversion: to sparse

4. **unsafe.h** - Unsafe raw struct API (ABI unstable)
   - Raw struct layouts: `scl_sparse_raw_t`, `scl_dense_raw_t`
   - Unsafe conversions: handle ↔ raw struct

### Internal Headers

5. **internal.hpp** - C++ implementation details
   - `SparseWrapper`: variant<CSR, CSC> with dynamic dispatch
   - Error state management: thread-local error messages
   - Exception to error code conversion

### Implementation Files

6. **core.cpp** - Error handling implementation
   - Thread-local error state
   - Exception to error code mapping
   - Error message management

7. **sparse_matrix.cpp** - Sparse matrix implementation
   - Factory functions using CSR/CSC::from_traditional
   - Dynamic dispatch via SparseWrapper::visit
   - Registry integration for memory management

8. **dense_matrix.cpp** - Dense matrix implementation
   - Row-major dense matrix operations
   - Conversion to sparse matrices
   - Stride support for non-contiguous data

9. **unsafe.cpp** - Unsafe API implementation
   - Direct struct access without registry
   - Raw pointer conversions
   - ABI-unstable but zero-overhead

### Documentation

10. **docs/cpp/c_api/overview.md** - Comprehensive guide
11. **docs/cpp/c_api/README.md** - Quick start and index

## Design Highlights

### Opaque Handle Pattern

```c
typedef struct scl_sparse_matrix* scl_sparse_t;
```

- Type-safe at compile time
- ABI stable across versions
- Clear ownership semantics

### Dynamic Dispatch

```cpp
template<typename Func>
auto visit(Func&& func) {
    if (is_csr) {
        return func(std::get<CSR>(matrix));
    } else {
        return func(std::get<CSC>(matrix));
    }
}
```

- Single handle for CSR/CSC
- Zero runtime overhead for inline lambdas
- Clean separation of concerns

### Registry Integration

All allocations go through `get_registry()`:
- Automatic reference counting
- Safe cleanup on destroy
- Memory leak detection (debug)
- Thread-safe operations

### Two-Layer Safety

1. **Safe API**: Memory-safe, ABI stable
2. **Unsafe API**: Zero-overhead, ABI unstable

Users choose based on performance vs safety tradeoffs.

## Usage Examples

### Safe API (Recommended)

```c
scl_sparse_t matrix;
scl_sparse_create(&matrix, rows, cols, nnz, indptr, indices, data, 1);
// Use matrix...
scl_sparse_destroy(&matrix);
```

### Unsafe API (Advanced)

```c
scl_sparse_raw_t raw;
scl_sparse_unsafe_get_raw(matrix, &raw);
// Direct access: raw.data_ptrs[i][j]
```

## Next Steps

1. Add language-specific bindings (Python, Julia, R)
2. Expand unsafe API with more operations
3. Add performance benchmarks
4. Write comprehensive test suite
5. Add examples for common use cases

## Notes

- All headers follow .cursorignore rules
- Error handling uses thread-local state
- Dynamic dispatch compiles to efficient code
- Registry handles complex memory scenarios (views, slices)
