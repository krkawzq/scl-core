# C API Documentation

This directory contains documentation for the SCL-Core C API bindings.

## Quick Start

### Headers

Include the appropriate headers for your use case:

```c
#include "scl/binding/c_api/core.h"    // Basic types and error handling
#include "scl/binding/c_api/sparse.h"  // Sparse matrix operations
#include "scl/binding/c_api/dense.h"   // Dense matrix operations
#include "scl/binding/c_api/unsafe.h"  // Direct struct access (advanced)
```

### Basic Example

```c
#include "scl/binding/c_api/sparse.h"

int main() {
    // Traditional CSR data
    scl_index_t indptr[] = {0, 2, 3, 5};
    scl_index_t indices[] = {0, 2, 1, 0, 2};
    scl_real_t data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Create sparse matrix
    scl_sparse_t matrix;
    scl_error_t err = scl_sparse_create(
        &matrix, 3, 3, 5,
        indptr, indices, data,
        1  // CSR format
    );
    
    if (err != SCL_OK) {
        fprintf(stderr, "Error: %s\n", scl_get_last_error());
        return 1;
    }
    
    // Use matrix...
    
    // Cleanup
    scl_sparse_destroy(&matrix);
    return 0;
}
```

## Documentation Files

- [**Overview**](overview.md) - Architecture, design principles, and examples
- [**Sparse API**](sparse.md) - Sparse matrix operations reference
- [**Dense API**](dense.md) - Dense matrix operations reference  
- [**Unsafe API**](unsafe.md) - Direct struct access reference
- [**Error Handling**](errors.md) - Error codes and handling guide

## Key Features

### Two-Layer Design

1. **Safe API** (Recommended)
   - Opaque handles
   - Automatic memory management
   - Type safety
   - ABI stable

2. **Unsafe API** (Advanced)
   - Direct struct access
   - Zero overhead
   - Manual memory management
   - ABI unstable

### Dynamic Dispatch

Single handle type supports both formats:
- CSR (Compressed Sparse Row)
- CSC (Compressed Sparse Column)

### Memory Management

- **Registry-based**: All allocations tracked
- **Reference counted**: Safe shared data
- **Thread-safe**: Fine-grained locking
- **Leak detection**: Debug builds report leaks

### Error Handling

- Return codes for all operations
- Thread-local error messages
- Detailed error descriptions

## Language Bindings

The C API is the foundation for bindings to other languages:

- **Python**: Use ctypes or cffi
- **Julia**: Use ccall
- **R**: Use .Call
- **C#**: Use P/Invoke
- **Java**: Use JNI

## Build Integration

### CMake

```cmake
find_package(scl-core REQUIRED)
target_link_libraries(your_target PRIVATE scl::c_api)
```

### Pkg-config

```bash
gcc your_code.c $(pkg-config --cflags --libs scl-core) -o your_program
```

## Performance Notes

### When to Use Safe API

- Default choice for most applications
- Automatic memory management
- Safe interop with other languages
- Minimal overhead for typical workloads

### When to Use Unsafe API

- Performance-critical inner loops
- Custom memory management required
- Direct access to internal layout needed
- Understanding of ABI stability implications

## Support and Issues

For questions and bug reports:
- GitHub Issues: https://github.com/your-org/scl-core/issues
- Documentation: https://scl-core.readthedocs.io

## License

Same as SCL-Core main library.

