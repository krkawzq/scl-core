# Error Handling

SCL-Core exception system with error codes for C-ABI compatibility.

## Overview

Error handling provides:

- **Exception Classes** - Typed exceptions with error codes
- **Error Codes** - Numeric identifiers for C-ABI compatibility
- **Assertions** - Debug assertions and runtime checks
- **Python Mapping** - Error codes map to Python exceptions

## Error Codes

### ErrorCode Enum

Numeric identifiers for exception types, compatible with C-ABI:

```cpp
enum class ErrorCode : int32_t {
    OK = 0,                     // Success
    
    UNKNOWN = 1,                // Generic error
    INTERNAL_ERROR = 2,         // Internal logic failure
    
    INVALID_ARGUMENT = 10,      // Invalid parameter value
    DIMENSION_MISMATCH = 11,    // Dimension incompatibility
    DOMAIN_ERROR = 12,          // Mathematical domain violation
    
    TYPE_ERROR = 20,            // Data type mismatch
    
    IO_ERROR = 30,              // File I/O failure
    
    UNREGISTERED_POINTER = 35,  // Unregistered pointer access
    
    NOT_IMPLEMENTED = 40,       // Feature not implemented
};
```

**Python Mapping:**

| ErrorCode | Python Exception |
|-----------|------------------|
| OK | No exception |
| UNKNOWN | RuntimeError |
| INTERNAL_ERROR | RuntimeError or AssertionError |
| INVALID_ARGUMENT | ValueError |
| DIMENSION_MISMATCH | ValueError |
| DOMAIN_ERROR | ValueError |
| TYPE_ERROR | TypeError |
| IO_ERROR | IOError/OSError |
| UNREGISTERED_POINTER | RuntimeError |
| NOT_IMPLEMENTED | NotImplementedError |

**Invariants:**
- Codes are stable across versions (ABI compatibility)
- OK is always zero
- Related errors are grouped by tens (10-19 for validation, etc.)

## Base Exception Class

### Exception

Base class for all SCL exceptions:

```cpp
#include "scl/core/error.hpp"

class Exception : public std::exception {
public:
    ErrorCode code() const noexcept;
    const char* what() const noexcept override;
};
```

**Members:**
- `_code` [protected] - ErrorCode enum value
- `_msg` [protected] - Human-readable error description

**Methods:**
- `code()` - Returns error code
- `what()` - Returns error message (C++ standard interface)

**Inheritance:**
- Inherits from `std::exception` for standard C++ compatibility

## Typed Exceptions

### InvalidArgumentError

Thrown when function receives invalid parameter value:

```cpp
throw scl::InvalidArgumentError("Index must be non-negative");
```

**Error Code:** `INVALID_ARGUMENT` (10)

**Python Mapping:** `ValueError`

### DimensionMismatchError

Thrown when tensor/matrix dimensions are incompatible:

```cpp
throw scl::DimensionMismatchError(
    "Matrix dimensions must match for multiplication");
```

**Error Code:** `DIMENSION_MISMATCH` (11)

**Python Mapping:** `ValueError`

### DomainError

Thrown for mathematical domain violations:

```cpp
if (x < 0) {
    throw scl::DomainError("Cannot take sqrt of negative number");
}
```

**Error Code:** `DOMAIN_ERROR` (12)

**Python Mapping:** `ValueError`

### TypeError

Thrown for data type mismatches:

```cpp
if (!std::is_same_v<T, Real>) {
    throw scl::TypeError("Expected Real type");
}
```

**Error Code:** `TYPE_ERROR` (20)

**Python Mapping:** `TypeError`

### IOError

Thrown for file I/O failures:

```cpp
if (!file.good()) {
    throw scl::IOError("Failed to read file");
}
```

**Error Code:** `IO_ERROR` (30)

**Python Mapping:** `IOError`/`OSError`

### UnregisteredPointerError

Thrown when accessing unregistered memory pointer:

```cpp
if (!reg.is_registered(ptr)) {
    throw scl::UnregisteredPointerError("Pointer not in registry");
}
```

**Error Code:** `UNREGISTERED_POINTER` (35)

**Python Mapping:** `RuntimeError`

### NotImplementedError

Thrown for unimplemented features:

```cpp
throw scl::NotImplementedError("Feature not yet implemented");
```

**Error Code:** `NOT_IMPLEMENTED` (40)

**Python Mapping:** `NotImplementedError`

### InternalError

Thrown for internal library logic failures:

```cpp
throw scl::InternalError("Unexpected internal state");
```

**Error Code:** `INTERNAL_ERROR` (2)

**Python Mapping:** `RuntimeError` or `AssertionError`

## Assertions and Checks

### SCL_ASSERT

Debug assertion (compiled out in release builds):

```cpp
#include "scl/core/macros.hpp"

SCL_ASSERT(i >= 0 && i < n, "Index out of bounds");
```

**Behavior:**
- Debug builds: Checks condition, throws `InternalError` if false
- Release builds: Compiled out (zero overhead)

**Use Cases:**
- Internal invariants
- Development-time debugging
- Performance-critical code paths

### SCL_CHECK_ARG

Runtime argument validation:

```cpp
SCL_CHECK_ARG(data != nullptr, "Null pointer");
SCL_CHECK_ARG(n > 0, "Size must be positive");
```

**Behavior:**
- Always enabled (even in release builds)
- Throws `InvalidArgumentError` if condition is false

**Use Cases:**
- User input validation
- Public API preconditions
- Safety-critical checks

### SCL_CHECK_DIM

Dimension mismatch check:

```cpp
SCL_CHECK_DIM(output.size() == n, "Size mismatch");
SCL_CHECK_DIM(A.cols() == B.rows(), "Matrix dimensions incompatible");
```

**Behavior:**
- Always enabled
- Throws `DimensionMismatchError` if condition is false

**Use Cases:**
- Matrix operation dimension checks
- Array size validation
- Tensor shape compatibility

### SCL_CHECK_DOMAIN

Mathematical domain check:

```cpp
SCL_CHECK_DOMAIN(x >= 0, "Cannot take sqrt of negative");
SCL_CHECK_DOMAIN(prob >= 0 && prob <= 1, "Probability out of range");
```

**Behavior:**
- Always enabled
- Throws `DomainError` if condition is false

**Use Cases:**
- Mathematical function domain validation
- Probability/statistics range checks
- Numerical stability checks

## Error Handling Patterns

### Exception Safety

```cpp
void process_data(Array<Real> data) {
    SCL_CHECK_ARG(data.ptr != nullptr, "Null data pointer");
    
    try {
        // Process data...
    } catch (const scl::Exception& e) {
        // Handle SCL exception
        std::cerr << "SCL Error: " << e.what() << "\n";
        throw;  // Re-throw
    } catch (const std::exception& e) {
        // Handle other exceptions
        throw scl::InternalError("Unexpected exception");
    }
}
```

### Error Code Extraction

```cpp
try {
    some_operation();
} catch (const scl::Exception& e) {
    ErrorCode code = e.code();
    
    if (code == ErrorCode::INVALID_ARGUMENT) {
        // Handle invalid argument
    } else if (code == ErrorCode::DIMENSION_MISMATCH) {
        // Handle dimension mismatch
    }
}
```

## C-ABI Compatibility

Error codes enable translation between C++ exceptions and Python exceptions:

```cpp
// C++ side
try {
    scl_function();
} catch (const scl::Exception& e) {
    return e.code();  // Return error code to C-ABI
}

// Python binding side (pseudocode)
error_code = c_function()
if error_code != 0:
    raise map_error_code_to_python_exception(error_code)
```

---

::: tip Debug vs Release
Use `SCL_ASSERT` for internal invariants (compiled out in release). Use `SCL_CHECK_*` for user input validation (always enabled for safety).
:::

