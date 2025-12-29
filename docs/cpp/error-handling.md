---
title: Error Handling
description: Exception system, error codes, and validation macros
---

# Error Handling

SCL-Core provides a comprehensive error handling system with typed exceptions, error codes, and validation macros for robust error reporting.

## Exception Hierarchy

### Base Exception

```cpp
namespace scl {
    class Exception : public std::exception {
    public:
        explicit Exception(ErrorCode code, std::string msg);
        
        const char* what() const noexcept override;
        ErrorCode code() const noexcept;
        const std::string& message() const noexcept;
    };
}
```

### Exception Types

```cpp
// Runtime errors
class RuntimeError : public Exception;
class OutOfMemoryError : public RuntimeError;
class NullPointerError : public RuntimeError;
class InternalError : public RuntimeError;

// Value errors
class ValueError : public Exception;
class DimensionError : public ValueError;
class DomainError : public ValueError;
class RangeError : public ValueError;
class IndexOutOfBoundsError : public ValueError;

// Type errors
class TypeError : public Exception;
class TypeMismatchError : public TypeError;

// I/O errors
class IOError : public Exception;
class FileNotFoundError : public IOError;
class PermissionDeniedError : public IOError;
class ReadError : public IOError;
class WriteError : public IOError;

// Feature errors
class NotImplementedError : public Exception;
class FeatureUnavailableError : public Exception;

// Numerical errors
class NumericalError : public Exception;
class DivisionByZeroError : public NumericalError;
class OverflowError : public NumericalError;
class UnderflowError : public NumericalError;
class ConvergenceError : public NumericalError;
```

## Error Codes

```cpp
enum class ErrorCode : std::int32_t {
    OK = 0,
    
    // General errors
    UNKNOWN = 1,
    INTERNAL_ERROR = 2,
    OUT_OF_MEMORY = 3,
    NULL_POINTER = 4,
    
    // Argument errors
    INVALID_ARGUMENT = 10,
    DIMENSION_MISMATCH = 11,
    DOMAIN_ERROR = 12,
    RANGE_ERROR = 13,
    INDEX_OUT_OF_BOUNDS = 14,
    
    // Type errors
    TYPE_ERROR = 20,
    TYPE_MISMATCH = 21,
    
    // I/O errors
    IO_ERROR = 30,
    FILE_NOT_FOUND = 31,
    PERMISSION_DENIED = 32,
    READ_ERROR = 33,
    WRITE_ERROR = 34,
    
    // Registry errors
    UNREGISTERED_POINTER = 35,
    BUFFER_NOT_FOUND = 36,
    
    // Feature errors
    NOT_IMPLEMENTED = 40,
    FEATURE_UNAVAILABLE = 41,
    
    // Numerical errors
    NUMERICAL_ERROR = 50,
    DIVISION_BY_ZERO = 51,
    OVERFLOW = 52,
    UNDERFLOW = 53,
    CONVERGENCE_ERROR = 54,
};
```

## Validation Macros

### SCL_ASSERT

For internal invariants (active in all builds):

```cpp
SCL_ASSERT(condition, "Error message");
```

**Usage**:
```cpp
void process(Array<Real> data) {
    SCL_ASSERT(data.ptr != nullptr, "Data pointer is null");
    SCL_ASSERT(data.len > 0, "Data is empty");
    // Process data
}
```

### SCL_CHECK_ARG

For user input validation:

```cpp
SCL_CHECK_ARG(condition, "Error message");
```

**Usage**:
```cpp
void normalize(Array<Real> data, Real target_sum) {
    SCL_CHECK_ARG(data.ptr != nullptr, "Data pointer cannot be null");
    SCL_CHECK_ARG(target_sum > 0, "Target sum must be positive");
    // Normalize data
}
```

### SCL_CHECK_DIM

For dimension mismatches:

```cpp
SCL_CHECK_DIM(condition, "Error message");
```

**Usage**:
```cpp
void add_matrices(const CSR& a, const CSR& b, CSR& result) {
    SCL_CHECK_DIM(a.rows() == b.rows(), "Row dimension mismatch");
    SCL_CHECK_DIM(a.cols() == b.cols(), "Column dimension mismatch");
    // Add matrices
}
```

### SCL_CHECK_NULL

For null pointer validation:

```cpp
SCL_CHECK_NULL(ptr, "Error message");
```

**Usage**:
```cpp
void process(Real* data, Size n) {
    SCL_CHECK_NULL(data, "Data pointer cannot be null");
    // Process data
}
```

### SCL_CHECK_BOUNDS

For index bounds checking:

```cpp
SCL_CHECK_BOUNDS(index, size, "Error message");
```

**Usage**:
```cpp
Real get_element(Array<Real> data, Index idx) {
    SCL_CHECK_BOUNDS(idx, data.size(), "Index out of bounds");
    return data[idx];
}
```

### SCL_CHECK_RANGE

For value range validation:

```cpp
SCL_CHECK_RANGE(value, min_val, max_val, "Error message");
```

**Usage**:
```cpp
void set_alpha(Real alpha) {
    SCL_CHECK_RANGE(alpha, Real(0), Real(1), "Alpha must be in [0, 1]");
    // Set alpha
}
```

## Error Handling Patterns

### Basic Exception Handling

```cpp
try {
    auto matrix = CSR::create(rows, cols, nnz);
    kernel::normalize::normalize_rows_inplace(matrix, target_sum);
} catch (const scl::ValueError& e) {
    std::cerr << "Value error: " << e.what() << std::endl;
    // Handle error
} catch (const scl::DimensionError& e) {
    std::cerr << "Dimension error: " << e.what() << std::endl;
    // Handle error
} catch (const scl::Exception& e) {
    std::cerr << "SCL error (" << static_cast<int>(e.code()) << "): " << e.what() << std::endl;
    // Handle error
} catch (const std::exception& e) {
    std::cerr << "Standard error: " << e.what() << std::endl;
    // Handle error
}
```

### Error Code Checking

```cpp
try {
    process_data(data);
} catch (const scl::Exception& e) {
    ErrorCode code = e.code();
    switch (code) {
        case ErrorCode::DIMENSION_MISMATCH:
            // Handle dimension error
            break;
        case ErrorCode::OUT_OF_MEMORY:
            // Handle memory error
            break;
        default:
            // Handle other errors
            break;
    }
}
```

### Custom Error Messages

```cpp
void validate_input(Array<Real> data, Real threshold) {
    if (data.empty()) {
        throw scl::ValueError("Input data cannot be empty");
    }
    if (threshold <= 0) {
        throw scl::ValueError("Threshold must be positive, got " + std::to_string(threshold));
    }
    // Validate data
}
```

## Best Practices

### 1. Use Appropriate Exception Types

```cpp
// Good: Specific exception type
void process(Array<Real> data) {
    if (data.empty()) {
        throw scl::ValueError("Data cannot be empty");
    }
    // Process
}

// Avoid: Generic exception
void process(Array<Real> data) {
    if (data.empty()) {
        throw std::runtime_error("Error");  // Less informative
    }
}
```

### 2. Validate Early

```cpp
// Good: Validate at function entry
void compute(Array<Real> input, Array<Real> output) {
    SCL_CHECK_ARG(input.ptr != nullptr, "Input cannot be null");
    SCL_CHECK_ARG(output.ptr != nullptr, "Output cannot be null");
    SCL_CHECK_DIM(input.size() == output.size(), "Size mismatch");
    
    // Process with confidence
    for (Size i = 0; i < input.size(); ++i) {
        output[i] = compute(input[i]);
    }
}
```

### 3. Provide Context in Error Messages

```cpp
// Good: Informative error message
void set_dimension(Index dim) {
    SCL_CHECK_ARG(dim > 0, "Dimension must be positive, got " + std::to_string(dim));
    // Set dimension
}

// Avoid: Generic error message
void set_dimension(Index dim) {
    SCL_CHECK_ARG(dim > 0, "Invalid dimension");  // Less helpful
}
```

### 4. Use Macros for Consistency

```cpp
// Good: Use validation macros
void process(Array<Real> data, Index idx) {
    SCL_CHECK_NULL(data.ptr, "Data pointer is null");
    SCL_CHECK_BOUNDS(idx, data.size(), "Index out of bounds");
    // Process
}

// Avoid: Manual checks
void process(Array<Real> data, Index idx) {
    if (data.ptr == nullptr) {
        throw scl::NullPointerError("Data pointer is null");
    }
    if (idx < 0 || static_cast<Size>(idx) >= data.size()) {
        throw scl::IndexOutOfBoundsError("Index out of bounds");
    }
    // Process
}
```

### 5. Handle Exceptions at Appropriate Levels

```cpp
// Low-level: Throw exceptions
void kernel_operation(Array<Real> data) {
    SCL_CHECK_ARG(data.ptr != nullptr, "Data cannot be null");
    // Perform operation, throw on error
}

// High-level: Catch and handle
void user_function(Array<Real> data) {
    try {
        kernel_operation(data);
    } catch (const scl::Exception& e) {
        // Log error, provide user-friendly message
        log_error(e);
        throw;  // Re-throw or handle
    }
}
```

## Thread Safety

Exceptions are thread-safe:

```cpp
// Safe: Each thread throws independently
threading::parallel_for(0, n, [&](size_t i) {
    try {
        process(data[i]);
    } catch (const scl::Exception& e) {
        // Handle error in this thread
        handle_error(i, e);
    }
});
```

## Performance Considerations

### Exception Overhead

Exceptions have minimal overhead when not thrown:

```cpp
// No overhead if condition is true
SCL_CHECK_ARG(ptr != nullptr, "Pointer is null");
// Fast path: no exception thrown
```

### Debug vs Release

Assertions are active in all builds, but bounds checking can be disabled:

```cpp
// In release builds (NDEBUG defined), bounds checking is disabled
Array<Real> data = {ptr, n};
Real val = data[i];  // No bounds check in release
```

## Related Documentation

- [Core Types](./core/types.md) - Type system
- [Threading](./threading.md) - Thread-safe error handling
- [Kernels](./kernels/) - Kernel error handling

