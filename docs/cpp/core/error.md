# error.hpp

> scl/core/error.hpp · SCL Core Exception System

## Overview

This file provides the exception system for SCL-Core, with error codes compatible with C-ABI for Python integration. All exceptions inherit from the base Exception class.

Key features:
- Unified exception interface with error codes
- C-ABI compatible (for Python bindings)
- Human-readable error messages
- Type-safe exception hierarchy

**Header**: `#include "scl/core/error.hpp"`

---

## Main APIs

### Exception

Base class for all SCL exceptions.

::: source_code file="scl/core/error.hpp" symbol="Exception" collapsed
:::

**Algorithm Description**

Base exception class that provides:
- Error code (ErrorCode enum) for C-ABI compatibility
- Human-readable message string
- Standard exception interface (inherits from std::exception)

All SCL exceptions inherit from this class.

**Edge Cases**

- **Empty message**: Valid but not informative
- **Invalid error code**: Undefined behavior

**Data Guarantees (Preconditions)**

- Error code must be valid ErrorCode value
- Message should be descriptive

**Complexity Analysis**

- **Time**: O(1) for construction
- **Space**: O(n) where n is message length

**Example**

```cpp
#include "scl/core/error.hpp"

throw scl::DimensionError("Matrix dimensions must match", 
                         rows, expected_rows);
```

---

### ErrorCode

Numeric identifiers for exception types.

```cpp
enum class ErrorCode {
    OK = 0,
    UNKNOWN = 1,
    INTERNAL_ERROR = 2,
    INVALID_ARGUMENT = 10,
    DIMENSION_MISMATCH = 11,
    DOMAIN_ERROR = 12,
    TYPE_ERROR = 20,
    IO_ERROR = 30,
    UNREGISTERED_POINTER = 35,
    NOT_IMPLEMENTED = 40
};
```

---

## Exception Types

### DimensionError

Thrown for dimension mismatches.

### ValueError

Thrown for invalid argument values.

### DomainError

Thrown for mathematical domain violations.

### TypeError

Thrown for data type mismatches.

## See Also

- [Macros](./macros) - Assertion macros (SCL_ASSERT, SCL_CHECK_*)
